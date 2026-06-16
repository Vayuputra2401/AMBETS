"""
train.py — CCR-Net Phase 2 training entry point.

Usage
-----
    python pipeline/train.py --env local          # paths from configs/env/local.yaml
    python pipeline/train.py --env gcp            # paths from configs/env/gcp.yaml
    python pipeline/train.py --env local --smoke_test  # 2-batch pipeline check
    python pipeline/train.py --env gcp  --resume /gcs/ccr-brats-training/checkpoints/epoch_0050.pth

Paths are loaded from configs/env/{env}.yaml — edit that file, not this script.
Base hyperparameters live in configs/brats_phase2.yaml (never env-specific).

CLI args
--------
    --env         environment name — loads configs/env/{env}.yaml (default: local)
    --config      path to base YAML (default: configs/brats_phase2.yaml, auto-resolved)
    --resume      path to .pth checkpoint to resume from
    --smoke_test  run 2 batches then exit
    --device      cuda / cpu (default: auto-detect)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.ccrnet import CCRNet
from ccrnet.utils.token_labels import downsample_labels_to_tokens
from ccrnet.utils.checkpoint import save_checkpoint, load_checkpoint, reinitialize_experts, sync_checkpoint_to_gcs
from brats_dataset import get_dataloader

from ccr.losses.total import CCRTotalLoss
from ccr.utils.metrics import ConceptAlignmentScore, ExpertUtilizationTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def validate(
    model: CCRNet,
    val_loader,
    cas: ConceptAlignmentScore,
    device: torch.device,
) -> dict:
    model.eval()
    cas.reset()
    total_dice = {k: 0.0 for k in ["WT", "TC", "ET"]}
    n_batches = 0

    for batch in val_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        out = model(image)
        token_labels = downsample_labels_to_tokens(label, model.get_grid_shape())
        cas.update(out["routing_probs"], token_labels)

        # Hard-argmax segmentation for Dice
        pred = out["seg_logits"].argmax(dim=1)   # [B, H, W, D]
        total_dice["WT"] += _dice_wt(pred, label)
        total_dice["TC"] += _dice_tc(pred, label)
        total_dice["ET"] += _dice_et(pred, label)
        n_batches += 1

    cas_fg = cas.compute_fg()
    dice   = {k: v / max(n_batches, 1) for k, v in total_dice.items()}
    return {"cas_fg": cas_fg, "dice": dice}


def _dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    p = pred_mask.float().reshape(-1)
    g = gt_mask.float().reshape(-1)
    intersection = (p * g).sum()
    return (2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth)


def _dice_wt(pred, label):
    return _dice((pred > 0), (label > 0)).item()


def _dice_tc(pred, label):
    return _dice(pred.eq(1) | pred.eq(3), label.eq(1) | label.eq(3)).item()


def _dice_et(pred, label):
    return _dice(pred.eq(3), label.eq(3)).item()


def _log_metrics(epoch: int, losses: dict, val_metrics: dict, utilization: dict) -> None:
    cas_fg = val_metrics.get("cas_fg", {})
    dice   = val_metrics.get("dice", {})
    _log(
        f"Epoch {epoch:3d} | "
        f"phase={losses['phase']} τ={losses.get('tau_target', 0):.2f} | "
        f"L={losses['total'].item():.4f} seg={losses['seg'].item():.4f} "
        f"align={losses['align'].item():.4f} | "
        f"CAS_fg ncr={cas_fg.get('necrotic_core', 0):.3f} "
        f"edema={cas_fg.get('edema', 0):.3f} "
        f"et={cas_fg.get('enhancing_tumor', 0):.3f} | "
        f"Dice WT={dice.get('WT', 0):.3f} TC={dice.get('TC', 0):.3f} ET={dice.get('ET', 0):.3f} | "
        f"util={utilization}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")

    parser = argparse.ArgumentParser(description="CCR-Net Phase 2 training")
    parser.add_argument("--env",    type=str, default="local",
                        help="Environment: 'local' or 'gcp' (loads configs/env/{env}.yaml)")
    parser.add_argument("--config", type=str, default=_default_config,
                        help="Path to base YAML config (default: configs/brats_phase2.yaml)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint .pth to resume training from")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 2 batches then exit (pipeline verification)")
    parser.add_argument("--device", type=str, default="",
                        help="Force device: cuda or cpu (default: auto-detect)")
    args = parser.parse_args()

    # --- Config: base YAML merged with env-specific paths ---
    config  = Phase2Config.from_yaml(args.config)
    env_cfg = apply_env(config, env=args.env,
                        config_dir=Path(args.config).parent)

    _log(f"Environment : {args.env}")
    if args.env == "gcp" and "gcp" in env_cfg:
        g = env_cfg["gcp"]
        _log(f"GCP project={g.get('project')}  bucket={g.get('bucket')}  "
             f"instance={g.get('instance')}  zone={g.get('zone')}")
    gcs_checkpoint_dir = env_cfg.get("paths", {}).get("gcs_checkpoint_dir", "")

    _log(f"data_root   : {config.data.data_root}")
    _log(f"checkpoint  : {config.training.checkpoint_dir}")
    if gcs_checkpoint_dir:
        _log(f"gcs sync    : {gcs_checkpoint_dir}")
    _log(f"pretrained  : {config.swin.pretrained_path}")

    if not config.data.data_root:
        parser.error("data_root is empty — set it in configs/env/{env}.yaml")

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # --- Model ---
    model = CCRNet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    _log(f"CCRNet parameters: {total_params:,}")

    # --- Loss, optimizer, scheduler ---
    loss_fn   = CCRTotalLoss(config.ccr)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.ccr.curriculum.total_epochs,
        eta_min=1e-6,
    )
    scaler = GradScaler(enabled=config.training.amp and device.type == "cuda")

    # --- Resume ---
    start_epoch = 1
    best_metrics: dict = {}
    if args.resume:
        start_epoch, best_metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        _log(f"Resumed from {args.resume}, starting epoch {start_epoch}")

    # --- Data ---
    _log("Building dataloaders ...")
    train_loader = get_dataloader(
        data_root    = config.data.data_root,
        split        = "train",
        batch_size   = config.training.batch_size,
        num_workers  = config.data.num_workers,
        spatial_size = config.data.spatial_size,
        brats_version= config.data.brats_version,
        val_fraction = config.data.val_fraction,
        test_fraction= config.data.test_fraction,
        seed         = config.data.seed,
    )
    val_loader = get_dataloader(
        data_root    = config.data.data_root,
        split        = "val",
        batch_size   = 1,
        num_workers  = config.data.num_workers,
        spatial_size = config.data.spatial_size,
        brats_version= config.data.brats_version,
        val_fraction = config.data.val_fraction,
        test_fraction= config.data.test_fraction,
        seed         = config.data.seed,
    )
    _log(f"Train: {len(train_loader.dataset)} patients | Val: {len(val_loader.dataset)} patients")

    # --- Metrics / Tracking ---
    concept_names = tuple(config.ccr.concept_names)
    cas     = ConceptAlignmentScore(config.ccr.router.num_concepts, concept_names)
    tracker = ExpertUtilizationTracker(
        num_concepts     = config.ccr.router.num_concepts,
        concept_names    = concept_names,
        warmup_end_epoch = config.ccr.curriculum.warmup_end_epoch,
        min_utilization_pct = 5.0,
    )

    total_epochs = config.ccr.curriculum.total_epochs

    # --- Training loop ---
    for epoch in range(start_epoch, total_epochs + 1):
        loss_fn.set_epoch(epoch)
        model.train()

        epoch_losses: dict = {}
        for batch_idx, batch in enumerate(train_loader):
            if args.smoke_test and batch_idx >= 2:
                break

            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=config.training.amp and device.type == "cuda"):
                out          = model(image)
                token_labels = downsample_labels_to_tokens(label, model.get_grid_shape())
                losses       = loss_fn(
                    pred_logits   = out["seg_logits"],
                    labels        = label,
                    routing_probs = out["routing_probs"],
                    token_labels  = token_labels,
                    tau_current   = model.ccr.router.temperature,
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tracker.update(out["assignments"])
            epoch_losses = losses

            if batch_idx % config.training.log_every == 0:
                _log(
                    f"  epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                    f"loss={losses['total'].item():.4f} "
                    f"phase={losses['phase']}"
                )

        # --- Collapse check ---
        collapsed = tracker.check_collapse(epoch)
        if collapsed:
            _log(f"  Expert collapse detected: {collapsed} — reinitializing fc2")
            reinitialize_experts(model.ccr, collapsed)

        # Snapshot utilization BEFORE reset so _log_metrics can report it
        epoch_utilization = tracker.compute()
        tracker.reset()

        scheduler.step()

        # --- Validation + checkpoint ---
        if epoch % config.training.checkpoint_every == 0 or args.smoke_test:
            val_metrics = validate(model, val_loader, cas, device)
            _log_metrics(epoch, epoch_losses, val_metrics, epoch_utilization)
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, config.training.checkpoint_dir,
            )
            _log(f"  Checkpoint saved: {ckpt_path}")
            if gcs_checkpoint_dir:
                if sync_checkpoint_to_gcs(ckpt_path, gcs_checkpoint_dir):
                    _log(f"  Synced to {gcs_checkpoint_dir}/")

        if args.smoke_test:
            _log("Smoke test complete — 2 batches ran successfully.")
            break

    _log("Training complete.")


if __name__ == "__main__":
    main()
