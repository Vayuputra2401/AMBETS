"""
train.py — CCR-Net Phase 2 training entry point.

Usage
-----
    python pipeline/train.py --env local          # paths from configs/env/local.yaml
    python pipeline/train.py --env gcp            # paths from configs/env/gcp.yaml
    python pipeline/train.py --env local --smoke_test  # 2-batch pipeline check
    python pipeline/train.py --env gcp  --epochs 2     # 2 full real epochs, train+val tracked
    python pipeline/train.py --env gcp  --resume /gcs/ccr-brats-training/checkpoints/epoch_0050.pth

Paths are loaded from configs/env/{env}.yaml — edit that file, not this script.
Base hyperparameters live in configs/brats_phase2.yaml (never env-specific).

CLI args
--------
    --env         environment name — loads configs/env/{env}.yaml (default: local)
    --config      path to base YAML (default: configs/brats_phase2.yaml, auto-resolved)
    --resume      path to .pth checkpoint to resume from
    --smoke_test  run 2 batches then exit
    --epochs      run only this many full epochs then stop (train+val every epoch,
                   metrics written to <run_dir>/metrics.json). 0 = full curriculum.
    --device      cuda / cpu (default: auto-detect)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from tqdm import tqdm

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

    for batch in tqdm(val_loader, desc="  validating", leave=False):
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


_CONCEPT_SHORT = {
    "background":       "bg",
    "necrotic_core":     "ncr",
    "edema":             "edema",
    "enhancing_tumor":   "et",
}


def _kv(d: dict, fmt: str = ".4f") -> str:
    return " ".join(f"{k}={v:{fmt}}" for k, v in d.items())


def _log_train_metrics(epoch: int, phase: str, tau_target: float, train_avg: dict, train_time: float) -> None:
    _log(
        f"[epoch {epoch:3d}] train | phase={phase} tau={tau_target:.2f} | "
        f"{_kv(train_avg)} | time={train_time:.1f}s"
    )


def _log_val_metrics(epoch: int, val_metrics: dict, utilization: dict, val_time: float) -> None:
    cas_fg = {f"cas_fg_{_CONCEPT_SHORT.get(k, k)}": v for k, v in val_metrics.get("cas_fg", {}).items()}
    dice   = {f"dice_{k.lower()}": v for k, v in val_metrics.get("dice", {}).items()}
    util   = {f"util_{_CONCEPT_SHORT.get(k, k)}": v * 100.0 for k, v in utilization.items()}
    _log(
        f"[epoch {epoch:3d}] val   | "
        f"{_kv(cas_fg, '.3f')} {_kv(dice, '.3f')} {_kv(util, '.1f')} | time={val_time:.1f}s"
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
    parser.add_argument("--epochs", type=int, default=0,
                        help="Run only this many full epochs then stop (train+val "
                             "tracked every epoch, written to <run_dir>/metrics.json). "
                             "0 = run the full curriculum (config total_epochs).")
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

    # --- Run folder: one dated sub-directory per training run, so checkpoints
    # from different launches never mix. Resuming reuses the run folder the
    # checkpoint came from instead of starting a new one.
    if args.resume:
        run_dir = os.path.dirname(os.path.abspath(args.resume))
        run_id  = os.path.basename(run_dir)
    else:
        run_id  = "smoke_test" if args.smoke_test else time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.training.checkpoint_dir, run_id)
    gcs_run_dir = f"{gcs_checkpoint_dir.rstrip('/')}/{run_id}" if gcs_checkpoint_dir else ""

    _log(f"data_root   : {config.data.data_root}")
    _log(f"run id      : {run_id}")
    _log(f"checkpoint  : {run_dir}")
    if gcs_run_dir:
        _log(f"gcs sync    : {gcs_run_dir}")
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
    scaler = GradScaler("cuda", enabled=config.training.amp and device.type == "cuda")

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
    end_epoch = min(total_epochs, start_epoch + args.epochs - 1) if args.epochs > 0 else total_epochs
    force_validate_every_epoch = args.smoke_test or args.epochs > 0

    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")
    all_epoch_metrics: list = []

    # --- Training loop ---
    epoch_bar = tqdm(range(start_epoch, end_epoch + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.time()
        loss_fn.set_epoch(epoch)
        model.train()

        loss_sums = {k: 0.0 for k in ["total", "seg", "align", "diversity", "entropy", "boundary", "contrastive", "tau_reg"]}
        n_train_batches = 0
        epoch_losses: dict = {}
        accum_steps = max(config.training.grad_accum_steps, 1)
        accum_count = 0
        optimizer.zero_grad()
        batch_bar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"  epoch {epoch}", unit="batch", leave=False,
        )
        for batch_idx, batch in batch_bar:
            if args.smoke_test and batch_idx >= 2:
                break

            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with autocast("cuda", enabled=config.training.amp and device.type == "cuda"):
                out          = model(image)
                token_labels = downsample_labels_to_tokens(label, model.get_grid_shape())
                losses       = loss_fn(
                    pred_logits   = out["seg_logits"],
                    labels        = label,
                    routing_probs = out["routing_probs"],
                    token_labels  = token_labels,
                    tau_current   = model.ccr.router.temperature,
                )

            scaler.scale(losses["total"] / accum_steps).backward()
            accum_count += 1

            is_last_batch = (batch_idx + 1) == len(train_loader)
            if accum_count == accum_steps or is_last_batch:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

            tracker.update(out["assignments"])
            epoch_losses = losses
            n_train_batches += 1
            for k in loss_sums:
                loss_sums[k] += losses[k].item()

            batch_bar.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                phase=losses["phase"],
                bs=config.training.batch_size,
                accum=accum_steps,
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        # Flush leftover accumulated gradients (e.g. smoke_test break mid-window)
        if accum_count > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_avg = {k: v / max(n_train_batches, 1) for k, v in loss_sums.items()}
        train_time = time.time() - epoch_start
        _log_train_metrics(epoch, epoch_losses["phase"], epoch_losses["tau_target"], train_avg, train_time)

        # --- Collapse check ---
        collapsed = tracker.check_collapse(epoch)
        if collapsed:
            _log(f"  Expert collapse detected: {collapsed} — reinitializing fc2")
            reinitialize_experts(model.ccr, collapsed)

        # Snapshot utilization BEFORE reset so logging/metrics can report it
        epoch_utilization = tracker.compute()
        tracker.reset()

        scheduler.step()

        # --- Validation + checkpoint ---
        val_metrics = None
        val_time = 0.0
        if epoch % config.training.checkpoint_every == 0 or force_validate_every_epoch:
            val_start   = time.time()
            val_metrics = validate(model, val_loader, cas, device)
            val_time    = time.time() - val_start
            _log_val_metrics(epoch, val_metrics, epoch_utilization, val_time)
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, run_dir,
            )
            _log(f"  Checkpoint saved: {ckpt_path}")
            if gcs_run_dir:
                if sync_checkpoint_to_gcs(ckpt_path, gcs_run_dir):
                    _log(f"  Synced to {gcs_run_dir}/")

        epoch_total_time = time.time() - epoch_start
        _log(f"[epoch {epoch:3d}] done  | sec/epoch={epoch_total_time:.1f}s ({epoch_total_time/60:.1f}min)")

        # --- Per-epoch metrics record (always written; val included only when run) ---
        epoch_record = {
            "epoch":                epoch,
            "phase":                epoch_losses.get("phase"),
            "tau_target":           epoch_losses.get("tau_target"),
            "lr":                   optimizer.param_groups[0]["lr"],
            "batch_size":           config.training.batch_size,
            "grad_accum_steps":     accum_steps,
            "effective_batch_size": config.training.batch_size * accum_steps,
            "n_train_batches":      n_train_batches,
            "train_loss":           train_avg,
            "val":                  val_metrics,
            "expert_utilization":   epoch_utilization,
            "train_time_sec":       round(train_time, 1),
            "val_time_sec":         round(val_time, 1),
            "total_time_sec":       round(epoch_total_time, 1),
            "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        all_epoch_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_epoch_metrics, f, indent=2, default=str)

        # Mirror metrics.json to GCS every epoch (tiny file). Keeps the training
        # curve current without a manual pull and survives instance loss. Syncs
        # whether or not this was a checkpoint epoch, since metrics are written
        # every epoch. Failures are logged, never fatal.
        if gcs_run_dir:
            sync_checkpoint_to_gcs(metrics_path, gcs_run_dir)

        postfix = {"train_loss": f"{train_avg['total']:.4f}"}
        if val_metrics is not None:
            postfix["val_dice_wt"] = f"{val_metrics['dice'].get('WT', 0):.3f}"
        epoch_bar.set_postfix(postfix)

        if args.smoke_test:
            _log("Smoke test complete — 2 batches ran successfully.")
            break

    _log(f"Training complete. Per-epoch metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
