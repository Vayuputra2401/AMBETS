"""
conftest.py — Shared pytest fixtures for all CCR Phase 1 tests.

Dimensional constants are chosen to reflect realistic BraTS inference:
  B = 2    batch size
  N = 4096 token count (16×16×16 spatial grid with patch_size=8, input 128³)
  D = 192  Swin-B embed_dim
  K = 4    BraTS concepts: Background, NCR, Edema, ET

Spatial fixtures use 128³ volumes (not 240×240×155) so tests finish in < 10 s
on CPU.  The math is identical; only the volume size differs from full BraTS.
"""

import sys
import os

# Ensure src/ is on the Python path so "import ccr" resolves correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from ccr.config.ccr_config import CCRConfig, RouterConfig, ExpertConfig
from ccr.modules.router     import ClinicalConceptRouter
from ccr.modules.expert     import ClinicalConceptExpert
from ccr.modules.dispatcher import CCRBottleneckModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH      = 2
N_TOKENS   = 4096    # 16×16×16 = 4096 tokens for 128³ volume with patch_size=8
EMBED_DIM  = 192
K          = 4
H = W = D  = 128     # Spatial dimensions of the volume


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_config() -> CCRConfig:
    """Default BraTS CCR configuration (K=4, embed_dim=192)."""
    return CCRConfig()


@pytest.fixture(scope="session")
def router_config() -> RouterConfig:
    return RouterConfig()


@pytest.fixture(scope="session")
def expert_config() -> ExpertConfig:
    return ExpertConfig()


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router(default_config) -> ClinicalConceptRouter:
    """Fresh ClinicalConceptRouter in training mode."""
    m = ClinicalConceptRouter(default_config.router)
    m.train()
    return m


@pytest.fixture
def dispatcher(default_config) -> CCRBottleneckModule:
    """Fresh CCRBottleneckModule in training mode."""
    m = CCRBottleneckModule(default_config)
    m.train()
    return m


# ---------------------------------------------------------------------------
# Tensor fixtures — realistic BraTS-like data
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_tokens() -> torch.Tensor:
    """
    Bottleneck tokens: [B=2, N=4096, D=192].
    Values drawn from N(0,1), consistent with typical encoder outputs after
    LayerNorm normalisation.
    """
    torch.manual_seed(0)
    return torch.randn(BATCH, N_TOKENS, EMBED_DIM)


@pytest.fixture
def token_labels() -> torch.Tensor:
    """
    Per-token ground-truth labels: [B=2, N=4096] with approximate BraTS class
    balance:
        Background (0): ~80 %  — 3276 tokens
        NCR        (1): ~ 8 %  —  328 tokens
        Edema      (2): ~ 7 %  —  286 tokens
        ET         (3): ~ 5 %  —  206 tokens
    """
    labels = torch.zeros(BATCH, N_TOKENS, dtype=torch.long)
    labels[:, 3276:3604] = 1   # NCR   tokens
    labels[:, 3604:3890] = 2   # Edema tokens
    labels[:, 3890:4096] = 3   # ET    tokens
    return labels


@pytest.fixture
def spatial_pred_logits() -> torch.Tensor:
    """
    3D segmentation logits: [B=2, K=4, H=128, W=128, D=128].
    Raw (pre-softmax) values.
    """
    torch.manual_seed(1)
    return torch.randn(BATCH, K, H, W, D)


@pytest.fixture
def spatial_labels() -> torch.Tensor:
    """
    3D ground-truth label volume: [B=2, H=128, W=128, D=128].
    Simulates a central tumor with NCR core, edema ring, and ET sub-region.
    """
    labels = torch.zeros(BATCH, H, W, D, dtype=torch.long)

    # Edema region: a 60³ cube centred in the volume
    labels[:, 34:94, 34:94, 34:94] = 2

    # NCR core inside edema: 40³ cube
    labels[:, 44:84, 44:84, 44:84] = 1

    # ET inside NCR: 20³ cube
    labels[:, 54:74, 54:74, 54:74] = 3

    return labels
