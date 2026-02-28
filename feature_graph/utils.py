"""
Shared utilities for the feature interaction graph library.
"""

from __future__ import annotations

import random
import logging
from typing import Optional

import numpy as np
import torch


logger = logging.getLogger("feature_graph")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the library."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    """Get torch device, falling back gracefully."""
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def feature_id(layer: int, feature_idx: int) -> str:
    """Create a canonical string ID for a feature."""
    return f"L{layer}_F{feature_idx}"


def parse_feature_id(fid: str) -> tuple[int, int]:
    """Parse a feature ID string back to (layer, feature_idx)."""
    parts = fid.split("_")
    layer = int(parts[0][1:])
    feature_idx = int(parts[1][1:])
    return layer, feature_idx


@torch.no_grad()
def get_nonzero_activation_stats(
    activations: torch.Tensor,
) -> dict[str, float]:
    """Compute statistics of non-zero activations for a single feature.

    Args:
        activations: 1D tensor of feature activations across many tokens.

    Returns:
        Dict with keys: mean, std, p10, p50, p90, freq (fraction nonzero).
    """
    nonzero_mask = activations > 0
    freq = nonzero_mask.float().mean().item()

    if freq == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "freq": 0.0}

    nz = activations[nonzero_mask]
    return {
        "mean": nz.mean().item(),
        "std": nz.std().item() if len(nz) > 1 else 0.0,
        "p10": torch.quantile(nz, 0.1).item(),
        "p50": torch.quantile(nz, 0.5).item(),
        "p90": torch.quantile(nz, 0.9).item(),
        "freq": freq,
    }


def batched_cosine_similarity(
    a: torch.Tensor, b: torch.Tensor, batch_size: int = 1024
) -> torch.Tensor:
    """Compute cosine similarity between all pairs of row vectors.

    Args:
        a: (N, D) tensor
        b: (M, D) tensor
        batch_size: Process in batches to avoid OOM.

    Returns:
        (N, M) tensor of cosine similarities.
    """
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)

    results = []
    for i in range(0, len(a_norm), batch_size):
        batch = a_norm[i : i + batch_size]
        sim = batch @ b_norm.T
        results.append(sim.cpu())

    return torch.cat(results, dim=0)
