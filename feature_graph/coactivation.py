"""
Co-activation atlas construction.

Computes pairwise co-activation statistics between SAE features across layers.
This is the correlational scaffold — it identifies which feature pairs *might*
interact, so we only run expensive causal interventions on plausible candidates.

Key statistics computed:
- Co-activation frequency: P(f_j > 0 | f_i > 0)
- Conditional mean activation: E[f_j | f_i > 0] vs E[f_j]
- Pointwise mutual information: log P(i,j) / (P(i) * P(j))
- Anomalously low co-activation (candidate inhibitory interactions)

Design: We work with sparse representations wherever possible, since SAE
features are sparse by construction (typically 0.1-5% activation frequency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy import sparse
from tqdm import tqdm

from feature_graph.config import Config
from feature_graph.activations import ActivationStore

logger = logging.getLogger("feature_graph")


@dataclass
class CoactivationAtlas:
    """Co-activation statistics between feature pairs.

    For each layer pair (l_src, l_tgt), stores sparse matrices of
    co-activation statistics for the feature pairs that pass threshold.
    """

    # Mapping: (l_src, l_tgt) -> sparse matrix of PMI values
    pmi: dict[tuple[int, int], sparse.csr_matrix] = field(default_factory=dict)

    # Mapping: (l_src, l_tgt) -> sparse matrix of co-activation probabilities
    coact_prob: dict[tuple[int, int], sparse.csr_matrix] = field(default_factory=dict)

    # Mapping: (l_src, l_tgt) -> sparse matrix of conditional activation ratios
    # Value = P(f_j > 0 | f_i > 0) / P(f_j > 0). High = co-excitation, Low = possible inhibition
    coact_ratio: dict[tuple[int, int], sparse.csr_matrix] = field(default_factory=dict)

    # Per-layer feature frequencies: layer -> (n_features,) array
    feature_freqs: dict[int, np.ndarray] = field(default_factory=dict)

    # Layer pairs that were analyzed
    layer_pairs: list[tuple[int, int]] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save atlas to HDF5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.attrs["layer_pairs"] = np.array(self.layer_pairs)

            for layer in self.feature_freqs:
                f.create_dataset(f"freq/layer_{layer}", data=self.feature_freqs[layer])

            for lp in self.layer_pairs:
                key = f"pair_{lp[0]}_{lp[1]}"
                grp = f.create_group(key)

                for name, matrix_dict in [
                    ("pmi", self.pmi),
                    ("coact_prob", self.coact_prob),
                    ("coact_ratio", self.coact_ratio),
                ]:
                    if lp in matrix_dict:
                        mat = matrix_dict[lp]
                        sub = grp.create_group(name)
                        sub.create_dataset("data", data=mat.data)
                        sub.create_dataset("indices", data=mat.indices)
                        sub.create_dataset("indptr", data=mat.indptr)
                        sub.attrs["shape"] = mat.shape

    @classmethod
    def load(cls, path: str | Path) -> "CoactivationAtlas":
        """Load atlas from HDF5."""
        atlas = cls()
        with h5py.File(path, "r") as f:
            layer_pairs_raw = f.attrs["layer_pairs"]
            atlas.layer_pairs = [tuple(lp) for lp in layer_pairs_raw]

            for key in f["freq"]:
                layer = int(key.split("_")[1])
                atlas.feature_freqs[layer] = f[f"freq/{key}"][:]

            for lp in atlas.layer_pairs:
                key = f"pair_{lp[0]}_{lp[1]}"
                if key in f:
                    grp = f[key]
                    for name, matrix_dict in [
                        ("pmi", atlas.pmi),
                        ("coact_prob", atlas.coact_prob),
                        ("coact_ratio", atlas.coact_ratio),
                    ]:
                        if name in grp:
                            sub = grp[name]
                            shape = tuple(sub.attrs["shape"])
                            mat = sparse.csr_matrix(
                                (sub["data"][:], sub["indices"][:], sub["indptr"][:]),
                                shape=shape,
                            )
                            matrix_dict[lp] = mat

        return atlas


def build_coactivation_atlas(
    activation_store: ActivationStore,
    cfg: Config,
) -> CoactivationAtlas:
    """Build the co-activation atlas from collected activations.

    For each layer pair (l_src, l_tgt) within the layer window, compute
    pairwise co-activation statistics for all feature pairs.

    Args:
        activation_store: Collected activations from collect_activations().
        cfg: Configuration.

    Returns:
        CoactivationAtlas with co-activation statistics.
    """
    layers = activation_store.layers
    n_tokens = activation_store.n_tokens

    # Compute per-feature binary activation masks and frequencies
    binary_masks = {}
    feature_freqs = {}
    for layer in layers:
        acts = activation_store.activations[layer]  # (n_tokens, n_features)
        mask = (acts > 0).astype(np.float32)
        binary_masks[layer] = mask
        feature_freqs[layer] = mask.mean(axis=0)

    # Identify layer pairs within window
    layer_pairs = []
    for i, l_src in enumerate(layers):
        for l_tgt in layers:
            if l_tgt > l_src and (l_tgt - l_src) <= cfg.layer_window:
                layer_pairs.append((l_src, l_tgt))

    logger.info(f"Computing co-activation for {len(layer_pairs)} layer pairs")

    atlas = CoactivationAtlas(
        feature_freqs=feature_freqs,
        layer_pairs=layer_pairs,
    )

    for l_src, l_tgt in tqdm(layer_pairs, desc="Co-activation atlas"):
        pmi_mat, coact_prob_mat, coact_ratio_mat = _compute_coactivation_pair(
            binary_masks[l_src],
            binary_masks[l_tgt],
            feature_freqs[l_src],
            feature_freqs[l_tgt],
            n_tokens,
            cfg,
        )
        atlas.pmi[(l_src, l_tgt)] = pmi_mat
        atlas.coact_prob[(l_src, l_tgt)] = coact_prob_mat
        atlas.coact_ratio[(l_src, l_tgt)] = coact_ratio_mat

    return atlas


def _compute_coactivation_pair(
    mask_src: np.ndarray,   # (n_tokens, n_src)
    mask_tgt: np.ndarray,   # (n_tokens, n_tgt)
    freq_src: np.ndarray,   # (n_src,)
    freq_tgt: np.ndarray,   # (n_tgt,)
    n_tokens: int,
    cfg: Config,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Compute co-activation statistics for one layer pair.

    Returns sparse matrices of (PMI, co-activation probability, co-activation ratio)
    with entries only for pairs that pass the thresholds.
    """
    n_src = mask_src.shape[1]
    n_tgt = mask_tgt.shape[1]

    # Co-occurrence counts: (n_src, n_tgt) = mask_src.T @ mask_tgt
    # This is the core computation. For large dictionaries, do it in batches.
    cooccur = _batched_matmul(mask_src.T, mask_tgt, batch_size=512)  # (n_src, n_tgt)

    # Marginal counts
    count_src = mask_src.sum(axis=0)  # (n_src,)
    count_tgt = mask_tgt.sum(axis=0)  # (n_tgt,)

    # Avoid division by zero
    count_src_safe = np.maximum(count_src, 1)
    count_tgt_safe = np.maximum(count_tgt, 1)
    freq_src_safe = np.maximum(freq_src, 1e-10)
    freq_tgt_safe = np.maximum(freq_tgt, 1e-10)

    # P(j > 0 | i > 0) = cooccur[i,j] / count_src[i]
    coact_prob_dense = cooccur / count_src_safe[:, None]

    # Ratio: P(j>0 | i>0) / P(j>0)
    coact_ratio_dense = coact_prob_dense / freq_tgt_safe[None, :]

    # PMI: log(P(i,j) / (P(i) * P(j))) = log(cooccur * n_tokens / (count_src * count_tgt))
    joint = cooccur / n_tokens
    expected = freq_src_safe[:, None] * freq_tgt_safe[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi_dense = np.log(np.maximum(joint / expected, 1e-30))

    # Filter: keep only entries that pass thresholds
    keep_mask = np.zeros((n_src, n_tgt), dtype=bool)

    # Keep high co-activation
    keep_mask |= coact_prob_dense > cfg.coactivation_threshold

    # Keep anomalously low co-activation (candidate inhibition)
    keep_mask |= (coact_ratio_dense < cfg.low_coactivation_ratio) & (freq_src > 0.001)[:, None] & (freq_tgt > 0.001)[None, :]

    # Keep high PMI
    keep_mask |= pmi_dense > cfg.pmi_threshold

    # Convert to sparse, keeping only passing entries
    pmi_sparse = sparse.csr_matrix(pmi_dense * keep_mask)
    coact_prob_sparse = sparse.csr_matrix(coact_prob_dense * keep_mask)
    coact_ratio_sparse = sparse.csr_matrix(coact_ratio_dense * keep_mask)

    n_kept = keep_mask.sum()
    n_total = n_src * n_tgt
    logger.debug(f"  Kept {n_kept}/{n_total} pairs ({100*n_kept/max(n_total,1):.2f}%)")

    return pmi_sparse, coact_prob_sparse, coact_ratio_sparse


def _batched_matmul(a: np.ndarray, b: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Batched matrix multiply to avoid memory issues with large feature dicts."""
    n_rows = a.shape[0]
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)

    for i in range(0, n_rows, batch_size):
        end = min(i + batch_size, n_rows)
        result[i:end] = a[i:end] @ b

    return result
