"""
Configuration dataclasses for the feature interaction graph pipeline.

We use plain dataclasses rather than Hydra/OmegaConf because:
1. Lower dependency footprint — no YAML parsing framework to learn.
2. Full IDE support for autocomplete and type checking.
3. Easy to serialize to/from JSON for reproducibility.
4. Other researchers can read the config definition and understand every option
   without learning a config framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Master configuration for the feature interaction graph pipeline."""

    # ── Model & SAE ──────────────────────────────────────────────────────
    model_name: str = "gpt2-small"
    """TransformerLens model name. Supports any model in the TL registry."""

    sae_release: str = "gpt2-small-res-jb"
    """SAELens release name for loading pre-trained SAEs."""

    sae_id_template: str = "blocks.{layer}.hook_resid_post"
    """Template for SAE hook IDs, with {layer} as placeholder."""

    layers: list[int] = field(default_factory=lambda: list(range(12)))
    """Which transformer layers to include in the analysis."""

    device: str = "cuda"
    """Device for computation. 'cuda', 'cpu', or 'mps'."""

    dtype: str = "float32"
    """Computation dtype. 'float32' or 'float16' (use float16 for large models)."""

    # ── Activation Collection ────────────────────────────────────────────
    dataset_name: str = "monology/pile-uncopyrighted"
    """HuggingFace dataset for activation collection."""

    dataset_split: str = "train"
    """Dataset split to use."""

    n_tokens: int = 1_000_000
    """Total number of tokens to process for activation collection."""

    context_length: int = 128
    """Sequence length for each batch element."""

    batch_size: int = 32
    """Batch size for forward passes."""

    # ── Co-Activation Atlas ──────────────────────────────────────────────
    coactivation_threshold: float = 0.01
    """Minimum co-activation probability to retain a pair as a candidate.
    Pairs with P(f_j > 0 | f_i > 0) < this are pruned."""

    low_coactivation_ratio: float = 0.1
    """Flag pairs where P(f_j>0 | f_i>0) / P(f_j>0) < this ratio.
    These are candidate inhibitory interactions."""

    pmi_threshold: float = 1.0
    """Minimum pointwise mutual information to retain a pair."""

    # ── Candidate Filtering ──────────────────────────────────────────────
    top_k_features: int = 200
    """Number of top features (by importance) to include in the analysis.
    Start small (200) for tractability, scale up as compute allows."""

    layer_window: int = 3
    """Maximum layer distance for candidate pairs. |l' - l| <= layer_window.
    Most causal interactions are local; increase for long-range analysis."""

    decoder_alignment_threshold: float = 0.05
    """Minimum |e_j · d_i| to retain a pair based on decoder alignment.
    Only applied for adjacent layers where this is informative."""

    importance_method: str = "activation_freq_x_causal"
    """How to compute feature importance. Options:
    - 'activation_freq': frequency of non-zero activations
    - 'causal_effect': mean absolute change in loss when feature is ablated
    - 'activation_freq_x_causal': product of both (recommended)
    """

    n_importance_samples: int = 256
    """Number of samples for estimating feature importance (causal effect)."""

    # ── Interaction Measurement ──────────────────────────────────────────
    interaction_method: str = "clamping"
    """Method for measuring interactions. Options:
    - 'clamping': Gold standard. Modify source feature, measure downstream.
    - 'jacobian': Fast approximate. Compute df_j/df_i via backprop.
    - 'both': Run both and compare.
    """

    n_intervention_samples: int = 100
    """Number of input samples per candidate pair for clamping interventions."""

    clamp_percentile_low: float = 10.0
    """Lower percentile of non-zero activations for clamping range."""

    clamp_percentile_high: float = 90.0
    """Upper percentile of non-zero activations for clamping range."""

    # ── Statistical Testing ──────────────────────────────────────────────
    significance_level: float = 0.01
    """P-value threshold for retaining an edge (before correction)."""

    correction_method: str = "fdr_bh"
    """Multiple comparison correction. Options:
    - 'bonferroni': Conservative. Use when few candidate pairs.
    - 'fdr_bh': Benjamini-Hochberg FDR control. Recommended.
    - 'none': No correction. Only for exploratory analysis.
    """

    null_permutations: int = 1000
    """Number of permutations for null distribution estimation."""

    gating_variance_threshold: float = 2.0
    """Ratio of interaction variance to mean^2 above which we flag gating.
    A high coefficient of variation suggests context-dependent interaction."""

    # ── Graph Construction ───────────────────────────────────────────────
    edge_strength_threshold: float = 0.0
    """Minimum |interaction strength| to include an edge. 0 means use
    statistical significance only. Set > 0 for sparser graphs."""

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "outputs"
    """Directory for all outputs."""

    seed: int = 42
    """Random seed for reproducibility."""

    def save(self, path: str | Path) -> None:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def get_sae_id(self, layer: int) -> str:
        """Get the SAE hook ID for a given layer."""
        return self.sae_id_template.format(layer=layer)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def get_dtype(self):
        import torch
        return getattr(torch, self.dtype)
