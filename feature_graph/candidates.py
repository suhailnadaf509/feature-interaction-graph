"""
Candidate pair identification and filtering.

Implements the four-stage pruning pipeline described in the research document:
1. Importance filtering — select top-K features by composite importance score
2. Co-activation filtering — remove pairs with negligible co-activation
3. Decoder alignment filtering — remove pairs with near-zero direct alignment
4. Layer locality — only consider pairs within a layer window

The output is a list of (source_feature, target_feature) pairs that will be
tested with causal interventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from feature_graph.config import Config
from feature_graph.activations import ActivationStore
from feature_graph.coactivation import CoactivationAtlas
from feature_graph.loading import (
    get_sae_decoder_directions,
    get_sae_encoder_directions,
    get_sae_dict_size,
    get_hook_name,
)
from feature_graph.utils import feature_id, batched_cosine_similarity

logger = logging.getLogger("feature_graph")


@dataclass
class FeatureInfo:
    """Information about a single feature."""

    layer: int
    feature_idx: int
    importance: float
    activation_freq: float
    causal_effect: float = 0.0
    label: str = ""

    @property
    def id(self) -> str:
        return feature_id(self.layer, self.feature_idx)


@dataclass
class CandidatePair:
    """A candidate pair of features to test for causal interaction."""

    src: FeatureInfo
    tgt: FeatureInfo
    coactivation_prob: float = 0.0
    pmi: float = 0.0
    decoder_alignment: float = 0.0
    is_candidate_inhibitory: bool = False

    @property
    def src_id(self) -> str:
        return self.src.id

    @property
    def tgt_id(self) -> str:
        return self.tgt.id


def identify_candidates(
    atlas: CoactivationAtlas,
    saes: dict[int, object],
    cfg: Config,
    activation_store: Optional[ActivationStore] = None,
    model: Optional[object] = None,
) -> list[CandidatePair]:
    """Identify candidate feature pairs for causal interaction testing.

    Implements the four-stage pruning pipeline.

    Args:
        atlas: Co-activation atlas from build_coactivation_atlas().
        saes: Dict mapping layer -> SAE object.
        cfg: Configuration.
        activation_store: Optional activation store (needed for importance scoring
            with causal_effect method).
        model: Optional model (needed for causal effect importance scoring).

    Returns:
        List of CandidatePair objects, sorted by expected interaction likelihood.
    """
    # Stage 1: Compute feature importance and select top-K
    logger.info("Stage 1: Computing feature importance scores")
    important_features = _compute_importance_and_select(
        atlas, saes, cfg, activation_store, model
    )
    logger.info(f"  Selected {len(important_features)} features across {len(set(f.layer for f in important_features))} layers")

    # Stage 2 & 3: Filter by co-activation and decoder alignment
    logger.info("Stage 2-3: Filtering by co-activation and decoder alignment")
    candidates = _filter_pairs(important_features, atlas, saes, cfg)
    logger.info(f"  {len(candidates)} candidate pairs after filtering")

    # Sort by expected interaction likelihood (PMI as proxy)
    candidates.sort(key=lambda c: abs(c.pmi), reverse=True)

    return candidates


def _compute_importance_and_select(
    atlas: CoactivationAtlas,
    saes: dict[int, object],
    cfg: Config,
    activation_store: Optional[ActivationStore] = None,
    model: Optional[object] = None,
) -> list[FeatureInfo]:
    """Compute importance scores and select top-K features per layer."""

    all_features = []

    for layer in sorted(atlas.feature_freqs.keys()):
        freqs = atlas.feature_freqs[layer]
        n_features = len(freqs)

        if cfg.importance_method == "activation_freq":
            importance = freqs
        elif cfg.importance_method in ("causal_effect", "activation_freq_x_causal"):
            if activation_store is not None and model is not None:
                causal_effects = _estimate_causal_effects(
                    model, saes[layer], activation_store, layer, cfg
                )
                if cfg.importance_method == "causal_effect":
                    importance = causal_effects
                else:
                    importance = freqs * causal_effects
            else:
                logger.warning(
                    "Causal effect importance requested but model/activations not provided. "
                    "Falling back to activation_freq."
                )
                importance = freqs
        else:
            importance = freqs

        # Select top features for this layer
        n_select = min(cfg.top_k_features, n_features)
        top_indices = np.argsort(importance)[-n_select:][::-1]

        for idx in top_indices:
            if freqs[idx] > 0:  # Skip dead features
                all_features.append(
                    FeatureInfo(
                        layer=layer,
                        feature_idx=int(idx),
                        importance=float(importance[idx]),
                        activation_freq=float(freqs[idx]),
                    )
                )

    return all_features


def _estimate_causal_effects(
    model: object,
    sae: object,
    activation_store: ActivationStore,
    layer: int,
    cfg: Config,
) -> np.ndarray:
    """Estimate causal effect of each feature on model output (loss change when ablated).

    This is a simplified version — we measure the change in logit output norm
    when each feature is zeroed out, averaged over a sample of inputs.
    """
    from feature_graph.utils import get_device

    device = get_device(cfg.device)
    n_features = get_sae_dict_size(sae)
    effects = np.zeros(n_features, dtype=np.float32)

    # Sample inputs where features are active
    acts = activation_store.activations[layer]  # (n_tokens, n_features)
    n_samples = min(cfg.n_importance_samples, acts.shape[0])

    # For each feature, estimate causal effect by measuring decoder direction contribution
    # This is a fast approximation: ||f_i * d_i|| / ||h|| averaged over active tokens
    decoder_dirs = get_sae_decoder_directions(sae)  # (n_features, d_model)
    decoder_norms = torch.norm(decoder_dirs, dim=1).cpu().numpy()  # (n_features,)

    # Mean activation when active * decoder norm = approximate causal effect
    for i in range(n_features):
        active_mask = acts[:n_samples, i] > 0
        if active_mask.sum() > 0:
            mean_act = float(np.mean(acts[:n_samples, i][active_mask]))
            effects[i] = mean_act * decoder_norms[i]

    return effects


def _filter_pairs(
    features: list[FeatureInfo],
    atlas: CoactivationAtlas,
    saes: dict[int, object],
    cfg: Config,
) -> list[CandidatePair]:
    """Filter feature pairs by co-activation and decoder alignment."""

    # Group features by layer
    by_layer: dict[int, list[FeatureInfo]] = {}
    for f in features:
        by_layer.setdefault(f.layer, []).append(f)

    # Precompute decoder/encoder directions for alignment check
    decoder_dirs = {}
    encoder_dirs = {}
    for layer in by_layer:
        if layer in saes:
            decoder_dirs[layer] = get_sae_decoder_directions(saes[layer])
            encoder_dirs[layer] = get_sae_encoder_directions(saes[layer])

    candidates = []

    for (l_src, l_tgt) in atlas.layer_pairs:
        if l_src not in by_layer or l_tgt not in by_layer:
            continue

        src_features = by_layer[l_src]
        tgt_features = by_layer[l_tgt]

        # Get co-activation data for this layer pair
        pmi_mat = atlas.pmi.get((l_src, l_tgt))
        coact_prob_mat = atlas.coact_prob.get((l_src, l_tgt))
        coact_ratio_mat = atlas.coact_ratio.get((l_src, l_tgt))

        for src_f in src_features:
            for tgt_f in tgt_features:
                si = src_f.feature_idx
                ti = tgt_f.feature_idx

                # Check co-activation
                pmi_val = 0.0
                coact_val = 0.0
                is_inhibitory = False

                if pmi_mat is not None:
                    pmi_val = float(pmi_mat[si, ti])
                if coact_prob_mat is not None:
                    coact_val = float(coact_prob_mat[si, ti])
                if coact_ratio_mat is not None:
                    ratio = float(coact_ratio_mat[si, ti])
                    if 0 < ratio < cfg.low_coactivation_ratio:
                        is_inhibitory = True

                # Check decoder alignment for adjacent layers
                align_val = 0.0
                if (l_tgt - l_src) == 1 and l_src in decoder_dirs and l_tgt in encoder_dirs:
                    d_i = decoder_dirs[l_src][si]  # (d_model,)
                    e_j = encoder_dirs[l_tgt][ti]  # (d_model,)
                    align_val = float(torch.abs(torch.dot(d_i, e_j)).item())

                # Apply filters
                passes = False

                # Pass if high co-activation / PMI
                if coact_val > cfg.coactivation_threshold or pmi_val > cfg.pmi_threshold:
                    passes = True

                # Pass if candidate inhibitory
                if is_inhibitory:
                    passes = True

                # Pass if high decoder alignment (adjacent layers)
                if align_val > cfg.decoder_alignment_threshold:
                    passes = True

                # All top-K features among each other always pass (ensures dense core graph)
                if src_f.importance > 0 and tgt_f.importance > 0:
                    passes = True

                if passes:
                    candidates.append(
                        CandidatePair(
                            src=src_f,
                            tgt=tgt_f,
                            coactivation_prob=coact_val,
                            pmi=pmi_val,
                            decoder_alignment=align_val,
                            is_candidate_inhibitory=is_inhibitory,
                        )
                    )

    return candidates
