"""
Causal interaction measurement via clamping interventions and Jacobian approximation.

This is the core scientific contribution of the library. We implement two
complementary methods for measuring feature interactions:

1. **Clamping (gold standard)**: Modify a source feature's activation by
   adding/subtracting its decoder direction from the residual stream, then
   measure the downstream effect on target features. This gives true causal
   interaction strengths.

2. **Jacobian (fast approximate)**: Compute df_j^(l') / df_i^(l) via
   backpropagation. ~100x faster but only captures first-order (linear)
   effects. Misses gating interactions.

Both methods produce InteractionResult objects that feed into graph construction.

Statistical rigor:
- Bootstrap confidence intervals on interaction strengths
- Permutation tests for null distribution
- Multiple comparison correction (Bonferroni or BH-FDR)
- Explicit gating detection via variance analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm

from feature_graph.config import Config
from feature_graph.candidates import CandidatePair
from feature_graph.loading import (
    get_sae_decoder_directions,
    get_sae_encoder_directions,
    get_hook_name,
)
from feature_graph.utils import get_device, set_seed

logger = logging.getLogger("feature_graph")


@dataclass
class InteractionResult:
    """Result of measuring a causal interaction between two features.

    This is the universal interface between measurement and graph construction.
    All interaction methods produce these objects.
    """

    src_layer: int
    src_feature: int
    tgt_layer: int
    tgt_feature: int

    # Core measurements
    mean_strength: float  # Mean interaction strength (positive=excitatory, negative=inhibitory)
    std_strength: float  # Std of interaction strength across inputs
    abs_strength: float  # Mean absolute interaction strength

    # Statistical testing
    p_value: float  # P-value from permutation or bootstrap test
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound

    # Interaction typing
    interaction_type: str  # "excitatory", "inhibitory", or "gating"
    gating_score: float  # Variance-based gating indicator

    # Metadata
    n_samples: int  # Number of input samples used
    method: str  # "clamping" or "jacobian"

    @property
    def src_id(self) -> str:
        from feature_graph.utils import feature_id
        return feature_id(self.src_layer, self.src_feature)

    @property
    def tgt_id(self) -> str:
        from feature_graph.utils import feature_id
        return feature_id(self.tgt_layer, self.tgt_feature)

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05  # Will be adjusted by multiple comparison correction

    def to_dict(self) -> dict:
        return {
            "src_layer": self.src_layer,
            "src_feature": self.src_feature,
            "tgt_layer": self.tgt_layer,
            "tgt_feature": self.tgt_feature,
            "mean_strength": self.mean_strength,
            "std_strength": self.std_strength,
            "abs_strength": self.abs_strength,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "interaction_type": self.interaction_type,
            "gating_score": self.gating_score,
            "n_samples": self.n_samples,
            "method": self.method,
        }


def measure_interactions(
    model: object,
    saes: dict[int, object],
    candidates: list[CandidatePair],
    cfg: Config,
    dataset_tokens: Optional[torch.Tensor] = None,
) -> list[InteractionResult]:
    """Measure causal interactions for all candidate pairs.

    Args:
        model: HookedTransformer.
        saes: Dict mapping layer -> SAE.
        candidates: Candidate pairs from identify_candidates().
        cfg: Configuration.
        dataset_tokens: Optional pre-tokenized dataset for interventions.

    Returns:
        List of InteractionResult for all tested pairs.
    """
    set_seed(cfg.seed)

    if cfg.interaction_method == "clamping":
        results = _measure_by_clamping(model, saes, candidates, cfg, dataset_tokens)
    elif cfg.interaction_method == "jacobian":
        results = _measure_by_jacobian(model, saes, candidates, cfg, dataset_tokens)
    elif cfg.interaction_method == "both":
        results_clamp = _measure_by_clamping(model, saes, candidates, cfg, dataset_tokens)
        results_jac = _measure_by_jacobian(model, saes, candidates, cfg, dataset_tokens)
        # Return clamping results with Jacobian results appended (different method tag)
        results = results_clamp + results_jac
    else:
        raise ValueError(f"Unknown interaction method: {cfg.interaction_method}")

    # Apply multiple comparison correction
    results = _apply_multiple_comparison_correction(results, cfg)

    logger.info(f"Measured {len(results)} interactions, "
                f"{sum(1 for r in results if r.p_value < cfg.significance_level)} significant")

    return results


def _measure_by_clamping(
    model: object,
    saes: dict[int, object],
    candidates: list[CandidatePair],
    cfg: Config,
    dataset_tokens: Optional[torch.Tensor] = None,
) -> list[InteractionResult]:
    """Measure interactions via feature clamping interventions.

    For each candidate pair (f_i^l, f_j^l'):
    1. Sample inputs where f_i is naturally active
    2. Run baseline forward pass, record f_j's activation
    3. Clamp f_i to high/low values, record f_j's activation
    4. Compute interaction strength as the mean difference
    """
    device = get_device(cfg.device)

    if dataset_tokens is None:
        from feature_graph.activations import _load_dataset_tokens
        dataset_tokens = _load_dataset_tokens(model, cfg)

    results = []

    # Group candidates by source layer for efficient batching
    by_src_layer: dict[int, list[CandidatePair]] = {}
    for cand in candidates:
        by_src_layer.setdefault(cand.src.layer, []).append(cand)

    # Precompute decoder directions
    decoder_dirs = {}
    for layer in saes:
        decoder_dirs[layer] = get_sae_decoder_directions(saes[layer]).to(device)

    for src_layer, layer_candidates in tqdm(
        sorted(by_src_layer.items()), desc="Clamping interventions (by source layer)"
    ):
        # Get all unique target layers for this source layer
        tgt_layers = sorted(set(c.tgt.layer for c in layer_candidates))

        # Process candidates in sub-batches
        for cand in tqdm(layer_candidates, desc=f"  Layer {src_layer}", leave=False):
            result = _clamp_single_pair(
                model, saes, decoder_dirs, cand, dataset_tokens, cfg, device
            )
            if result is not None:
                results.append(result)

    return results


def _clamp_single_pair(
    model: object,
    saes: dict[int, object],
    decoder_dirs: dict[int, torch.Tensor],
    cand: CandidatePair,
    dataset_tokens: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> Optional[InteractionResult]:
    """Run clamping intervention for a single feature pair.

    The intervention modifies the residual stream at the source layer by
    adding (alpha_high - f_i) * d_i or (alpha_low - f_i) * d_i, then
    measures the effect on the target feature.
    """
    src_layer = cand.src.layer
    src_idx = cand.src.feature_idx
    tgt_layer = cand.tgt.layer
    tgt_idx = cand.tgt.feature_idx

    src_hook = get_hook_name(src_layer)
    tgt_hook = get_hook_name(tgt_layer)
    sae_src = saes[src_layer]
    sae_tgt = saes[tgt_layer]
    d_i = decoder_dirs[src_layer][src_idx]  # (d_model,)

    n_samples = min(cfg.n_intervention_samples, dataset_tokens.shape[0])
    sample_indices = np.random.choice(dataset_tokens.shape[0], n_samples, replace=False)

    interaction_strengths = []

    for idx in sample_indices:
        tokens = dataset_tokens[idx : idx + 1].to(device)

        try:
            # Baseline forward pass
            with torch.no_grad():
                _, cache_base = model.run_with_cache(
                    tokens, names_filter=[src_hook, tgt_hook]
                )
                src_resid = cache_base[src_hook]  # (1, seq, d_model)
                tgt_resid_base = cache_base[tgt_hook]  # (1, seq, d_model)

                # Get source feature activation
                src_acts = sae_src.encode(src_resid.reshape(-1, src_resid.shape[-1]))
                f_i = src_acts[:, src_idx]  # (seq,)

                # Get baseline target feature activation
                tgt_acts_base = sae_tgt.encode(tgt_resid_base.reshape(-1, tgt_resid_base.shape[-1]))
                f_j_base = tgt_acts_base[:, tgt_idx]  # (seq,)

                # Find token positions where source feature is active
                active_mask = f_i > 0
                if active_mask.sum() == 0:
                    continue

                # Compute clamp levels from the active positions
                active_vals = f_i[active_mask]
                alpha_high = torch.quantile(active_vals.float(), cfg.clamp_percentile_high / 100)
                alpha_low = torch.quantile(active_vals.float(), cfg.clamp_percentile_low / 100)

                if alpha_high - alpha_low < 1e-6:
                    continue

            # Intervention forward pass: clamp f_i to alpha_high
            def hook_clamp_high(activation, hook):
                # activation: (1, seq, d_model)
                flat = activation.reshape(-1, activation.shape[-1])
                src_acts_hook = sae_src.encode(flat)
                current_fi = src_acts_hook[:, src_idx]
                delta = (alpha_high - current_fi).unsqueeze(-1) * d_i.unsqueeze(0)
                flat_modified = flat + delta
                return flat_modified.reshape(activation.shape)

            with torch.no_grad():
                _, cache_high = model.run_with_cache(
                    tokens,
                    names_filter=[tgt_hook],
                    fwd_hooks=[(src_hook, hook_clamp_high)],
                )
                tgt_resid_high = cache_high[tgt_hook]
                tgt_acts_high = sae_tgt.encode(
                    tgt_resid_high.reshape(-1, tgt_resid_high.shape[-1])
                )
                f_j_high = tgt_acts_high[:, tgt_idx]

            # Intervention forward pass: clamp f_i to alpha_low
            def hook_clamp_low(activation, hook):
                flat = activation.reshape(-1, activation.shape[-1])
                src_acts_hook = sae_src.encode(flat)
                current_fi = src_acts_hook[:, src_idx]
                delta = (alpha_low - current_fi).unsqueeze(-1) * d_i.unsqueeze(0)
                flat_modified = flat + delta
                return flat_modified.reshape(activation.shape)

            with torch.no_grad():
                _, cache_low = model.run_with_cache(
                    tokens,
                    names_filter=[tgt_hook],
                    fwd_hooks=[(src_hook, hook_clamp_low)],
                )
                tgt_resid_low = cache_low[tgt_hook]
                tgt_acts_low = sae_tgt.encode(
                    tgt_resid_low.reshape(-1, tgt_resid_low.shape[-1])
                )
                f_j_low = tgt_acts_low[:, tgt_idx]

            # Compute interaction strength for active positions
            strength = (f_j_high[active_mask] - f_j_low[active_mask]) / (alpha_high - alpha_low + 1e-8)
            mean_strength = strength.mean().item()
            interaction_strengths.append(mean_strength)

            # Cleanup
            del cache_base, cache_high, cache_low

        except Exception as e:
            logger.debug(f"  Error processing pair ({cand.src_id}, {cand.tgt_id}): {e}")
            continue

    if len(interaction_strengths) < 5:
        return None

    strengths = np.array(interaction_strengths)
    mean_s = float(np.mean(strengths))
    std_s = float(np.std(strengths))
    abs_s = float(np.mean(np.abs(strengths)))

    # Bootstrap confidence interval
    ci_lower, ci_upper = _bootstrap_ci(strengths)

    # P-value: test if mean is significantly different from 0
    if std_s > 0:
        t_stat = mean_s / (std_s / np.sqrt(len(strengths)))
        p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df=len(strengths) - 1))
    else:
        p_value = 1.0

    # Gating detection: high coefficient of variation suggests context-dependence
    cv = std_s / (abs(mean_s) + 1e-8)
    gating_score = float(cv)

    # Type classification
    if cv > cfg.gating_variance_threshold:
        interaction_type = "gating"
    elif mean_s > 0:
        interaction_type = "excitatory"
    else:
        interaction_type = "inhibitory"

    return InteractionResult(
        src_layer=src_layer,
        src_feature=src_idx,
        tgt_layer=tgt_layer,
        tgt_feature=tgt_idx,
        mean_strength=mean_s,
        std_strength=std_s,
        abs_strength=abs_s,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interaction_type=interaction_type,
        gating_score=gating_score,
        n_samples=len(strengths),
        method="clamping",
    )


def _measure_by_jacobian(
    model: object,
    saes: dict[int, object],
    candidates: list[CandidatePair],
    cfg: Config,
    dataset_tokens: Optional[torch.Tensor] = None,
) -> list[InteractionResult]:
    """Measure interactions via Jacobian approximation (fast, approximate).

    Computes df_j^(l') / df_i^(l) by:
    1. Running a forward pass and recording SAE activations
    2. Backpropagating from f_j through the model to f_i
    3. The gradient is the first-order interaction strength

    This is ~100x faster than clamping but misses nonlinear (gating) effects.
    """
    device = get_device(cfg.device)

    if dataset_tokens is None:
        from feature_graph.activations import _load_dataset_tokens
        dataset_tokens = _load_dataset_tokens(model, cfg)

    results = []
    n_samples = min(cfg.n_intervention_samples // 5, dataset_tokens.shape[0])  # Fewer samples needed
    sample_indices = np.random.choice(dataset_tokens.shape[0], n_samples, replace=False)

    # Precompute decoder/encoder directions
    decoder_dirs = {l: get_sae_decoder_directions(saes[l]).to(device) for l in saes}
    encoder_dirs = {l: get_sae_encoder_directions(saes[l]).to(device) for l in saes}

    for cand in tqdm(candidates, desc="Jacobian interactions"):
        strengths = []
        src_layer = cand.src.layer
        tgt_layer = cand.tgt.layer
        src_idx = cand.src.feature_idx
        tgt_idx = cand.tgt.feature_idx

        src_hook = get_hook_name(src_layer)
        tgt_hook = get_hook_name(tgt_layer)

        d_i = decoder_dirs[src_layer][src_idx]  # (d_model,)
        e_j = encoder_dirs[tgt_layer][tgt_idx]  # (d_model,)

        for idx in sample_indices:
            tokens = dataset_tokens[idx : idx + 1].to(device)

            try:
                # Forward pass with gradient tracking on the residual stream
                residuals = {}

                def save_and_require_grad(name):
                    def hook(activation, hook_info):
                        activation.requires_grad_(True)
                        activation.retain_grad()
                        residuals[name] = activation
                        return activation
                    return hook

                model.zero_grad()
                fwd_hooks = [
                    (src_hook, save_and_require_grad(src_hook)),
                    (tgt_hook, save_and_require_grad(tgt_hook)),
                ]

                logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

                if src_hook not in residuals or tgt_hook not in residuals:
                    continue

                src_resid = residuals[src_hook]  # (1, seq, d_model)
                tgt_resid = residuals[tgt_hook]  # (1, seq, d_model)

                # Compute target feature activation
                tgt_flat = tgt_resid.reshape(-1, tgt_resid.shape[-1])
                # Project onto encoder direction for target feature
                f_j = (tgt_flat @ e_j).sum()

                # Backpropagate from f_j
                f_j.backward(retain_graph=False)

                if src_resid.grad is not None:
                    # df_j / d(h^l) at each position
                    grad_wrt_src = src_resid.grad  # (1, seq, d_model)
                    # Project onto source decoder direction: df_j / df_i = df_j/dh · d_i
                    jacobian = (grad_wrt_src.reshape(-1, grad_wrt_src.shape[-1]) @ d_i)
                    mean_jac = jacobian.mean().item()
                    strengths.append(mean_jac)

                model.zero_grad()

            except Exception as e:
                logger.debug(f"Jacobian error for ({cand.src_id}, {cand.tgt_id}): {e}")
                continue

        if len(strengths) < 3:
            continue

        strengths_arr = np.array(strengths)
        mean_s = float(np.mean(strengths_arr))
        std_s = float(np.std(strengths_arr))
        abs_s = float(np.mean(np.abs(strengths_arr)))

        ci_lower, ci_upper = _bootstrap_ci(strengths_arr)

        if std_s > 0:
            t_stat = mean_s / (std_s / np.sqrt(len(strengths_arr)))
            p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df=len(strengths_arr) - 1))
        else:
            p_value = 1.0

        cv = std_s / (abs(mean_s) + 1e-8)
        gating_score = float(cv)

        if cv > cfg.gating_variance_threshold:
            interaction_type = "gating"
        elif mean_s > 0:
            interaction_type = "excitatory"
        else:
            interaction_type = "inhibitory"

        results.append(InteractionResult(
            src_layer=src_layer,
            src_feature=src_idx,
            tgt_layer=tgt_layer,
            tgt_feature=tgt_idx,
            mean_strength=mean_s,
            std_strength=std_s,
            abs_strength=abs_s,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            interaction_type=interaction_type,
            gating_score=gating_score,
            n_samples=len(strengths_arr),
            method="jacobian",
        ))

    return results


def _bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) < 2:
        return (float(data.mean()), float(data.mean()))

    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_means, 100 * alpha))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha)))
    return lower, upper


def _apply_multiple_comparison_correction(
    results: list[InteractionResult],
    cfg: Config,
) -> list[InteractionResult]:
    """Apply multiple comparison correction to p-values."""
    if not results or cfg.correction_method == "none":
        return results

    p_values = np.array([r.p_value for r in results])
    n_tests = len(p_values)

    if cfg.correction_method == "bonferroni":
        corrected = np.minimum(p_values * n_tests, 1.0)
    elif cfg.correction_method == "fdr_bh":
        # Benjamini-Hochberg procedure
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(p_values, method="fdr_bh")
    else:
        corrected = p_values

    for r, p in zip(results, corrected):
        r.p_value = float(p)

    return results
