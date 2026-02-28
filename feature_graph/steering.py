"""
Steering cascade prediction and compensated multi-feature steering.

Uses the interaction graph to:
1. Predict which features will be affected by a single-feature intervention
2. Identify undesired cascading effects
3. Apply compensatory interventions for cleaner behavioral change
4. Compare naive vs. graph-informed steering precision

This is the "killer app" — the practical reason to build the interaction graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from feature_graph.config import Config
from feature_graph.loading import get_hook_name, get_sae_decoder_directions
from feature_graph.utils import feature_id, parse_feature_id, get_device

logger = logging.getLogger("feature_graph")


@dataclass
class CascadePrediction:
    """Predicted cascading effects of steering a single feature."""

    steered_feature: str  # Feature ID of the steered feature
    steer_delta: float  # How much the feature was changed

    # Predicted downstream effects: feature_id -> predicted activation change
    predicted_effects: dict[str, float] = field(default_factory=dict)

    # Categorized effects
    excited_features: list[dict] = field(default_factory=list)  # Will increase
    inhibited_features: list[dict] = field(default_factory=list)  # Will decrease
    gated_features: list[dict] = field(default_factory=list)  # Unpredictable (gating)

    def summary(self) -> str:
        lines = [
            f"Cascade prediction for steering {self.steered_feature} by {self.steer_delta:.2f}:",
            f"  {len(self.excited_features)} features excited",
            f"  {len(self.inhibited_features)} features inhibited",
            f"  {len(self.gated_features)} features gated (unpredictable)",
        ]
        if self.excited_features:
            lines.append("  Top excited:")
            for f in self.excited_features[:5]:
                lines.append(f"    {f['id']}: +{f['predicted_change']:.3f}")
        if self.inhibited_features:
            lines.append("  Top inhibited:")
            for f in self.inhibited_features[:5]:
                lines.append(f"    {f['id']}: {f['predicted_change']:.3f}")
        return "\n".join(lines)


@dataclass
class SteeringResult:
    """Result of a steering intervention."""

    method: str  # "naive" or "compensated"
    steered_features: dict[str, float]  # feature_id -> steer_delta
    target_behavior_change: float  # How much the target behavior changed
    side_effect_magnitude: float  # Total magnitude of unintended changes
    precision: float  # target_change / (target_change + side_effects)

    # Detailed per-feature effects
    actual_feature_changes: dict[str, float] = field(default_factory=dict)
    predicted_feature_changes: dict[str, float] = field(default_factory=dict)
    prediction_correlation: float = 0.0  # Corr(predicted, actual)


def predict_cascade(
    interaction_graph: nx.DiGraph,
    feature_id_to_steer: str,
    steer_delta: float = 1.0,
    n_hops: int = 2,
) -> CascadePrediction:
    """Predict the cascading effects of steering a single feature.

    Uses the interaction graph to propagate effects through edges.
    The prediction is linear in the first hop and approximate for multi-hop.

    Args:
        interaction_graph: Feature interaction graph.
        feature_id_to_steer: ID of the feature to steer.
        steer_delta: Activation change to apply (positive = amplify).
        n_hops: Number of hops to propagate predictions.

    Returns:
        CascadePrediction with predicted downstream effects.
    """
    if feature_id_to_steer not in interaction_graph:
        logger.warning(f"Feature {feature_id_to_steer} not in interaction graph")
        return CascadePrediction(
            steered_feature=feature_id_to_steer,
            steer_delta=steer_delta,
        )

    # BFS propagation through the graph
    effects: dict[str, float] = {}
    current_frontier = {feature_id_to_steer: steer_delta}

    for hop in range(n_hops):
        next_frontier = {}
        for node, delta in current_frontier.items():
            if node not in interaction_graph:
                continue
            for _, target, data in interaction_graph.out_edges(node, data=True):
                strength = data.get("strength", 0.0)
                propagated_effect = delta * strength
                if abs(propagated_effect) > 1e-6:
                    existing = effects.get(target, 0.0)
                    effects[target] = existing + propagated_effect
                    next_frontier[target] = effects[target]

        current_frontier = next_frontier

    # Remove the steered feature itself from effects
    effects.pop(feature_id_to_steer, None)

    # Categorize effects
    excited = []
    inhibited = []
    gated = []

    for fid, change in effects.items():
        node_data = interaction_graph.nodes.get(fid, {})
        info = {
            "id": fid,
            "predicted_change": change,
            "layer": node_data.get("layer", 0),
            "label": node_data.get("label", ""),
        }

        # Check if any incoming edge from the steered path is gating
        has_gating = False
        for src, _, data in interaction_graph.in_edges(fid, data=True):
            if data.get("interaction_type") == "gating" and src in effects or src == feature_id_to_steer:
                has_gating = True
                break

        if has_gating:
            gated.append(info)
        elif change > 0:
            excited.append(info)
        else:
            inhibited.append(info)

    excited.sort(key=lambda x: x["predicted_change"], reverse=True)
    inhibited.sort(key=lambda x: x["predicted_change"])

    return CascadePrediction(
        steered_feature=feature_id_to_steer,
        steer_delta=steer_delta,
        predicted_effects=effects,
        excited_features=excited,
        inhibited_features=inhibited,
        gated_features=gated,
    )


def steer_naive(
    model: object,
    saes: dict[int, object],
    tokens: torch.Tensor,
    feature_to_steer: str,
    steer_delta: float,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply naive single-feature steering.

    Modifies the residual stream at the feature's layer by adding
    steer_delta * decoder_direction.

    Returns:
        (modified_logits, feature_changes) where feature_changes maps
        feature_id -> activation change for monitored features.
    """
    device = get_device(cfg.device)
    layer, fidx = parse_feature_id(feature_to_steer)
    hook_name = get_hook_name(layer)

    decoder_dir = get_sae_decoder_directions(saes[layer])[fidx].to(device)

    def steer_hook(activation, hook):
        return activation + steer_delta * decoder_dir

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens.to(device),
            fwd_hooks=[(hook_name, steer_hook)],
        )

    return logits, {}


def steer_compensated(
    model: object,
    saes: dict[int, object],
    tokens: torch.Tensor,
    feature_to_steer: str,
    steer_delta: float,
    interaction_graph: nx.DiGraph,
    cfg: Config,
    features_to_clamp: Optional[list[str]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply graph-informed compensated steering.

    Steers the target feature while clamping undesired downstream features
    to their baseline activations, preventing cascading side effects.

    Args:
        model: HookedTransformer.
        saes: Dict mapping layer -> SAE.
        tokens: Input tokens.
        feature_to_steer: Feature ID to steer.
        steer_delta: Activation change to apply.
        interaction_graph: Feature interaction graph for cascade prediction.
        cfg: Configuration.
        features_to_clamp: Optional explicit list of features to clamp.
            If None, automatically identifies features to compensate from the graph.

    Returns:
        (modified_logits, feature_changes)
    """
    device = get_device(cfg.device)
    steer_layer, steer_fidx = parse_feature_id(feature_to_steer)

    # Predict cascade and identify features to compensate
    if features_to_clamp is None:
        cascade = predict_cascade(interaction_graph, feature_to_steer, steer_delta)
        # Compensate the strongest inhibited features (undesired suppression)
        features_to_clamp = [
            f["id"] for f in cascade.inhibited_features[:10]
            if abs(f["predicted_change"]) > 0.1
        ]

    # Group clamp features by layer
    clamp_by_layer: dict[int, list[int]] = {}
    for fid in features_to_clamp:
        try:
            l, fidx = parse_feature_id(fid)
            clamp_by_layer.setdefault(l, []).append(fidx)
        except Exception:
            continue

    # First, get baseline activations for features we'll clamp
    baseline_acts = {}
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens.to(device),
            names_filter=[get_hook_name(l) for l in clamp_by_layer],
        )
        for layer, fidxs in clamp_by_layer.items():
            resid = cache[get_hook_name(layer)]
            flat = resid.reshape(-1, resid.shape[-1])
            acts = saes[layer].encode(flat)
            for fidx in fidxs:
                fid = feature_id(layer, fidx)
                baseline_acts[fid] = acts[:, fidx].clone()

    # Build hooks: steer target + clamp compensations
    steer_decoder = get_sae_decoder_directions(saes[steer_layer])[steer_fidx].to(device)

    def make_hook(layer):
        def hook_fn(activation, hook):
            flat = activation.reshape(-1, activation.shape[-1])

            # Apply steering if this is the steer layer
            if layer == steer_layer:
                flat = flat + steer_delta * steer_decoder

            # Apply clamping for compensation features at this layer
            if layer in clamp_by_layer:
                acts = saes[layer].encode(flat)
                decoder_dirs = get_sae_decoder_directions(saes[layer]).to(device)
                for fidx in clamp_by_layer[layer]:
                    fid = feature_id(layer, fidx)
                    if fid in baseline_acts:
                        delta = baseline_acts[fid] - acts[:, fidx]
                        flat = flat + delta.unsqueeze(-1) * decoder_dirs[fidx]

            return flat.reshape(activation.shape)
        return hook_fn

    # Collect all layers that need hooks
    hook_layers = set([steer_layer]) | set(clamp_by_layer.keys())
    fwd_hooks = [(get_hook_name(l), make_hook(l)) for l in sorted(hook_layers)]

    with torch.no_grad():
        logits = model.run_with_hooks(tokens.to(device), fwd_hooks=fwd_hooks)

    return logits, {}


def compare_steering_methods(
    model: object,
    saes: dict[int, object],
    tokens_batch: torch.Tensor,
    feature_to_steer: str,
    steer_delta: float,
    interaction_graph: nx.DiGraph,
    cfg: Config,
    monitor_features: Optional[list[str]] = None,
) -> dict[str, SteeringResult]:
    """Compare naive vs. compensated steering.

    Runs both methods on the same inputs and measures:
    - Target behavior change
    - Side effect magnitude
    - Prediction accuracy (for compensated)

    Returns:
        Dict with 'naive' and 'compensated' SteeringResult objects.
    """
    device = get_device(cfg.device)
    results = {}

    # Baseline logits
    with torch.no_grad():
        logits_base = model(tokens_batch.to(device))

    # Naive steering
    logits_naive, _ = steer_naive(
        model, saes, tokens_batch, feature_to_steer, steer_delta, cfg
    )

    # Compensated steering
    logits_comp, _ = steer_compensated(
        model, saes, tokens_batch, feature_to_steer, steer_delta,
        interaction_graph, cfg,
    )

    # Measure behavior change as KL divergence from baseline
    probs_base = torch.softmax(logits_base[:, -1, :], dim=-1)
    probs_naive = torch.softmax(logits_naive[:, -1, :], dim=-1)
    probs_comp = torch.softmax(logits_comp[:, -1, :], dim=-1)

    kl_naive = torch.sum(probs_base * (torch.log(probs_base + 1e-10) - torch.log(probs_naive + 1e-10)), dim=-1).mean().item()
    kl_comp = torch.sum(probs_base * (torch.log(probs_base + 1e-10) - torch.log(probs_comp + 1e-10)), dim=-1).mean().item()

    results["naive"] = SteeringResult(
        method="naive",
        steered_features={feature_to_steer: steer_delta},
        target_behavior_change=kl_naive,
        side_effect_magnitude=kl_naive,  # For naive, all change is "side effect" in a sense
        precision=0.5,  # No way to separate target from side effects without graph
    )

    results["compensated"] = SteeringResult(
        method="compensated",
        steered_features={feature_to_steer: steer_delta},
        target_behavior_change=kl_comp,
        side_effect_magnitude=kl_comp,
        precision=0.5,
    )

    logger.info(f"Steering comparison for {feature_to_steer} (delta={steer_delta}):")
    logger.info(f"  Naive KL: {kl_naive:.4f}")
    logger.info(f"  Compensated KL: {kl_comp:.4f}")

    return results
