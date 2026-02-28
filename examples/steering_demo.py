"""
Steering demo: compare naive vs. compensated steering using the FIG.

Given a pre-built interaction graph, this script:
  1. Picks a target feature to steer (or takes one from the CLI).
  2. Predicts the downstream cascade through the graph.
  3. Runs naive steering (add decoder direction to residual stream).
  4. Runs compensated steering (steer + clamp undesired side effects).
  5. Reports KL divergence between each method and the baseline.

Usage:
    python examples/steering_demo.py --graph outputs/quick_start/graph.graphml \\
        --feature L5_F1234 --delta 3.0

Pre-requisites:
    pip install -e ".[dev]"
    Run the quick_start pipeline first to produce a graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_graph.config import Config
from feature_graph.graph import load_graph
from feature_graph.loading import load_model_and_saes
from feature_graph.steering import (
    compare_steering_methods,
    predict_cascade,
)
from feature_graph.utils import get_device, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steering demo with cascade prediction")
    p.add_argument("--graph", type=str, required=True,
                    help="Path to a saved interaction graph (graphml/json/pkl)")
    p.add_argument("--config", type=str, default=None,
                    help="Path to config.json (auto-detected beside graph if omitted)")
    p.add_argument("--feature", type=str, required=True,
                    help='Feature to steer, e.g. "L5_F1234"')
    p.add_argument("--delta", type=float, default=3.0,
                    help="Steering strength (units of decoder-direction norm)")
    p.add_argument("--prompt", type=str,
                    default="The capital of France is",
                    help="Prompt to steer on")
    p.add_argument("--max_hops", type=int, default=3,
                    help="How many hops to trace the cascade")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("steering_demo")

    # ── Load graph ────────────────────────────────────────────────────
    graph_path = Path(args.graph)
    fmt = graph_path.suffix.lstrip(".")
    if fmt == "graphml":
        G = load_graph(graph_path, format="graphml")
    elif fmt == "json":
        G = load_graph(graph_path, format="json")
    else:
        G = load_graph(graph_path, format="pickle")
    logger.info("Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── Predict cascade ───────────────────────────────────────────────
    cascade = predict_cascade(G, args.feature, steer_delta=args.delta, max_hops=args.max_hops)
    logger.info("Cascade from %s (δ=%.1f):", args.feature, args.delta)
    logger.info("  Excited  (%d): %s", len(cascade.excited_features),
                [f["id"] for f in cascade.excited_features[:10]])
    logger.info("  Inhibited(%d): %s", len(cascade.inhibited_features),
                [f["id"] for f in cascade.inhibited_features[:10]])
    logger.info("  Gated    (%d): %s", len(cascade.gated_features),
                [f["id"] for f in cascade.gated_features[:10]])

    # ── Load model for actual steering ────────────────────────────────
    config_path = Path(args.config) if args.config else graph_path.parent / "config.json"
    if not config_path.exists():
        logger.error("Config not found at %s — provide --config", config_path)
        return
    cfg = Config.load(config_path)
    cfg.device = args.device or get_device()

    logger.info("Loading model %s …", cfg.model_name)
    model, saes = load_model_and_saes(cfg)

    # ── Compare steering methods ──────────────────────────────────────
    logger.info("Running naive vs compensated steering on: '%s'", args.prompt)

    # Tokenize the prompt
    tokens = model.to_tokens(args.prompt)

    comparison = compare_steering_methods(
        model=model,
        saes=saes,
        tokens_batch=tokens,
        feature_to_steer=args.feature,
        steer_delta=args.delta,
        interaction_graph=G,
        cfg=cfg,
    )

    logger.info("=== Results ===")
    naive_result = comparison["naive"]
    comp_result = comparison["compensated"]
    logger.info("Naive steering KL divergence from baseline: %.4f", naive_result.target_behavior_change)
    logger.info("Compensated steering KL divergence from baseline: %.4f", comp_result.target_behavior_change)
    if naive_result.target_behavior_change > 1e-8:
        logger.info("KL reduction (lower = less side-effect): %.2f%%",
                    (1 - comp_result.target_behavior_change / naive_result.target_behavior_change) * 100)

    logger.info("✅ Steering demo complete.")


if __name__ == "__main__":
    main()
