"""
Behaviour-subgraph extraction example.

Finds the subgraph responsible for a specific behaviour (factual recall,
refusal, sentiment, etc.) by comparing feature activations on positive-
vs. negative-example prompts and then extracting the neighbourhood from
the global interaction graph.

Usage:
    python examples/extract_behavior.py --graph outputs/quick_start/graph.graphml \\
        --behavior factual_recall

    python examples/extract_behavior.py --graph outputs/quick_start/graph.graphml \\
        --positive "The Eiffel Tower is in" "Rome was the capital of" \\
        --negative "Once upon a time" "The weather today"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_graph.config import Config
from feature_graph.graph import load_graph
from feature_graph.loading import load_model_and_saes
from feature_graph.subgraphs import (
    BehaviorContrast,
    PREDEFINED_BEHAVIORS,
    extract_behavior_subgraph,
)
from feature_graph.utils import get_device, setup_logging
from feature_graph.visualization import render_interactive_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract behaviour subgraph from FIG")
    p.add_argument("--graph", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--behavior", type=str, choices=list(PREDEFINED_BEHAVIORS.keys()),
                      help="Use a predefined behaviour contrast")
    grp.add_argument("--positive", type=str, nargs="+",
                      help="Positive-example prompts (custom behaviour)")
    p.add_argument("--negative", type=str, nargs="+", default=None,
                    help="Negative-example prompts (required if --positive given)")
    p.add_argument("--top_k", type=int, default=20,
                    help="Number of contrastive features to seed the subgraph")
    p.add_argument("--hops", type=int, default=2,
                    help="BFS expansion hops from seed features")
    p.add_argument("--output_dir", type=str, default="outputs/behavior",
                    help="Output directory")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("extract_behavior")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    # ── Resolve prompts ───────────────────────────────────────────────
    if args.behavior:
        contrast = PREDEFINED_BEHAVIORS[args.behavior]
        positive_prompts = contrast["positive"]
        negative_prompts = contrast["negative"]
        behavior_name = args.behavior
    else:
        if not args.negative:
            logger.error("--negative prompts required when using --positive")
            return
        positive_prompts = args.positive
        negative_prompts = args.negative
        behavior_name = "custom"

    logger.info("Behaviour: %s  |  %d positive, %d negative prompts",
                behavior_name, len(positive_prompts), len(negative_prompts))

    # ── Load model ────────────────────────────────────────────────────
    config_path = Path(args.config) if args.config else graph_path.parent / "config.json"
    if not config_path.exists():
        logger.error("Config not found at %s", config_path)
        return
    cfg = Config.load(config_path)
    cfg.device = args.device or get_device()

    logger.info("Loading model %s …", cfg.model_name)
    model, saes = load_model_and_saes(cfg)

    # ── Extract subgraph ──────────────────────────────────────────────
    logger.info("Extracting subgraph …")

    if args.behavior:
        behavior_contrast: BehaviorContrast | str = args.behavior
    else:
        behavior_contrast = BehaviorContrast(
            name="custom",
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
        )

    behavior_sub = extract_behavior_subgraph(
        model=model,
        saes=saes,
        interaction_graph=G,
        behavior=behavior_contrast,
        cfg=cfg,
        top_k_seeds=args.top_k,
        n_hops=args.hops,
    )

    sub_G = behavior_sub.subgraph
    logger.info("Subgraph: %d nodes, %d edges", sub_G.number_of_nodes(), sub_G.number_of_edges())
    logger.info("Seed features: %s", behavior_sub.seed_features[:10])

    # ── Save outputs ──────────────────────────────────────────────────
    # Save subgraph metadata
    meta = {
        "behavior": behavior_name,
        "seed_features": behavior_sub.seed_features,
        "n_nodes": sub_G.number_of_nodes(),
        "n_edges": sub_G.number_of_edges(),
        "positive_prompts": positive_prompts,
        "negative_prompts": negative_prompts,
    }
    with open(out / f"{behavior_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Render interactive HTML
    html_path = out / f"{behavior_name}_subgraph.html"
    render_interactive_graph(
        sub_G,
        str(html_path),
        seed_nodes=[sf["feature_id"] for sf in behavior_sub.seed_features],
    )
    logger.info("Visualisation → %s", html_path)
    logger.info("✅ Behaviour subgraph extraction complete.  Outputs in %s", out)


if __name__ == "__main__":
    main()
