"""
Quick-start example: build a feature interaction graph for GPT-2-small.

This script runs the full pipeline end-to-end with sensible defaults,
producing a NetworkX graph, statistics summary, and interactive HTML
visualisation.

Usage:
    python examples/quick_start.py                   # defaults
    python examples/quick_start.py --n_tokens 5000   # faster smoke test
    python examples/quick_start.py --output_dir runs/my_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_graph import (
    Config,
    build_coactivation_atlas,
    build_interaction_graph,
    collect_activations,
    identify_candidates,
    load_model_and_saes,
    measure_interactions,
)
from feature_graph.analysis import compute_graph_statistics, find_hubs
from feature_graph.graph import save_graph
from feature_graph.utils import get_device, seed_everything, setup_logging
from feature_graph.visualization import render_interactive_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick-start: GPT-2-small FIG pipeline")
    p.add_argument("--n_tokens", type=int, default=50_000,
                    help="Number of tokens to collect activations over")
    p.add_argument("--top_k", type=int, default=200,
                    help="Top-K features per layer to retain")
    p.add_argument("--output_dir", type=str, default="outputs/quick_start",
                    help="Directory for all outputs")
    p.add_argument("--method", choices=["clamping", "jacobian"], default="clamping",
                    help="Interaction measurement method")
    p.add_argument("--device", type=str, default=None,
                    help="Device override (auto-detected if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("quick_start")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Configuration ──────────────────────────────────────────────
    device = args.device or get_device()
    cfg = Config(
        model_name="gpt2",
        sae_release="gpt2-small-res-jb",
        sae_id_template="blocks.{layer}.hook_resid_post",
        hook_template="blocks.{layer}.hook_resid_post",
        layers=list(range(12)),
        n_tokens=args.n_tokens,
        activation_batch_size=32,
        top_k_features=args.top_k,
        interaction_method=args.method,
        n_inputs_per_pair=50,
        significance_level=0.01,
        correction_method="bh",
        edge_strength_threshold=0.05,
        output_dir=str(out),
        device=device,
    )
    cfg.save(out / "config.json")
    logger.info("Config saved → %s", out / "config.json")

    # ── 2. Load model + SAEs ──────────────────────────────────────────
    logger.info("Loading model and SAEs …")
    model, saes = load_model_and_saes(cfg)
    logger.info("Model: %s  |  SAE layers: %s", cfg.model_name, list(saes.keys()))

    # ── 3. Collect activations ────────────────────────────────────────
    logger.info("Collecting activations over %d tokens …", cfg.n_tokens)
    store = collect_activations(model, saes, cfg)
    store.save(out / "activations.h5")
    logger.info("Activation store saved → %s", out / "activations.h5")

    # ── 4. Co-activation atlas ────────────────────────────────────────
    logger.info("Building co-activation atlas …")
    atlas = build_coactivation_atlas(store, cfg)
    atlas.save(out / "atlas.h5")
    logger.info("Atlas saved → %s", out / "atlas.h5")

    # ── 5. Candidate identification ───────────────────────────────────
    logger.info("Identifying candidate pairs …")
    candidates = identify_candidates(atlas, saes, cfg, activation_store=store, model=model)
    logger.info("Candidate pairs: %d", len(candidates))

    # ── 6. Measure interactions ───────────────────────────────────────
    logger.info("Measuring interactions via %s …", cfg.interaction_method)
    results = measure_interactions(model, saes, candidates, cfg)
    logger.info("Significant results: %d / %d", sum(1 for r in results if r.significant), len(results))

    # ── 7. Build graph ────────────────────────────────────────────────
    logger.info("Building interaction graph …")
    G = build_interaction_graph(results, cfg)
    save_graph(G, out / "graph.graphml", format="graphml")
    logger.info("Graph: %d nodes, %d edges → %s", G.number_of_nodes(), G.number_of_edges(), out / "graph.graphml")

    # ── 8. Analyse ────────────────────────────────────────────────────
    stats = compute_graph_statistics(G)
    logger.info("\n%s", stats.summary())
    with open(out / "statistics.json", "w") as f:
        json.dump({
            "n_nodes": stats.n_nodes,
            "n_edges": stats.n_edges,
            "density": stats.density,
            "clustering_coefficient": stats.clustering_coefficient,
            "reciprocity": stats.reciprocity,
            "n_communities": stats.n_communities,
            "edge_type_counts": stats.edge_type_counts,
        }, f, indent=2)
    hubs = find_hubs(G, top_k=10)
    logger.info("Top-10 hubs (by degree): %s", [h["id"] for h in hubs])

    # ── 9. Visualise ──────────────────────────────────────────────────
    html_path = out / "graph.html"
    render_interactive_graph(G, str(html_path))
    logger.info("Interactive visualisation → %s", html_path)

    logger.info("✅ Pipeline complete.  All outputs in %s", out)


if __name__ == "__main__":
    main()
