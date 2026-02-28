"""
CLI for the Feature Interaction Graph pipeline.

Provides commands for each major stage, plus a full pipeline command.
Uses argparse with subcommands — simple, no framework dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from feature_graph.config import Config
from feature_graph.utils import setup_logging, set_seed


def main():
    parser = argparse.ArgumentParser(
        prog="feature-graph",
        description="Feature Interaction Graph: Map causal interactions between SAE features",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── run-pipeline ─────────────────────────────────────────────────────
    p_pipeline = subparsers.add_parser("run-pipeline", help="Run the full pipeline end-to-end")
    _add_common_args(p_pipeline)
    p_pipeline.add_argument("--skip-viz", action="store_true", help="Skip visualization generation")

    # ── collect-activations ──────────────────────────────────────────────
    p_collect = subparsers.add_parser("collect-activations", help="Collect SAE feature activations")
    _add_common_args(p_collect)
    p_collect.add_argument("--output", default="outputs/activations.h5", help="Output HDF5 path")

    # ── build-atlas ──────────────────────────────────────────────────────
    p_atlas = subparsers.add_parser("build-atlas", help="Build co-activation atlas")
    p_atlas.add_argument("--activations", required=True, help="Path to activations HDF5")
    p_atlas.add_argument("--output", default="outputs/atlas.h5", help="Output HDF5 path")
    _add_common_args(p_atlas)

    # ── measure-interactions ─────────────────────────────────────────────
    p_measure = subparsers.add_parser("measure-interactions", help="Measure causal interactions")
    p_measure.add_argument("--atlas", required=True, help="Path to atlas HDF5")
    p_measure.add_argument("--output", default="outputs/interactions.json", help="Output JSON path")
    _add_common_args(p_measure)

    # ── build-graph ──────────────────────────────────────────────────────
    p_graph = subparsers.add_parser("build-graph", help="Build interaction graph from measurements")
    p_graph.add_argument("--interactions", required=True, help="Path to interactions JSON")
    p_graph.add_argument("--output", default="outputs/graph.graphml", help="Output GraphML path")
    p_graph.add_argument("--format", default="graphml", choices=["graphml", "json", "pickle"])
    _add_common_args(p_graph)

    # ── analyze ──────────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Compute graph statistics and analysis")
    p_analyze.add_argument("--graph", required=True, help="Path to graph file")
    p_analyze.add_argument("--format", default="graphml", choices=["graphml", "json", "pickle"])
    p_analyze.add_argument("--output", default="outputs/analysis.json", help="Output analysis JSON")

    # ── visualize ────────────────────────────────────────────────────────
    p_viz = subparsers.add_parser("visualize", help="Generate interactive visualization")
    p_viz.add_argument("--graph", required=True, help="Path to graph file")
    p_viz.add_argument("--format", default="graphml", choices=["graphml", "json", "pickle"])
    p_viz.add_argument("--output", default="outputs/graph.html", help="Output HTML path")

    # ── extract-subgraph ─────────────────────────────────────────────────
    p_sub = subparsers.add_parser("extract-subgraph", help="Extract behavior-specific subgraph")
    p_sub.add_argument("--graph", required=True, help="Path to graph file")
    p_sub.add_argument("--behavior", required=True, help="Behavior name (e.g., 'factual_recall', 'refusal')")
    p_sub.add_argument("--output", default="outputs/subgraph.html", help="Output HTML path")
    _add_common_args(p_sub)

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run-pipeline":
        _cmd_run_pipeline(args)
    elif args.command == "collect-activations":
        _cmd_collect_activations(args)
    elif args.command == "build-atlas":
        _cmd_build_atlas(args)
    elif args.command == "measure-interactions":
        _cmd_measure_interactions(args)
    elif args.command == "build-graph":
        _cmd_build_graph(args)
    elif args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "visualize":
        _cmd_visualize(args)
    elif args.command == "extract-subgraph":
        _cmd_extract_subgraph(args)


def _add_common_args(parser):
    """Add common arguments shared across commands."""
    parser.add_argument("--model", default="gpt2-small", help="Model name")
    parser.add_argument("--sae-release", default="gpt2-small-res-jb", help="SAE release name")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layers to analyze")
    parser.add_argument("--n-tokens", type=int, default=100_000, help="Number of tokens")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K features by importance")
    parser.add_argument("--layer-window", type=int, default=3, help="Layer window for interactions")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default=None, help="Path to config JSON (overrides other args)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--method", default="clamping", choices=["clamping", "jacobian", "both"],
                       help="Interaction measurement method")


def _build_config(args) -> Config:
    """Build Config from CLI arguments."""
    if hasattr(args, "config") and args.config is not None:
        cfg = Config.load(args.config)
    else:
        kwargs = {}
        if hasattr(args, "model"):
            kwargs["model_name"] = args.model
        if hasattr(args, "sae_release"):
            kwargs["sae_release"] = args.sae_release
        if hasattr(args, "layers") and args.layers is not None:
            kwargs["layers"] = args.layers
        if hasattr(args, "n_tokens"):
            kwargs["n_tokens"] = args.n_tokens
        if hasattr(args, "top_k"):
            kwargs["top_k_features"] = args.top_k
        if hasattr(args, "layer_window"):
            kwargs["layer_window"] = args.layer_window
        if hasattr(args, "device"):
            kwargs["device"] = args.device
        if hasattr(args, "seed"):
            kwargs["seed"] = args.seed
        if hasattr(args, "output_dir"):
            kwargs["output_dir"] = args.output_dir
        if hasattr(args, "method"):
            kwargs["interaction_method"] = args.method
        cfg = Config(**kwargs)

    set_seed(cfg.seed)
    return cfg


def _cmd_run_pipeline(args):
    """Run the full pipeline end-to-end."""
    cfg = _build_config(args)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(output_dir / "config.json")

    logger = logging.getLogger("feature_graph")
    logger.info("=" * 60)
    logger.info("Feature Interaction Graph — Full Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"SAE: {cfg.sae_release}")
    logger.info(f"Layers: {cfg.layers}")
    logger.info(f"Tokens: {cfg.n_tokens:,}")
    logger.info(f"Top-K features: {cfg.top_k_features}")

    # Step 1: Load model and SAEs
    from feature_graph.loading import load_model_and_saes
    logger.info("\n[1/6] Loading model and SAEs...")
    model, saes = load_model_and_saes(cfg)

    # Step 2: Collect activations
    from feature_graph.activations import collect_activations
    logger.info("\n[2/6] Collecting activations...")
    acts = collect_activations(model, saes, cfg)
    acts.save(output_dir / "activations.h5")

    # Step 3: Build co-activation atlas
    from feature_graph.coactivation import build_coactivation_atlas
    logger.info("\n[3/6] Building co-activation atlas...")
    atlas = build_coactivation_atlas(acts, cfg)
    atlas.save(output_dir / "atlas.h5")

    # Step 4: Identify candidates and measure interactions
    from feature_graph.candidates import identify_candidates
    from feature_graph.interactions import measure_interactions
    logger.info("\n[4/6] Identifying candidates...")
    candidates = identify_candidates(atlas, saes, cfg, activation_store=acts, model=model)

    logger.info(f"\n[5/6] Measuring interactions ({len(candidates)} candidates)...")
    interactions = measure_interactions(model, saes, candidates, cfg)

    # Save interactions
    import json
    interactions_data = [r.to_dict() for r in interactions]
    with open(output_dir / "interactions.json", "w") as f:
        json.dump(interactions_data, f, indent=2)

    # Step 5: Build graph
    from feature_graph.graph import build_interaction_graph, save_graph
    logger.info("\n[6/6] Building interaction graph...")
    G = build_interaction_graph(interactions, cfg)
    save_graph(G, output_dir / "graph.graphml", format="graphml")
    save_graph(G, output_dir / "graph.json", format="json")

    # Analysis
    from feature_graph.analysis import compute_graph_statistics, find_hubs, count_motifs
    stats = compute_graph_statistics(G)
    logger.info("\n" + stats.summary())

    hubs = find_hubs(G, top_k=10)
    if hubs:
        logger.info("\nTop 10 Hub Features:")
        for h in hubs:
            logger.info(f"  {h['id']}: degree={h.get('total_degree_score', 0)}, "
                       f"in={h['in_degree']}, out={h['out_degree']}")

    motifs = count_motifs(G)
    logger.info(f"\nMotif counts: {motifs}")

    # Visualization
    if not args.skip_viz:
        from feature_graph.visualization import (
            render_interactive_graph,
            plot_degree_distribution,
            plot_layer_distance_decay,
            plot_interaction_type_breakdown,
        )
        render_interactive_graph(G, output_dir / "graph.html")
        plot_degree_distribution(G, output_dir / "degree_distribution.html")
        plot_layer_distance_decay(G, output_dir / "layer_decay.html")
        plot_interaction_type_breakdown(G, output_dir / "type_breakdown.html")

    logger.info(f"\nPipeline complete! Outputs in {output_dir}/")


def _cmd_collect_activations(args):
    """Collect SAE feature activations."""
    cfg = _build_config(args)
    from feature_graph.loading import load_model_and_saes
    from feature_graph.activations import collect_activations

    model, saes = load_model_and_saes(cfg)
    acts = collect_activations(model, saes, cfg)
    acts.save(args.output)
    logging.getLogger("feature_graph").info(f"Saved activations to {args.output}")


def _cmd_build_atlas(args):
    """Build co-activation atlas."""
    cfg = _build_config(args)
    from feature_graph.activations import ActivationStore
    from feature_graph.coactivation import build_coactivation_atlas

    acts = ActivationStore.load(args.activations)
    atlas = build_coactivation_atlas(acts, cfg)
    atlas.save(args.output)
    logging.getLogger("feature_graph").info(f"Saved atlas to {args.output}")


def _cmd_measure_interactions(args):
    """Measure causal interactions."""
    cfg = _build_config(args)
    from feature_graph.loading import load_model_and_saes
    from feature_graph.activations import ActivationStore
    from feature_graph.coactivation import CoactivationAtlas
    from feature_graph.candidates import identify_candidates
    from feature_graph.interactions import measure_interactions

    model, saes = load_model_and_saes(cfg)
    atlas = CoactivationAtlas.load(args.atlas)
    candidates = identify_candidates(atlas, saes, cfg)
    interactions = measure_interactions(model, saes, candidates, cfg)

    interactions_data = [r.to_dict() for r in interactions]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(interactions_data, f, indent=2)


def _cmd_build_graph(args):
    """Build interaction graph from measurements."""
    cfg = _build_config(args)
    from feature_graph.interactions import InteractionResult
    from feature_graph.graph import build_interaction_graph, save_graph

    with open(args.interactions) as f:
        data = json.load(f)
    interactions = [InteractionResult(**d) for d in data]
    G = build_interaction_graph(interactions, cfg)
    save_graph(G, args.output, format=args.format)


def _cmd_analyze(args):
    """Compute graph statistics."""
    from feature_graph.graph import load_graph
    from feature_graph.analysis import compute_graph_statistics, find_hubs, count_motifs

    G = load_graph(args.graph, format=args.format)
    stats = compute_graph_statistics(G)
    print(stats.summary())

    hubs = find_hubs(G, top_k=20)
    motifs = count_motifs(G)

    output = {
        "statistics": {k: v for k, v in stats.__dict__.items()},
        "hubs": hubs,
        "motifs": motifs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def _cmd_visualize(args):
    """Generate interactive visualization."""
    from feature_graph.graph import load_graph
    from feature_graph.visualization import render_interactive_graph

    G = load_graph(args.graph, format=args.format)
    render_interactive_graph(G, output_path=args.output)


def _cmd_extract_subgraph(args):
    """Extract behavior-specific subgraph."""
    cfg = _build_config(args)
    from feature_graph.loading import load_model_and_saes
    from feature_graph.graph import load_graph
    from feature_graph.subgraphs import extract_behavior_subgraph
    from feature_graph.visualization import render_interactive_graph

    model, saes = load_model_and_saes(cfg)
    G = load_graph(args.graph)
    result = extract_behavior_subgraph(model, saes, G, args.behavior, cfg)
    render_interactive_graph(
        result.subgraph,
        output_path=args.output,
        title=f"Subgraph: {result.behavior.name}",
        seed_nodes=[sf["feature_id"] for sf in result.seed_features],
    )


if __name__ == "__main__":
    main()
