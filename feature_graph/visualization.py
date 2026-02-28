"""
Interactive visualization of feature interaction graphs.

Provides multiple visualization modes:
1. Full graph visualization (interactive HTML via pyvis)
2. Feature neighborhood explorer
3. Behavior subgraph diagrams
4. Steering cascade overlay
5. Layer-organized layout
6. Static plots for papers (via plotly)

Design decision: We use pyvis for interactive graph visualization (produces
standalone HTML files that work in any browser) and plotly for statistical
plots (degree distributions, layer decay curves, etc.). This avoids requiring
a running web server while still providing rich interactivity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from feature_graph.utils import parse_feature_id

logger = logging.getLogger("feature_graph")

# Color scheme for interaction types
COLORS = {
    "excitatory": "#2ecc71",   # Green
    "inhibitory": "#e74c3c",   # Red
    "gating": "#f39c12",       # Orange
    "default_node": "#3498db", # Blue
    "hub_node": "#9b59b6",     # Purple
    "seed_node": "#e67e22",    # Dark orange
}


def render_interactive_graph(
    G: nx.DiGraph,
    output_path: str | Path = "interaction_graph.html",
    title: str = "Feature Interaction Graph",
    height: str = "900px",
    width: str = "100%",
    highlight_hubs: bool = True,
    hub_threshold_percentile: float = 90,
    physics: bool = True,
    seed_nodes: Optional[list[str]] = None,
) -> Path:
    """Render the interaction graph as an interactive HTML file.

    Args:
        G: NetworkX DiGraph.
        output_path: Path for the output HTML file.
        title: Title shown in the visualization.
        height: CSS height of the visualization.
        width: CSS width of the visualization.
        highlight_hubs: Whether to make hub nodes larger/different color.
        hub_threshold_percentile: Percentile of degree above which a node is a "hub".
        physics: Whether to enable physics simulation for layout.
        seed_nodes: Optional list of feature IDs to highlight as seed nodes.

    Returns:
        Path to the generated HTML file.
    """
    from pyvis.network import Network

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        heading=title,
        bgcolor="#ffffff",
        font_color="#333333",
    )

    if not physics:
        net.toggle_physics(False)

    # Determine hub threshold
    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        hub_degree = np.percentile(degrees, hub_threshold_percentile) if degrees else 0
    else:
        hub_degree = 0

    seed_set = set(seed_nodes) if seed_nodes else set()

    # Add nodes
    for node_id, attrs in G.nodes(data=True):
        layer = attrs.get("layer", 0)
        feature_idx = attrs.get("feature_idx", 0)
        label_text = attrs.get("label", "")
        importance = attrs.get("importance", 0)
        freq = attrs.get("activation_freq", 0)
        degree = G.degree(node_id)

        # Determine size and color
        if node_id in seed_set:
            color = COLORS["seed_node"]
            size = 25
        elif highlight_hubs and degree >= hub_degree and hub_degree > 0:
            color = COLORS["hub_node"]
            size = 15 + min(degree * 2, 30)
        else:
            color = COLORS["default_node"]
            size = 10

        # Build hover title
        title_parts = [
            f"<b>{node_id}</b>",
            f"Layer: {layer}, Feature: {feature_idx}",
        ]
        if label_text:
            title_parts.append(f"Label: {label_text}")
        title_parts.extend([
            f"Importance: {importance:.4f}",
            f"Activation freq: {freq:.4f}",
            f"In-degree: {G.in_degree(node_id)}, Out-degree: {G.out_degree(node_id)}",
        ])
        hover_title = "<br>".join(title_parts)

        display_label = label_text if label_text else node_id

        net.add_node(
            node_id,
            label=display_label,
            title=hover_title,
            color=color,
            size=size,
            level=layer,  # For hierarchical layout
        )

    # Add edges
    for src, tgt, attrs in G.edges(data=True):
        interaction_type = attrs.get("interaction_type", "excitatory")
        strength = attrs.get("strength", 0)
        abs_strength = attrs.get("abs_strength", 0)
        p_value = attrs.get("p_value", 1)

        color = COLORS.get(interaction_type, "#999999")
        width = max(1, min(abs_strength * 10, 8))

        title_parts = [
            f"<b>{src} → {tgt}</b>",
            f"Type: {interaction_type}",
            f"Strength: {strength:.4f}",
            f"|Strength|: {abs_strength:.4f}",
            f"p-value: {p_value:.4f}",
            f"Gating score: {attrs.get('gating_score', 0):.2f}",
        ]
        hover_title = "<br>".join(title_parts)

        # Dashed lines for gating
        dashes = interaction_type == "gating"

        net.add_edge(
            src, tgt,
            color=color,
            width=width,
            title=hover_title,
            dashes=dashes,
            arrows="to",
        )

    # Configure physics and layout
    net.set_options(json.dumps({
        "nodes": {
            "font": {"size": 12},
            "borderWidth": 2,
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"scaleFactor": 0.5}},
        },
        "physics": {
            "enabled": physics,
            "hierarchicalRepulsion": {
                "nodeDistance": 150,
                "centralGravity": 0.0,
            },
            "solver": "hierarchicalRepulsion" if physics else "barnesHut",
        },
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",  # Up-Down: earlier layers at top
                "sortMethod": "directed",
                "levelSeparation": 200,
            }
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True,
        },
    }))

    net.save_graph(str(output_path))
    logger.info(f"Saved interactive visualization to {output_path}")
    return output_path


def render_neighborhood(
    G: nx.DiGraph,
    center_feature: str,
    n_hops: int = 1,
    output_path: str | Path = "neighborhood.html",
) -> Path:
    """Render the local neighborhood of a specific feature.

    Shows all features within n_hops and their interactions.
    """
    from feature_graph.graph import get_subgraph

    subgraph = get_subgraph(G, [center_feature], n_hops=n_hops)

    return render_interactive_graph(
        subgraph,
        output_path=output_path,
        title=f"Neighborhood of {center_feature}",
        seed_nodes=[center_feature],
        physics=True,
    )


def render_cascade_overlay(
    G: nx.DiGraph,
    cascade_prediction,
    output_path: str | Path = "cascade.html",
) -> Path:
    """Render the interaction graph with cascade prediction overlay.

    Shows which features are predicted to be affected by a steering intervention,
    with color coding for excited (green), inhibited (red), and gated (orange).
    """
    # Create a subgraph containing affected features
    affected_ids = set(cascade_prediction.predicted_effects.keys())
    affected_ids.add(cascade_prediction.steered_feature)

    # Expand to include connecting edges
    subgraph = G.subgraph(
        [n for n in G.nodes() if n in affected_ids]
    ).copy()

    return render_interactive_graph(
        subgraph,
        output_path=output_path,
        title=f"Cascade from steering {cascade_prediction.steered_feature}",
        seed_nodes=[cascade_prediction.steered_feature],
        physics=True,
    )


def plot_degree_distribution(G: nx.DiGraph, output_path: Optional[str | Path] = None):
    """Plot degree distribution (in, out, and total) with power-law fit overlay."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=["In-Degree", "Out-Degree", "Total Degree"])

    for col, (title, degrees) in enumerate([
        ("In-Degree", [d for _, d in G.in_degree()]),
        ("Out-Degree", [d for _, d in G.out_degree()]),
        ("Total Degree", [d for _, d in G.degree()]),
    ], 1):
        if not degrees or max(degrees) == 0:
            continue

        # Histogram
        fig.add_trace(
            go.Histogram(x=degrees, nbinsx=50, name=title, marker_color=COLORS["default_node"]),
            row=1, col=col,
        )

    fig.update_layout(
        title="Degree Distribution",
        showlegend=False,
        height=400,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig


def plot_layer_distance_decay(
    G: nx.DiGraph,
    output_path: Optional[str | Path] = None,
):
    """Plot mean interaction strength vs. layer distance."""
    import plotly.graph_objects as go

    from feature_graph.analysis import compute_layer_distance_decay

    decay = compute_layer_distance_decay(G)
    if not decay:
        logger.warning("No layer distance data to plot")
        return None

    distances = sorted(decay.keys())
    strengths = [decay[d] for d in distances]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=distances,
        y=strengths,
        marker_color=COLORS["default_node"],
    ))
    fig.update_layout(
        title="Interaction Strength vs. Layer Distance",
        xaxis_title="Layer Distance",
        yaxis_title="Mean |Interaction Strength|",
        height=400,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig


def plot_interaction_type_breakdown(
    G: nx.DiGraph,
    output_path: Optional[str | Path] = None,
):
    """Plot breakdown of interaction types (excitatory, inhibitory, gating)."""
    import plotly.graph_objects as go

    type_counts = {"excitatory": 0, "inhibitory": 0, "gating": 0}
    for _, _, data in G.edges(data=True):
        t = data.get("interaction_type", "excitatory")
        if t in type_counts:
            type_counts[t] += 1

    fig = go.Figure(data=[go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        marker_colors=[COLORS[t] for t in type_counts.keys()],
        hole=0.4,
    )])
    fig.update_layout(
        title="Interaction Type Breakdown",
        height=400,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig
