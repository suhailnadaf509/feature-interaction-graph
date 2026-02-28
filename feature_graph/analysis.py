"""
Graph-theoretic analysis utilities.

Implements the analyses from Experiment 3 of the research document:
- Degree distribution analysis (scale-free testing)
- Clustering coefficient and modularity
- Community detection (Louvain/Leiden)
- Hub identification
- Excitatory vs. inhibitory subgraph comparison
- Layer-distance decay analysis
- Motif counting (excitatory chains, inhibitory competition, gating triangles)
- Small-world and assortativity metrics
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy import stats as scipy_stats

from feature_graph.utils import parse_feature_id

logger = logging.getLogger("feature_graph")


@dataclass
class GraphStatistics:
    """Comprehensive statistics about the interaction graph."""

    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0

    # Degree statistics
    mean_in_degree: float = 0.0
    mean_out_degree: float = 0.0
    max_in_degree: int = 0
    max_out_degree: int = 0
    degree_distribution_power_law_alpha: float = 0.0  # Fitted power-law exponent
    degree_distribution_power_law_p: float = 0.0  # Goodness of fit p-value

    # Clustering
    avg_clustering: float = 0.0
    transitivity: float = 0.0

    # Path statistics
    n_weakly_connected_components: int = 0
    largest_wcc_size: int = 0
    avg_shortest_path_in_largest_wcc: float = 0.0

    # Assortativity
    degree_assortativity: float = 0.0

    # Reciprocity
    reciprocity: float = 0.0

    # Edge type breakdown
    n_excitatory: int = 0
    n_inhibitory: int = 0
    n_gating: int = 0

    # Layer distance statistics
    mean_edge_layer_distance: float = 0.0
    layer_distance_distribution: dict[int, int] = field(default_factory=dict)

    # Community structure
    n_communities: int = 0
    modularity: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=== Feature Interaction Graph Statistics ===",
            f"Nodes: {self.n_nodes}",
            f"Edges: {self.n_edges} (density: {self.density:.4f})",
            f"  Excitatory: {self.n_excitatory}",
            f"  Inhibitory: {self.n_inhibitory}",
            f"  Gating: {self.n_gating}",
            "",
            f"Mean in-degree: {self.mean_in_degree:.2f}, max: {self.max_in_degree}",
            f"Mean out-degree: {self.mean_out_degree:.2f}, max: {self.max_out_degree}",
            f"Degree distribution power-law α: {self.degree_distribution_power_law_alpha:.2f} "
            f"(p={self.degree_distribution_power_law_p:.3f})",
            "",
            f"Avg clustering coefficient: {self.avg_clustering:.4f}",
            f"Transitivity: {self.transitivity:.4f}",
            f"Reciprocity: {self.reciprocity:.4f}",
            f"Degree assortativity: {self.degree_assortativity:.4f}",
            "",
            f"Weakly connected components: {self.n_weakly_connected_components}",
            f"Largest WCC size: {self.largest_wcc_size}",
            f"Avg shortest path (largest WCC): {self.avg_shortest_path_in_largest_wcc:.2f}",
            "",
            f"Mean edge layer distance: {self.mean_edge_layer_distance:.2f}",
            f"Communities detected: {self.n_communities}",
            f"Modularity: {self.modularity:.4f}",
        ]
        return "\n".join(lines)


def compute_graph_statistics(G: nx.DiGraph) -> GraphStatistics:
    """Compute comprehensive statistics about the interaction graph."""
    stats = GraphStatistics()

    if G.number_of_nodes() == 0:
        return stats

    stats.n_nodes = G.number_of_nodes()
    stats.n_edges = G.number_of_edges()
    stats.density = nx.density(G)

    # Degree statistics
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    stats.mean_in_degree = float(np.mean(in_degrees)) if in_degrees else 0
    stats.mean_out_degree = float(np.mean(out_degrees)) if out_degrees else 0
    stats.max_in_degree = max(in_degrees) if in_degrees else 0
    stats.max_out_degree = max(out_degrees) if out_degrees else 0

    # Power-law fit on total degree
    total_degrees = [d for _, d in G.degree()]
    nonzero_degrees = [d for d in total_degrees if d > 0]
    if len(nonzero_degrees) > 10:
        try:
            alpha, loc, scale = scipy_stats.powerlaw.fit(nonzero_degrees)
            stats.degree_distribution_power_law_alpha = float(alpha)
            _, p = scipy_stats.kstest(nonzero_degrees, "powerlaw", args=(alpha, loc, scale))
            stats.degree_distribution_power_law_p = float(p)
        except Exception:
            pass

    # Clustering
    G_undirected = G.to_undirected()
    stats.avg_clustering = float(nx.average_clustering(G_undirected))
    stats.transitivity = float(nx.transitivity(G_undirected))

    # Reciprocity
    stats.reciprocity = float(nx.reciprocity(G)) if G.number_of_edges() > 0 else 0

    # Assortativity
    try:
        stats.degree_assortativity = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        stats.degree_assortativity = 0.0

    # Connected components
    wccs = list(nx.weakly_connected_components(G))
    stats.n_weakly_connected_components = len(wccs)
    if wccs:
        largest_wcc = max(wccs, key=len)
        stats.largest_wcc_size = len(largest_wcc)

        # Average shortest path in largest WCC
        if len(largest_wcc) > 1 and len(largest_wcc) < 5000:
            subg = G.subgraph(largest_wcc)
            try:
                stats.avg_shortest_path_in_largest_wcc = float(
                    nx.average_shortest_path_length(subg)
                )
            except nx.NetworkXError:
                stats.avg_shortest_path_in_largest_wcc = float("inf")

    # Edge type breakdown
    for _, _, data in G.edges(data=True):
        t = data.get("interaction_type", "unknown")
        if t == "excitatory":
            stats.n_excitatory += 1
        elif t == "inhibitory":
            stats.n_inhibitory += 1
        elif t == "gating":
            stats.n_gating += 1

    # Layer distance distribution
    distances = []
    for u, v, data in G.edges(data=True):
        try:
            l_u = G.nodes[u].get("layer", 0)
            l_v = G.nodes[v].get("layer", 0)
            # Try parsing from node ID if not in attributes
            if l_u == 0 and l_v == 0:
                l_u, _ = parse_feature_id(u)
                l_v, _ = parse_feature_id(v)
            dist = abs(l_v - l_u)
            distances.append(dist)
        except Exception:
            continue

    if distances:
        stats.mean_edge_layer_distance = float(np.mean(distances))
        stats.layer_distance_distribution = dict(Counter(distances))

    # Community detection
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
        stats.n_communities = len(communities)
        stats.modularity = float(nx.community.modularity(G_undirected, communities))
    except Exception:
        stats.n_communities = 0
        stats.modularity = 0.0

    return stats


def find_hubs(
    G: nx.DiGraph,
    top_k: int = 20,
    metric: str = "total_degree",
) -> list[dict]:
    """Find hub features in the interaction graph.

    Args:
        G: Interaction graph.
        top_k: Number of top hubs to return.
        metric: 'total_degree', 'in_degree', 'out_degree', 'betweenness', or 'pagerank'.

    Returns:
        List of dicts with hub feature info, sorted by importance.
    """
    if G.number_of_nodes() == 0:
        return []

    if metric == "total_degree":
        scores = dict(G.degree())
    elif metric == "in_degree":
        scores = dict(G.in_degree())
    elif metric == "out_degree":
        scores = dict(G.out_degree())
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "pagerank":
        scores = nx.pagerank(G)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    hubs = []
    for node_id, score in sorted_nodes:
        attrs = G.nodes[node_id]
        hubs.append({
            "id": node_id,
            "layer": attrs.get("layer", 0),
            "feature_idx": attrs.get("feature_idx", 0),
            "label": attrs.get("label", ""),
            f"{metric}_score": score,
            "in_degree": G.in_degree(node_id),
            "out_degree": G.out_degree(node_id),
            "importance": attrs.get("importance", 0),
        })

    return hubs


def detect_communities(G: nx.DiGraph) -> list[set[str]]:
    """Detect communities using the Louvain algorithm.

    Returns:
        List of sets, each set containing feature IDs in one community.
    """
    G_undirected = G.to_undirected()
    communities = nx.community.louvain_communities(G_undirected, seed=42)
    return [set(c) for c in communities]


def get_edge_type_subgraph(G: nx.DiGraph, edge_type: str) -> nx.DiGraph:
    """Extract subgraph containing only edges of a specific type.

    Args:
        G: Full interaction graph.
        edge_type: 'excitatory', 'inhibitory', or 'gating'.

    Returns:
        Subgraph with only the specified edge type.
    """
    edges = [
        (u, v, data)
        for u, v, data in G.edges(data=True)
        if data.get("interaction_type") == edge_type
    ]
    subG = nx.DiGraph()
    subG.add_nodes_from(G.nodes(data=True))
    subG.add_edges_from(edges)
    # Remove isolated nodes
    isolates = list(nx.isolates(subG))
    subG.remove_nodes_from(isolates)
    return subG


def compute_layer_distance_decay(G: nx.DiGraph) -> dict[int, float]:
    """Compute mean interaction strength as a function of layer distance.

    Returns:
        Dict mapping layer_distance -> mean_abs_strength.
    """
    distance_strengths: dict[int, list[float]] = {}

    for u, v, data in G.edges(data=True):
        try:
            l_u, _ = parse_feature_id(u)
            l_v, _ = parse_feature_id(v)
            dist = abs(l_v - l_u)
            strength = abs(data.get("strength", 0))
            distance_strengths.setdefault(dist, []).append(strength)
        except Exception:
            continue

    return {
        dist: float(np.mean(strengths))
        for dist, strengths in sorted(distance_strengths.items())
    }


def count_motifs(G: nx.DiGraph) -> dict[str, int]:
    """Count recurring interaction motifs in the graph.

    Motifs detected:
    - excitatory_chain: A →+ B →+ C (chain of excitation)
    - inhibitory_competition: A →- B, A →- C (one feature inhibits multiple)
    - gating_triangle: A → B, C → B where one edge is gating
    - mutual_excitation: A →+ B, B →+ A (positive feedback)
    - feedforward_fan: A →+ B, A →+ C, A →+ D (one feature excites many)
    """
    motifs = {
        "excitatory_chain": 0,
        "inhibitory_competition": 0,
        "gating_triangle": 0,
        "mutual_excitation": 0,
        "feedforward_fan": 0,
    }

    # Excitatory chains: A →+ B →+ C
    exc_edges = {
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("interaction_type") == "excitatory"
    }
    for a, b in exc_edges:
        for _, c in G.out_edges(b):
            if (b, c) in exc_edges and c != a:
                motifs["excitatory_chain"] += 1

    # Mutual excitation: A →+ B and B →+ A
    for a, b in exc_edges:
        if (b, a) in exc_edges:
            motifs["mutual_excitation"] += 1
    motifs["mutual_excitation"] //= 2  # Each pair counted twice

    # Inhibitory competition: A →- B and A →- C
    inh_targets: dict[str, list[str]] = {}
    for u, v, d in G.edges(data=True):
        if d.get("interaction_type") == "inhibitory":
            inh_targets.setdefault(u, []).append(v)
    for src, targets in inh_targets.items():
        n = len(targets)
        motifs["inhibitory_competition"] += n * (n - 1) // 2

    # Feedforward fan: node with out-degree >= 3 in excitatory subgraph
    exc_out: dict[str, int] = {}
    for a, b in exc_edges:
        exc_out[a] = exc_out.get(a, 0) + 1
    for node, deg in exc_out.items():
        if deg >= 3:
            motifs["feedforward_fan"] += 1

    # Gating triangles: A → B, C → B where at least one edge is gating
    for b in G.nodes():
        predecessors = list(G.predecessors(b))
        gating_preds = [
            p for p in predecessors
            if G.edges[p, b].get("interaction_type") == "gating"
        ]
        non_gating_preds = [
            p for p in predecessors
            if G.edges[p, b].get("interaction_type") != "gating"
        ]
        motifs["gating_triangle"] += len(gating_preds) * len(non_gating_preds)

    return motifs
