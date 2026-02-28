"""
Interaction graph construction, typed edges, and serialization.

Takes InteractionResult objects and builds a NetworkX DiGraph with:
- Nodes: SAE features (with layer, feature_idx, importance, label attributes)
- Edges: Typed interactions (excitatory, inhibitory, gating) with strength, p-value, etc.

Serialization formats:
- GraphML: Standard, interoperable, human-readable XML
- JSON: Easy to load in web visualizations
- Pickle: Fast Python-native serialization
- PyTorch Geometric: For GNN-based downstream analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from feature_graph.config import Config
from feature_graph.interactions import InteractionResult
from feature_graph.candidates import FeatureInfo
from feature_graph.utils import feature_id

logger = logging.getLogger("feature_graph")


def build_interaction_graph(
    interactions: list[InteractionResult],
    cfg: Config,
    feature_info: Optional[dict[str, FeatureInfo]] = None,
) -> nx.DiGraph:
    """Build the feature interaction graph from measured interactions.

    Args:
        interactions: List of InteractionResult from measure_interactions().
        cfg: Configuration.
        feature_info: Optional dict mapping feature_id -> FeatureInfo for node attributes.

    Returns:
        NetworkX DiGraph with typed, weighted edges.
    """
    G = nx.DiGraph()

    # Filter by significance and strength
    significant = [
        r for r in interactions
        if r.p_value < cfg.significance_level and r.abs_strength > cfg.edge_strength_threshold
    ]

    logger.info(
        f"Building graph: {len(significant)}/{len(interactions)} interactions pass "
        f"significance (p<{cfg.significance_level}) and strength (>{cfg.edge_strength_threshold}) thresholds"
    )

    # Add nodes and edges
    for r in significant:
        src_id = r.src_id
        tgt_id = r.tgt_id

        # Add source node if not present
        if src_id not in G:
            node_attrs = {
                "layer": r.src_layer,
                "feature_idx": r.src_feature,
                "label": "",
                "importance": 0.0,
                "activation_freq": 0.0,
            }
            if feature_info and src_id in feature_info:
                fi = feature_info[src_id]
                node_attrs["importance"] = fi.importance
                node_attrs["activation_freq"] = fi.activation_freq
                node_attrs["label"] = fi.label
            G.add_node(src_id, **node_attrs)

        # Add target node if not present
        if tgt_id not in G:
            node_attrs = {
                "layer": r.tgt_layer,
                "feature_idx": r.tgt_feature,
                "label": "",
                "importance": 0.0,
                "activation_freq": 0.0,
            }
            if feature_info and tgt_id in feature_info:
                fi = feature_info[tgt_id]
                node_attrs["importance"] = fi.importance
                node_attrs["activation_freq"] = fi.activation_freq
                node_attrs["label"] = fi.label
            G.add_node(tgt_id, **node_attrs)

        # Add edge
        edge_attrs = {
            "interaction_type": r.interaction_type,
            "strength": r.mean_strength,
            "abs_strength": r.abs_strength,
            "p_value": r.p_value,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "std_strength": r.std_strength,
            "gating_score": r.gating_score,
            "n_samples": r.n_samples,
            "method": r.method,
        }
        G.add_edge(src_id, tgt_id, **edge_attrs)

    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Log type breakdown
    type_counts = {}
    for _, _, data in G.edges(data=True):
        t = data["interaction_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        logger.info(f"  {t}: {c} edges")

    return G


def save_graph(G: nx.DiGraph, path: str | Path, format: str = "graphml") -> None:
    """Save interaction graph to file.

    Args:
        G: NetworkX DiGraph.
        path: Output file path.
        format: 'graphml', 'json', or 'pickle'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "graphml":
        nx.write_graphml(G, path)
    elif format == "json":
        data = nx.node_link_data(G)
        # Convert numpy types to Python types for JSON serialization
        data = _convert_numpy_types(data)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    elif format == "pickle":
        import pickle
        with open(path, "wb") as f:
            pickle.dump(G, f)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved graph to {path} ({format})")


def load_graph(path: str | Path, format: str = "graphml") -> nx.DiGraph:
    """Load interaction graph from file."""
    path = Path(path)

    if format == "graphml":
        return nx.read_graphml(path)
    elif format == "json":
        with open(path) as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    elif format == "pickle":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def to_pytorch_geometric(G: nx.DiGraph):
    """Convert interaction graph to PyTorch Geometric Data object.

    Requires torch_geometric to be installed.

    Returns:
        torch_geometric.data.Data with:
            - x: Node features (layer, importance, activation_freq)
            - edge_index: Edge connectivity
            - edge_attr: Edge features (strength, abs_strength, type_encoding)
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.utils import from_networkx
    except ImportError:
        raise ImportError("PyTorch Geometric is required. Install with: pip install torch-geometric")

    # Create numeric node features
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.zeros(len(node_list), 3)
    for i, node in enumerate(node_list):
        attrs = G.nodes[node]
        x[i, 0] = attrs.get("layer", 0)
        x[i, 1] = attrs.get("importance", 0)
        x[i, 2] = attrs.get("activation_freq", 0)

    # Create edge features
    edge_index = []
    edge_attr = []
    type_to_idx = {"excitatory": 0, "inhibitory": 1, "gating": 2}

    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_attr.append([
            data.get("strength", 0),
            data.get("abs_strength", 0),
            type_to_idx.get(data.get("interaction_type", "excitatory"), 0),
            data.get("gating_score", 0),
        ])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_ids=node_list,
    )


def get_subgraph(
    G: nx.DiGraph,
    seed_nodes: list[str],
    n_hops: int = 2,
    min_strength: float = 0.0,
) -> nx.DiGraph:
    """Extract a subgraph around seed nodes.

    Args:
        G: Full interaction graph.
        seed_nodes: Starting feature IDs.
        n_hops: Number of hops to expand.
        min_strength: Minimum edge strength to follow.

    Returns:
        Subgraph as a new DiGraph.
    """
    nodes = set(seed_nodes)

    for _ in range(n_hops):
        new_nodes = set()
        for node in nodes:
            if node not in G:
                continue
            # Follow outgoing edges
            for _, tgt, data in G.out_edges(node, data=True):
                if data.get("abs_strength", 0) >= min_strength:
                    new_nodes.add(tgt)
            # Follow incoming edges
            for src, _, data in G.in_edges(node, data=True):
                if data.get("abs_strength", 0) >= min_strength:
                    new_nodes.add(src)
        nodes |= new_nodes

    return G.subgraph(nodes).copy()


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
