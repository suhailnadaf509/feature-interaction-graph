"""
Feature Interaction Graph: Mapping compositional computation in SAE feature space.

This library provides tools for constructing, analyzing, and visualizing
causal interaction graphs between SAE features in transformer language models.
"""

from feature_graph.config import Config
from feature_graph.loading import load_model_and_saes
from feature_graph.activations import collect_activations
from feature_graph.coactivation import build_coactivation_atlas
from feature_graph.candidates import identify_candidates
from feature_graph.interactions import measure_interactions
from feature_graph.graph import build_interaction_graph

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_model_and_saes",
    "collect_activations",
    "build_coactivation_atlas",
    "identify_candidates",
    "measure_interactions",
    "build_interaction_graph",
]
