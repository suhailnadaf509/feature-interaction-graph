"""
Tests for core numerical components of the feature interaction graph library.

These tests use synthetic data to validate the critical computations without
requiring GPU access or model downloads. They test:
1. Co-activation computation correctness
2. Interaction measurement statistics
3. Graph construction and serialization
4. Analysis utilities
5. Candidate filtering logic
"""

import json
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from feature_graph.config import Config
from feature_graph.utils import (
    feature_id,
    parse_feature_id,
    get_nonzero_activation_stats,
    batched_cosine_similarity,
)


class TestUtils:
    """Test utility functions."""

    def test_feature_id_roundtrip(self):
        """Feature ID creation and parsing should be inverses."""
        for layer in [0, 5, 25]:
            for fidx in [0, 100, 65535]:
                fid = feature_id(layer, fidx)
                parsed_layer, parsed_fidx = parse_feature_id(fid)
                assert parsed_layer == layer
                assert parsed_fidx == fidx

    def test_feature_id_format(self):
        """Feature IDs should have the expected format."""
        assert feature_id(3, 42) == "L3_F42"
        assert feature_id(0, 0) == "L0_F0"
        assert feature_id(25, 100000) == "L25_F100000"

    def test_nonzero_activation_stats_all_zero(self):
        """Stats for all-zero activations should be all zeros."""
        import torch
        acts = torch.zeros(100)
        stats = get_nonzero_activation_stats(acts)
        assert stats["freq"] == 0.0
        assert stats["mean"] == 0.0

    def test_nonzero_activation_stats_some_active(self):
        """Stats should reflect non-zero activations only."""
        import torch
        acts = torch.zeros(100)
        acts[10:20] = torch.linspace(1, 10, 10)
        stats = get_nonzero_activation_stats(acts)
        assert stats["freq"] == pytest.approx(0.1, abs=0.01)
        assert stats["mean"] > 0
        assert stats["p10"] <= stats["p50"] <= stats["p90"]

    def test_batched_cosine_similarity(self):
        """Cosine similarity should work correctly in batches."""
        import torch
        a = torch.randn(10, 64)
        b = torch.randn(20, 64)
        sim = batched_cosine_similarity(a, b, batch_size=3)
        assert sim.shape == (10, 20)

        # Self-similarity should be ~1 on diagonal
        self_sim = batched_cosine_similarity(a, a, batch_size=5)
        for i in range(10):
            assert self_sim[i, i].item() == pytest.approx(1.0, abs=1e-5)

    def test_batched_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have ~0 cosine similarity."""
        import torch
        a = torch.eye(64)[:10]  # First 10 basis vectors
        b = torch.eye(64)[10:20]  # Next 10 basis vectors
        sim = batched_cosine_similarity(a, b)
        assert torch.abs(sim).max().item() < 1e-5


class TestConfig:
    """Test configuration management."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        cfg = Config()
        assert cfg.model_name == "gpt2-small"
        assert cfg.top_k_features == 200
        assert cfg.layer_window == 3
        assert len(cfg.layers) == 12

    def test_config_save_load_roundtrip(self):
        """Config should survive JSON serialization."""
        cfg = Config(
            model_name="test-model",
            top_k_features=100,
            n_tokens=50000,
            layers=[0, 1, 2, 3],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        cfg.save(path)
        loaded = Config.load(path)

        assert loaded.model_name == "test-model"
        assert loaded.top_k_features == 100
        assert loaded.n_tokens == 50000
        assert loaded.layers == [0, 1, 2, 3]

    def test_config_get_sae_id(self):
        """SAE ID template should be correctly formatted."""
        cfg = Config(sae_id_template="blocks.{layer}.hook_resid_post")
        assert cfg.get_sae_id(5) == "blocks.5.hook_resid_post"


class TestCoactivation:
    """Test co-activation computation with synthetic data."""

    def test_perfect_coactivation(self):
        """Features that always co-activate should have high PMI."""
        from feature_graph.coactivation import _compute_coactivation_pair

        n_tokens = 10000
        n_src = 5
        n_tgt = 5
        cfg = Config(coactivation_threshold=0.001, pmi_threshold=0.0, low_coactivation_ratio=0.1)

        # Features 0 and 0 always co-activate
        mask_src = np.zeros((n_tokens, n_src), dtype=np.float32)
        mask_tgt = np.zeros((n_tokens, n_tgt), dtype=np.float32)

        # Both active on first 1000 tokens
        mask_src[:1000, 0] = 1
        mask_tgt[:1000, 0] = 1

        freq_src = mask_src.mean(axis=0)
        freq_tgt = mask_tgt.mean(axis=0)

        pmi, coact_prob, coact_ratio = _compute_coactivation_pair(
            mask_src, mask_tgt, freq_src, freq_tgt, n_tokens, cfg
        )

        # Co-activation prob should be 1.0 (every time src[0] is active, tgt[0] is too)
        assert coact_prob[0, 0] == pytest.approx(1.0, abs=0.01)
        # PMI should be positive
        assert pmi[0, 0] > 0

    def test_no_coactivation(self):
        """Features that never co-activate should have low/negative PMI."""
        from feature_graph.coactivation import _compute_coactivation_pair

        n_tokens = 10000
        n_src = 5
        n_tgt = 5
        cfg = Config(coactivation_threshold=0.001, pmi_threshold=-100, low_coactivation_ratio=0.5)

        mask_src = np.zeros((n_tokens, n_src), dtype=np.float32)
        mask_tgt = np.zeros((n_tokens, n_tgt), dtype=np.float32)

        # Active on completely different tokens
        mask_src[:1000, 0] = 1
        mask_tgt[1000:2000, 0] = 1

        freq_src = mask_src.mean(axis=0)
        freq_tgt = mask_tgt.mean(axis=0)

        pmi, coact_prob, coact_ratio = _compute_coactivation_pair(
            mask_src, mask_tgt, freq_src, freq_tgt, n_tokens, cfg
        )

        # Co-activation prob should be ~0
        assert coact_prob[0, 0] == pytest.approx(0.0, abs=0.01)

    def test_independent_features(self):
        """Independent features should have PMI near 0."""
        from feature_graph.coactivation import _compute_coactivation_pair

        rng = np.random.default_rng(42)
        n_tokens = 100000
        n_src = 3
        n_tgt = 3
        cfg = Config(coactivation_threshold=0.001, pmi_threshold=-100, low_coactivation_ratio=0.5)

        # Generate independent binary activations
        mask_src = (rng.random((n_tokens, n_src)) < 0.1).astype(np.float32)
        mask_tgt = (rng.random((n_tokens, n_tgt)) < 0.1).astype(np.float32)

        freq_src = mask_src.mean(axis=0)
        freq_tgt = mask_tgt.mean(axis=0)

        pmi, coact_prob, coact_ratio = _compute_coactivation_pair(
            mask_src, mask_tgt, freq_src, freq_tgt, n_tokens, cfg
        )

        # PMI should be close to 0 for independent features
        for i in range(n_src):
            for j in range(n_tgt):
                if pmi[i, j] != 0:
                    assert abs(pmi[i, j]) < 0.5  # Allow some noise


class TestGraph:
    """Test graph construction and analysis."""

    def _make_synthetic_graph(self) -> nx.DiGraph:
        """Create a synthetic interaction graph for testing."""
        G = nx.DiGraph()

        # Add features across 4 layers
        features = []
        for layer in range(4):
            for fidx in range(5):
                fid = feature_id(layer, fidx)
                features.append(fid)
                G.add_node(fid, layer=layer, feature_idx=fidx,
                          importance=np.random.random(),
                          activation_freq=np.random.random() * 0.1,
                          label=f"feature_{layer}_{fidx}")

        # Add edges (excitatory chain, some inhibitory, one gating)
        edges = [
            ("L0_F0", "L1_F0", "excitatory", 0.5),
            ("L1_F0", "L2_F0", "excitatory", 0.3),
            ("L2_F0", "L3_F0", "excitatory", 0.7),
            ("L0_F1", "L1_F1", "inhibitory", -0.4),
            ("L0_F2", "L1_F2", "gating", 0.2),
            ("L1_F1", "L2_F1", "excitatory", 0.6),
            ("L0_F0", "L1_F1", "excitatory", 0.15),
            ("L0_F0", "L1_F2", "excitatory", 0.1),
            ("L0_F0", "L1_F3", "excitatory", 0.08),
            ("L1_F0", "L2_F1", "inhibitory", -0.25),
        ]

        for src, tgt, itype, strength in edges:
            G.add_edge(src, tgt,
                       interaction_type=itype,
                       strength=strength,
                       abs_strength=abs(strength),
                       p_value=0.001,
                       ci_lower=strength - 0.1,
                       ci_upper=strength + 0.1,
                       std_strength=0.1,
                       gating_score=3.0 if itype == "gating" else 0.5,
                       n_samples=100,
                       method="clamping")

        return G

    def test_graph_construction(self):
        """Graph should have correct number of nodes and edges."""
        G = self._make_synthetic_graph()
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() == 10

    def test_graph_save_load_graphml(self):
        """Graph should survive GraphML serialization."""
        from feature_graph.graph import save_graph, load_graph

        G = self._make_synthetic_graph()
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            path = f.name
        save_graph(G, path, format="graphml")
        loaded = load_graph(path, format="graphml")
        assert loaded.number_of_nodes() == G.number_of_nodes()
        assert loaded.number_of_edges() == G.number_of_edges()

    def test_graph_save_load_json(self):
        """Graph should survive JSON serialization."""
        from feature_graph.graph import save_graph, load_graph

        G = self._make_synthetic_graph()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_graph(G, path, format="json")
        loaded = load_graph(path, format="json")
        assert loaded.number_of_nodes() == G.number_of_nodes()
        assert loaded.number_of_edges() == G.number_of_edges()

    def test_subgraph_extraction(self):
        """Subgraph should contain seed nodes and their neighbors."""
        from feature_graph.graph import get_subgraph

        G = self._make_synthetic_graph()
        sub = get_subgraph(G, ["L0_F0"], n_hops=1)

        assert "L0_F0" in sub
        assert "L1_F0" in sub  # Direct neighbor
        assert "L1_F1" in sub  # Direct neighbor

    def test_subgraph_depth(self):
        """Multi-hop subgraph should go deeper."""
        from feature_graph.graph import get_subgraph

        G = self._make_synthetic_graph()
        sub_1 = get_subgraph(G, ["L0_F0"], n_hops=1)
        sub_2 = get_subgraph(G, ["L0_F0"], n_hops=2)
        assert sub_2.number_of_nodes() >= sub_1.number_of_nodes()


class TestAnalysis:
    """Test graph analysis utilities."""

    def _make_graph(self) -> nx.DiGraph:
        return TestGraph()._make_synthetic_graph()

    def test_compute_statistics(self):
        """Statistics should be computed without errors."""
        from feature_graph.analysis import compute_graph_statistics

        G = self._make_graph()
        stats = compute_graph_statistics(G)

        assert stats.n_nodes == 20
        assert stats.n_edges == 10
        assert stats.n_excitatory > 0
        assert stats.n_inhibitory > 0
        assert stats.n_gating > 0
        assert stats.density > 0

    def test_find_hubs(self):
        """Hub finding should return features sorted by centrality."""
        from feature_graph.analysis import find_hubs

        G = self._make_graph()
        hubs = find_hubs(G, top_k=5)

        assert len(hubs) == 5
        # L0_F0 has the most edges (4 outgoing), should be top hub
        assert hubs[0]["id"] == "L0_F0"

    def test_community_detection(self):
        """Community detection should produce non-empty communities."""
        from feature_graph.analysis import detect_communities

        G = self._make_graph()
        communities = detect_communities(G)

        assert len(communities) > 0
        total_nodes = sum(len(c) for c in communities)
        assert total_nodes == G.number_of_nodes()

    def test_edge_type_subgraph(self):
        """Edge type filtering should produce correct subgraphs."""
        from feature_graph.analysis import get_edge_type_subgraph

        G = self._make_graph()
        exc = get_edge_type_subgraph(G, "excitatory")
        inh = get_edge_type_subgraph(G, "inhibitory")

        for _, _, d in exc.edges(data=True):
            assert d["interaction_type"] == "excitatory"
        for _, _, d in inh.edges(data=True):
            assert d["interaction_type"] == "inhibitory"

    def test_motif_counting(self):
        """Motif counter should find the excitatory chain."""
        from feature_graph.analysis import count_motifs

        G = self._make_graph()
        motifs = count_motifs(G)

        # There's an excitatory chain: L0_F0 →+ L1_F0 →+ L2_F0 →+ L3_F0
        assert motifs["excitatory_chain"] >= 1
        # L0_F0 has 4 excitatory outputs → feedforward fan
        assert motifs["feedforward_fan"] >= 1

    def test_layer_distance_decay(self):
        """Layer distance decay should be computable."""
        from feature_graph.analysis import compute_layer_distance_decay

        G = self._make_graph()
        decay = compute_layer_distance_decay(G)

        assert len(decay) > 0
        assert 1 in decay  # All edges are distance 1 in our synthetic graph


class TestInteractionResult:
    """Test InteractionResult dataclass."""

    def test_interaction_result_serialization(self):
        """InteractionResult should serialize to dict."""
        from feature_graph.interactions import InteractionResult

        result = InteractionResult(
            src_layer=0,
            src_feature=10,
            tgt_layer=1,
            tgt_feature=20,
            mean_strength=0.5,
            std_strength=0.1,
            abs_strength=0.5,
            p_value=0.001,
            ci_lower=0.3,
            ci_upper=0.7,
            interaction_type="excitatory",
            gating_score=0.2,
            n_samples=100,
            method="clamping",
        )

        d = result.to_dict()
        assert d["src_layer"] == 0
        assert d["mean_strength"] == 0.5
        assert d["interaction_type"] == "excitatory"

        # Roundtrip
        result2 = InteractionResult(**d)
        assert result2.src_layer == result.src_layer
        assert result2.mean_strength == result.mean_strength

    def test_bootstrap_ci(self):
        """Bootstrap CI should be reasonable."""
        from feature_graph.interactions import _bootstrap_ci

        data = np.random.normal(1.0, 0.1, size=100)
        lower, upper = _bootstrap_ci(data)
        assert lower < 1.0 < upper
        assert upper - lower < 0.5  # CI shouldn't be too wide for this much data


class TestSteering:
    """Test steering cascade prediction."""

    def _make_graph(self) -> nx.DiGraph:
        return TestGraph()._make_synthetic_graph()

    def test_cascade_prediction(self):
        """Cascade prediction should propagate through edges."""
        from feature_graph.steering import predict_cascade

        G = self._make_graph()
        cascade = predict_cascade(G, "L0_F0", steer_delta=1.0, n_hops=2)

        assert cascade.steered_feature == "L0_F0"
        assert len(cascade.predicted_effects) > 0
        # L1_F0 should be excited (positive edge from L0_F0)
        assert "L1_F0" in cascade.predicted_effects
        assert cascade.predicted_effects["L1_F0"] > 0

    def test_cascade_inhibitory_propagation(self):
        """Inhibitory edges should produce negative predicted effects."""
        from feature_graph.steering import predict_cascade

        G = self._make_graph()
        cascade = predict_cascade(G, "L0_F1", steer_delta=1.0, n_hops=1)

        # L1_F1 should be inhibited (negative edge from L0_F1)
        if "L1_F1" in cascade.predicted_effects:
            assert cascade.predicted_effects["L1_F1"] < 0

    def test_cascade_empty_for_unknown_feature(self):
        """Cascade for a feature not in the graph should be empty."""
        from feature_graph.steering import predict_cascade

        G = self._make_graph()
        cascade = predict_cascade(G, "L99_F99", steer_delta=1.0)
        assert len(cascade.predicted_effects) == 0


class TestBuildInteractionGraph:
    """Test the build_interaction_graph function."""

    def test_build_from_results(self):
        """Graph should be built correctly from InteractionResults."""
        from feature_graph.interactions import InteractionResult
        from feature_graph.graph import build_interaction_graph

        results = [
            InteractionResult(
                src_layer=0, src_feature=0, tgt_layer=1, tgt_feature=0,
                mean_strength=0.5, std_strength=0.1, abs_strength=0.5,
                p_value=0.001, ci_lower=0.3, ci_upper=0.7,
                interaction_type="excitatory", gating_score=0.2,
                n_samples=100, method="clamping",
            ),
            InteractionResult(
                src_layer=0, src_feature=1, tgt_layer=1, tgt_feature=1,
                mean_strength=-0.3, std_strength=0.1, abs_strength=0.3,
                p_value=0.005, ci_lower=-0.5, ci_upper=-0.1,
                interaction_type="inhibitory", gating_score=0.3,
                n_samples=100, method="clamping",
            ),
            InteractionResult(
                src_layer=0, src_feature=2, tgt_layer=1, tgt_feature=2,
                mean_strength=0.01, std_strength=0.1, abs_strength=0.01,
                p_value=0.5, ci_lower=-0.1, ci_upper=0.1,
                interaction_type="excitatory", gating_score=0.5,
                n_samples=100, method="clamping",
            ),
        ]

        cfg = Config(significance_level=0.01, edge_strength_threshold=0.0)
        G = build_interaction_graph(results, cfg)

        # Third result should be filtered out (p=0.5 > 0.01)
        assert G.number_of_edges() == 2
        assert G.has_edge("L0_F0", "L1_F0")
        assert G.has_edge("L0_F1", "L1_F1")
        assert not G.has_edge("L0_F2", "L1_F2")

    def test_edge_attributes(self):
        """Edges should have correct typed attributes."""
        from feature_graph.interactions import InteractionResult
        from feature_graph.graph import build_interaction_graph

        results = [
            InteractionResult(
                src_layer=0, src_feature=0, tgt_layer=1, tgt_feature=0,
                mean_strength=0.5, std_strength=0.1, abs_strength=0.5,
                p_value=0.001, ci_lower=0.3, ci_upper=0.7,
                interaction_type="excitatory", gating_score=0.2,
                n_samples=100, method="clamping",
            ),
        ]

        cfg = Config(significance_level=0.01)
        G = build_interaction_graph(results, cfg)

        edge_data = G.edges["L0_F0", "L1_F0"]
        assert edge_data["interaction_type"] == "excitatory"
        assert edge_data["strength"] == 0.5
        assert edge_data["p_value"] == 0.001
