# Feature Interaction Graph

**Mapping compositional computation in SAE feature space.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What This Is

Sparse autoencoders (SAEs) give us dictionaries of interpretable features inside neural networks — "Golden Gate Bridge," "legal terminology," "Python function definitions." But knowing the parts is not the same as knowing the wiring. **Feature Interaction Graph** (`feature_graph`) maps the causal relationships *between* SAE features: which features excite, inhibit, or gate each other across layers.

This is the difference between a gene catalog and a gene regulatory network. Between a parts list and a circuit schematic.

## Why This Methodology

There are several ways to estimate feature interactions. We implement two complementary approaches and made deliberate choices about each:

### 1. Causal Clamping (Gold Standard, Slower)

We modify a source feature's activation by adding a scaled perturbation to the residual stream, then measure the downstream effect on target features. This is the only method that gives **true causal interaction strengths** — not correlations, not local linear approximations, but measured responses to interventions.

**Why not pure gradient/Jacobian methods?** Gradients give the *local linear approximation* of the interaction at the current operating point. They miss nonlinearities (gating interactions through GeLU/SiLU), they conflate direct and indirect effects, and they don't account for the fact that feature activations are ReLU-thresholded (a gradient can be nonzero even when the downstream feature is firmly at zero). Clamping measures the *actual* response.

**Why not path patching?** Path patching decomposes effects along specific computational paths (through specific attention heads or MLP layers). This is excellent for mechanistic understanding of *how* an interaction is implemented, but it requires enumerating paths — which is combinatorially expensive and doesn't scale to the full interaction graph. We use clamping for the graph and recommend path patching for deep-diving into specific edges.

**Why not crosscoder weights?** Crosscoder weight analysis (looking at decoder-encoder alignment across layers) gives a fast *structural prior* on which features might interact. We use this as a **pruning filter** (Section 4.3 of the methodology), not as the interaction measure itself — because weight alignment doesn't account for the nonlinear transformations between layers.

### 2. Jacobian Approximation (Fast, Approximate)

For rapid exploration, we also provide a Jacobian-based estimator that computes $\partial f_j^{(l')} / \partial f_i^{(l)}$ via backpropagation. This is ~100x faster than clamping (one backward pass vs. many forward passes) and gives a reasonable first-order approximation. Use it for initial exploration; use clamping for publication-quality results.

### Why Not Just Use Anthropic's Attribution Graphs?

Anthropic's attribution graphs trace computation *for a specific prompt*. They answer: "on this input, which features contributed to which?" They do **not** answer: "is this feature connection a structural property of the model?" Our graph aggregates across thousands of inputs to find **structural** interactions — the model's wiring, not one input's activation pattern. We also explicitly type edges (excitatory/inhibitory/gating) and compute global graph statistics, neither of which attribution graphs provide.

## Architecture

```
feature_graph/
├── config.py          # Dataclass-based configuration
├── loading.py         # Model + SAE loading (TransformerLens + SAELens)
├── activations.py     # Activation collection over corpora
├── coactivation.py    # Co-activation atlas construction
├── candidates.py      # Candidate pair identification + filtering
├── interactions.py    # Causal interaction measurement (clamping + Jacobian)
├── graph.py           # Interaction graph construction + serialization
├── analysis.py        # Graph-theoretic analysis utilities
├── subgraphs.py       # Behavior-specific subgraph extraction
├── steering.py        # Cascade prediction + compensated steering
├── visualization.py   # Interactive graph visualization
└── utils.py           # Shared utilities
```

The library is designed so that **each module is independently useful**. You can use `coactivation.py` without ever building the full graph. You can use `steering.py` with a hand-constructed graph. Every component has a clean API that takes explicit inputs and produces explicit outputs — no hidden global state.

## Installation

```bash
git clone https://github.com/your-username/feature-interaction-graph.git
cd feature-interaction-graph
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.1
- transformer-lens ≥ 2.0
- sae-lens ≥ 4.0
- networkx ≥ 3.0
- scipy, numpy, h5py, plotly, tqdm

## Quick Start

```python
from feature_graph import (
    Config, load_model_and_saes, collect_activations,
    build_coactivation_atlas, identify_candidates,
    measure_interactions, build_interaction_graph
)

# Configure
cfg = Config(
    model_name="gpt2-small",
    sae_release="gpt2-small-res-jb",
    layers=list(range(12)),
    n_tokens=1_000_000,
    top_k_features=200,
    layer_window=3,
)

# Load model + SAEs
model, saes = load_model_and_saes(cfg)

# Collect activations over a corpus
acts = collect_activations(model, saes, cfg)

# Build co-activation atlas
atlas = build_coactivation_atlas(acts, cfg)

# Identify candidate pairs
candidates = identify_candidates(atlas, saes, cfg)

# Measure causal interactions
interactions = measure_interactions(model, saes, candidates, cfg)

# Build the graph
G = build_interaction_graph(interactions, cfg)

# Analyze
from feature_graph.analysis import compute_graph_statistics, find_hubs
stats = compute_graph_statistics(G)
hubs = find_hubs(G, top_k=20)

# Visualize
from feature_graph.visualization import render_interactive_graph
render_interactive_graph(G, output_path="interaction_graph.html")
```

## CLI Usage

```bash
# Full pipeline
python -m feature_graph.cli run-pipeline --model gpt2-small --n-tokens 1000000

# Individual stages
python -m feature_graph.cli collect-activations --model gpt2-small --output acts.h5
python -m feature_graph.cli build-atlas --activations acts.h5 --output atlas.h5
python -m feature_graph.cli measure-interactions --atlas atlas.h5 --output interactions.json
python -m feature_graph.cli build-graph --interactions interactions.json --output graph.graphml
python -m feature_graph.cli analyze --graph graph.graphml
python -m feature_graph.cli visualize --graph graph.graphml --output graph.html
```

## Graph Format

The interaction graph is stored as a NetworkX DiGraph with typed edges, serializable to GraphML, JSON, or pickle. Each node has attributes:

- `layer`: int — which transformer layer
- `feature_idx`: int — SAE feature index
- `label`: str — auto-interpretability label (if available)
- `importance`: float — composite importance score
- `activation_freq`: float — fraction of tokens where feature fires

Each edge has attributes:

- `interaction_type`: str — "excitatory", "inhibitory", or "gating"
- `strength`: float — signed interaction strength (positive=excitatory, negative=inhibitory)
- `abs_strength`: float — absolute interaction strength
- `p_value`: float — statistical significance
- `variance`: float — variance of interaction across inputs (high = gating)
- `n_samples`: int — number of inputs used for measurement

## Extending

The library is designed for extension:

- **New models**: Implement a loader in `loading.py` that returns a `HookedTransformer` + dict of SAEs. The rest of the pipeline is model-agnostic.
- **New interaction measures**: Add methods to `interactions.py`. The `InteractionResult` dataclass is the universal interface.
- **New graph analyses**: Add functions to `analysis.py` that take a NetworkX graph.
- **New pruning strategies**: Add filters to `candidates.py` that take the co-activation atlas and return candidate pairs.

## Citation

If you use this library, please cite:

```bibtex
@software{feature_interaction_graph,
  title={Feature Interaction Graph: Mapping Compositional Computation in SAE Feature Space},
  author={Nadaf, Mohammed Suhail B},
  year={2026},
  url={https://github.com/your-username/feature-interaction-graph}
}
```

## License

MIT
