# Contributing to Feature Interaction Graph

Thank you for your interest in contributing! This project follows a standard open-source workflow.

## Development setup

```bash
git clone https://github.com/your-username/feature-interaction-graph.git
cd feature-interaction-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                     # all tests
pytest -x                  # stop on first failure
pytest tests/test_core.py  # specific file
pytest -k "test_graph"     # by name pattern
```

## Code style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check .           # lint
ruff check --fix .     # auto-fix
ruff format .          # format
```

Line length limit: 100. Target Python: 3.10+.

## Project structure

```
feature_graph/
├── __init__.py          # Public API re-exports
├── config.py            # Config dataclass
├── utils.py             # Shared utilities
├── loading.py           # Model / SAE loading
├── activations.py       # Activation collection + HDF5 cache
├── coactivation.py      # Co-activation atlas
├── candidates.py        # Candidate pair pruning
├── interactions.py      # Causal measurement (clamping + Jacobian)
├── graph.py             # Graph construction + serialization
├── analysis.py          # Graph statistics + motifs
├── subgraphs.py         # Behavior subgraph extraction
├── steering.py          # Cascade prediction + compensated steering
├── visualization.py     # Plotly / pyvis rendering
└── cli.py               # argparse CLI
```

## Adding a new interaction measurement method

1. Add the method function in `interactions.py` following the `_measure_by_clamping` pattern.
2. Have it return `list[InteractionResult]`.
3. Add a dispatch branch in `measure_interactions()`.
4. Add the method name to the `interaction_method` choices in `config.py` and `cli.py`.

## Adding a new predefined behavior

Add a `BehaviorContrast` to the `PREDEFINED_BEHAVIORS` dict in `subgraphs.py`:

```python
PREDEFINED_BEHAVIORS["my_behavior"] = BehaviorContrast(
    name="my_behavior",
    description="...",
    positive_prompts=["..."],
    negative_prompts=["..."],
)
```

## Pull request checklist

- [ ] All existing tests pass (`pytest`)
- [ ] New code has tests
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] Docstrings on public functions
- [ ] Updated `README.md` if the public API changed
