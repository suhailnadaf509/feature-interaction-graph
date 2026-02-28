"""
Activation collection over token corpora.

Collects SAE feature activations for all specified layers across a large
token corpus. Stores results in an efficient format (HDF5 with sparse
representation) for downstream co-activation analysis.

Design decisions:
- We store activations as dense numpy arrays on disk (HDF5), but only for
  the top-K features by activation frequency. This keeps memory bounded
  while retaining the features we care about.
- Activations are collected per-token (not per-sequence), because feature
  interactions happen at the token level.
- We stream through the dataset to avoid loading everything into memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from feature_graph.config import Config
from feature_graph.loading import get_hook_name, get_sae_dict_size
from feature_graph.utils import get_device, set_seed

logger = logging.getLogger("feature_graph")


@dataclass
class ActivationStore:
    """Container for collected activations.

    Attributes:
        activations: dict mapping layer -> (n_tokens, n_features) numpy array
        feature_frequencies: dict mapping layer -> (n_features,) array of activation frequencies
        feature_stats: dict mapping layer -> dict of per-feature statistics
        n_tokens: total number of tokens processed
        layers: list of layer indices
    """

    activations: dict[int, np.ndarray] = field(default_factory=dict)
    feature_frequencies: dict[int, np.ndarray] = field(default_factory=dict)
    feature_stats: dict[int, dict] = field(default_factory=dict)
    n_tokens: int = 0
    layers: list[int] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save to HDF5."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.attrs["n_tokens"] = self.n_tokens
            f.attrs["layers"] = self.layers

            for layer in self.layers:
                grp = f.create_group(f"layer_{layer}")
                grp.create_dataset(
                    "activations",
                    data=self.activations[layer],
                    compression="gzip",
                    compression_opts=4,
                )
                grp.create_dataset("frequencies", data=self.feature_frequencies[layer])

    @classmethod
    def load(cls, path: str | Path) -> "ActivationStore":
        """Load from HDF5."""
        store = cls()
        with h5py.File(path, "r") as f:
            store.n_tokens = int(f.attrs["n_tokens"])
            store.layers = list(f.attrs["layers"])

            for layer in store.layers:
                grp = f[f"layer_{layer}"]
                store.activations[layer] = grp["activations"][:]
                store.feature_frequencies[layer] = grp["frequencies"][:]

        return store


def collect_activations(
    model: object,
    saes: dict[int, object],
    cfg: Config,
    dataset_tokens: Optional[torch.Tensor] = None,
) -> ActivationStore:
    """Collect SAE feature activations over a token corpus.

    Args:
        model: HookedTransformer model.
        saes: Dict mapping layer index -> SAE object.
        cfg: Configuration.
        dataset_tokens: Optional pre-tokenized tensor of shape (n_sequences, seq_len).
            If None, loads from cfg.dataset_name.

    Returns:
        ActivationStore with collected activations.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    if dataset_tokens is None:
        dataset_tokens = _load_dataset_tokens(model, cfg)

    n_sequences = dataset_tokens.shape[0]
    seq_len = dataset_tokens.shape[1]
    total_tokens = n_sequences * seq_len

    logger.info(f"Collecting activations: {n_sequences} sequences × {seq_len} tokens = {total_tokens} tokens")

    layers = sorted(saes.keys())
    hook_names = {layer: get_hook_name(layer) for layer in layers}

    # Pre-allocate storage
    # We collect binary activation indicators and running sums for efficiency
    n_features_per_layer = {layer: get_sae_dict_size(saes[layer]) for layer in layers}

    # Accumulators
    act_counts = {layer: np.zeros(n_features_per_layer[layer], dtype=np.int64) for layer in layers}
    act_sums = {layer: np.zeros(n_features_per_layer[layer], dtype=np.float64) for layer in layers}
    act_sq_sums = {layer: np.zeros(n_features_per_layer[layer], dtype=np.float64) for layer in layers}

    # We'll store a sampled subset of full activation vectors for co-activation analysis
    max_stored_tokens = min(total_tokens, cfg.n_tokens)
    # For memory: store activations as sparse-ish booleans (which features are active)
    # and a smaller set of full activation values
    stored_activations = {
        layer: np.zeros((max_stored_tokens, n_features_per_layer[layer]), dtype=np.float16)
        for layer in layers
    }

    token_idx = 0

    for batch_start in tqdm(range(0, n_sequences, cfg.batch_size), desc="Collecting activations"):
        batch_end = min(batch_start + cfg.batch_size, n_sequences)
        batch_tokens = dataset_tokens[batch_start:batch_end].to(device)

        # Run forward pass and cache residual stream activations
        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=list(hook_names.values()),
        )

        for layer in layers:
            hook_name = hook_names[layer]
            residual = cache[hook_name]  # (batch, seq, d_model)

            # Flatten to (batch * seq, d_model)
            flat_residual = residual.reshape(-1, residual.shape[-1])

            # Encode through SAE
            with torch.no_grad():
                feature_acts = saes[layer].encode(flat_residual)  # (batch*seq, n_features)

            feature_acts_np = feature_acts.cpu().float().numpy()

            # Update accumulators
            active = feature_acts_np > 0
            act_counts[layer] += active.sum(axis=0)
            act_sums[layer] += feature_acts_np.sum(axis=0)
            act_sq_sums[layer] += (feature_acts_np ** 2).sum(axis=0)

            # Store activations
            n_new = feature_acts_np.shape[0]
            end_idx = min(token_idx + n_new, max_stored_tokens)
            n_store = end_idx - token_idx
            if n_store > 0:
                stored_activations[layer][token_idx:end_idx] = feature_acts_np[:n_store].astype(np.float16)

        token_idx += batch_tokens.shape[0] * batch_tokens.shape[1]

        # Clean up cache
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if token_idx >= max_stored_tokens:
            break

    actual_tokens = min(token_idx, max_stored_tokens)
    logger.info(f"Collected activations for {actual_tokens} tokens")

    # Compute feature frequencies
    feature_frequencies = {}
    for layer in layers:
        feature_frequencies[layer] = act_counts[layer] / actual_tokens

    # Trim stored activations to actual size
    for layer in layers:
        stored_activations[layer] = stored_activations[layer][:actual_tokens]

    store = ActivationStore(
        activations=stored_activations,
        feature_frequencies=feature_frequencies,
        n_tokens=actual_tokens,
        layers=layers,
    )

    return store


def _load_dataset_tokens(model: object, cfg: Config) -> torch.Tensor:
    """Load and tokenize a dataset for activation collection."""
    from datasets import load_dataset

    logger.info(f"Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)

    # Collect enough text to get n_tokens
    n_sequences_needed = (cfg.n_tokens // cfg.context_length) + 1
    all_tokens = []
    n_collected = 0

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for example in dataset:
        text = example.get("text", example.get("content", ""))
        if not text:
            continue

        tokens = tokenizer.encode(text, return_tensors="pt", truncation=True,
                                  max_length=cfg.context_length, padding="max_length")
        all_tokens.append(tokens.squeeze(0))
        n_collected += 1

        if n_collected >= n_sequences_needed:
            break

    tokens_tensor = torch.stack(all_tokens)
    logger.info(f"Tokenized {tokens_tensor.shape[0]} sequences of length {tokens_tensor.shape[1]}")
    return tokens_tensor
