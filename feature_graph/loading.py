"""
Model and SAE loading infrastructure.

Supports:
- GPT-2-small + SAELens community SAEs (Joseph Bloom's residual stream SAEs)
- Gemma-2-2B + Gemma Scope SAEs
- Any TransformerLens-compatible model with SAELens-compatible SAEs

Design decision: We return (HookedTransformer, dict[int, SAE]) — a model and
a dict mapping layer indices to SAE objects. This is the universal interface
that all downstream code depends on. Adding a new model is just writing a new
loader that returns this pair.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformer_lens import HookedTransformer

from feature_graph.config import Config
from feature_graph.utils import get_device

logger = logging.getLogger("feature_graph")


def load_model_and_saes(
    cfg: Config,
) -> tuple[HookedTransformer, dict[int, object]]:
    """Load a transformer model and its pre-trained SAEs.

    Args:
        cfg: Configuration specifying model, SAE release, layers, device, etc.

    Returns:
        (model, saes) where:
            - model is a HookedTransformer
            - saes is a dict mapping layer index -> SAE object
    """
    device = get_device(cfg.device)
    dtype = cfg.get_dtype()

    logger.info(f"Loading model: {cfg.model_name}")
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        device=str(device),
        dtype=dtype,
    )
    model.eval()

    logger.info(f"Loading SAEs from release: {cfg.sae_release}")
    saes = {}
    for layer in cfg.layers:
        sae_id = cfg.get_sae_id(layer)
        logger.info(f"  Loading SAE for layer {layer}: {sae_id}")
        try:
            sae = _load_sae(cfg.sae_release, sae_id, device, dtype)
            saes[layer] = sae
        except Exception as e:
            logger.warning(f"  Failed to load SAE for layer {layer}: {e}")

    logger.info(f"Loaded {len(saes)} SAEs for layers {sorted(saes.keys())}")
    return model, saes


def _load_sae(
    release: str, sae_id: str, device: torch.device, dtype: torch.dtype
) -> object:
    """Load a single SAE from SAELens.

    Returns an SAE object with .encode() and .decode() methods,
    plus .W_enc, .W_dec, .b_enc, .b_dec attributes.
    """
    from sae_lens import SAE

    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=str(device),
    )
    sae.eval()
    return sae


def get_sae_encoder_directions(sae: object) -> torch.Tensor:
    """Extract encoder weight matrix (feature directions in activation space).

    Returns: (n_features, d_model) tensor where each row is an encoder direction.
    """
    # SAELens SAE objects store encoder weights as W_enc of shape (d_model, n_features)
    W_enc = sae.W_enc  # (d_model, n_features)
    return W_enc.T  # (n_features, d_model)


def get_sae_decoder_directions(sae: object) -> torch.Tensor:
    """Extract decoder weight matrix (feature directions for reconstruction).

    Returns: (n_features, d_model) tensor where each row is a decoder direction.
    """
    # SAELens SAE objects store decoder weights as W_dec of shape (n_features, d_model)
    return sae.W_dec  # (n_features, d_model)


def get_sae_dict_size(sae: object) -> int:
    """Get the dictionary size (number of features) of an SAE."""
    return sae.W_dec.shape[0]


def get_model_d_model(model: HookedTransformer) -> int:
    """Get the residual stream dimension of the model."""
    return model.cfg.d_model


def get_hook_name(layer: int, hook_type: str = "resid_post") -> str:
    """Get the TransformerLens hook name for a given layer and hook type."""
    return f"blocks.{layer}.hook_{hook_type}"
