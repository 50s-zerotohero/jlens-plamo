"""LensModel adapter for pfnet/plamo-3-nict-8b-base.

jlens.hf.from_hf() auto-detects standard HF layouts by walking dotted
attribute paths, but PLaMo-3's decoder wraps its block list in an extra
Plamo3Decoder module: ``model.model.layers`` is a Plamo3Decoder instance, not
the nn.ModuleList of blocks. The real ModuleList is
``model.model.layers.layers``. This mismatch surfaced in the Phase 3a smoke
test (see scripts/smoke_test.py). jlens's LensModel protocol
(jlens.protocol.LensModel) exists precisely so callers can plug in a model
this way instead of forcing it through from_hf()'s single-attribute-per-hop
Layout; the actual lens fitting/application math (jlens.fitting,
jlens.lens.JacobianLens) is untouched and reused as-is.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PlamoLensModel:
    """:class:`jlens.protocol.LensModel` over a loaded Plamo3ForCausalLM."""

    def __init__(self, hf_model: nn.Module, tokenizer: Any) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer

        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        text_module = hf_model.model
        self.layers: nn.ModuleList = text_module.layers.layers  # unwrap Plamo3Decoder
        self._final_norm: nn.Module = text_module.norm
        self._embed_tokens: nn.Module = text_module.embed_tokens
        self._lm_head: nn.Module = hf_model.lm_head

        self.n_layers: int = hf_model.config.num_hidden_layers
        self.d_model: int = hf_model.config.hidden_size
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config.num_hidden_layers={self.n_layers} but found "
                f"{len(self.layers)} blocks at model.layers.layers"
            )

    def __repr__(self) -> str:
        return (
            f"PlamoLensModel({type(self._hf_model).__name__}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )

    @property
    def input_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self._hf_model.model(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        return self._lm_head(
            self._final_norm(residual.to(target_dtype).to(target_device))
        )


def from_plamo(hf_model: nn.Module, tokenizer: Any) -> PlamoLensModel:
    """Wrap a loaded ``Plamo3ForCausalLM`` as a :class:`PlamoLensModel`."""
    return PlamoLensModel(hf_model, tokenizer)
