"""Loads pfnet/plamo-3-nict-8b-base, working around known compatibility gaps
between its `trust_remote_code` modeling file and current `transformers`.

Found during the Phase 3a smoke test (see scripts/smoke_test.py):

1. PLaMo's modeling_plamo.py sets ``Plamo3ForCausalLM._tied_weights_keys =
   ["lm_head.weight"]`` (a list), the pre-5.5 `transformers` convention.
   `transformers` >= 5.5 (required by `jlens`) expects a ``{target: source}``
   dict instead and crashes in ``get_expanded_tied_weights_keys`` before
   that. We patch the class attribute to the new dict form after the first
   failed load triggers the dynamic module import, then retry. This only
   affects how the tied lm_head/embedding weights are *loaded*; it does not
   touch model weights, architecture, or jlens's fitting/application logic.

2. Each `RotaryEmbedding`'s `inv_freq` / `cos_cached` / `sin_cached` buffers
   are registered ``persistent=False`` and computed once in
   `RotaryEmbedding.__init__` from (dim, base) alone -- they hold no learned
   data. Under `from_pretrained`'s meta-device skeleton + checkpoint-dispatch
   loading path, a random subset of these buffers (varies run to run, ~0-15
   of the 32 layers, reproduced both on CPU and CUDA, with and without
   `low_cpu_mem_usage`) come out as uninitialized garbage instead of the
   values `__init__` computed, and propagate NaNs through the whole forward
   pass starting at whichever layer was affected. Since these buffers are a
   pure function of static config (not checkpoint data), it's always safe to
   rebuild them after loading; we do that unconditionally rather than only
   after detecting NaN, since corruption was observed on every load attempt
   in some layer. This looks like an upstream `transformers`/accelerate
   bug with non-persistent computed buffers, not a jlens or PLaMo modeling
   bug in the ordinary sense -- worth a minimal repro report upstream, but
   out of scope to fix here.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "pfnet/plamo-3-nict-8b-base"


def _patch_tied_weights_keys(model_id: str) -> bool:
    """Fix up `_tied_weights_keys` on the dynamically-imported PLaMo class,
    if it was imported in the old list format. Returns True if a patch was
    applied."""
    patched = False
    for name, module in list(sys.modules.items()):
        if "modeling_plamo" not in name:
            continue
        cls = getattr(module, "Plamo3ForCausalLM", None)
        if cls is not None and isinstance(
            getattr(cls, "_tied_weights_keys", None), list
        ):
            cls._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
            patched = True
    return patched


def _rebuild_rotary_embeddings(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """Recompute every RotaryEmbedding's inv_freq/cos_cached/sin_cached from
    scratch on the model's current device. See module docstring point 2.
    Returns the number of modules rebuilt."""
    device = next(model.parameters()).device
    n_rebuilt = 0
    for _, mod in model.named_modules():
        if type(mod).__name__ != "RotaryEmbedding":
            continue
        inv_freq = 1.0 / (
            mod.base ** (torch.arange(0, mod.dim, 2, device=device).float() / mod.dim)
        )
        mod.inv_freq = inv_freq
        mod._set_cos_sin_cache(
            seq_len=mod.max_position_embeddings, device=device, dtype=dtype
        )
        n_rebuilt += 1
    return n_rebuilt


def load_plamo(
    *,
    model_id: str = MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Load the PLaMo tokenizer and model, applying the compatibility
    patches described in the module docstring. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype
        )
    except AttributeError:
        if not _patch_tied_weights_keys(model_id):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype
        )
    model = model.to(device)
    _rebuild_rotary_embeddings(model, dtype)
    return model, tokenizer
