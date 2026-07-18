"""Phase 3a smoke test: verify jlens compatibility with PLaMo-3-NICT-8B before
committing to a multi-hour production fit (Phase 3b).

Checks (per CLAUDE.md Phase 3a):
  1. Hooks attach and a minimal fit runs on a couple of prompts.
  2. The resulting J_l has shape [d_model, d_model].
  3. Gradients are non-zero / non-NaN at both an SWA layer and a full-attention
     layer (PLaMo-3 interleaves sliding-window and full attention every 8
     layers; see interleaved_sliding_window in the model config).

Usage:
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys

import torch

from jlens_plamo.model_loading import MODEL_ID, load_plamo

SMOKE_PROMPTS = [
    "日本の四季は春夏秋冬に分かれており、それぞれの季節に特有の行事や食べ物がある。"
    "春には桜が咲き、夏には花火大会が開かれ、秋には紅葉が美しく、冬には雪が降る地域も多い。",
    "県庁所在地が松山である県は愛媛県で、瀬戸内海に面した温暖な気候で知られている。"
    "みかんの生産量が多く、道後温泉をはじめとする観光地でも知られる。",
]


def find_swa_and_full_layers(config) -> tuple[int, int]:
    pattern = config.interleaved_sliding_window
    swa_layer = next(i for i, w in enumerate(pattern) if w is not None)
    full_layer = next(i for i, w in enumerate(pattern) if w is None)
    return swa_layer, full_layer


def main() -> None:
    print(f"Loading tokenizer and model: {MODEL_ID}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_plamo(dtype=torch.bfloat16, device=device)
    print(f"Model loaded on {device}: {type(model).__name__}")

    swa_layer, full_layer = find_swa_and_full_layers(model.config)
    print(f"SWA layer under test: {swa_layer}; full-attention layer under test: {full_layer}")

    # --- Step 0: does jlens.from_hf()'s generic HF layout auto-detection work? ---
    import jlens

    try:
        lens_model = jlens.from_hf(model, tokenizer)
        print(f"jlens.from_hf() succeeded directly: {lens_model}")
    except Exception as e:
        print(
            f"jlens.from_hf() failed as expected for PLaMo's wrapped decoder "
            f"({type(e).__name__}: {e}). Falling back to the PlamoLensModel "
            f"adapter (jlens_plamo/plamo_adapter.py)."
        )
        from jlens_plamo.plamo_adapter import from_plamo

        lens_model = from_plamo(model, tokenizer)
        print(f"PlamoLensModel adapter succeeded: {lens_model}")

    assert lens_model.n_layers == 32, lens_model.n_layers
    assert lens_model.d_model == 4096, lens_model.d_model

    # --- Step 1+2: minimal fit on 2-3 prompts, check J_l shape ---
    target_layer = max(swa_layer, full_layer) + 4
    source_layers = sorted({swa_layer, full_layer})
    print(
        f"Running jlens.fit() on {len(SMOKE_PROMPTS)} prompts, "
        f"source_layers={source_layers}, target_layer={target_layer}"
    )
    lens = jlens.fit(
        lens_model,
        SMOKE_PROMPTS,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=8,
        max_seq_len=128,
    )
    print(f"Fit result: {lens}")

    for layer in source_layers:
        J = lens.jacobians[layer]
        expected_shape = (lens_model.d_model, lens_model.d_model)
        assert J.shape == expected_shape, (layer, J.shape, expected_shape)
        print(f"  layer {layer}: J shape={tuple(J.shape)} OK")

        # --- Step 3: gradient sanity ---
        if torch.isnan(J).any():
            print(f"  FAIL: layer {layer} J_l contains NaNs")
            sys.exit(1)
        norm = J.norm().item()
        if norm == 0.0:
            print(f"  FAIL: layer {layer} J_l is exactly zero (gradient did not flow)")
            sys.exit(1)
        kind = "SWA" if layer == swa_layer else "full-attention"
        print(f"  layer {layer} ({kind}): ||J_l||={norm:.4f} — gradient flowed OK")

    print("\nSmoke test PASSED: hooks attach, J_l shape is correct, and "
          "gradients flow through both SWA and full-attention layers.")


if __name__ == "__main__":
    main()
