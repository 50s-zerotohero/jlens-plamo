"""Phase 3b completion check: does the fitted lens read out meaningful
Japanese tokens at intermediate layers, before Phase 4 builds real tooling
around this?

Runs the two-step-reasoning prompt named in CLAUDE.md's Phase 3b checkpoint
("県庁所在地が松山である県は" -> expect "愛媛" to surface mid-stack before the
final layer) through JacobianLens.apply() at the last prompt position, and
prints each fitted layer's top-k next-token readout next to the model's own
final-layer prediction.

Usage:
    uv run python scripts/qualitative_check.py
    uv run python scripts/qualitative_check.py --prompt "..." --top-k 5
"""

from __future__ import annotations

import argparse

import torch

import jlens

from jlens_plamo.model_loading import MODEL_ID, load_plamo

DEFAULT_LENS_PATH = "data/lens/lens.pt"

DEFAULT_PROMPTS = [
    "県庁所在地が松山である県は",
    "日本で一番高い山は富士山で、二番目に高い山は",
    "The capital of France is",
]


def build_lens_model(model, tokenizer):
    try:
        return jlens.from_hf(model, tokenizer)
    except Exception:
        from jlens_plamo.plamo_adapter import from_plamo

        return from_plamo(model, tokenizer)


def topk_tokens(logits: torch.Tensor, tokenizer, k: int) -> list[str]:
    values, indices = logits.topk(k)
    return [
        f"{tokenizer.decode([idx.item()])!r}({val.item():.2f})"
        for val, idx in zip(values, indices, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", default=DEFAULT_LENS_PATH)
    parser.add_argument("--prompt", action="append", dest="prompts", default=None)
    parser.add_argument("--layers", default=None, help="comma-separated; default: every 4th fitted layer")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print(f"Loading lens: {args.lens}")
    lens = jlens.JacobianLens.load(args.lens)
    print(f"  {lens}")

    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load_plamo(device=args.device)
    lens_model = build_lens_model(model, tokenizer)

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else lens.source_layers[::4] + [lens.source_layers[-1]]
    )
    layers = sorted(set(layers) & set(lens.source_layers))

    for prompt in prompts:
        print(f"\n{'=' * 70}\nprompt: {prompt!r}")
        lens_logits, model_logits, input_ids = lens.apply(
            lens_model, prompt, layers=layers, positions=[-1]
        )
        print(f"  tokenized as {input_ids.shape[1]} tokens")
        for layer in layers:
            top = topk_tokens(lens_logits[layer][0], tokenizer, args.top_k)
            print(f"  layer {layer:>2}: {', '.join(top)}")
        top_final = topk_tokens(model_logits[0], tokenizer, args.top_k)
        print(f"  model (L31, true): {', '.join(top_final)}")


if __name__ == "__main__":
    main()
