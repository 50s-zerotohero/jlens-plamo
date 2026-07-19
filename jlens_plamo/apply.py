"""Phase 4: apply a fitted lens to a prompt and read out top-k tokens per
(layer, position).

This is a thin wrapper around ``jlens.lens.JacobianLens.apply()`` — the
transport/readout math lives entirely in `jlens`, not here. What this module
adds: loading the PLaMo model + lens together, decoding readouts into
readable Japanese-safe token strings, and a layer x position CLI table.

Library usage::

    from jlens_plamo.apply import load_lens_and_model, read_layers

    lens, lens_model = load_lens_and_model()
    result = read_layers(lens, lens_model, "県庁所在地が松山である県は")

CLI usage::

    uv run python -m jlens_plamo.apply "県庁所在地が松山である県は"
    uv run python -m jlens_plamo.apply "県庁所在地が松山である県は" --layers 20,24,28,30 --top-k 3
    uv run python -m jlens_plamo.apply "県庁所在地が松山である県は" --layer 24 --position -1 --top-k 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import jlens
import torch

from jlens_plamo.model_loading import MODEL_ID, load_plamo

DEFAULT_LENS_PATH = "data/lens/lens.pt"


@dataclass
class TokenReadout:
    """One ranked token from a top-k readout."""

    token: str
    logit: float

    def __str__(self) -> str:
        return f"{self.token!r}({self.logit:.2f})"


@dataclass
class LensReadout:
    """Full result of applying a lens to one prompt.

    Attributes:
        tokens: Decoded string for each input token, in prompt order.
        positions: The sequence positions that were read out (Python
            indexing; matches the order of each layer's row in `by_layer`).
        by_layer: ``{layer: [top-k readouts for each position]}``.
        model_readout: The model's own final-layer top-k readout at the same
            positions (``use_jacobian=True`` lenses only add a comparison
            point; this is what the model actually predicts).
    """

    tokens: list[str]
    positions: list[int]
    by_layer: dict[int, list[list[TokenReadout]]]
    model_readout: list[list[TokenReadout]]


def build_lens_model(model: torch.nn.Module, tokenizer: Any):
    """Same from_hf()-then-adapter-fallback pattern used across this project
    (see scripts/smoke_test.py); PLaMo-3's wrapped decoder always falls
    through to the PlamoLensModel adapter."""
    try:
        return jlens.from_hf(model, tokenizer)
    except Exception:
        from jlens_plamo.plamo_adapter import from_plamo

        return from_plamo(model, tokenizer)


def load_lens_and_model(
    *, lens_path: str = DEFAULT_LENS_PATH, device: str = "cuda"
) -> tuple[jlens.JacobianLens, Any]:
    """Load a fitted lens and the PLaMo model together, wrapped for jlens."""
    lens = jlens.JacobianLens.load(lens_path)
    model, tokenizer = load_plamo(device=device)
    lens_model = build_lens_model(model, tokenizer)
    return lens, lens_model


def _topk(logits: torch.Tensor, tokenizer: Any, k: int) -> list[TokenReadout]:
    values, indices = logits.topk(min(k, logits.shape[-1]))
    return [
        TokenReadout(tokenizer.decode([idx.item()]), val.item())
        for val, idx in zip(values, indices, strict=True)
    ]


def read_layers(
    lens: jlens.JacobianLens,
    lens_model: Any,
    prompt: str,
    *,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
    top_k: int = 5,
    max_seq_len: int = 512,
    use_jacobian: bool = True,
) -> LensReadout:
    """Apply `lens` to `prompt` and return top-k readouts at every requested
    (layer, position). `layers` defaults to every fitted layer;
    `positions` defaults to every token in the (possibly truncated) prompt.
    """
    if layers is None:
        layers = lens.source_layers
    lens_logits, model_logits, input_ids = lens.apply(
        lens_model,
        prompt,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
        use_jacobian=use_jacobian,
    )
    tokenizer = lens_model.tokenizer
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    resolved_positions = (
        list(positions) if positions is not None else list(range(len(tokens)))
    )

    by_layer: dict[int, list[list[TokenReadout]]] = {}
    for layer in layers:
        logits = lens_logits[layer]  # [n_positions, vocab_size]
        by_layer[layer] = [_topk(logits[i], tokenizer, top_k) for i in range(logits.shape[0])]

    model_readout = [_topk(model_logits[i], tokenizer, top_k) for i in range(model_logits.shape[0])]

    return LensReadout(
        tokens=tokens,
        positions=resolved_positions,
        by_layer=by_layer,
        model_readout=model_readout,
    )


def format_top1_table(result: LensReadout, *, cell_width: int = 12) -> str:
    """Render a layer x position grid of top-1 tokens as plain text.

    Not column-aligned by display width (CJK characters are double-width in
    most terminals, so naive `str.ljust` misaligns on mixed Japanese/ASCII
    tokens) — cells are simply padded/truncated by character count and
    separated with `|`, which stays readable without pulling in a table
    library.
    """
    header_cells = [
        (result.tokens[p] if p < len(result.tokens) else str(p))[:cell_width].ljust(cell_width)
        for p in result.positions
    ]
    lines = [f"{'layer':>6} | " + " | ".join(header_cells)]
    lines.append("-" * len(lines[0]))
    for layer in sorted(result.by_layer):
        cells = [
            readouts[0].token[:cell_width].ljust(cell_width) if readouts else "".ljust(cell_width)
            for readouts in result.by_layer[layer]
        ]
        lines.append(f"{layer:>6} | " + " | ".join(cells))
    model_cells = [
        readouts[0].token[:cell_width].ljust(cell_width) if readouts else "".ljust(cell_width)
        for readouts in result.model_readout
    ]
    lines.append(f"{'model':>6} | " + " | ".join(model_cells))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--lens", default=DEFAULT_LENS_PATH)
    parser.add_argument(
        "--layers", default=None, help="comma-separated layer indices; default: every 4th fitted layer"
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="single layer for detailed --top-k output at --position"
    )
    parser.add_argument(
        "--position",
        type=int,
        default=None,
        help="single position (Python indexing) for detailed --top-k output; requires --layer",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--no-jacobian", action="store_true", help="vanilla logit-lens baseline (skip J_l transport)")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lens, lens_model = load_lens_and_model(lens_path=args.lens, device=args.device)
    print(f"lens: {lens}")

    if args.layer is not None and args.position is not None:
        result = read_layers(
            lens,
            lens_model,
            args.prompt,
            layers=[args.layer],
            positions=[args.position],
            top_k=args.top_k,
            max_seq_len=args.max_seq_len,
            use_jacobian=not args.no_jacobian,
        )
        token = result.tokens[args.position] if -len(result.tokens) <= args.position < len(result.tokens) else "?"
        print(f"prompt: {args.prompt!r}")
        print(f"position {args.position} ({token!r}), layer {args.layer}:")
        for readout in result.by_layer[args.layer][0]:
            print(f"  {readout}")
        print("model (final layer, true):")
        for readout in result.model_readout[0]:
            print(f"  {readout}")
        return

    layers = [int(x) for x in args.layers.split(",")] if args.layers else lens.source_layers[::4] + [lens.source_layers[-1]]
    layers = sorted(set(layers) & set(lens.source_layers))
    result = read_layers(
        lens,
        lens_model,
        args.prompt,
        layers=layers,
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
        use_jacobian=not args.no_jacobian,
    )
    print(f"prompt: {args.prompt!r}  ({len(result.tokens)} tokens)")
    print(format_top1_table(result))


if __name__ == "__main__":
    main()
