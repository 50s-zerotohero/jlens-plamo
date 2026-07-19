"""Phase 3b: production Jacobian-lens fitting for pfnet/plamo-3-nict-8b-base.

Fits J_l for every source layer against the final-layer residual, over the
Phase 2 corpus (data/corpus/prompts.jsonl, n=100 by default). This is a
multi-hour GPU job; run it manually in the background and let it resume if
interrupted.

Resumability is two-layered:
  1. Prompts are split into disjoint chunks (--chunk-size). Each chunk is
     fit independently with jlens.fit() and saved to
     <checkpoint-dir>/chunk_NNN.pt; a chunk whose file already exists is
     loaded instead of re-fit. This is the jlens_plamo.merge()-based scheme
     CLAUDE.md asks for, and it means a crash never loses more than one
     in-progress chunk.
  2. Within a chunk, jlens.fit()'s own checkpoint_path/resume (see
     jlens.fitting.fit) saves the running per-prompt sum, so even an
     in-progress chunk resumes mid-way rather than from its own start.

Finished chunk lenses are combined with jlens.JacobianLens.merge() (an
n_prompts-weighted mean, exact regardless of chunking) into the final lens.

Usage:
    uv run python scripts/run_fit.py
    uv run python scripts/run_fit.py --limit 4 --chunk-size 2  # quick script smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jlens

from jlens_plamo.model_loading import MODEL_ID, load_plamo

logger = logging.getLogger("run_fit")

DEFAULT_PROMPTS_PATH = "data/corpus/prompts.jsonl"
DEFAULT_OUTPUT_PATH = "data/lens/lens.pt"
DEFAULT_CHECKPOINT_DIR = "data/lens/checkpoints"


def _human_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def load_prompts(path: str, *, limit: int | None = None) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["text"])
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def build_lens_model(model, tokenizer):
    """Same from_hf-then-adapter-fallback pattern as scripts/smoke_test.py."""
    try:
        lens_model = jlens.from_hf(model, tokenizer)
        logger.info("jlens.from_hf() succeeded directly: %s", lens_model)
    except Exception as e:
        logger.info(
            "jlens.from_hf() failed as expected for PLaMo's wrapped decoder "
            "(%s: %s); using PlamoLensModel adapter.",
            type(e).__name__,
            e,
        )
        from jlens_plamo.plamo_adapter import from_plamo

        lens_model = from_plamo(model, tokenizer)
    return lens_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="prompts per disjoint fit-then-merge chunk (default: 10)",
    )
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--source-layers",
        default=None,
        help="comma-separated layer indices; default: every layer below --target-layer",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help="default: final layer (n_layers - 1)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only fit on the first N prompts (for a quick script smoke-test, "
        "not the production run)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    jlens.configure_logging(level=args.log_level)

    prompts = load_prompts(args.prompts, limit=args.limit)
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    logger.info("Loading model %s ...", MODEL_ID)
    model, tokenizer = load_plamo(device=args.device)
    lens_model = build_lens_model(model, tokenizer)
    logger.info("%s", lens_model)

    source_layers = (
        [int(x) for x in args.source_layers.split(",")] if args.source_layers else None
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        prompts[i : i + args.chunk_size] for i in range(0, len(prompts), args.chunk_size)
    ]
    n_chunks = len(chunks)
    logger.info(
        "Fitting %d prompts in %d chunks of <=%d prompts (checkpoint dir: %s)",
        len(prompts),
        n_chunks,
        args.chunk_size,
        checkpoint_dir,
    )

    chunk_lenses = []
    chunk_durations: list[float] = []
    run_start = time.perf_counter()

    for chunk_idx, chunk_prompts in enumerate(chunks):
        chunk_final_path = checkpoint_dir / f"chunk_{chunk_idx:03d}.pt"
        chunk_inprogress_path = checkpoint_dir / f"chunk_{chunk_idx:03d}.inprogress.pt"
        first = chunk_idx * args.chunk_size
        last = first + len(chunk_prompts) - 1

        if chunk_final_path.exists():
            logger.info(
                "[chunk %d/%d] already fitted (prompts %d-%d), loading %s",
                chunk_idx + 1,
                n_chunks,
                first,
                last,
                chunk_final_path,
            )
            chunk_lenses.append(jlens.JacobianLens.load(str(chunk_final_path)))
            continue

        if chunk_durations:
            avg = sum(chunk_durations) / len(chunk_durations)
            eta = _human_duration(avg * (n_chunks - chunk_idx))
        else:
            eta = "unknown"
        logger.info(
            "[chunk %d/%d] fitting prompts %d-%d  elapsed=%s  eta=%s",
            chunk_idx + 1,
            n_chunks,
            first,
            last,
            _human_duration(time.perf_counter() - run_start),
            eta,
        )

        chunk_start = time.perf_counter()
        chunk_lens = jlens.fit(
            lens_model,
            chunk_prompts,
            source_layers=source_layers,
            target_layer=args.target_layer,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            checkpoint_path=str(chunk_inprogress_path),
            checkpoint_every=1,
            resume=True,
        )
        chunk_duration = time.perf_counter() - chunk_start
        chunk_durations.append(chunk_duration)

        chunk_lens.save(str(chunk_final_path))
        chunk_inprogress_path.unlink(missing_ok=True)
        logger.info(
            "[chunk %d/%d] done in %s -> %s",
            chunk_idx + 1,
            n_chunks,
            _human_duration(chunk_duration),
            chunk_final_path,
        )
        chunk_lenses.append(chunk_lens)

    logger.info("All chunks fitted; merging %d chunk lenses", len(chunk_lenses))
    merged = jlens.JacobianLens.merge(chunk_lenses)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(output_path))
    logger.info(
        "Saved merged lens to %s (%s)  total elapsed=%s",
        output_path,
        merged,
        _human_duration(time.perf_counter() - run_start),
    )


if __name__ == "__main__":
    main()
