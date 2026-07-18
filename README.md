# jlens-plamo

> This is not evidence of consciousness. This is not proof that PLaMo has a Claude-like
> global workspace. This is not yet a robust reproduction of the Anthropic paper.

A Japanese-language reproduction of Anthropic's [Jacobian Lens (J-lens)](https://transformer-circuits.pub/2026/workspace/index.html)
on [`pfnet/plamo-3-nict-8b-base`](https://huggingface.co/pfnet/plamo-3-nict-8b-base) (32 layers, GQA,
Gemma-style SWA/full-attention hybrid, no GDN), with an original extension: probing whether
haiku lookahead planning (kigo / kireji / mora count) is visible through J-space.

## Status

**Demo-quality, research preview.** This is a hobbyist reproduction, not a research-grade
replication of the Anthropic paper. See [Limitations](#limitations) below.

## Quick start

```bash
uv sync
# Agree to the PLaMo Community License on Hugging Face first, then:
uv run huggingface-cli login

# Phase 2 — build the fitting corpus
uv run python data/corpus/build_corpus.py --config data/corpus/config.yaml

# Phase 3 — fit the lens (see data/lens/README.md once fitting has run)
# Phase 4 — apply the lens
# Phase 5 — haiku probing + UI
```

## How it works

The Jacobian lens reads out what a mid-layer residual-stream activation is "disposed to say"
by linearly transporting it into the final-layer basis with a layer-averaged Jacobian `J_ℓ`,
then decoding it with the model's own unembedding into a ranked list of vocabulary tokens.
`J_ℓ` is fit (not hand-derived) from a corpus of ordinary text by summing cotangents at the
target position and averaging over source positions — the fitting and application logic here
is provided by [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) and is
not reimplemented.

The fitting corpus is mostly [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
(`jpn_Jpan` config) with a Japanese Wikipedia supplement for long-document coherence. Haiku data
is kept out of the fitting corpus entirely and used only as a held-out probe set (see
`data/probes/`).

## Project layout

```
data/corpus/    fitting corpus builder + config.yaml (recipe, not raw text)
data/probes/    held-out haiku probe set + loader
data/lens/      fitted lens artifacts (not committed) + fitting README
jlens_plamo/    apply / intervention library
scripts/        run_fit.py and other one-off scripts
web/            FastAPI backend + self-contained HTML slice-grid viewer
```

## Limitations

- Initial fitting corpus is n=100 prompts — small relative to the paper's n=1000; readouts may
  be noisy. See `data/lens/README.md` for the actual corpus size used for the fitted lens.
- PLaMo-3's `trust_remote_code` attention implementation (SWA/full hybrid) has not been
  independently verified upstream for autograd/VJP compatibility with `jacobian-lens`; this is
  checked with a smoke test before any production fit (see Phase 3a in `CLAUDE.md`).
- The haiku lookahead probing (Phase 5) is an original extension, not part of the Anthropic
  paper. Findings are reported as "interpretable but noisy" vs. "clear lookahead pattern"
  without exaggeration, in the spirit of `jlens-qwen36`'s "hypothesis-generating rather than a
  robust reproduction" framing.

## Acknowledgements

- [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) — the fitting and
  application logic this project depends on and does not reimplement.
- [`WeZZard/jlens-qwen36`](https://github.com/WeZZard/jlens-qwen36) — corpus loader design, UI,
  and project layout reference.

## License

Code is Apache-2.0 (see `LICENSE`), matching `anthropics/jacobian-lens`.

`pfnet/plamo-3-nict-8b-base` itself is distributed under the separate
[PLaMo Community License](https://huggingface.co/pfnet/plamo-3-nict-8b-base) — not Apache-2.0.
Per that license: outputs/derivatives must display "Built with PLaMo" where applicable, and
commercial use above the license's revenue threshold requires a separate commercial agreement
(see the license page for current terms). Read the license on the model page before commercial
use.

Raw corpus text (fineweb-2, Wikipedia) is never committed to this repository — only
`data/corpus/config.yaml` and `data/corpus/build_corpus.py`, which regenerate an equivalent
corpus deterministically via a fixed seed.
