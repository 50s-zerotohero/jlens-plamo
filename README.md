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
uv run hf auth login

# Phase 2 — build the fitting corpus
uv run python data/corpus/build_corpus.py --config data/corpus/config.yaml

# Phase 3 — fit the lens (multi-hour GPU job; see data/lens/README.md for the recipe used)
uv run python scripts/run_fit.py

# Phase 4 — apply the fitted lens: layer x position top-k readout
uv run python -m jlens_plamo.apply "県庁所在地が松山である県は" --layers 16,20,24,28,30

# Phase 5 — haiku probing + UI
```

`uv sync` pulls a CUDA-12.9-linked (`cu129`) torch build (see
[Environment notes](#environment-notes--known-compatibility-issues) below) — if your GPU driver
supports a different CUDA version, adjust the `[[tool.uv.index]]` entry in `pyproject.toml`
accordingly. Always load the model via `jlens_plamo.model_loading.load_plamo()`, not
`AutoModelForCausalLM.from_pretrained()` directly — see the same section for why.

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

## Demos

Run these yourself with `uv run python -m jlens_plamo.apply "<prompt>" --layers ...` (see
`jlens_plamo/apply.py`). Output below is real, from the fitted `data/lens/lens.pt` described in
`data/lens/README.md` — not fabricated or cherry-picked past what's shown.

**Two-step factual reasoning** (Japanese "the prefecture whose capital is Matsuyama"):
the correct answer surfaces from layer 20 onward, well before the final layer.

```
uv run python -m jlens_plamo.apply "県庁所在地が松山である県は" --layer 24 --position -1 --top-k 5
  '愛媛県'(42.75)  '愛媛'(37.00)  '松山市'(36.25)  '高知県'(34.25)  '四国'(33.25)
model (final layer, true): '愛媛県'(15.88)  '愛媛'(14.25)  '四国'(12.50)  '松山市'(12.25)  '松山'(11.81)
```

**"Spider → legs", Japanese analogue** of the Anthropic paper's headline example
(`クモは節足動物で、脚の本数は` — "the spider is an arthropod, the number of legs is"): the count
`8` is already the top readout by layer 28, two layers before the model's own final answer.

```
 layer | ...本数           | は
    24 | 本数           | 蜘
    28 | 数は           | 8
    30 | 数は           | 8
 model | 数            | 8
```

**Cross-lingual concept binding** (`パリはフランスの首都です。日本の首都は` — "Paris is the
capital of France. Japan's capital is"): at layer 20, the readout at the *French* capital's own
position briefly surfaces the English token `" Paris"` / `" in Paris"` even though the whole
prompt is Japanese, and by the same layer `東京` (Tokyo) is already the top readout at the final
position — the underlying concept looks language-agnostic before either surface-form token is
chosen.

```
 layer | 首都(after フランス)   | ... | 首都(after 日本の)  | は
    20 |  Paris          |     | 東京            |  Tokyo
    24 | 首都             |     | 東京            |  Tokyo
    30 | 首都             |     | は             | 東京
 model | の              |     | 東京            | 東京
```

These are three prompts, not a systematic evaluation — see [Limitations](#limitations).

## Project layout

```
data/corpus/    fitting corpus builder + config.yaml (recipe, not raw text)
data/probes/    held-out haiku probe set + loader
data/lens/      fitted lens artifacts (not committed) + fitting README
jlens_plamo/    apply / intervention library
scripts/        run_fit.py and other one-off scripts
web/            FastAPI backend + self-contained HTML slice-grid viewer
```

## Environment notes / known compatibility issues

Phase 3a's smoke test (`scripts/smoke_test.py`) surfaced three real incompatibilities between
`pfnet/plamo-3-nict-8b-base`'s `trust_remote_code` model and the rest of this stack — none of
them the SWA/GDN autograd concern originally anticipated. All three are worked around in
`jlens_plamo/model_loading.py` (`load_plamo()`); use that instead of calling
`AutoModelForCausalLM.from_pretrained()` directly, or you'll hit them yourself.

1. **`jlens.from_hf()`'s layout auto-detection doesn't work on PLaMo-3.** Its decoder wraps the
   block list in an extra `Plamo3Decoder` module — `model.model.layers` is that wrapper, not the
   `nn.ModuleList` of blocks (`model.model.layers.layers` is). `jlens_plamo/plamo_adapter.py`
   implements `jlens.protocol.LensModel` directly instead, which is jlens's own documented
   extension point for exactly this case. `jlens`'s fitting/application math is untouched.
2. **`transformers` version mismatch on tied weights.** `transformers>=5.5` (required by `jlens`)
   expects `_tied_weights_keys: dict[str, str]`; PLaMo's modeling file still sets the pre-5.5 list
   form and crashes in `get_expanded_tied_weights_keys()`. Patched to the dict form at import time.
3. **RoPE cache corruption on load (the one to actually watch out for).** Each layer's
   `RotaryEmbedding` holds `inv_freq`/`cos_cached`/`sin_cached` as `persistent=False` buffers —
   pure functions of static config, no learned data. On every load we tried, a random subset of
   the 32 layers' worth of these buffers (varies per run, sometimes zero, sometimes over a third
   of the model) come out as uninitialized garbage instead of the values `__init__` computed,
   producing NaNs partway through the forward pass. Reproduced on CPU and CUDA, with and without
   `low_cpu_mem_usage`. This looks like an upstream `transformers`/`accelerate` bug with
   non-persistent computed buffers surviving the meta-device-skeleton + checkpoint-dispatch
   loading path, not something specific to PLaMo or jlens. `load_plamo()` unconditionally rebuilds
   these buffers after loading (cheap, deterministic, safe regardless of root cause). If you ever
   load this model without going through `load_plamo()`, sanity-check
   `torch.isnan(model.model.layers.layers[i].mixer.rotary_emb.inv_freq).any()` before trusting any
   output.

Separately, `uv sync` pins `torch` to the `cu129` wheel index (see `pyproject.toml`): the default
PyPI torch build at the time of writing links against CUDA 13, newer than what this project's dev
GPU (RTX 5090, driver 576.88, CUDA 12.9) supports. If you're on different hardware/drivers, adjust
or remove that pin.

## Limitations

- Initial fitting corpus is n=100 prompts — small relative to the paper's n=1000; readouts may
  be noisy. See `data/lens/README.md` for the actual corpus size used for the fitted lens.
- The fineweb-2 portion of the corpus is filtered with simple heuristics (NSFW keyword denylist,
  boilerplate keywords, nav-menu line-shape detection, sentence-ending-punctuation density to
  catch EC/listing dumps) rather than a learned quality classifier. A human review of 5 random
  samples at each generation caught and removed adult-content and product-listing-dump documents
  across two filter iterations; one residual machine-translation-flavored ad-copy document was
  knowingly left in the accepted n=100 as a known, tolerated source of noise at this corpus size.
- PLaMo-3's `trust_remote_code` attention implementation (SWA/full hybrid) autograd/VJP behavior
  was checked in the Phase 3a smoke test and gradients flow correctly at both an SWA and a
  full-attention layer — see [Environment notes](#environment-notes--known-compatibility-issues)
  above for the (different) issues that smoke test actually caught.
- The haiku lookahead probing (Phase 5) is an original extension, not part of the Anthropic
  paper. Findings are reported as "interpretable but noisy" vs. "clear lookahead pattern"
  without exaggeration, in the spirit of `jlens-qwen36`'s "hypothesis-generating rather than a
  robust reproduction" framing.
- The [Demos](#demos) above are three hand-picked prompts run once each, not a systematic
  evaluation — they show the lens *can* produce clean, meaningful readouts, not that it reliably
  does so across arbitrary prompts. `data/lens/README.md`'s qualitative check includes a fourth,
  noisier example (Japan's second-tallest mountain) deliberately kept in for balance.

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
