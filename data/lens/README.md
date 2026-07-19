# Fitted lens — `lens.pt`

Neuronpedia-style record of exactly what produced this artifact, per `CLAUDE.md`'s Phase 3
requirements. `lens.pt` itself is not committed (see repo root `.gitignore`); regenerate with the
command below.

## Recipe

| | |
|---|---|
| Model | `pfnet/plamo-3-nict-8b-base`, loaded via `jlens_plamo.model_loading.load_plamo()` |
| Corpus | `data/corpus/prompts.jsonl`, n=100 (fineweb-2 `jpn_Jpan` ×83, Wikipedia ja ×17 — see `data/corpus/config.yaml`) |
| Source layers | 0–30 (31 of the model's 32 decoder layers) |
| Target layer | 31 (final decoder layer; its own residual is already in the final basis, so it has no `J_l`) |
| `dim_batch` | 8 |
| `max_seq_len` | 128 tokens |
| Chunking | 10 chunks of 10 prompts each, fit independently and combined with `JacobianLens.merge()` (see `scripts/run_fit.py`) |
| Fitting time | 1h59m wall clock (100 prompts, ~75s/prompt), RTX 5090 32GB, bf16 |
| Output | `data/lens/lens.pt`, ~993 MB (fp16-packed `J_l` matrices, `JacobianLens.save()` default) |
| Command | `uv run python scripts/run_fit.py` (defaults: `--prompts data/corpus/prompts.jsonl --output data/lens/lens.pt --checkpoint-dir data/lens/checkpoints --chunk-size 10 --dim-batch 8 --max-seq-len 128`) |

Why 31 source layers, not 32: `jlens.fit()` requires every source layer to be strictly below
`target_layer`, and `target_layer` defaults to the model's final layer. The final layer's residual
is already expressed in the final basis (it's what the model itself decodes), so a `J_l` for it
would be a no-op transport, not a genuine reproduction gap.

## Qualitative check (Phase 3b completion condition)

Run via `uv run python scripts/qualitative_check.py`, reading out top-5 next-token predictions at
every 4th fitted layer (via `JacobianLens.apply()`) for a few two-step-reasoning prompts.

**`県庁所在地が松山である県は` (the prefecture whose capital is Matsuyama is):**

By layer 20, `愛媛県`/`愛媛` (Ehime, the correct answer) is already the top readout and stays
dominant through the final layer — a clean, unambiguous lookahead signal:

```
layer 20: '愛媛県'(38.00), '愛媛'(32.75), '高知県'(32.50), '松山市'(32.50), '四国'(31.88)
layer 24: '愛媛県'(42.75), '愛媛'(37.00), '松山市'(36.25), '高知県'(34.25), '四国'(33.25)
layer 30: '愛媛県'(16.88), '愛媛'(15.06), '四国'(13.25), '松山市'(13.00), '松山'(12.38)
model L31 (true): '愛媛県'(15.88), '愛媛'(14.25), '四国'(12.50), '松山市'(12.25), '松山'(11.81)
```

**`The capital of France is`:** same pattern — `Paris`/`パリ` dominates from layer 20 onward,
matching the model's own final answer.

**`日本で一番高い山は富士山で、二番目に高い山は` (Japan's tallest mountain is Fuji, the
second-tallest is):** noisier and worth reporting honestly rather than cherry-picking. Layers
20–28 do surface mountain-related tokens (`富士山`, `山頂`, `立山`, `穂高`), but the model's own
final-layer answer diverges to `北`/`南`/`御` (plausibly continuing toward a place-name compound
like "北岳" rather than answering with a bare mountain name) — the lens is reading out what the
model is actually disposed to say, which here is not a clean single-mountain-name answer.

**Reading**: the lens produces genuinely meaningful, non-noise Japanese lookahead readouts on
factual two-step-reasoning prompts, at n=100. It is not uniformly clean — see the mountain example
— which is consistent with `CLAUDE.md`'s expectation that n=100 may need expansion to n≈300 if
noise turns out to dominate. On this small qualitative sample it does not look like it does; a
larger, systematic Phase 4/5 evaluation would be needed to say more than that.

## Regenerating

```bash
uv run python scripts/run_fit.py
```

Resumable: prompts are fit in disjoint chunks under `data/lens/checkpoints/` (gitignored); a chunk
whose output file already exists is loaded instead of re-fit, and `jlens.fit()`'s own
per-prompt checkpoint further protects an in-progress chunk. Delete `data/lens/checkpoints/` to
force a full refit.
