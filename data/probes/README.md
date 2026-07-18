# Haiku held-out probe set

This is the held-out probe set for Phase 5 (haiku lookahead probing). It is kept
**separate from the fitting corpus** (`data/corpus/`) — no haiku or verse data is
mixed into the J-lens fitting data.

`haiku_prompts.jsonl` (the real 10-20 curated haiku) is curated by hand by a
human (季語・切れ字・五七五構造 selection requires editorial judgment that a
script shouldn't make). This directory only provides:

- `haiku_prompts.example.jsonl` — 2 **placeholder** rows showing the exact
  schema. Do not treat their content as real curated haiku; they exist only to
  document the format.
- The loader in `jlens_plamo/probes.py` (`load_haiku_probes`), which validates
  and reads `haiku_prompts.jsonl` once it exists.

## Schema (one JSON object per line)

| field           | type          | required | notes                                                   |
|-----------------|---------------|----------|----------------------------------------------------------|
| `id`            | string        | yes      | stable identifier, e.g. `"haiku-001"`                     |
| `text`          | string        | yes      | the poem, newline-separated per line if multi-line        |
| `kigo`          | string        | no       | the season word (季語), if present                        |
| `kigo_season`   | string        | no       | one of `spring`/`summer`/`autumn`/`winter`/`new_year`      |
| `kireji`        | string        | no       | the cutting word (切れ字), e.g. `"や"`, `"かな"`, `"けり"`, if present |
| `mora_breakdown`| list[int]     | yes      | mora count per line, e.g. `[5, 7, 5]`                      |
| `notes`         | string        | no       | free-text annotation (e.g. why this probe was chosen)      |

To add the real set, create `data/probes/haiku_prompts.jsonl` following this
schema.
