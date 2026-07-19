"""Phase 5 UI backend: minimal FastAPI wrapping jlens_plamo.apply for the
web/index.html slice-grid viewer.

Model + lens are loaded once, lazily, on first request (not at import time,
so `uvicorn --reload` doesn't reload the model on every source change).

Endpoints (per CLAUDE.md's Phase 5 spec, plus /api/ask):
  GET  /             -- serves the self-contained HTML viewer
  GET  /api/lens      -- fitted-lens metadata
  POST /api/slice     -- layer x position top-k readout grid for one prompt
  POST /generate      -- plain model continuation (no lens), for comparison
  POST /api/ask       -- free-chat: generate a real answer, then run the
                         slice grid over the whole question+answer exchange

/api/ask exists because a public J-lens demo (jlens.wezzard.com) turned out
to serve precomputed example sessions rather than live chat -- reasonably,
since live-inferencing a public site's traffic is a cost/abuse concern that
doesn't apply to a single local GPU. There's nothing stopping live free-chat
here: /api/ask just runs generate() then feeds the model's own real answer
back through read_layers(), so the grid covers the model's actual output,
not only the prompt.

Run:
    uv run uvicorn web.app:app --port 8420
Then open http://localhost:8420/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jlens
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jlens_plamo.apply import DEFAULT_LENS_PATH, build_lens_model, read_layers
from jlens_plamo.model_loading import MODEL_ID, load_plamo

app = FastAPI(title="jlens-plamo viewer")

_state: dict[str, Any] = {}


def _get_state() -> dict[str, Any]:
    """Lazily load the model + lens once and cache them in-process."""
    if not _state:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_plamo(device=device)
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["lens_model"] = build_lens_model(model, tokenizer)
        _state["lens"] = jlens.JacobianLens.load(DEFAULT_LENS_PATH)
    return _state


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/lens")
def lens_info() -> dict:
    lens = _get_state()["lens"]
    return {
        "model_id": MODEL_ID,
        "d_model": lens.d_model,
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
    }


class SliceRequest(BaseModel):
    prompt: str
    top_k: int = 10
    max_seq_len: int = 512
    layers: list[int] | None = None
    use_jacobian: bool = True


def _serialize_slice(result) -> dict:
    layers_sorted = sorted(result.by_layer)
    return {
        "tokens": result.tokens,
        "positions": result.positions,
        "layers": layers_sorted,
        "grid": {
            str(layer): [
                [{"token": r.token, "logit": r.logit} for r in readouts]
                for readouts in result.by_layer[layer]
            ]
            for layer in layers_sorted
        },
        "model_readout": [
            [{"token": r.token, "logit": r.logit} for r in readouts]
            for readouts in result.model_readout
        ],
    }


@app.post("/api/slice")
def slice_(req: SliceRequest) -> dict:
    state = _get_state()
    result = read_layers(
        state["lens"],
        state["lens_model"],
        req.prompt,
        layers=req.layers,
        top_k=req.top_k,
        max_seq_len=req.max_seq_len,
        use_jacobian=req.use_jacobian,
    )
    return _serialize_slice(result)


def _run_generate_samples(
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
    num_samples: int = 1,
) -> dict:
    """Generate one or more continuations via HF's own `model.generate()` --
    no hand-rolled sampling. `temperature <= 0` forces greedy decoding
    (`do_sample=False`, the tool's original deterministic behavior);
    `num_samples` is forced to 1 in that case since repeated greedy calls are
    always identical. `temperature > 0` samples `num_samples` sequences in a
    single batched call via `num_return_sequences`, and (via
    `output_scores=True`) records each sample's first generated token's
    logprob -- for later correlation against the lens's own readout ranking
    at that position.

    Returns {"prompt_token_count": int, "do_sample": bool, "samples": [...]}
    where each sample is {"full_text", "continuation", "first_token",
    "first_token_logprob"}. `prompt_token_count` uses the same encode()
    convention (a single leading BOS) that read_layers()/JacobianLens.apply()
    use, so it lines up with the position index in a later /api/slice-style
    call over a sample's full_text.
    """
    state = _get_state()
    model, tokenizer = state["model"], state["tokenizer"]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    do_sample = temperature > 0
    effective_num_samples = num_samples if do_sample else 1

    if seed is not None:
        torch.manual_seed(seed)

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
        # use_cache=False: PLaMo's custom Plamo3Cache.finalize() does
        # `self[layer_idx]`, but transformers' generate() loop hands it a
        # plain Cache object that isn't subscriptable in this transformers
        # version -> TypeError. Not a jlens or lens-fitting issue (plain
        # .generate() never runs during fitting/apply); see README
        # "Environment notes". Disabling the KV cache sidesteps it at the
        # cost of recomputing attention over the whole prefix each step --
        # fine for this UI's short demo continuations.
        use_cache=False,
    )
    if do_sample:
        gen_kwargs.update(
            temperature=temperature, top_p=top_p, num_return_sequences=effective_num_samples
        )

    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)

    sequences = outputs.sequences  # [effective_num_samples, prompt_len + generated_len]
    # outputs.scores[0]: logits for the first *generated* token, one row per
    # sample -- exactly what output_scores=True is for; no custom sampling.
    first_step_logprobs = F.log_softmax(outputs.scores[0].float(), dim=-1)

    samples = []
    for i in range(sequences.shape[0]):
        first_token_id = sequences[i, prompt_len].item()
        samples.append(
            {
                "full_text": tokenizer.decode(sequences[i], skip_special_tokens=True),
                "continuation": tokenizer.decode(sequences[i, prompt_len:], skip_special_tokens=True),
                "first_token": tokenizer.decode([first_token_id]),
                "first_token_logprob": first_step_logprobs[i, first_token_id].item(),
            }
        )

    return {"prompt_token_count": prompt_len, "do_sample": do_sample, "samples": samples}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int | None = None
    num_samples: int = 1


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    return _run_generate_samples(
        req.prompt,
        req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        num_samples=req.num_samples,
    )


class AskRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30
    top_k: int = 10
    layers: list[int] | None = None
    use_jacobian: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int | None = None
    num_samples: int = 1


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Free-chat: generate the model's real answer, then run the slice grid
    over the whole question+answer exchange (re-tokenized from the decoded
    text, since JacobianLens.apply() takes a prompt string, not raw ids).
    When num_samples > 1, all samples are returned (for the frequency-table
    view) but the grid is only computed for the first sample -- running the
    full 31-layer readout per sample would be wasteful and there's no single
    grid that could represent several different continuations at once."""
    state = _get_state()
    gen_result = _run_generate_samples(
        req.prompt,
        req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        num_samples=req.num_samples,
    )
    primary = gen_result["samples"][0]
    prompt_token_count = gen_result["prompt_token_count"]
    result = read_layers(
        state["lens"],
        state["lens_model"],
        primary["full_text"],
        layers=req.layers,
        top_k=req.top_k,
        max_seq_len=prompt_token_count + req.max_new_tokens + 8,
        use_jacobian=req.use_jacobian,
    )
    response = _serialize_slice(result)
    response["prompt_token_count"] = prompt_token_count
    response["full_text"] = primary["full_text"]
    response["samples"] = gen_result["samples"]
    response["do_sample"] = gen_result["do_sample"]
    return response
