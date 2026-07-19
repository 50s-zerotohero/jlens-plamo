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


def _run_generate(prompt: str, max_new_tokens: int) -> tuple[str, int]:
    """Greedy-generate a continuation. Returns (full_text_no_special_tokens,
    prompt_token_count) -- the token count uses the same encode() convention
    (a single leading BOS) that read_layers()/JacobianLens.apply() use, so it
    lines up with the position index in a later /api/slice-style call over
    the full text."""
    state = _get_state()
    model, tokenizer = state["model"], state["tokenizer"]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        # use_cache=False: PLaMo's custom Plamo3Cache.finalize() does
        # `self[layer_idx]`, but transformers' generate() loop hands it a
        # plain Cache object that isn't subscriptable in this transformers
        # version -> TypeError. Not a jlens or lens-fitting issue (plain
        # .generate() never runs during fitting/apply); see README
        # "Environment notes". Disabling the KV cache sidesteps it at the
        # cost of recomputing attention over the whole prefix each step --
        # fine for this UI's short demo continuations.
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False
        )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text, input_ids.shape[1]


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    full_text, _ = _run_generate(req.prompt, req.max_new_tokens)
    return {"full_text": full_text}


class AskRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30
    top_k: int = 10
    layers: list[int] | None = None
    use_jacobian: bool = True


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Free-chat: generate the model's real answer, then run the slice grid
    over the whole question+answer exchange (re-tokenized from the decoded
    text, since JacobianLens.apply() takes a prompt string, not raw ids)."""
    state = _get_state()
    full_text, prompt_token_count = _run_generate(req.prompt, req.max_new_tokens)
    result = read_layers(
        state["lens"],
        state["lens_model"],
        full_text,
        layers=req.layers,
        top_k=req.top_k,
        max_seq_len=prompt_token_count + req.max_new_tokens + 8,
        use_jacobian=req.use_jacobian,
    )
    response = _serialize_slice(result)
    response["prompt_token_count"] = prompt_token_count
    response["full_text"] = full_text
    return response
