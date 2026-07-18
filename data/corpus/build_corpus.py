"""Build the J-lens fitting corpus for jlens-plamo (Phase 2).

Pulls documents from HuggingFaceFW/fineweb-2 (jpn_Jpan) and wikimedia/wikipedia
(ja), filters for length / boilerplate / near-duplication, and writes
data/corpus/prompts.jsonl. See data/corpus/config.yaml for the recipe; this
script and that file together are the only things committed to the repo — the
generated prompts.jsonl is gitignored.

Usage:
    uv run python data/corpus/build_corpus.py --config data/corpus/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import yaml

BOILERPLATE_KEYWORDS = [
    "サイトマップ", "トップページ", "利用規約", "プライバシーポリシー",
    "全て表示", "コメントを投稿", "この記事をシェア", "シェアする",
    "Copyright ©", "All Rights Reserved", "無断転載を禁じます",
    "ログイン", "会員登録", "カートに入れる", "お問い合わせ", "cookie",
]

# Hard reject, unlike BOILERPLATE_KEYWORDS: any single hit disqualifies the
# document rather than counting toward a threshold. Deliberately targets
# explicit-content web-scrape artifacts, not general adult-themed topics.
NSFW_KEYWORDS = [
    "アダルト動画", "エロ動画", "エロ画像", "無修正", "潮吹き", "緊縛",
    "AV女優", "風俗", "出会い系", "ヌード写真", "セックス", "アダルトビデオ",
    "素人撮影", "人妻", "熟女", "巨乳美女",
]

# EC / product-listing dumps read as concatenated noun phrases (item names,
# brand tags, prices, sizes) rather than sentences, so they carry almost no
# sentence-ending punctuation relative to their length.
SENTENCE_END_RE = re.compile(r"[。！？]")

JAPANESE_CHAR_RE = re.compile(
    r"[぀-ゟ゠-ヿ一-鿿]"
)


def contains_nsfw(text: str) -> bool:
    return any(kw in text for kw in NSFW_KEYWORDS)


def sentence_end_density(text: str) -> float:
    if not text:
        return 0.0
    return len(SENTENCE_END_RE.findall(text)) / len(text)


def japanese_char_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    jp = sum(1 for c in non_ws if JAPANESE_CHAR_RE.match(c))
    return jp / len(non_ws)


def looks_like_boilerplate(text: str, cfg: dict) -> bool:
    if contains_nsfw(text):
        return True

    hits = sum(1 for kw in BOILERPLATE_KEYWORDS if kw in text)
    if hits > cfg["max_boilerplate_keyword_hits"]:
        return True

    if text.count("|") > cfg["max_pipe_char_count"]:
        return True

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) >= cfg["min_line_count_for_nav_check"]:
        lengths = sorted(len(ln) for ln in lines)
        median_len = lengths[len(lengths) // 2]
        if median_len < cfg["min_median_line_length"]:
            return True

    if (
        len(text) >= cfg["min_length_for_density_check"]
        and sentence_end_density(text) < cfg["min_sentence_end_density"]
    ):
        # Listing dumps (product names / brand tags / prices strung together)
        # carry almost no 。！？ relative to length, unlike real prose.
        return True

    if japanese_char_ratio(text) < cfg["min_japanese_char_ratio"]:
        return True

    return False


def shingles(text: str, n: int) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def doc_id(source: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}-{h}"


def load_source_stream(source_cfg: dict, seed: int):
    from datasets import load_dataset

    ds = load_dataset(
        source_cfg["dataset"],
        name=source_cfg["config"],
        split=source_cfg["split"],
        streaming=source_cfg["streaming"],
    )
    ds = ds.shuffle(seed=seed, buffer_size=source_cfg["shuffle_buffer_size"])
    return ds


def extract_text(example: dict) -> str | None:
    for key in ("text", "content"):
        if key in example and example[key]:
            return example[key]
    return None


def collect_from_source(
    source_cfg: dict,
    tokenizer,
    filters: dict,
    seed: int,
    accepted_shingles: list[set[str]],
) -> Iterator[dict]:
    ds = load_source_stream(source_cfg, seed)
    target = source_cfg["target_count"]
    max_scan = source_cfg["max_candidates_scanned"]
    accepted = 0
    scanned = 0
    start = time.time()

    for example in ds:
        if accepted >= target or scanned >= max_scan:
            break
        scanned += 1

        text = extract_text(example)
        if not text:
            continue
        text = text.strip()
        if not text:
            continue

        if looks_like_boilerplate(text, filters):
            continue

        n_tokens_full = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens_full < filters["min_tokens"]:
            continue

        stored_text = text[: filters["max_stored_chars"]]
        n_tokens_stored = len(
            tokenizer(stored_text, add_special_tokens=False)["input_ids"]
        )
        if n_tokens_stored < filters["min_tokens"]:
            # cap trimmed us below the threshold; fall back to full text
            stored_text = text
            n_tokens_stored = n_tokens_full

        cand_shingles = shingles(stored_text, filters["near_dup_shingle_size"])
        if any(
            jaccard(cand_shingles, prior) >= filters["near_dup_jaccard_threshold"]
            for prior in accepted_shingles
        ):
            continue

        accepted_shingles.append(cand_shingles)
        accepted += 1

        if accepted % 10 == 0 or accepted == target:
            elapsed = time.time() - start
            print(
                f"[{source_cfg['name']}] accepted {accepted}/{target} "
                f"(scanned {scanned}, {elapsed:.1f}s)",
                file=sys.stderr,
            )

        yield {
            "text": stored_text,
            "source": source_cfg["name"],
            "n_tokens": n_tokens_stored,
            "doc_id": doc_id(source_cfg["name"], stored_text),
        }

    if accepted < target:
        print(
            f"WARNING: [{source_cfg['name']}] only found {accepted}/{target} "
            f"after scanning {scanned} candidates (max_scan={max_scan}). "
            "Consider raising max_candidates_scanned or relaxing filters.",
            file=sys.stderr,
        )


def build_corpus(config: dict) -> list[dict]:
    from transformers import AutoTokenizer

    seed = config["seed"]
    random.seed(seed)

    tok_cfg = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg["model_id"], trust_remote_code=tok_cfg["trust_remote_code"]
    )

    filters = config["filters"]
    accepted_shingles: list[set[str]] = []
    docs: list[dict] = []

    for source_cfg in config["sources"]:
        docs.extend(
            collect_from_source(source_cfg, tokenizer, filters, seed, accepted_shingles)
        )

    rng = random.Random(seed)
    rng.shuffle(docs)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("data/corpus/config.yaml")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_path = args.output or Path(config["output"]["prompts_path"])

    try:
        docs = build_corpus(config)
    except OSError as e:
        print(
            "ERROR: failed to load the PLaMo tokenizer. If this is a "
            "gated-repo / authentication error, agree to the PLaMo Community "
            "License at https://huggingface.co/pfnet/plamo-3-nict-8b-base "
            "and run `uv run huggingface-cli login` first.\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    total_target = config["total_target_count"]
    if len(docs) != total_target:
        print(
            f"WARNING: generated {len(docs)} documents, expected exactly "
            f"{total_target}. See per-source warnings above.",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Wrote {len(docs)} documents to {output_path}")

    by_source: dict[str, int] = {}
    for doc in docs:
        by_source[doc["source"]] = by_source.get(doc["source"], 0) + 1
    print(f"By source: {by_source}")

    preview_n = config["output"]["sample_preview_count"]
    sample = random.Random(config["seed"]).sample(docs, min(preview_n, len(docs)))
    print(f"\n--- {len(sample)} random samples for human review ---")
    for doc in sample:
        preview = doc["text"][:200].replace("\n", " ")
        print(f"\n[{doc['doc_id']}] source={doc['source']} n_tokens={doc['n_tokens']}")
        print(preview + ("..." if len(doc["text"]) > 200 else ""))


if __name__ == "__main__":
    main()
