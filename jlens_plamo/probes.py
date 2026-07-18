"""Loader for the Phase 5 haiku held-out probe set.

The probe content itself (data/probes/haiku_prompts.jsonl) is hand-curated by
a human, not generated here — see data/probes/README.md for the schema and
curation notes. This module only reads and validates that file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("id", "text", "mora_breakdown")
OPTIONAL_FIELDS = ("kigo", "kigo_season", "kireji", "notes")
VALID_SEASONS = {"spring", "summer", "autumn", "winter", "new_year", None}


@dataclass
class HaikuProbe:
    id: str
    text: str
    mora_breakdown: list[int]
    kigo: str | None = None
    kigo_season: str | None = None
    kireji: str | None = None
    notes: str | None = None


def _validate(row: dict, line_no: int) -> None:
    for field in REQUIRED_FIELDS:
        if field not in row:
            raise ValueError(f"line {line_no}: missing required field {field!r}")
    if not isinstance(row["mora_breakdown"], list) or not all(
        isinstance(x, int) for x in row["mora_breakdown"]
    ):
        raise ValueError(f"line {line_no}: mora_breakdown must be a list of ints")
    if row.get("kigo_season") not in VALID_SEASONS:
        raise ValueError(
            f"line {line_no}: kigo_season must be one of {sorted(s for s in VALID_SEASONS if s)} or null"
        )


def load_haiku_probes(path: str | Path) -> list[HaikuProbe]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. It must be hand-curated — see "
            f"{path.parent / 'README.md'} for the schema, or "
            f"{path.parent / 'haiku_prompts.example.jsonl'} for a format example."
        )

    probes: list[HaikuProbe] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _validate(row, line_no)
            probes.append(
                HaikuProbe(
                    id=row["id"],
                    text=row["text"],
                    mora_breakdown=row["mora_breakdown"],
                    kigo=row.get("kigo"),
                    kigo_season=row.get("kigo_season"),
                    kireji=row.get("kireji"),
                    notes=row.get("notes"),
                )
            )
    return probes
