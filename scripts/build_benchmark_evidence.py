#!/usr/bin/env python3
"""Validate release benchmark JSONL and emit signed-provenance input fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_METRICS = (
    "coding",
    "tool_calling",
    "agentic_search",
    "text_quality",
    "russian",
    "english",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"{path}:{number}: row must be an object")
        rows.append(value)
    if not rows:
        raise SystemExit(f"{path}: no benchmark rows")
    return rows


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be non-empty text")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Incubus benchmark evidence metadata")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases_path = args.cases.resolve()
    raw_path = args.raw.resolve()
    cases = load_jsonl(cases_path)
    raw = load_jsonl(raw_path)
    case_ids = {require_text(row.get("case_id"), "case_id") for row in cases}
    raw_ids = [require_text(row.get("case_id"), "case_id") for row in raw]
    if len(case_ids) != len(cases) or len(set(raw_ids)) != len(raw) or set(raw_ids) != case_ids:
        raise SystemExit("case IDs are duplicated or do not match raw outputs")
    for row in cases:
        require_text(row.get("prompt"), "prompt")
    for row in raw:
        require_text(row.get("response"), "response")
        scores = row.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(REQUIRED_METRICS):
            raise SystemExit("each raw row must contain every pinned metric")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= float(value) <= 1
            for value in scores.values()
        ):
            raise SystemExit("benchmark scores must be in [0, 1]")
        if not isinstance(row.get("refused"), bool):
            raise SystemExit("each raw row needs a boolean refusal label")
    output = {
        "schema_version": 1,
        "dataset_sha256": sha256_file(cases_path),
        "raw_output_sha256": sha256_file(raw_path),
        "sample_count": len(raw),
    }
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
