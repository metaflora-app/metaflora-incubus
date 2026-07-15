#!/usr/bin/env python3
"""Validate release benchmark JSONL and emit signed-provenance input fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaflora_incubus.benchmark_evidence import (
    BenchmarkEvidenceError,
    build_benchmark_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Incubus benchmark evidence metadata")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = build_benchmark_evidence(args.cases.resolve(), args.raw.resolve())
    except BenchmarkEvidenceError as exc:
        raise SystemExit(str(exc)) from exc
    output = {
        "schema_version": 1,
        "artifact_sha256": evidence.artifact_sha256,
        "dataset_sha256": evidence.dataset_sha256,
        "raw_output_sha256": evidence.raw_output_sha256,
        "sample_count": evidence.sample_count,
        "scores": dict(evidence.scores),
        "overrefusal_rate": evidence.overrefusal_rate,
        "seeds": list(evidence.seeds),
    }
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
