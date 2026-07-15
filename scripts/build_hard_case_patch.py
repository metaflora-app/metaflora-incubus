#!/usr/bin/env python3
"""Build a deterministic private SFT/preference patch from benchmark failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaflora_incubus.hard_case_distillation import (
    HardCaseDistillationError,
    build_hard_case_patch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-cases", type=Path, required=True)
    parser.add_argument("--benchmark-raw", type=Path, required=True)
    parser.add_argument("--teacher-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-output-root", type=Path, required=True)
    parser.add_argument("--expected-cases-sha256", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument(
        "--prohibited-identifier",
        action="append",
        default=[],
        help="Identifier forbidden in private patch records; repeat as needed.",
    )
    parser.add_argument("--failure-score-threshold", type=float, default=0.75)
    arguments = parser.parse_args()
    try:
        result = build_hard_case_patch(
            benchmark_cases=arguments.benchmark_cases,
            benchmark_raw=arguments.benchmark_raw,
            teacher_corpus=arguments.teacher_corpus,
            output_dir=arguments.output_dir,
            private_output_root=arguments.private_output_root,
            expected_cases_sha256=arguments.expected_cases_sha256,
            expected_raw_sha256=arguments.expected_raw_sha256,
            expected_teacher_sha256=arguments.expected_teacher_sha256,
            prohibited_identifiers=tuple(arguments.prohibited_identifier),
            failure_score_threshold=arguments.failure_score_threshold,
        )
    except HardCaseDistillationError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "candidate_artifact_sha256": result.candidate_artifact_sha256,
                "failure_count": result.failure_count,
                "manifest_sha256": result.manifest_sha256,
                "patch_count": result.patch_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
