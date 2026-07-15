#!/usr/bin/env python3
"""Create a deterministic private raw dataset with an additional teacher corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaflora_incubus.teacher_augmentation import augment_prepared_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dataset", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--private-config", required=True)
    parser.add_argument("--benchmark-cases", required=True)
    parser.add_argument("--private-output-root", required=True)
    parser.add_argument("--train-count", required=True, type=int)
    parser.add_argument("--validation-count", required=True, type=int)
    parser.add_argument("--minimum-output-chars", type=int, default=80)
    parser.add_argument("--maximum-output-chars", type=int, default=16_000)
    arguments = parser.parse_args()
    private_config_path = Path(arguments.private_config)
    if private_config_path.is_symlink() or not private_config_path.is_file():
        raise SystemExit("private config is invalid")
    if private_config_path.stat().st_mode & 0o077:
        raise SystemExit("private config permissions must be 0600")
    private_config = json.loads(private_config_path.read_text(encoding="utf-8"))
    result = augment_prepared_dataset(
        prepared_dataset=arguments.prepared_dataset,
        teacher_jsonl=arguments.teacher_jsonl,
        output_path=arguments.output,
        source_url=private_config["source_url"],
        source_revision=private_config["source_revision"],
        collected_at=private_config["collected_at"],
        license_id=private_config["license_id"],
        capability=private_config["capability"],
        train_count=arguments.train_count,
        validation_count=arguments.validation_count,
        allowed_domains=tuple(private_config["allowed_domains"]),
        minimum_output_chars=arguments.minimum_output_chars,
        maximum_output_chars=arguments.maximum_output_chars,
        prohibited_identifiers=tuple(private_config["prohibited_identifiers"]),
        expected_prepared_sha256=private_config["expected_prepared_sha256"],
        expected_teacher_sha256=private_config["expected_teacher_sha256"],
        benchmark_cases=arguments.benchmark_cases,
        expected_benchmark_sha256=private_config["expected_benchmark_sha256"],
        private_output_root=arguments.private_output_root,
    )
    print(
        json.dumps(
            {
                "added_train_count": result.added_train_count,
                "added_validation_count": result.added_validation_count,
                "existing_count": result.existing_count,
                "output_sha256": result.output_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
