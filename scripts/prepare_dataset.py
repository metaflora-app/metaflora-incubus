#!/usr/bin/env python3
"""Prepare pinned maintainer training data for Metaflora Incubus v1."""

from __future__ import annotations

import argparse
import json

from metaflora_incubus.training_entrypoints import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Metaflora Incubus v1 training data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--config", default="configs/training/incubus-v1.json")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    result = prepare_dataset(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        expected_input_sha256=arguments.expected_input_sha256,
        config_path=arguments.config,
        dry_run=arguments.dry_run,
    )
    print(
        json.dumps(
            {
                "dataset_sha256": result.dataset_sha256,
                "dry_run": arguments.dry_run,
                "input_sha256": result.input_sha256,
                "product_id": "metaflora-incubus-v1",
                "record_counts": dict(result.record_counts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
