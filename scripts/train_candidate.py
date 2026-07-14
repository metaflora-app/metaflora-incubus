#!/usr/bin/env python3
"""Run or validate the Metaflora Incubus v1 maintainer training plan."""

from __future__ import annotations

import argparse
import json

from metaflora_incubus.training_entrypoints import train_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Metaflora Incubus v1 candidate")
    parser.add_argument("--config", default="configs/training/incubus-v1.json")
    parser.add_argument("--resume-metadata")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    run = train_candidate(
        config_path=arguments.config,
        resume_metadata_path=arguments.resume_metadata,
        dry_run=arguments.dry_run,
    )
    print(
        json.dumps(
            {
                "config_sha256": run.execution_config_sha256,
                "dry_run": run.dry_run,
                "candidate_state": run.candidate_state.value,
                "plan_sha256": run.execution_plan_sha256,
                "product_id": run.product_id,
                "release_ready": run.release_ready,
                "required_post_training": list(run.post_training.steps),
                "resume_next_step": (
                    run.resume_checkpoint.next_step if run.resume_checkpoint is not None else None
                ),
                "stages": [recipe.kind.value for recipe in run.stage_recipes],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
