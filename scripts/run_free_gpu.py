#!/usr/bin/env python3
"""Create or execute the fail-closed free-tier GPU plan."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from metaflora_incubus.cloud_failure_reporting import run_with_failure_reporting
from metaflora_incubus.cloud_training import (
    CheckpointBackend,
    CloudConstraintError,
    CloudExecutionPlan,
    RemoteCheckpointTarget,
    cloud_disk_budget,
    load_cloud_config,
)


def detect_vram_bytes() -> int:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(process.stdout.splitlines()[0].strip()) * 1024**2


def detect_code_revision() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = process.stdout.strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise CloudConstraintError("cloud code must run from a pinned Git commit")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description="Metaflora Incubus free GPU runner")
    parser.add_argument("--config", default="configs/cloud/free-gpu-v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parameter-count", required=True, type=int)
    parser.add_argument(
        "--checkpoint-backend",
        choices=[item.value for item in CheckpointBackend],
        required=True,
    )
    parser.add_argument("--checkpoint-location", required=True)
    parser.add_argument("--checkpoint-branch")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()

    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend(arguments.checkpoint_backend),
        location=arguments.checkpoint_location,
        branch=arguments.checkpoint_branch,
    )
    if arguments.execute:
        try:
            code_revision = detect_code_revision()
        except Exception as revision_failure:

            def repeat_revision_failure(failure: Exception = revision_failure) -> None:
                raise failure

            run_with_failure_reporting(
                repeat_revision_failure,
                target=target,
                run_id=arguments.run_id,
                code_revision="unknown",
                phase="post-bootstrap-execution",
                environment=os.environ,
            )
            raise AssertionError(
                "failure reporter returned after a revision failure"
            ) from revision_failure
        os.environ["INCUBUS_CODE_REVISION"] = code_revision
    else:
        code_revision = "unknown"

    def execute() -> dict[str, object] | None:
        config = load_cloud_config(Path(arguments.config))
        plan = CloudExecutionPlan.create(
            config=config,
            checkpoint_target=target,
            run_id=arguments.run_id,
            parameter_count=arguments.parameter_count,
            vram_bytes=detect_vram_bytes(),
        )
        summary = {
            "local_retention": plan.local_retention,
            "product_id": config.product_id,
            "public_upload": "blocked_until_eval_gates",
            "resume_enabled": plan.resume_enabled,
            "required_disk_bytes": cloud_disk_budget(plan).required_bytes,
            "run_id": plan.run_id,
            "training_mode": plan.training_mode,
            "workspace": str(plan.workspace),
        }
        print(json.dumps(summary, sort_keys=True))
        if not arguments.execute:
            return None
        required = (
            "INCUBUS_CHECKPOINT_HMAC_KEY",
            "INCUBUS_SOURCE_REPO",
            "INCUBUS_SOURCE_REVISION",
            "INCUBUS_DATASET_REPO",
            "INCUBUS_DATASET_REVISION",
            "INCUBUS_DATASET_SHA256",
            "INCUBUS_PARAMETER_COUNT",
            "INCUBUS_BENCHMARK_SIGNING_KEY",
        )
        missing = tuple(name for name in required if not os.environ.get(name))
        if missing:
            raise CloudConstraintError(f"missing cloud secret names: {', '.join(missing)}")
        try:
            bootstrap_parameter_count = int(os.environ["INCUBUS_PARAMETER_COUNT"])
        except ValueError as exc:
            raise CloudConstraintError("bootstrap parameter count must be an integer") from exc
        if not 0 < bootstrap_parameter_count <= 7_500_000_000:
            raise CloudConstraintError("bootstrap parameter count is outside the compact profile")
        if bootstrap_parameter_count != arguments.parameter_count:
            raise CloudConstraintError("command parameter count does not match the bootstrap")
        from metaflora_incubus.cloud_training_runtime import execute_training_and_build

        return execute_training_and_build(plan=plan, environment=os.environ)

    if arguments.execute:
        result = run_with_failure_reporting(
            execute,
            target=target,
            run_id=arguments.run_id,
            code_revision=code_revision,
            phase="post-bootstrap-execution",
            environment=os.environ,
        )
        print(json.dumps(result, sort_keys=True))
    else:
        execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
