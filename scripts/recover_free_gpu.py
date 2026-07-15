#!/usr/bin/env python3
"""Recover GGUF and benchmark evidence from an authenticated final adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from metaflora_incubus.cloud_failure_reporting import run_with_failure_reporting
from metaflora_incubus.cloud_training import (
    CheckpointBackend,
    CloudExecutionPlan,
    RemoteCheckpointTarget,
    load_cloud_config,
)
from metaflora_incubus.cloud_training_runtime import recover_trained_artifact


def detect_vram_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.splitlines()[0].strip()) * 1024**2


def detect_code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover Metaflora Incubus GGUF")
    parser.add_argument("--config", default="configs/cloud/free-gpu-v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parameter-count", required=True, type=int)
    parser.add_argument("--checkpoint-location", required=True)
    parser.add_argument("--checkpoint-branch", required=True)
    arguments = parser.parse_args()
    target = RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location=arguments.checkpoint_location,
        branch=arguments.checkpoint_branch,
    )
    config = load_cloud_config(Path(arguments.config))
    plan = CloudExecutionPlan.create(
        config=config,
        checkpoint_target=target,
        run_id=arguments.run_id,
        parameter_count=arguments.parameter_count,
        vram_bytes=detect_vram_bytes(),
    )
    code_revision = detect_code_revision()

    def recover() -> dict[str, object]:
        return recover_trained_artifact(plan=plan, environment=os.environ)

    result = run_with_failure_reporting(
        recover,
        target=target,
        run_id=arguments.run_id,
        code_revision=code_revision,
        phase="artifact-recovery",
        environment=os.environ,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
