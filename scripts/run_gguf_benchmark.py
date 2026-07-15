#!/usr/bin/env python3
"""Run the deterministic local GGUF release benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from metaflora_incubus.gguf_benchmark_runner import (
    BenchmarkRunnerConfig,
    run_gguf_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pinned local GGUF benchmark")
    parser.add_argument("--server-binary", required=True, type=Path)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("benchmarks/gguf-v1-cases.jsonl"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--runner-code-revision",
        default=os.environ.get("INCUBUS_CODE_REVISION"),
        required="INCUBUS_CODE_REVISION" not in os.environ,
    )
    arguments = parser.parse_args()

    config = BenchmarkRunnerConfig.create(
        server_binary=arguments.server_binary,
        server_sha256=arguments.server_sha256,
        model_path=arguments.model,
        model_sha256=arguments.model_sha256,
        cases_path=arguments.cases,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        port=arguments.port,
        health_timeout_seconds=arguments.health_timeout,
        request_timeout_seconds=arguments.request_timeout,
        runner_code_revision=arguments.runner_code_revision,
    )
    print(json.dumps(run_gguf_benchmark(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
