#!/usr/bin/env python3
"""Resume a completed Kaggle DPO export and produce an artifact-bound Q5_K_M GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaflora_incubus.kaggle_recovery_export import (
    GIB,
    ExportRecoveryError,
    RecoveryExportConfig,
    execute_recovery_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="incubus-v1-refine-001")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--merge-script", type=Path, required=True)
    parser.add_argument("--convert-script", type=Path, required=True)
    parser.add_argument("--quantize-binary", type=Path, required=True)
    parser.add_argument("--minimum-free-disk-gib", type=int, default=24)
    parser.add_argument("--minimum-ram-gib", type=int, default=20)
    arguments = parser.parse_args()
    try:
        config = RecoveryExportConfig.create(
            run_id=arguments.run_id,
            base_model=arguments.base,
            adapter=arguments.adapter,
            workspace=arguments.workspace,
            merge_script=arguments.merge_script,
            convert_script=arguments.convert_script,
            quantize_binary=arguments.quantize_binary,
            minimum_free_disk_bytes=arguments.minimum_free_disk_gib * GIB,
            minimum_ram_bytes=arguments.minimum_ram_gib * GIB,
        )
        result = execute_recovery_export(config)
    except ExportRecoveryError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "artifact": str(result.artifact),
                "artifact_sha256": result.artifact_sha256,
                "artifact_size_bytes": result.artifact_size_bytes,
                "gguf_quantization": "Q5_K_M",
                "manifest": str(result.manifest),
                "resumed": result.resumed,
                "run_id": arguments.run_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
