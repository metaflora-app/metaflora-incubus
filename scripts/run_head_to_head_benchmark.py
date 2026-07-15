#!/usr/bin/env python3
"""Run the private, paired GGUF head-to-head promotion benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaflora_incubus.head_to_head_benchmark import (
    load_head_to_head_config,
    run_head_to_head,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired private GGUF comparison")
    parser.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args()
    report = run_head_to_head(load_head_to_head_config(arguments.manifest))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
