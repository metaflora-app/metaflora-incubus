#!/usr/bin/env python3
"""Merge one local, completed adapter into its local base model for recovery export."""

from __future__ import annotations

import argparse
from pathlib import Path


def _directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_dir():
        raise SystemExit(f"{label} is invalid")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    base = _directory(arguments.base, "base model")
    adapter = _directory(arguments.adapter, "DPO adapter")
    output = arguments.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit("merge output already exists")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False
    )
    merged = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    merged = merged.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
