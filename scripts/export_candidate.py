#!/usr/bin/env python3
"""Merge a trained adapter, export GGUF, and build the release Q4 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} is not a file: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise SystemExit(f"{label} is not a directory: {resolved}")
    return resolved


def run_checked(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True, shell=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained Incubus adapter to Q4 GGUF")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--convert-script", type=Path, required=True)
    parser.add_argument("--quantize-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = require_directory(args.base, "base model")
    adapter = require_directory(args.adapter, "trained adapter")
    convert_script = require_file(args.convert_script, "GGUF converter")
    quantize_binary = require_file(args.quantize_binary, "quantizer")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = output / "merged-safetensors"
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
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
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    full_gguf = output / "incubus-v1-f16.gguf"
    q4_gguf = output / "metaflora-incubus-v1-q4.gguf"
    run_checked(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(full_gguf),
            "--outtype",
            "f16",
        ]
    )
    run_checked([str(quantize_binary), str(full_gguf), str(q4_gguf), "Q4_K_M"])
    with q4_gguf.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise SystemExit("quantizer did not produce a GGUF file")
    size = q4_gguf.stat().st_size
    if not 5 * 1024**3 <= size <= 6 * 1024**3:
        raise SystemExit(f"Q4 artifact is outside the 5-6 GiB release window: {size}")
    manifest = {
        "schema_version": 1,
        "candidate_state": "quantized_candidate",
        "release_ready": False,
        "artifact": {
            "path": q4_gguf.name,
            "sha256": sha256_file(q4_gguf),
            "size_bytes": size,
        },
        "required_next_step": "run_parity_and_release_gates",
    }
    (output / "candidate-export.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
