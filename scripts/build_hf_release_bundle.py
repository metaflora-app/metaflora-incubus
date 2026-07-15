#!/usr/bin/env python3
"""Build and cryptographically validate the measured Hugging Face bundle."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from metaflora_incubus.gguf_benchmark_runner import PRODUCTION_ATTESTATION_PUBLIC_KEY
from metaflora_incubus.huggingface_publication import PublicationPolicy
from metaflora_incubus.release_bundle import (
    BoundBenchmarkInput,
    ReleaseBundleInputs,
    build_release_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Incubus v1 HF release bundle")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--license", dest="license_path", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--prohibited-identifiers", type=Path, required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--harness-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = _json_object(args.reports)
    private_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(args.private_key.read_text(encoding="ascii").strip(), validate=True)
    )
    public_key = private_key.public_key()
    actual_public = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    expected_public = base64.urlsafe_b64decode(PRODUCTION_ATTESTATION_PUBLIC_KEY.encode("ascii"))
    if actual_public != expected_public:
        raise SystemExit("private key does not match the pinned production release key")

    def signer(_purpose: str, payload: bytes) -> bytes:
        return private_key.sign(payload)

    def verifier(_purpose: str, payload: bytes, signature: bytes) -> bool:
        try:
            public_key.verify(signature, payload)
        except InvalidSignature:
            return False
        return True

    base_policy = PublicationPolicy.default()
    prohibited = tuple(
        line.strip()
        for line in args.prohibited_identifiers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    policy = PublicationPolicy(
        repo_id=base_policy.repo_id,
        min_model_bytes=base_policy.min_model_bytes,
        max_model_bytes=base_policy.max_model_bytes,
        prohibited_identifiers=prohibited,
    )
    baselines_document = _object(reports.get("baselines"), "baselines")
    inputs = ReleaseBundleInputs(
        model_path=args.model,
        evidence_dir=args.evidence,
        output_dir=args.output,
        full_precision_report=_bound_report(
            _object(reports.get("full_precision_report"), "full_precision_report")
        ),
        baselines={
            name: _bound_report(_object(value, name))
            for name, value in baselines_document.items()
        },
        runtime_id=args.runtime,
        harness_revision=args.harness_revision,
        license_path=args.license_path,
        notices_path=args.notices,
    )
    result = build_release_bundle(
        inputs,
        signer=signer,
        signature_verifier=verifier,
        publication_policy=policy,
    )
    print(
        json.dumps(
            {
                "approved": result.release_decision.approved,
                "artifact_sha256": result.artifact_sha256,
                "output": str(result.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _bound_report(document: dict[str, object]) -> BoundBenchmarkInput:
    return BoundBenchmarkInput(
        artifact_path=Path(_text(document.get("artifact_path"), "artifact_path")),
        evidence_dir=Path(_text(document.get("evidence_dir"), "evidence_dir")),
        artifact_id=_text(document.get("artifact_id"), "artifact_id"),
        suite_id=_text(document.get("suite_id"), "suite_id"),
    )


def _json_object(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), path.name)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be non-empty text")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
