"""Immutable provenance records for every Incubus run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    base_model: str
    model_revision: str
    model_sha256: str
    dataset_sha256: str
    seed: int
    strength: float
    transform_version: str

    @classmethod
    def create(
        cls,
        *,
        base_model: str,
        model_revision: str,
        model_sha256: str,
        dataset_sha256: str,
        seed: int,
        strength: float,
        transform_version: str,
    ) -> RunManifest:
        temporary = cls(
            schema_version=1,
            run_id="",
            base_model=base_model,
            model_revision=model_revision,
            model_sha256=model_sha256,
            dataset_sha256=dataset_sha256,
            seed=seed,
            strength=strength,
            transform_version=transform_version,
        )
        return cls(**{**asdict(temporary), "run_id": temporary.compute_run_id()})

    def compute_run_id(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key != "run_id"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(canonical).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
