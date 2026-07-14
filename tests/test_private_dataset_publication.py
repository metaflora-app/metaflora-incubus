from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from metaflora_incubus.private_dataset import (
    DATASET_BUNDLE_FILES,
    PrivateDatasetError,
    publish_private_dataset_bundle,
)


def canonical_json(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def build_bundle(root: Path) -> str:
    payloads = {
        "sft.jsonl": b'{"capability":"code","messages":[]}\n',
        "preference.jsonl": b'{"capability":"code","prompt":[]}\n',
        "sft_validation.jsonl": b'{"capability":"code","messages":[]}\n',
        "preference_validation.jsonl": b'{"capability":"code","prompt":[]}\n',
        "provenance.jsonl": b'{"record_id":"one"}\n',
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    file_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    dataset_sha = hashlib.sha256(canonical_json(file_hashes)).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha,
        "input_sha256": "f" * 64,
        "files": file_hashes,
        "capability_counts": {
            name: {
                capability: 1
                for capability in ("code", "agentic_tools", "russian_text", "english_text")
            }
            for name in (
                "sft.jsonl",
                "preference.jsonl",
                "sft_validation.jsonl",
                "preference_validation.jsonl",
            )
        },
        "record_counts": {
            "sft": 1,
            "preference": 1,
            "sft_validation": 1,
            "preference_validation": 1,
        },
    }
    (root / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return dataset_sha


class RecordingPrivateUploader:
    def __init__(self, remote: dict[str, bytes], *, private: bool = True) -> None:
        self.remote = remote
        self.private = private
        self.calls: list[tuple[str, object]] = []

    def ensure_private_dataset(self, *, repo_id: str) -> None:
        self.calls.append(("ensure", repo_id))
        if not self.private:
            raise PrivateDatasetError("dataset repository is not private")

    def upload_bundle(self, *, repo_id: str, bundle: Path) -> str:
        self.calls.append(("upload", repo_id))
        self.remote = {name: (bundle / name).read_bytes() for name in DATASET_BUNDLE_FILES}
        return "a" * 40

    def snapshot(self, *, repo_id: str, revision: str) -> dict[str, bytes]:
        self.calls.append(("snapshot", revision))
        return dict(self.remote)


def test_private_bundle_upload_returns_exact_commit_and_verifies_remote_bytes(
    tmp_path: Path,
) -> None:
    dataset_sha = build_bundle(tmp_path)
    uploader = RecordingPrivateUploader({})

    result = publish_private_dataset_bundle(
        bundle=tmp_path,
        repo_id="private-owner/incubus-data",
        uploader=uploader,
    )

    assert result.revision == "a" * 40
    assert result.dataset_sha256 == dataset_sha
    assert result.verified is True
    assert result.repo_id_sha256 == hashlib.sha256(b"private-owner/incubus-data").hexdigest()
    assert [name for name, _value in uploader.calls] == ["ensure", "upload", "snapshot"]
    with pytest.raises(FrozenInstanceError):
        result.revision = "b" * 40  # type: ignore[misc]


def test_private_upload_fails_closed_for_public_repo_bad_revision_and_remote_drift(
    tmp_path: Path,
) -> None:
    build_bundle(tmp_path)
    with pytest.raises(PrivateDatasetError, match="private"):
        publish_private_dataset_bundle(
            bundle=tmp_path,
            repo_id="private-owner/incubus-data",
            uploader=RecordingPrivateUploader({}, private=False),
        )

    bad_revision = RecordingPrivateUploader({})
    bad_revision.upload_bundle = lambda **kwargs: "main"  # type: ignore[method-assign]
    with pytest.raises(PrivateDatasetError, match="40-hex"):
        publish_private_dataset_bundle(
            bundle=tmp_path,
            repo_id="private-owner/incubus-data",
            uploader=bad_revision,
        )

    class DriftingUploader(RecordingPrivateUploader):
        def snapshot(self, *, repo_id: str, revision: str) -> dict[str, bytes]:
            snapshot = super().snapshot(repo_id=repo_id, revision=revision)
            return {**snapshot, "sft.jsonl": b"tampered\n"}

    with pytest.raises(PrivateDatasetError, match="read-back"):
        publish_private_dataset_bundle(
            bundle=tmp_path,
            repo_id="private-owner/incubus-data",
            uploader=DriftingUploader({}),
        )


def test_private_upload_rejects_extra_files_and_public_model_destination(tmp_path: Path) -> None:
    build_bundle(tmp_path)
    (tmp_path / "notes.txt").write_text("must stay out")
    with pytest.raises(PrivateDatasetError, match="exactly"):
        publish_private_dataset_bundle(
            bundle=tmp_path,
            repo_id="private-owner/incubus-data",
            uploader=RecordingPrivateUploader({}),
        )
    (tmp_path / "notes.txt").unlink()
    with pytest.raises(PrivateDatasetError, match="separate"):
        publish_private_dataset_bundle(
            bundle=tmp_path,
            repo_id="metaflora/incubus",
            uploader=RecordingPrivateUploader({}),
        )
