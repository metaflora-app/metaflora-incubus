"""Private, byte-verified publication of prepared training datasets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from metaflora_incubus.training_entrypoints import _load_dataset_manifest

DATASET_BUNDLE_FILES = (
    "manifest.json",
    "sft.jsonl",
    "preference.jsonl",
    "sft_validation.jsonl",
    "preference_validation.jsonl",
    "provenance.jsonl",
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_AUTOMATIC_HUB_FILES = frozenset({".gitattributes"})


class PrivateDatasetError(ValueError):
    """Raised when a dataset could leak publicly or cannot be verified."""


class PrivateDatasetUploader(Protocol):
    def ensure_private_dataset(self, *, repo_id: str) -> None: ...

    def upload_bundle(self, *, repo_id: str, bundle: Path) -> str: ...

    def snapshot(self, *, repo_id: str, revision: str) -> dict[str, bytes]: ...


@dataclass(frozen=True)
class PrivateDatasetPublication:
    revision: str
    dataset_sha256: str
    repo_id_sha256: str
    verified: bool


def _local_snapshot(bundle: Path) -> dict[str, bytes]:
    if not bundle.is_dir() or bundle.is_symlink():
        raise PrivateDatasetError("prepared dataset bundle is invalid")
    entries = tuple(bundle.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise PrivateDatasetError("prepared dataset bundle must contain files only")
    if {entry.name for entry in entries} != set(DATASET_BUNDLE_FILES):
        raise PrivateDatasetError("prepared dataset bundle must contain exactly six files")
    return {name: (bundle / name).read_bytes() for name in DATASET_BUNDLE_FILES}


def _dataset_sha256(bundle: Path) -> str:
    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        value = manifest["dataset_sha256"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PrivateDatasetError("prepared dataset manifest is invalid") from exc
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PrivateDatasetError("prepared dataset sha256 is invalid")
    try:
        _load_dataset_manifest(bundle / "manifest.json", value)
    except ValueError as exc:
        raise PrivateDatasetError("prepared dataset bytes do not match the manifest") from exc
    return value


def _remote_bundle_names_are_valid(names: object) -> bool:
    try:
        remote_names = set(names)  # type: ignore[arg-type]
    except TypeError:
        return False
    return remote_names - _AUTOMATIC_HUB_FILES == set(DATASET_BUNDLE_FILES)


def publish_private_dataset_bundle(
    *, bundle: Path, repo_id: str, uploader: PrivateDatasetUploader
) -> PrivateDatasetPublication:
    """Upload a prepared bundle privately, then verify every byte at the returned commit."""

    normalized_repo = repo_id.strip()
    if "/" not in normalized_repo or normalized_repo == "metaflora/incubus":
        raise PrivateDatasetError("training data requires a separate private dataset repository")
    before = _local_snapshot(bundle)
    dataset_sha = _dataset_sha256(bundle)
    uploader.ensure_private_dataset(repo_id=normalized_repo)
    revision = uploader.upload_bundle(repo_id=normalized_repo, bundle=bundle)
    if _REVISION.fullmatch(revision) is None:
        raise PrivateDatasetError("Hub upload did not return an exact 40-hex revision")
    if before != _local_snapshot(bundle):
        raise PrivateDatasetError("local dataset bundle changed during upload")
    remote = uploader.snapshot(repo_id=normalized_repo, revision=revision)
    if remote != before:
        raise PrivateDatasetError("private dataset read-back verification failed")
    uploader.ensure_private_dataset(repo_id=normalized_repo)
    return PrivateDatasetPublication(
        revision=revision,
        dataset_sha256=dataset_sha,
        repo_id_sha256=hashlib.sha256(normalized_repo.encode()).hexdigest(),
        verified=True,
    )


class HuggingFacePrivateDatasetUploader:
    """Production Hub adapter; repository identity never appears in public output."""

    def __init__(self, *, token: str) -> None:
        if not token:
            raise PrivateDatasetError("Hugging Face token is required")
        from huggingface_hub import HfApi

        self._api = HfApi(token=token)
        self._token = token

    def ensure_private_dataset(self, *, repo_id: str) -> None:
        self._api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        info = self._api.dataset_info(repo_id=repo_id)
        if info.private is not True:
            raise PrivateDatasetError("dataset repository is not private")

    def upload_bundle(self, *, repo_id: str, bundle: Path) -> str:
        result = self._api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(bundle),
            commit_message="Upload private Metaflora Incubus training dataset",
        )
        revision = getattr(result, "oid", None)
        return revision if isinstance(revision, str) else ""

    def snapshot(self, *, repo_id: str, revision: str) -> dict[str, bytes]:
        from huggingface_hub import hf_hub_download

        names = self._api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
        if not _remote_bundle_names_are_valid(names):
            return {}
        return {
            name: Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=name,
                    revision=revision,
                    token=self._token,
                    force_download=True,
                )
            ).read_bytes()
            for name in DATASET_BUNDLE_FILES
        }
