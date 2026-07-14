#!/usr/bin/env python3
"""Upload a prepared training dataset to a byte-verified private Hub commit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from metaflora_incubus.private_dataset import (
    HuggingFacePrivateDatasetUploader,
    publish_private_dataset_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a private Incubus training dataset")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    arguments = parser.parse_args()
    token = os.environ.get(arguments.token_env, "")
    result = publish_private_dataset_bundle(
        bundle=Path(arguments.bundle),
        repo_id=arguments.repo_id,
        uploader=HuggingFacePrivateDatasetUploader(token=token),
    )
    print(
        json.dumps(
            {
                "dataset_sha256": result.dataset_sha256,
                "product_id": "metaflora-incubus-v1",
                "repo_id_sha256": result.repo_id_sha256,
                "revision": result.revision,
                "verified": result.verified,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
