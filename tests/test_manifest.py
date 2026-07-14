from metaflora_incubus.manifest import RunManifest


def test_manifest_is_stable_for_identical_inputs() -> None:
    manifest = RunManifest.create(
        base_model="org/model",
        model_revision="abc123",
        model_sha256="1" * 64,
        dataset_sha256="2" * 64,
        seed=42,
        strength=0.8,
        transform_version="0.1.0",
    )

    assert manifest.run_id == manifest.compute_run_id()
    assert manifest.schema_version == 1


def test_manifest_changes_when_transformation_strength_changes() -> None:
    common = {
        "base_model": "org/model",
        "model_revision": "abc123",
        "model_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "seed": 42,
        "transform_version": "0.1.0",
    }

    weaker = RunManifest.create(**common, strength=0.2)
    stronger = RunManifest.create(**common, strength=0.8)

    assert weaker.run_id != stronger.run_id
