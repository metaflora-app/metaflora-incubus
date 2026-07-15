from dataclasses import FrozenInstanceError, replace

import pytest

from metaflora_incubus.benchmark_harness import (
    BenchmarkProvenance,
    ProvenanceError,
    canonical_raw_output_digest,
    verify_benchmark_provenance,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def provenance(**overrides: object) -> BenchmarkProvenance:
    values: dict[str, object] = {
        "artifact_sha256": SHA_A,
        "dataset_sha256": SHA_B,
        "harness_revision": "8c7c74f3b118327f60a0dfd0ab9a5d467f2f2622",
        "prompt_template_sha256": SHA_C,
        "runtime_name": "llama.cpp",
        "runtime_version": "b7001",
        "seeds": (17, 29, 43),
        "sample_count": 2,
        "raw_output_sha256": SHA_D,
        "signer_id": "release-key-v1",
        "signature": "signed-payload-base64",
    }
    values.update(overrides)
    return BenchmarkProvenance.create(**values)


def test_provenance_contains_every_reproducibility_and_integrity_field() -> None:
    record = provenance()

    assert record.artifact_sha256 == SHA_A
    assert record.dataset_sha256 == SHA_B
    assert record.harness_revision == "8c7c74f3b118327f60a0dfd0ab9a5d467f2f2622"
    assert record.prompt_template_sha256 == SHA_C
    assert record.runtime_name == "llama.cpp"
    assert record.runtime_version == "b7001"
    assert record.seeds == (17, 29, 43)
    assert record.sample_count == 2
    assert record.raw_output_sha256 == SHA_D
    assert record.signer_id == "release-key-v1"
    assert record.signature == "signed-payload-base64"


def test_provenance_is_deeply_immutable() -> None:
    record = provenance()

    with pytest.raises(FrozenInstanceError):
        record.sample_count = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.seeds.append(59)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "field",
    ("artifact_sha256", "dataset_sha256", "prompt_template_sha256", "raw_output_sha256"),
)
@pytest.mark.parametrize("invalid", ("", "a" * 63, "g" * 64, "A" * 64))
def test_rejects_noncanonical_sha256_fields(field: str, invalid: str) -> None:
    with pytest.raises(ProvenanceError, match=field):
        provenance(**{field: invalid})


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"harness_revision": ""}, "harness_revision"),
        ({"runtime_name": ""}, "runtime_name"),
        ({"runtime_version": ""}, "runtime_version"),
        ({"seeds": ()}, "seeds"),
        ({"seeds": (17, 17)}, "seeds"),
        ({"seeds": (17, -1)}, "seeds"),
        ({"sample_count": 0}, "sample_count"),
        ({"signer_id": ""}, "signer_id"),
        ({"signature": ""}, "signature"),
    ),
)
def test_rejects_incomplete_or_ambiguous_provenance(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ProvenanceError, match=message):
        provenance(**override)


def test_signed_payload_is_canonical_and_excludes_the_signature() -> None:
    first = provenance(signature="first-signature")
    second = provenance(signature="second-signature")

    assert first.signed_payload() == second.signed_payload()
    assert b"first-signature" not in first.signed_payload()
    assert b"second-signature" not in second.signed_payload()
    assert first.signed_payload().startswith(b'{"artifact_sha256"')


def test_signature_verification_receives_pinned_signer_and_exact_payload() -> None:
    record = provenance()
    calls: list[tuple[str, bytes, str]] = []

    def verifier(signer_id: str, payload: bytes, signature: str) -> bool:
        calls.append((signer_id, payload, signature))
        return True

    assert verify_benchmark_provenance(record, verifier=verifier) is True
    assert calls == [(record.signer_id, record.signed_payload(), record.signature)]


def test_signature_verification_fails_closed_on_false_or_verifier_error() -> None:
    record = provenance()

    assert verify_benchmark_provenance(record, verifier=lambda *_: False) is False

    def unavailable(*_: object) -> bool:
        raise RuntimeError("verification backend unavailable")

    assert verify_benchmark_provenance(record, verifier=unavailable) is False


def test_raw_output_digest_is_ordered_canonical_and_detects_any_output_change() -> None:
    outputs = (
        {"case_id": "case-001", "output": "answer one", "seed": 17},
        {"case_id": "case-002", "output": "ответ два", "seed": 29},
    )

    digest = canonical_raw_output_digest(outputs)

    assert len(digest) == 64
    assert digest == canonical_raw_output_digest(tuple(dict(row) for row in outputs))
    assert digest != canonical_raw_output_digest(tuple(reversed(outputs)))
    changed = ({**outputs[0], "output": "changed"}, outputs[1])
    assert digest != canonical_raw_output_digest(changed)


def test_changing_any_signed_field_invalidates_a_signature_bound_to_payload() -> None:
    original = provenance()
    accepted_payload = original.signed_payload()

    def verifier(_: str, payload: bytes, __: str) -> bool:
        return payload == accepted_payload

    assert verify_benchmark_provenance(original, verifier=verifier) is True
    tampered = replace(original, sample_count=original.sample_count + 1)
    assert verify_benchmark_provenance(tampered, verifier=verifier) is False
