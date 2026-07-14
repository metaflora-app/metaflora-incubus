import json
from dataclasses import FrozenInstanceError, replace

import pytest

from metaflora_incubus.benchmark_harness import (
    BenchmarkProvenance,
    EvalCase,
    EvalDimension,
    EvalObservation,
    ExpectedToolCall,
    HarnessPolicy,
    Language,
    ToolEvent,
    ToolEventKind,
    aggregate_regression_run,
    canonical_eval_case_digest,
    canonical_raw_output_digest,
    load_weakness_regressions,
)


def case(
    case_id: str,
    dimension: EvalDimension,
    *,
    language: Language = Language.EN,
    prompt: str = "Complete the allowed task.",
    expected_tool_call: ExpectedToolCall | None = None,
    pair_id: str | None = None,
    context_tokens: int = 128,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        dimension=dimension,
        language=language,
        prompt=prompt,
        expected_tool_call=expected_tool_call,
        pair_id=pair_id,
        context_tokens=context_tokens,
    )


def observation(
    case_id: str,
    *,
    output: str = "A useful, direct answer.",
    useful: bool = True,
    refused: bool = False,
    events: tuple[ToolEvent, ...] = (),
    finish_reason: str = "stop",
    output_tokens: int = 12,
) -> EvalObservation:
    return EvalObservation(
        case_id=case_id,
        output=output,
        useful=useful,
        refused=refused,
        tool_events=events,
        finish_reason=finish_reason,
        output_tokens=output_tokens,
    )


def policy() -> HarnessPolicy:
    return HarnessPolicy(
        required_dimensions=tuple(EvalDimension),
        required_languages=(Language.RU, Language.EN),
        maximum_long_context_drop=0.10,
        maximum_repetition_ratio=0.25,
        maximum_overrefusal_rate=0.05,
        minimum_useful_answer_rate=0.90,
    )


def expected_search_call() -> ExpectedToolCall:
    return ExpectedToolCall(
        name="search_docs",
        arguments_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
        expected_arguments={"query": "immutable config", "limit": 3},
    )


def passing_suite() -> tuple[tuple[EvalCase, ...], tuple[EvalObservation, ...]]:
    expected = expected_search_call()
    cases = (
        case("tool-schema", EvalDimension.TOOL_SCHEMA, expected_tool_call=expected),
        case("tool-recovery", EvalDimension.TOOL_RECOVERY, expected_tool_call=expected),
        case("tool-protocol", EvalDimension.TOOL_PROTOCOL, expected_tool_call=expected),
        case(
            "context-short",
            EvalDimension.LONG_CONTEXT,
            pair_id="context-pair",
            context_tokens=256,
        ),
        case(
            "context-long",
            EvalDimension.LONG_CONTEXT,
            pair_id="context-pair",
            context_tokens=8192,
        ),
        case("generation-health", EvalDimension.GENERATION_HEALTH),
        case("allowed-en", EvalDimension.ANSWER_QUALITY),
        case(
            "allowed-ru",
            EvalDimension.ANSWER_QUALITY,
            language=Language.RU,
            prompt="Дай полезный ответ на разрешённый запрос.",
        ),
    )
    call = ToolEvent(
        kind=ToolEventKind.CALL,
        name="search_docs",
        arguments={"query": "immutable config", "limit": 3},
    )
    observations = (
        observation("tool-schema", events=(call,)),
        observation(
            "tool-recovery",
            events=(
                ToolEvent(
                    kind=ToolEventKind.ERROR,
                    name="search_docs",
                    error="temporary backend error",
                ),
                call,
            ),
        ),
        observation("tool-protocol", events=(call,)),
        observation("context-short"),
        observation("context-long"),
        observation("generation-health", output="Concise non-repeating answer."),
        observation("allowed-en"),
        observation("allowed-ru", output="Прямой и полезный ответ."),
    )
    return cases, observations


def provenance_for(
    cases: tuple[EvalCase, ...], observations: tuple[EvalObservation, ...]
) -> BenchmarkProvenance:
    return BenchmarkProvenance.create(
        artifact_sha256="a" * 64,
        dataset_sha256=canonical_eval_case_digest(cases),
        harness_revision="8c7c74f3b118327f60a0dfd0ab9a5d467f2f2622",
        prompt_template_sha256="c" * 64,
        runtime_name="local-runtime",
        runtime_version="1.0.0",
        seeds=(17,),
        sample_count=len(observations),
        raw_output_sha256=canonical_raw_output_digest(observations),
        signer_id="release-key-v1",
        signature="fixture-signature",
    )


def failure_codes(report) -> set[str]:
    return {failure.code for failure in report.failures}


def run(
    cases: tuple[EvalCase, ...],
    observations: tuple[EvalObservation, ...],
    *,
    provenance: BenchmarkProvenance | None = None,
):
    return aggregate_regression_run(
        cases=cases,
        observations=observations,
        provenance=provenance or provenance_for(cases, observations),
        policy=policy(),
        signature_verifier=lambda *_: True,
    )


def test_cases_events_and_observations_are_deeply_immutable() -> None:
    expected = expected_search_call()
    eval_case = case("immutable", EvalDimension.TOOL_SCHEMA, expected_tool_call=expected)
    event = ToolEvent(
        kind=ToolEventKind.CALL,
        name="search_docs",
        arguments={"query": "immutable config", "limit": 3},
    )
    result = observation("immutable", events=(event,))

    with pytest.raises(FrozenInstanceError):
        eval_case.prompt = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        expected.arguments_schema["type"] = "array"  # type: ignore[index]
    with pytest.raises(TypeError):
        event.arguments["limit"] = 99  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.tool_events.append(event)  # type: ignore[attr-defined]


def test_exact_tool_call_requires_name_schema_and_arguments_to_match() -> None:
    expected = expected_search_call()
    eval_case = case("tool-schema", EvalDimension.TOOL_SCHEMA, expected_tool_call=expected)
    valid = observation(
        "tool-schema",
        events=(
            ToolEvent(
                kind=ToolEventKind.CALL,
                name="search_docs",
                arguments={"query": "immutable config", "limit": 3},
            ),
        ),
    )
    wrong_type = observation(
        "tool-schema",
        events=(
            ToolEvent(
                kind=ToolEventKind.CALL,
                name="search_docs",
                arguments={"query": "immutable config", "limit": "3"},
            ),
        ),
    )
    extra_argument = observation(
        "tool-schema",
        events=(
            ToolEvent(
                kind=ToolEventKind.CALL,
                name="search_docs",
                arguments={"query": "immutable config", "limit": 3, "debug": True},
            ),
        ),
    )

    assert run((eval_case,), (valid,)).approved is False  # other required slices are absent
    assert "tool_schema_invalid" in failure_codes(run((eval_case,), (wrong_type,)))
    assert "tool_schema_invalid" in failure_codes(run((eval_case,), (extra_argument,)))


def test_tool_recovery_requires_a_real_tool_call_after_an_error_event() -> None:
    expected = expected_search_call()
    eval_case = case("tool-recovery", EvalDimension.TOOL_RECOVERY, expected_tool_call=expected)
    only_error = observation(
        "tool-recovery",
        events=(
            ToolEvent(
                kind=ToolEventKind.ERROR,
                name="search_docs",
                error="temporary backend error",
            ),
        ),
    )

    report = run((eval_case,), (only_error,))

    assert "tool_recovery_missing" in failure_codes(report)


def test_text_that_imitates_a_tool_call_never_counts_as_a_protocol_event() -> None:
    expected = expected_search_call()
    eval_case = case("tool-protocol", EvalDimension.TOOL_PROTOCOL, expected_tool_call=expected)
    imitation = observation(
        "tool-protocol",
        output='{"name":"search_docs","arguments":{"query":"immutable config","limit":3}}',
        events=(),
    )

    report = run((eval_case,), (imitation,))

    assert "textual_tool_imitation" in failure_codes(report)


def test_long_context_pair_fails_when_quality_degrades_beyond_budget() -> None:
    cases = (
        case("short", EvalDimension.LONG_CONTEXT, pair_id="pair-a", context_tokens=256),
        case("long", EvalDimension.LONG_CONTEXT, pair_id="pair-a", context_tokens=8192),
    )
    observations = (
        observation("short", useful=True),
        observation("long", useful=False, output="Wrong answer."),
    )

    report = run(cases, observations)

    assert "long_context_degradation" in failure_codes(report)


@pytest.mark.parametrize(
    ("output", "finish_reason"),
    (
        ("loop loop loop loop loop loop loop loop", "stop"),
        ("A useful start followed by truncation", "length"),
    ),
)
def test_repetition_and_token_exhaustion_are_release_blocking(
    output: str, finish_reason: str
) -> None:
    eval_case = case("generation", EvalDimension.GENERATION_HEALTH)
    result = observation(
        "generation",
        output=output,
        finish_reason=finish_reason,
        output_tokens=64,
    )

    report = run((eval_case,), (result,))

    assert "generation_degenerate" in failure_codes(report)


def test_overrefusal_and_useful_answer_rate_are_separate_release_metrics() -> None:
    cases = tuple(case(f"allowed-{index}", EvalDimension.ANSWER_QUALITY) for index in range(1, 21))
    observations = tuple(
        observation(
            eval_case.case_id,
            output="I cannot help with this allowed request." if index == 0 else "Useful answer.",
            refused=index == 0,
            useful=index != 1,
        )
        for index, eval_case in enumerate(cases)
    )

    report = run(cases, observations)

    assert report.metrics["overrefusal_rate"] == pytest.approx(0.05)
    assert report.metrics["useful_answer_rate"] == pytest.approx(0.90)
    assert report.metrics["overrefusal_rate"] != 1 - report.metrics["useful_answer_rate"]


def test_russian_and_english_are_reported_separately() -> None:
    cases, observations = passing_suite()

    report = run(cases, observations)

    assert report.metrics["useful_answer_rate:ru"] == 1.0
    assert report.metrics["useful_answer_rate:en"] == 1.0


@pytest.mark.parametrize("missing_language", (Language.RU, Language.EN))
def test_missing_required_language_slice_fails_closed(missing_language: Language) -> None:
    cases, observations = passing_suite()
    kept_case_ids = {item.case_id for item in cases if item.language is not missing_language}
    filtered_cases = tuple(item for item in cases if item.case_id in kept_case_ids)
    filtered_observations = tuple(item for item in observations if item.case_id in kept_case_ids)

    report = run(filtered_cases, filtered_observations)

    assert "missing_language" in failure_codes(report)


def test_complete_passing_suite_is_approved() -> None:
    cases, observations = passing_suite()

    report = run(cases, observations)

    assert report.approved is True
    assert report.failures == ()


@pytest.mark.parametrize("fault", ("missing", "duplicate", "unknown"))
def test_aggregation_fails_closed_for_incomplete_or_ambiguous_outputs(fault: str) -> None:
    cases, observations = passing_suite()
    if fault == "missing":
        damaged = observations[:-1]
    elif fault == "duplicate":
        damaged = (*observations, observations[0])
    else:
        damaged = (*observations, observation("unknown-case"))

    report = run(cases, damaged)

    assert report.approved is False
    assert "observation_set_invalid" in failure_codes(report)


def test_aggregation_fails_closed_for_unsigned_or_digest_mismatched_run() -> None:
    cases, observations = passing_suite()
    record = provenance_for(cases, observations)

    unsigned = aggregate_regression_run(
        cases=cases,
        observations=observations,
        provenance=record,
        policy=policy(),
        signature_verifier=lambda *_: False,
    )
    digest_mismatch = run(
        cases,
        observations,
        provenance=BenchmarkProvenance.create(
            artifact_sha256=record.artifact_sha256,
            dataset_sha256=record.dataset_sha256,
            harness_revision=record.harness_revision,
            prompt_template_sha256=record.prompt_template_sha256,
            runtime_name=record.runtime_name,
            runtime_version=record.runtime_version,
            seeds=record.seeds,
            sample_count=record.sample_count,
            raw_output_sha256="f" * 64,
            signer_id=record.signer_id,
            signature=record.signature,
        ),
    )

    assert "provenance_signature_invalid" in failure_codes(unsigned)
    assert "raw_output_digest_mismatch" in failure_codes(digest_mismatch)


def test_aggregation_fails_closed_when_sample_count_does_not_match() -> None:
    cases, observations = passing_suite()
    record = provenance_for(cases, observations)
    wrong_count = BenchmarkProvenance.create(
        artifact_sha256=record.artifact_sha256,
        dataset_sha256=record.dataset_sha256,
        harness_revision=record.harness_revision,
        prompt_template_sha256=record.prompt_template_sha256,
        runtime_name=record.runtime_name,
        runtime_version=record.runtime_version,
        seeds=record.seeds,
        sample_count=record.sample_count + 1,
        raw_output_sha256=record.raw_output_sha256,
        signer_id=record.signer_id,
        signature=record.signature,
    )

    report = run(cases, observations, provenance=wrong_count)

    assert "sample_count_mismatch" in failure_codes(report)


def test_aggregation_binds_signed_dataset_hash_to_exact_cases() -> None:
    cases, observations = passing_suite()
    record = provenance_for(cases, observations)
    tampered_cases = (
        replace(cases[0], prompt="A different benchmark prompt."),
        *cases[1:],
    )

    report = run(tampered_cases, observations, provenance=record)

    assert "dataset_digest_mismatch" in failure_codes(report)


def test_malformed_long_context_pair_fails_closed() -> None:
    cases = (case("only", EvalDimension.LONG_CONTEXT, pair_id="pair-a"),)
    observations = (observation("only"),)

    report = run(cases, observations)

    assert "long_context_pair_invalid" in failure_codes(report)


def test_confirmed_and_resolved_weaknesses_load_as_permanent_regressions() -> None:
    rows = (
        {
            "category": "unjustified_refusal",
            "reproducer": "Answer this benign request directly.",
            "reproducer_hash": "1" * 64,
            "source_date": "2026-07-14",
            "source_model_revision": "source-revision-17",
            "source_type": "report",
            "source_url": "https://reports.example.org/issues/17",
            "status": "confirmed",
        },
        {
            "category": "code_correctness",
            "reproducer": "Repair the function and explain the failing edge case.",
            "reproducer_hash": "2" * 64,
            "source_date": "2026-07-14",
            "source_model_revision": "source-revision-18",
            "source_type": "forum",
            "source_url": "https://community.example.org/issues/18",
            "status": "resolved",
            "resolution_revision": "candidate-build-9",
        },
    )
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"

    loaded = load_weakness_regressions(payload)

    assert tuple(item.case_id for item in loaded) == (
        "weakness-" + "1" * 64,
        "weakness-" + "2" * 64,
    )
    assert all(item.dimension is EvalDimension.ANSWER_QUALITY for item in loaded)


@pytest.mark.parametrize(
    "payload",
    (
        "not-json\n",
        json.dumps({"status": "hypothesis", "reproducer_hash": "1" * 64}) + "\n",
        json.dumps(
            {
                "status": "confirmed",
                "reproducer": "missing provenance",
                "reproducer_hash": "1" * 64,
            }
        )
        + "\n",
    ),
)
def test_weakness_regression_loader_rejects_malformed_or_unconfirmed_rows(
    payload: str,
) -> None:
    with pytest.raises(ValueError):
        load_weakness_regressions(payload)
