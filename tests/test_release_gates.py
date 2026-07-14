from dataclasses import replace
from math import inf, nan

import pytest

from metaflora_incubus.release_gates import (
    BenchmarkReport,
    ReleaseGatePolicy,
    evaluate_release,
)

CRITICAL_CAPABILITIES = (
    "coding",
    "agentic_search",
    "text_quality",
    "russian",
    "english",
)
REQUIRED_BASELINES = ("reference", "gfusion", "yandex")


def report(
    artifact_id: str,
    scores: dict[str, float],
    *,
    overrefusal_rate: float,
    asr_wer: float | None = None,
    suite_id: str = "incubus-release-v1",
) -> BenchmarkReport:
    return BenchmarkReport(
        artifact_id=artifact_id,
        suite_id=suite_id,
        scores=scores,
        overrefusal_rate=overrefusal_rate,
        asr_wer=asr_wer,
    )


@pytest.fixture
def policy() -> ReleaseGatePolicy:
    return ReleaseGatePolicy(
        required_baselines=REQUIRED_BASELINES,
        required_score_targets={
            "coding": 0.80,
            "agentic_search": 0.75,
            "text_quality": 0.82,
            "russian": 0.85,
            "english": 0.84,
        },
        minimum_lead_over_each_baseline=0.01,
        minimum_overrefusal_reduction=0.10,
        maximum_quantization_drop=0.02,
        require_asr=False,
        maximum_asr_wer=0.12,
        minimum_asr_lead_over_each_baseline=0.01,
    )


@pytest.fixture
def reference() -> BenchmarkReport:
    return report(
        "reference-full",
        {
            "coding": 0.80,
            "agentic_search": 0.75,
            "text_quality": 0.82,
            "russian": 0.84,
            "english": 0.85,
        },
        overrefusal_rate=0.22,
        asr_wer=0.15,
    )


@pytest.fixture
def gfusion() -> BenchmarkReport:
    return report(
        "competitor-gfusion",
        {
            "coding": 0.62,
            "agentic_search": 0.50,
            "text_quality": 0.74,
            "russian": 0.79,
            "english": 0.75,
        },
        overrefusal_rate=0.20,
        asr_wer=0.14,
    )


@pytest.fixture
def yandex() -> BenchmarkReport:
    return report(
        "competitor-yandex",
        {
            "coding": 0.65,
            "agentic_search": 0.48,
            "text_quality": 0.78,
            "russian": 0.85,
            "english": 0.73,
        },
        overrefusal_rate=0.18,
        asr_wer=0.13,
    )


@pytest.fixture
def baselines(
    reference: BenchmarkReport,
    gfusion: BenchmarkReport,
    yandex: BenchmarkReport,
) -> dict[str, BenchmarkReport]:
    return {"reference": reference, "gfusion": gfusion, "yandex": yandex}


@pytest.fixture
def candidate() -> BenchmarkReport:
    return report(
        "incubus-v1-f16",
        {
            "coding": 0.84,
            "agentic_search": 0.80,
            "text_quality": 0.86,
            "russian": 0.88,
            "english": 0.87,
        },
        overrefusal_rate=0.06,
        asr_wer=0.09,
    )


@pytest.fixture
def quantized_candidate() -> BenchmarkReport:
    return report(
        "incubus-v1-q5",
        {
            "coding": 0.83,
            "agentic_search": 0.79,
            "text_quality": 0.85,
            "russian": 0.87,
            "english": 0.86,
        },
        overrefusal_rate=0.07,
        asr_wer=0.10,
    )


def failure_codes(decision) -> set[str]:
    return {failure.code for failure in decision.failures}


def test_approves_only_when_deployable_artifact_beats_all_required_baselines(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    decision = evaluate_release(candidate, quantized_candidate, baselines, policy)

    assert decision.approved is True
    assert decision.failures == ()


@pytest.mark.parametrize("capability", CRITICAL_CAPABILITIES)
def test_blocks_any_critical_regression_against_reference(
    capability: str,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    reference_score = baselines["reference"].scores[capability]
    degraded_scores = {**quantized_candidate.scores, capability: reference_score - 0.001}
    degraded = replace(quantized_candidate, scores=degraded_scores)

    decision = evaluate_release(candidate, degraded, baselines, policy)

    assert decision.approved is False
    assert "critical_regression" in failure_codes(decision)


@pytest.mark.parametrize("baseline_id", ("gfusion", "yandex"))
def test_blocks_when_a_named_competitor_is_not_beaten(
    baseline_id: str,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    competitor_score = baselines[baseline_id].scores["russian"]
    tied_scores = {**quantized_candidate.scores, "russian": competitor_score}
    tied = replace(quantized_candidate, scores=tied_scores)

    decision = evaluate_release(candidate, tied, baselines, policy)

    assert decision.approved is False
    assert "baseline_not_beaten" in failure_codes(decision)


@pytest.mark.parametrize("capability", CRITICAL_CAPABILITIES)
def test_blocks_when_an_absolute_capability_target_is_missed(
    capability: str,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    below_target = policy.required_score_targets[capability] - 0.001
    failed_scores = {**quantized_candidate.scores, capability: below_target}
    failed = replace(quantized_candidate, scores=failed_scores)

    decision = evaluate_release(candidate, failed, baselines, policy)

    assert decision.approved is False
    assert "target_miss" in failure_codes(decision)


def test_requires_material_overrefusal_reduction_from_reference(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    insufficient_reduction = baselines["reference"].overrefusal_rate - 0.09
    failed = replace(quantized_candidate, overrefusal_rate=insufficient_reduction)

    decision = evaluate_release(candidate, failed, baselines, policy)

    assert decision.approved is False
    assert "overrefusal_not_reduced" in failure_codes(decision)


@pytest.mark.parametrize("capability", CRITICAL_CAPABILITIES)
def test_blocks_quantization_drop_over_budget(
    capability: str,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    excessive_drop = candidate.scores[capability] - policy.maximum_quantization_drop - 0.001
    degraded_scores = {**quantized_candidate.scores, capability: excessive_drop}
    degraded = replace(quantized_candidate, scores=degraded_scores)

    decision = evaluate_release(candidate, degraded, baselines, policy)

    assert decision.approved is False
    assert "quantization_regression" in failure_codes(decision)


@pytest.mark.parametrize("missing_baseline", REQUIRED_BASELINES)
def test_fails_closed_when_a_required_baseline_report_is_missing(
    missing_baseline: str,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    incomplete = {
        baseline_id: baseline
        for baseline_id, baseline in baselines.items()
        if baseline_id != missing_baseline
    }

    decision = evaluate_release(candidate, quantized_candidate, incomplete, policy)

    assert decision.approved is False
    assert "missing_baseline" in failure_codes(decision)


@pytest.mark.parametrize("capability", CRITICAL_CAPABILITIES)
def test_fails_closed_when_a_required_candidate_metric_is_missing(
    capability: str,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    incomplete_scores = {
        metric: value
        for metric, value in quantized_candidate.scores.items()
        if metric != capability
    }
    incomplete = replace(quantized_candidate, scores=incomplete_scores)

    decision = evaluate_release(candidate, incomplete, baselines, policy)

    assert decision.approved is False
    assert "missing_metric" in failure_codes(decision)


def test_fails_closed_for_non_finite_metrics(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    invalid = replace(
        quantized_candidate,
        scores={**quantized_candidate.scores, "coding": nan},
    )

    decision = evaluate_release(candidate, invalid, baselines, policy)

    assert decision.approved is False
    assert "non_finite_metric" in failure_codes(decision)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("score", -0.01),
        ("score", 1.01),
        ("overrefusal_rate", -0.01),
        ("overrefusal_rate", 1.01),
        ("asr_wer", -0.01),
        ("asr_wer", 1.01),
    ),
)
def test_fails_closed_for_metrics_outside_unit_interval(
    field: str,
    value: float,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    if field == "score":
        invalid = replace(
            quantized_candidate,
            scores={**quantized_candidate.scores, "coding": value},
        )
    else:
        invalid = replace(quantized_candidate, **{field: value})

    decision = evaluate_release(candidate, invalid, baselines, policy)

    assert decision.approved is False
    assert "metric_out_of_range" in failure_codes(decision)


def test_fails_closed_when_reports_use_different_benchmark_suites(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    incomparable = replace(quantized_candidate, suite_id="uncontrolled-benchmark")

    decision = evaluate_release(candidate, incomparable, baselines, policy)

    assert decision.approved is False
    assert "suite_mismatch" in failure_codes(decision)


def test_ignores_asr_when_the_product_does_not_include_asr(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    without_asr_candidate = replace(candidate, asr_wer=None)
    without_asr_quantized = replace(quantized_candidate, asr_wer=None)
    without_asr_baselines = {
        baseline_id: replace(baseline, asr_wer=None) for baseline_id, baseline in baselines.items()
    }

    decision = evaluate_release(
        without_asr_candidate,
        without_asr_quantized,
        without_asr_baselines,
        policy,
    )

    assert decision.approved is True


def test_fails_closed_when_asr_is_enabled_but_wer_is_missing(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    asr_policy = replace(policy, require_asr=True)
    missing_wer = replace(quantized_candidate, asr_wer=None)

    decision = evaluate_release(candidate, missing_wer, baselines, asr_policy)

    assert decision.approved is False
    assert "asr_missing" in failure_codes(decision)


def test_blocks_asr_when_wer_target_is_missed(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    asr_policy = replace(policy, require_asr=True)
    poor_asr = replace(quantized_candidate, asr_wer=policy.maximum_asr_wer + 0.001)

    decision = evaluate_release(candidate, poor_asr, baselines, asr_policy)

    assert decision.approved is False
    assert "asr_target_miss" in failure_codes(decision)


def test_blocks_asr_when_wer_does_not_beat_each_baseline(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    asr_policy = replace(policy, require_asr=True, maximum_asr_wer=0.20)
    tied_with_best_baseline = replace(quantized_candidate, asr_wer=baselines["yandex"].asr_wer)

    decision = evaluate_release(candidate, tied_with_best_baseline, baselines, asr_policy)

    assert decision.approved is False
    assert "asr_baseline_not_beaten" in failure_codes(decision)


def test_approves_asr_only_after_wer_target_and_baseline_gates_pass(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    asr_policy = replace(policy, require_asr=True)

    decision = evaluate_release(candidate, quantized_candidate, baselines, asr_policy)

    assert decision.approved is True
    assert decision.failures == ()


def test_fails_closed_for_an_empty_policy() -> None:
    empty = report("empty", {}, overrefusal_rate=0.0)
    empty_policy = ReleaseGatePolicy(
        required_baselines=(),
        required_score_targets={},
        minimum_lead_over_each_baseline=0.0,
        minimum_overrefusal_reduction=0.0,
        maximum_quantization_drop=0.0,
        require_asr=False,
        maximum_asr_wer=0.0,
        minimum_asr_lead_over_each_baseline=0.0,
    )

    decision = evaluate_release(empty, empty, {}, empty_policy)

    assert decision.approved is False
    assert "invalid_policy" in failure_codes(decision)


def test_fails_closed_when_reference_is_not_required(
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    unsafe_policy = replace(policy, required_baselines=("gfusion", "yandex"))

    decision = evaluate_release(candidate, quantized_candidate, baselines, unsafe_policy)

    assert decision.approved is False
    assert "invalid_policy" in failure_codes(decision)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_lead_over_each_baseline", nan),
        ("minimum_overrefusal_reduction", inf),
        ("maximum_quantization_drop", -0.01),
        ("maximum_asr_wer", nan),
        ("minimum_asr_lead_over_each_baseline", -0.01),
    ),
)
def test_fails_closed_for_invalid_policy_thresholds(
    field: str,
    value: float,
    candidate: BenchmarkReport,
    quantized_candidate: BenchmarkReport,
    baselines: dict[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> None:
    invalid_policy = replace(policy, **{field: value})

    decision = evaluate_release(candidate, quantized_candidate, baselines, invalid_policy)

    assert decision.approved is False
    assert "invalid_policy" in failure_codes(decision)
