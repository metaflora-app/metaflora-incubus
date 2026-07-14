"""Fail-closed promotion gates for Incubus release candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True)
class BenchmarkReport:
    artifact_id: str
    suite_id: str
    scores: Mapping[str, float]
    overrefusal_rate: float
    asr_wer: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(frozen=True)
class ReleaseGatePolicy:
    required_baselines: tuple[str, ...]
    required_score_targets: Mapping[str, float]
    minimum_lead_over_each_baseline: float
    minimum_overrefusal_reduction: float
    maximum_quantization_drop: float
    require_asr: bool
    maximum_asr_wer: float
    minimum_asr_lead_over_each_baseline: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_baselines", tuple(self.required_baselines))
        object.__setattr__(
            self,
            "required_score_targets",
            MappingProxyType(dict(self.required_score_targets)),
        )


@dataclass(frozen=True)
class GateFailure:
    code: str
    detail: str


@dataclass(frozen=True)
class ReleaseDecision:
    approved: bool
    failures: tuple[GateFailure, ...]


def evaluate_release(
    candidate: BenchmarkReport,
    deployable_candidate: BenchmarkReport,
    baselines: Mapping[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
) -> ReleaseDecision:
    """Evaluate every promotion rule and approve only a complete passing report."""
    failures: list[GateFailure] = []
    _check_policy(policy, failures)
    required_metrics = tuple(policy.required_score_targets)
    required_reports = _collect_required_reports(baselines, policy, failures)

    _check_suite_ids(candidate, deployable_candidate, required_reports, failures)
    _check_report_values(candidate, required_metrics, failures)
    _check_report_values(deployable_candidate, required_metrics, failures)
    for report in required_reports.values():
        _check_report_values(report, required_metrics, failures)

    reference = required_reports.get("reference")
    if reference is not None:
        _check_capability_gates(
            candidate,
            deployable_candidate,
            required_reports,
            reference,
            policy,
            failures,
        )
        _check_overrefusal(deployable_candidate, reference, policy, failures)

    if policy.require_asr:
        _check_asr(deployable_candidate, required_reports, policy, failures)

    result = tuple(failures)
    return ReleaseDecision(approved=not result, failures=result)


def _check_policy(policy: ReleaseGatePolicy, failures: list[GateFailure]) -> None:
    baselines = policy.required_baselines
    targets = policy.required_score_targets
    if not baselines or "reference" not in baselines or len(set(baselines)) != len(baselines):
        failures.append(GateFailure("invalid_policy", "required_baselines"))
    if not targets:
        failures.append(GateFailure("invalid_policy", "required_score_targets"))
    for metric, target in targets.items():
        if not isinstance(metric, str) or not metric.strip() or not _is_unit_interval(target):
            failures.append(GateFailure("invalid_policy", f"target:{metric}"))
    thresholds = {
        "minimum_lead_over_each_baseline": policy.minimum_lead_over_each_baseline,
        "minimum_overrefusal_reduction": policy.minimum_overrefusal_reduction,
        "maximum_quantization_drop": policy.maximum_quantization_drop,
        "maximum_asr_wer": policy.maximum_asr_wer,
        "minimum_asr_lead_over_each_baseline": policy.minimum_asr_lead_over_each_baseline,
    }
    for name, value in thresholds.items():
        if not _is_unit_interval(value):
            failures.append(GateFailure("invalid_policy", name))


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _is_unit_interval(value: object) -> bool:
    return _is_finite_number(value) and 0 <= float(value) <= 1


def _collect_required_reports(
    baselines: Mapping[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
    failures: list[GateFailure],
) -> dict[str, BenchmarkReport]:
    reports: dict[str, BenchmarkReport] = {}
    for baseline_id in policy.required_baselines:
        report = baselines.get(baseline_id)
        if report is None:
            failures.append(GateFailure("missing_baseline", baseline_id))
        else:
            reports[baseline_id] = report
    return reports


def _check_suite_ids(
    candidate: BenchmarkReport,
    deployable_candidate: BenchmarkReport,
    baselines: Mapping[str, BenchmarkReport],
    failures: list[GateFailure],
) -> None:
    reports = (deployable_candidate, *baselines.values())
    for report in reports:
        if report.suite_id != candidate.suite_id:
            failures.append(GateFailure("suite_mismatch", report.artifact_id))


def _check_report_values(
    report: BenchmarkReport,
    required_metrics: tuple[str, ...],
    failures: list[GateFailure],
) -> None:
    for metric in required_metrics:
        if metric not in report.scores:
            failures.append(GateFailure("missing_metric", f"{report.artifact_id}:{metric}"))
            continue
        if not isfinite(report.scores[metric]):
            failures.append(GateFailure("non_finite_metric", f"{report.artifact_id}:{metric}"))
        elif not 0 <= report.scores[metric] <= 1:
            failures.append(GateFailure("metric_out_of_range", f"{report.artifact_id}:{metric}"))
    if not isfinite(report.overrefusal_rate):
        failures.append(GateFailure("non_finite_metric", f"{report.artifact_id}:overrefusal"))
    elif not 0 <= report.overrefusal_rate <= 1:
        failures.append(GateFailure("metric_out_of_range", f"{report.artifact_id}:overrefusal"))
    if report.asr_wer is not None:
        if not isfinite(report.asr_wer):
            failures.append(GateFailure("non_finite_metric", f"{report.artifact_id}:asr_wer"))
        elif not 0 <= report.asr_wer <= 1:
            failures.append(GateFailure("metric_out_of_range", f"{report.artifact_id}:asr_wer"))


def _check_capability_gates(
    candidate: BenchmarkReport,
    deployable: BenchmarkReport,
    baselines: Mapping[str, BenchmarkReport],
    reference: BenchmarkReport,
    policy: ReleaseGatePolicy,
    failures: list[GateFailure],
) -> None:
    for metric, target in policy.required_score_targets.items():
        deployed_score = deployable.scores.get(metric)
        candidate_score = candidate.scores.get(metric)
        reference_score = reference.scores.get(metric)
        if deployed_score is None or not isfinite(deployed_score):
            continue
        if deployed_score < target:
            failures.append(GateFailure("target_miss", metric))
        if (
            reference_score is not None
            and isfinite(reference_score)
            and deployed_score < reference_score
        ):
            failures.append(GateFailure("critical_regression", metric))
        if candidate_score is not None and isfinite(candidate_score):
            if candidate_score - deployed_score > policy.maximum_quantization_drop:
                failures.append(GateFailure("quantization_regression", metric))
        for baseline_id, baseline in baselines.items():
            baseline_score = baseline.scores.get(metric)
            if baseline_score is None or not isfinite(baseline_score):
                continue
            if deployed_score - baseline_score < policy.minimum_lead_over_each_baseline:
                failures.append(GateFailure("baseline_not_beaten", f"{baseline_id}:{metric}"))


def _check_overrefusal(
    deployable: BenchmarkReport,
    reference: BenchmarkReport,
    policy: ReleaseGatePolicy,
    failures: list[GateFailure],
) -> None:
    values = (deployable.overrefusal_rate, reference.overrefusal_rate)
    if not all(isfinite(value) for value in values):
        return
    reduction = reference.overrefusal_rate - deployable.overrefusal_rate
    if reduction < policy.minimum_overrefusal_reduction:
        failures.append(GateFailure("overrefusal_not_reduced", f"reduction={reduction:.6f}"))


def _check_asr(
    deployable: BenchmarkReport,
    baselines: Mapping[str, BenchmarkReport],
    policy: ReleaseGatePolicy,
    failures: list[GateFailure],
) -> None:
    if deployable.asr_wer is None or not isfinite(deployable.asr_wer):
        failures.append(GateFailure("asr_missing", deployable.artifact_id))
        return
    if deployable.asr_wer > policy.maximum_asr_wer:
        failures.append(GateFailure("asr_target_miss", f"wer={deployable.asr_wer:.6f}"))
    for baseline_id, baseline in baselines.items():
        if baseline.asr_wer is None or not isfinite(baseline.asr_wer):
            failures.append(GateFailure("asr_missing", baseline_id))
            continue
        lead = baseline.asr_wer - deployable.asr_wer
        if lead < policy.minimum_asr_lead_over_each_baseline:
            failures.append(GateFailure("asr_baseline_not_beaten", baseline_id))
