"""Metrics helpers for GhostBot evaluation runs."""

from __future__ import annotations

from statistics import median

from ghostbot.eval.schema import EvalMetricSummary, EvalRunSummary, EvalScenarioResult


def redundant_tool_rate(result: EvalScenarioResult) -> float:
    total = result.tools.tool_call_count
    if total <= 0:
        return 0.0
    return result.tools.redundant_tool_call_count / total


def requirement_hit_rate(result: EvalScenarioResult) -> float:
    expected = result.metrics.requirement_hit_rate
    if expected:
        return expected
    checks_total = len(result.checks)
    checks_passed = sum(1 for check in result.checks if check.passed)
    required_files_total = len(result.files.changed_files)
    file_hits = max(0, required_files_total - len(result.files.scope_violations))
    total = checks_total + required_files_total
    if total <= 0:
        return 1.0 if result.deterministic.passed else 0.0
    return (checks_passed + file_hits) / total


def self_correction_rate(result: EvalScenarioResult) -> float:
    return result.metrics.self_correction_rate


def recovery_cost(result: EvalScenarioResult) -> int:
    return result.metrics.recovery_cost


def _check_pass_rate(result: EvalScenarioResult) -> float:
    if not result.checks:
        return 0.0
    return sum(1 for check in result.checks if check.passed) / len(result.checks)


def summarize_results(results: list[EvalScenarioResult]) -> dict[str, float | int]:
    total = len(results)
    passed = sum(1 for result in results if result.status == "pass")
    failed = sum(1 for result in results if result.status == "fail")
    errors = sum(1 for result in results if result.status == "error")
    latencies = [result.timing.latency_ms for result in results]
    token_totals = [sum(result.usage.values()) for result in results if result.usage]
    redundancy = [redundant_tool_rate(result) for result in results]
    tool_calls = [result.tools.tool_call_count for result in results]
    requirement_rates = [requirement_hit_rate(result) for result in results]
    self_correction_rates = [self_correction_rate(result) for result in results]
    recovery_costs = [recovery_cost(result) for result in results]
    check_rates = [_check_pass_rate(result) for result in results if result.checks]
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": (passed / total) if total else 0.0,
        "latency_p50_ms": int(median(latencies)) if latencies else 0,
        "latency_p90_ms": max(latencies) if latencies else 0,
        "token_p50": int(median(token_totals)) if token_totals else 0,
        "token_p90": max(token_totals) if token_totals else 0,
        "redundant_tool_rate_p50": float(median(redundancy)) if redundancy else 0.0,
        "tool_call_p50": int(median(tool_calls)) if tool_calls else 0,
        "requirement_hit_rate_p50": float(median(requirement_rates)) if requirement_rates else 0.0,
        "self_correction_rate_p50": float(median(self_correction_rates)) if self_correction_rates else 0.0,
        "recovery_cost_p50": int(median(recovery_costs)) if recovery_costs else 0,
        "check_pass_rate": float(median(check_rates)) if check_rates else 0.0,
    }


def attach_summary(run: EvalRunSummary) -> EvalRunSummary:
    run.summary = summarize_results(run.results)
    return run


def attach_result_metrics(
    result: EvalScenarioResult,
    *,
    requirement_hit_rate_value: float,
    self_correction_rate_value: float,
    recovery_cost_value: int,
) -> EvalScenarioResult:
    result.metrics = EvalMetricSummary(
        requirement_hit_rate=requirement_hit_rate_value,
        self_correction_rate=self_correction_rate_value,
        recovery_cost=recovery_cost_value,
        check_pass_rate=_check_pass_rate(result),
    )
    return result
