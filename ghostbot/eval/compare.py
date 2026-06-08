"""Comparison helpers for GhostBot evaluation runs."""

from __future__ import annotations

from ghostbot.eval.schema import EvalRunSummary, EvalScenarioResult


def compare_runs(baseline: EvalRunSummary, candidate: EvalRunSummary) -> dict:
    baseline_by_id = {result.scenario_id: result for result in baseline.results}
    candidate_by_id = {result.scenario_id: result for result in candidate.results}
    scenario_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    scenario_deltas: list[dict[str, object]] = []

    for scenario_id in scenario_ids:
        base = baseline_by_id.get(scenario_id)
        cand = candidate_by_id.get(scenario_id)
        scenario_deltas.append({
            "scenario_id": scenario_id,
            "baseline_status": base.status if base else "missing",
            "candidate_status": cand.status if cand else "missing",
            "baseline_judge": _judge_verdict(base),
            "candidate_judge": _judge_verdict(cand),
            "baseline_score": _judge_score(base),
            "candidate_score": _judge_score(cand),
            "judge_score_delta": _judge_score(cand) - _judge_score(base),
            "latency_delta_ms": _latency(cand) - _latency(base),
            "token_delta": _tokens(cand) - _tokens(base),
            "redundant_tool_rate_delta": _redundancy(cand) - _redundancy(base),
            "self_correction_rate_delta": _self_correction(cand) - _self_correction(base),
        })

    return {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "scenario_count": len(scenario_ids),
        "pass_delta": candidate.summary.get("passed", 0) - baseline.summary.get("passed", 0),
        "fail_delta": candidate.summary.get("failed", 0) - baseline.summary.get("failed", 0),
        "latency_p50_delta_ms": candidate.summary.get("latency_p50_ms", 0) - baseline.summary.get("latency_p50_ms", 0),
        "token_p50_delta": candidate.summary.get("token_p50", 0) - baseline.summary.get("token_p50", 0),
        "redundant_tool_rate_p50_delta": candidate.summary.get("redundant_tool_rate_p50", 0.0) - baseline.summary.get("redundant_tool_rate_p50", 0.0),
        "self_correction_rate_p50_delta": candidate.summary.get("self_correction_rate_p50", 0.0) - baseline.summary.get("self_correction_rate_p50", 0.0),
        "judge_pass_delta": _judge_passes(candidate.results) - _judge_passes(baseline.results),
        "judge_coverage_delta": _judge_coverage(candidate.results) - _judge_coverage(baseline.results),
        "judge_score_p50_delta": _median_judge_score(candidate.results) - _median_judge_score(baseline.results),
        "scenarios": scenario_deltas,
    }


def render_compare_markdown(comparison: dict) -> str:
    lines = [
        "# Eval Compare",
        "",
        f"- Baseline: `{comparison['baseline_run_id']}`",
        f"- Candidate: `{comparison['candidate_run_id']}`",
        f"- Scenario count: {comparison['scenario_count']}",
        f"- Pass delta: {comparison['pass_delta']:+d}",
        f"- Fail delta: {comparison['fail_delta']:+d}",
        f"- Latency p50 delta: {comparison['latency_p50_delta_ms']:+d} ms",
        f"- Token p50 delta: {comparison['token_p50_delta']:+d}",
        f"- Redundant tool rate p50 delta: {comparison['redundant_tool_rate_p50_delta']:+.0%}",
        f"- Self-correction rate p50 delta: {comparison['self_correction_rate_p50_delta']:+.0%}",
        f"- Judge pass delta: {comparison['judge_pass_delta']:+d}",
        f"- Judge coverage delta: {comparison['judge_coverage_delta']:+.0%}",
        f"- Judge score p50 delta: {comparison['judge_score_p50_delta']:+.2f}",
        "",
        "| Scenario | Baseline | Candidate | Judge baseline | Judge candidate | Judge score Δ | Latency Δ | Tokens Δ | Redundant Δ | Self-correction Δ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparison["scenarios"]:
        lines.append(
            f"| {item['scenario_id']} | {item['baseline_status']} | {item['candidate_status']} | {item['baseline_judge']} | {item['candidate_judge']} | {item['judge_score_delta']:+.2f} | {item['latency_delta_ms']:+d} ms | {item['token_delta']:+d} | {item['redundant_tool_rate_delta']:+.0%} | {item['self_correction_rate_delta']:+.0%} |"
        )
    return "\n".join(lines) + "\n"


def _latency(result: EvalScenarioResult | None) -> int:
    return result.timing.latency_ms if result else 0


def _tokens(result: EvalScenarioResult | None) -> int:
    return sum(result.usage.values()) if result else 0


def _redundancy(result: EvalScenarioResult | None) -> float:
    if not result or result.tools.tool_call_count <= 0:
        return 0.0
    return result.tools.redundant_tool_call_count / result.tools.tool_call_count


def _self_correction(result: EvalScenarioResult | None) -> float:
    return result.metrics.self_correction_rate if result else 0.0


def _judge_verdict(result: EvalScenarioResult | None) -> str:
    return result.judge.verdict if result and result.judge.verdict else "-"


def _judge_score(result: EvalScenarioResult | None) -> float:
    return result.judge.score if result and result.judge.score is not None else 0.0


def _judge_passes(results: list[EvalScenarioResult]) -> int:
    return sum(1 for result in results if result.judge.passed is True)


def _judge_coverage(results: list[EvalScenarioResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.judge.verdict) / len(results)


def _median_judge_score(results: list[EvalScenarioResult]) -> float:
    scores = sorted(result.judge.score for result in results if result.judge.score is not None)
    if not scores:
        return 0.0
    mid = len(scores) // 2
    if len(scores) % 2:
        return scores[mid]
    return (scores[mid - 1] + scores[mid]) / 2
