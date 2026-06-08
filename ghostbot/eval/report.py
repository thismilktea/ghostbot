"""Report rendering for GhostBot evaluation runs."""

from __future__ import annotations

from ghostbot.eval.schema import EvalRunSummary


def render_summary_markdown(run: EvalRunSummary) -> str:
    summary = run.summary
    judged = [result for result in run.results if result.judge.verdict]
    judged_passes = sum(1 for result in judged if result.judge.passed is True)
    lines = [
        f"# Eval Report: {run.suite_name}",
        "",
        f"Run ID: `{run.run_id}`",
        "",
        f"- Total: {summary.get('total', 0)}",
        f"- Passed: {summary.get('passed', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Errors: {summary.get('errors', 0)}",
        f"- Pass rate: {summary.get('pass_rate', 0.0):.0%}",
        f"- Median latency: {summary.get('latency_p50_ms', 0)} ms",
        f"- P90 latency: {summary.get('latency_p90_ms', 0)} ms",
        f"- Median tokens: {summary.get('token_p50', 0)}",
        f"- P90 tokens: {summary.get('token_p90', 0)}",
        f"- Median tool calls: {summary.get('tool_call_p50', 0)}",
        f"- Median redundant tool rate: {summary.get('redundant_tool_rate_p50', 0.0):.0%}",
        f"- Requirement hit rate: {summary.get('requirement_hit_rate_p50', 0.0):.0%}",
        f"- Self-correction rate: {summary.get('self_correction_rate_p50', 0.0):.0%}",
        f"- Recovery cost p50: {summary.get('recovery_cost_p50', 0)}",
        f"- Check pass rate: {summary.get('check_pass_rate', 0.0):.0%}",
        f"- Judged scenarios: {len(judged)}",
        f"- Judge pass rate: {(judged_passes / len(judged)):.0%}" if judged else "- Judge pass rate: n/a",
        "",
        "| Scenario | Status | Judge | Score | Stop reason | Tool calls | Redundant | Requirement hit | Self-correction | Checks passed | Changed files |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in run.results:
        passed_checks = sum(1 for check in result.checks if check.passed)
        judge_verdict = result.judge.verdict or "-"
        judge_score = f"{result.judge.score:.2f}" if result.judge.score is not None else "-"
        lines.append(
            f"| {result.scenario_id} | {result.status} | {judge_verdict} | {judge_score} | {result.agent.stop_reason} | {result.tools.tool_call_count} | {result.tools.redundant_tool_call_count} | {result.metrics.requirement_hit_rate:.0%} | {result.metrics.self_correction_rate:.0%} | {passed_checks}/{len(result.checks)} | {len(result.files.changed_files)} |"
        )
    return "\n".join(lines) + "\n"
