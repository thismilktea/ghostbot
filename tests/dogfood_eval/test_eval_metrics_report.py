from __future__ import annotations

from datetime import UTC, datetime

from ghostbot.eval.compare import compare_runs, render_compare_markdown
from ghostbot.eval.metrics import summarize_results
from ghostbot.eval.report import render_summary_markdown
from ghostbot.eval.schema import (
    DeterministicCheckResult,
    EvalAgentSummary,
    EvalCommandResult,
    EvalFileSummary,
    EvalRunSummary,
    EvalScenarioResult,
    EvalTiming,
    EvalToolSummary,
)


def _result(
    *,
    run_id: str,
    scenario_id: str,
    status: str,
    latency_ms: int,
    tokens: int,
    tool_calls: int,
    redundant_calls: int,
    check_passed: bool,
    requirement_hit_rate: float = 1.0,
    self_correction_rate: float = 0.0,
    judge_verdict: str | None = None,
    judge_passed: bool | None = None,
    judge_score: float | None = None,
) -> EvalScenarioResult:
    result = EvalScenarioResult(
        run_id=run_id,
        scenario_id=scenario_id,
        category="smoke",
        fixture="empty",
        model="test-model",
        status=status,
        timing=EvalTiming(
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            latency_ms=latency_ms,
        ),
        agent=EvalAgentSummary(stop_reason="completed", iterations=1, final_content="ok"),
        usage={"total_tokens": tokens},
        tools=EvalToolSummary(
            tool_call_count=tool_calls,
            redundant_tool_call_count=redundant_calls,
        ),
        files=EvalFileSummary(changed_files=[]),
        checks=[
            EvalCommandResult(
                name="pytest",
                command="pytest",
                cwd=".",
                exit_code=0 if check_passed else 1,
                passed=check_passed,
                stdout="",
                stderr="",
            )
        ],
        deterministic=DeterministicCheckResult(passed=status == "pass", failures=[]),
    )
    result.metrics.requirement_hit_rate = requirement_hit_rate
    result.metrics.self_correction_rate = self_correction_rate
    result.metrics.check_pass_rate = 1.0 if check_passed else 0.0
    result.judge.verdict = judge_verdict
    result.judge.passed = judge_passed
    result.judge.score = judge_score
    return result


def test_summarize_results_aggregates_upgraded_metrics():
    results = [
        _result(
            run_id="baseline",
            scenario_id="a",
            status="pass",
            latency_ms=100,
            tokens=10,
            tool_calls=2,
            redundant_calls=0,
            check_passed=True,
            requirement_hit_rate=1.0,
            self_correction_rate=0.0,
        ),
        _result(
            run_id="baseline",
            scenario_id="b",
            status="fail",
            latency_ms=300,
            tokens=30,
            tool_calls=4,
            redundant_calls=2,
            check_passed=False,
            requirement_hit_rate=0.5,
            self_correction_rate=1.0,
        ),
    ]

    summary = summarize_results(results)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["latency_p50_ms"] == 200
    assert summary["latency_p90_ms"] == 300
    assert summary["token_p50"] == 20
    assert summary["token_p90"] == 30
    assert summary["tool_call_p50"] == 3
    assert summary["redundant_tool_rate_p50"] == 0.25
    assert summary["requirement_hit_rate_p50"] == 0.75
    assert summary["self_correction_rate_p50"] == 0.5
    assert summary["check_pass_rate"] == 0.5


def test_render_summary_markdown_contains_upgraded_fields():
    results = [
        _result(
            run_id="run-1",
            scenario_id="a",
            status="pass",
            latency_ms=100,
            tokens=10,
            tool_calls=2,
            redundant_calls=1,
            check_passed=True,
            requirement_hit_rate=1.0,
            self_correction_rate=1.0,
            judge_verdict="pass",
            judge_passed=True,
            judge_score=0.9,
        ),
    ]
    run = EvalRunSummary(
        run_id="run-1",
        suite_name="smoke",
        generated_at=datetime.now(UTC),
        results=results,
        summary=summarize_results(results),
    )

    markdown = render_summary_markdown(run)

    assert "Requirement hit rate" in markdown
    assert "Self-correction rate" in markdown
    assert "Recovery cost p50" in markdown
    assert "Judged scenarios: 1" in markdown
    assert "Judge pass rate: 100%" in markdown
    assert "| Scenario | Status | Judge | Score | Stop reason | Tool calls | Redundant | Requirement hit | Self-correction | Checks passed | Changed files |" in markdown
    assert "| a | pass | pass | 0.90 | completed | 2 | 1 | 100% | 100% | 1/1 | 0 |" in markdown


def test_compare_runs_summarizes_deltas():
    baseline_results = [
        _result(
            run_id="baseline",
            scenario_id="a",
            status="fail",
            latency_ms=200,
            tokens=40,
            tool_calls=4,
            redundant_calls=2,
            check_passed=False,
            requirement_hit_rate=0.5,
            self_correction_rate=0.0,
            judge_verdict="fail",
            judge_passed=False,
            judge_score=0.2,
        ),
    ]
    candidate_results = [
        _result(
            run_id="candidate",
            scenario_id="a",
            status="pass",
            latency_ms=120,
            tokens=20,
            tool_calls=2,
            redundant_calls=0,
            check_passed=True,
            requirement_hit_rate=1.0,
            self_correction_rate=1.0,
            judge_verdict="pass",
            judge_passed=True,
            judge_score=0.9,
        ),
    ]
    baseline = EvalRunSummary(
        run_id="baseline",
        suite_name="smoke",
        generated_at=datetime.now(UTC),
        results=baseline_results,
        summary=summarize_results(baseline_results),
    )
    candidate = EvalRunSummary(
        run_id="candidate",
        suite_name="smoke",
        generated_at=datetime.now(UTC),
        results=candidate_results,
        summary=summarize_results(candidate_results),
    )

    comparison = compare_runs(baseline, candidate)
    markdown = render_compare_markdown(comparison)

    assert comparison["pass_delta"] == 1
    assert comparison["fail_delta"] == -1
    assert comparison["latency_p50_delta_ms"] == -80
    assert comparison["token_p50_delta"] == -20
    assert comparison["redundant_tool_rate_p50_delta"] == -0.5
    assert comparison["self_correction_rate_p50_delta"] == 1.0
    assert comparison["judge_pass_delta"] == 1
    assert comparison["judge_coverage_delta"] == 0.0
    assert comparison["judge_score_p50_delta"] == 0.7
    assert "# Eval Compare" in markdown
    assert "Judge pass delta: +1" in markdown
    assert "Judge score p50 delta: +0.70" in markdown
    assert "| a | fail | pass | fail | pass | +0.70 | -80 ms | -20 | -50% | +100% |" in markdown
