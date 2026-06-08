from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from ghostbot.cli.commands import app
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

runner = CliRunner()


def _result(*, run_id: str, scenario_id: str, status: str, latency_ms: int, tokens: int) -> EvalScenarioResult:
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
        tools=EvalToolSummary(tool_call_count=2, redundant_tool_call_count=0),
        files=EvalFileSummary(changed_files=[]),
        checks=[
            EvalCommandResult(
                name="pytest",
                command="pytest",
                cwd=".",
                exit_code=0,
                passed=True,
                stdout="",
                stderr="",
            )
        ],
        deterministic=DeterministicCheckResult(passed=status == "pass", failures=[]),
    )
    result.metrics.requirement_hit_rate = 1.0 if status == "pass" else 0.5
    result.metrics.check_pass_rate = 1.0
    result.judge.verdict = "pass" if status == "pass" else "fail"
    result.judge.passed = status == "pass"
    result.judge.score = 0.9 if status == "pass" else 0.3
    return result


def _write_run(run_dir: Path, run_id: str, status: str, latency_ms: int, tokens: int) -> None:
    results = [_result(run_id=run_id, scenario_id="scenario.a", status=status, latency_ms=latency_ms, tokens=tokens)]
    summary = EvalRunSummary(
        run_id=run_id,
        suite_name="smoke",
        generated_at=datetime.now(UTC),
        results=results,
        summary=summarize_results(results),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def test_eval_cli_end_to_end_smoke(monkeypatch, tmp_path):
    suite_path = Path("tests/dogfood_eval/suites/smoke.json")
    baseline_out = tmp_path / "baseline-runs"
    candidate_out = tmp_path / "candidate-runs"

    from ghostbot.eval.harness import EvalHarness

    async def _fake_run(self, scenario_ids=None):
        status = "fail" if "baseline" in str(self.out_dir) else "pass"
        latency_ms = 200 if status == "fail" else 120
        tokens = 40 if status == "fail" else 20
        run_id = f"run-{status}"
        run_root = self.out_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        results = [
            _result(
                run_id=run_id,
                scenario_id="plan.empty.001",
                status=status,
                latency_ms=latency_ms,
                tokens=tokens,
            )
        ]
        summary = EvalRunSummary(
            run_id=run_id,
            suite_name=self.suite.name,
            generated_at=datetime.now(UTC),
            results=results,
            summary=summarize_results(results),
        )
        (run_root / "suite.json").write_text(self.suite.model_dump_json(indent=2), encoding="utf-8")
        (run_root / "results.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        (run_root / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(EvalHarness, "run", _fake_run)

    list_result = runner.invoke(app, ["eval", "list", "--suite", str(suite_path)])
    assert list_result.exit_code == 0
    assert "plan.empty.001" in list_result.stdout

    baseline_run = runner.invoke(
        app,
        ["eval", "run", "--suite", str(suite_path), "--out", str(baseline_out)],
    )
    assert baseline_run.exit_code == 0
    assert "Run ID:" in baseline_run.stdout

    candidate_run = runner.invoke(
        app,
        ["eval", "run", "--suite", str(suite_path), "--out", str(candidate_out)],
    )
    assert candidate_run.exit_code == 0
    assert "Results:" in candidate_run.stdout

    baseline_run_dir = baseline_out / "run-fail"
    candidate_run_dir = candidate_out / "run-pass"

    report_result = runner.invoke(app, ["eval", "report", "--run", str(candidate_run_dir)])
    assert report_result.exit_code == 0
    assert "Eval Report" in report_result.stdout
    assert "Judge pass rate" in report_result.stdout

    compare_result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(baseline_run_dir),
            "--candidate",
            str(candidate_run_dir),
        ],
    )
    assert compare_result.exit_code == 0
    assert "Eval Compare" in compare_result.stdout
    assert "Judge pass delta: +1" in compare_result.stdout


def test_eval_compare_cli_outputs_summary(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_run(baseline_dir, "baseline", "fail", 200, 40)
    _write_run(candidate_dir, "candidate", "pass", 120, 20)

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(baseline_dir),
            "--candidate",
            str(candidate_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Eval Compare" in result.stdout
    assert "Pass delta: +1" in result.stdout
    assert "Judge pass delta: +1" in result.stdout
    assert "fail" in result.stdout
    assert "pass" in result.stdout
