"""Dogfooding evaluation helpers for GhostBot."""

from ghostbot.eval.compare import compare_runs, render_compare_markdown
from ghostbot.eval.harness import EvalHarness, load_suite
from ghostbot.eval.metrics import summarize_results
from ghostbot.eval.report import render_summary_markdown
from ghostbot.eval.schema import EvalRunSummary, EvalScenarioResult, EvalSuite

__all__ = [
    "EvalHarness",
    "EvalRunSummary",
    "EvalScenarioResult",
    "EvalSuite",
    "compare_runs",
    "load_suite",
    "render_compare_markdown",
    "render_summary_markdown",
    "summarize_results",
]
