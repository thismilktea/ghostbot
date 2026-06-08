"""LLM-as-judge helpers for GhostBot evaluation runs."""

from __future__ import annotations

from ghostbot.eval.schema import EvalJudgeResult, EvalScenario, EvalScenarioResult
from ghostbot.utils.prompt_templates import render_template

_EVALUATE_RUN_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_run",
            "description": "Score whether the GhostBot eval run satisfied the scenario requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "passed": {"type": "boolean"},
                    "verdict": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["score", "passed", "verdict", "reasoning"],
            },
        },
    }
]


def default_judge_result() -> EvalJudgeResult:
    return EvalJudgeResult(
        score=None,
        passed=None,
        verdict="not_run",
        reasoning="LLM judging is planned for a later phase.",
    )


async def judge_scenario_result(
    provider,
    model: str,
    scenario: EvalScenario,
    result: EvalScenarioResult,
) -> EvalJudgeResult:
    try:
        response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": render_template("agent/eval_judge.md", part="system")},
                {
                    "role": "user",
                    "content": render_template(
                        "agent/eval_judge.md",
                        part="user",
                        prompt=scenario.prompt or (scenario.turns[0].input if scenario.turns else ""),
                        acceptance_criteria=_format_acceptance_criteria(scenario),
                        changed_files=_format_changed_files(result),
                        checks=_format_checks(result),
                        deterministic_failures="; ".join(result.deterministic.failures) or "none",
                        final_response=result.agent.final_content or "",
                    ),
                },
            ],
            tools=_EVALUATE_RUN_TOOL,
            model=model,
            max_tokens=512,
            temperature=0.0,
        )
        if not response.has_tool_calls:
            return EvalJudgeResult(
                score=0.0,
                passed=False,
                verdict="judge_error",
                reasoning="Judge returned no tool call.",
            )
        args = response.tool_calls[0].arguments
        judge = EvalJudgeResult(
            score=float(args.get("score", 0.0)),
            passed=bool(args.get("passed", False)),
            verdict=str(args.get("verdict", "judge_error")),
            reasoning=str(args.get("reasoning", "")),
        )
        if result.deterministic.failures and judge.passed:
            judge.passed = False
            judge.verdict = "fail"
            judge.reasoning = f"Deterministic failures override judge pass. {judge.reasoning}".strip()
        return judge
    except Exception as exc:
        return EvalJudgeResult(
            score=0.0,
            passed=False,
            verdict="judge_error",
            reasoning=f"Judge failed: {exc}",
        )


def _format_acceptance_criteria(scenario: EvalScenario) -> str:
    criteria = list(scenario.expected.acceptance_criteria)
    if scenario.expected.required_phrases:
        criteria.extend(f"include phrase: {phrase}" for phrase in scenario.expected.required_phrases)
    if scenario.expected.required_changed_files:
        criteria.extend(f"modify file: {path}" for path in scenario.expected.required_changed_files)
    return "\n".join(criteria) if criteria else "none"


def _format_changed_files(result: EvalScenarioResult) -> str:
    return "\n".join(result.files.changed_files) if result.files.changed_files else "none"


def _format_checks(result: EvalScenarioResult) -> str:
    if not result.checks:
        return "none"
    return "\n".join(
        f"{check.name}: {'pass' if check.passed else 'fail'} (exit={check.exit_code})"
        for check in result.checks
    )
