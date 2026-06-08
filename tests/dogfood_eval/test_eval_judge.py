from __future__ import annotations

import pytest

from ghostbot.eval.judge import default_judge_result, judge_scenario_result
from ghostbot.eval.schema import (
    DeterministicCheckResult,
    EvalAgentSummary,
    EvalCheck,
    EvalCommandResult,
    EvalExpected,
    EvalFileSummary,
    EvalScenario,
    EvalScenarioResult,
    EvalTiming,
    EvalToolSummary,
)
from ghostbot.providers.base import LLMResponse, ToolCallRequest
from datetime import UTC, datetime


class _FakeJudgeProvider:
    def __init__(self, response: LLMResponse | Exception):
        self.response = response

    async def chat_with_retry(self, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _scenario() -> EvalScenario:
    return EvalScenario(
        id="bugfix.cart_negative_quantity.001",
        category="bugfix",
        fixture="shop_project",
        prompt="修复负数数量问题",
        expected=EvalExpected(
            required_phrases=["负数数量"],
            acceptance_criteria=["pytest tests/test_cart.py"],
            checks=[EvalCheck(name="pytest-cart", command="python -m pytest tests/test_cart.py")],
        ),
    )


def _result(*, deterministic_failures: list[str]) -> EvalScenarioResult:
    return EvalScenarioResult(
        run_id="run-1",
        scenario_id="bugfix.cart_negative_quantity.001",
        category="bugfix",
        fixture="shop_project",
        model="test-model",
        status="pass" if not deterministic_failures else "fail",
        timing=EvalTiming(
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            latency_ms=100,
        ),
        agent=EvalAgentSummary(stop_reason="completed", iterations=1, final_content="已修复负数数量问题并运行 pytest tests/test_cart.py"),
        usage={"total_tokens": 20},
        tools=EvalToolSummary(tool_call_count=2),
        files=EvalFileSummary(changed_files=["shop/cart.py"]),
        checks=[
            EvalCommandResult(
                name="pytest-cart",
                command="python -m pytest tests/test_cart.py",
                cwd=".",
                exit_code=0,
                passed=True,
                stdout="",
                stderr="",
            )
        ],
        deterministic=DeterministicCheckResult(passed=not deterministic_failures, failures=deterministic_failures),
    )


def test_default_judge_result_is_stub():
    result = default_judge_result()

    assert result.verdict == "not_run"
    assert result.passed is None


@pytest.mark.asyncio
async def test_judge_scenario_result_parses_tool_call():
    provider = _FakeJudgeProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="1",
                    name="evaluate_run",
                    arguments={
                        "score": 0.9,
                        "passed": True,
                        "verdict": "pass",
                        "reasoning": "Requirements satisfied.",
                    },
                )
            ],
            usage={},
        )
    )

    result = await judge_scenario_result(provider, "test-model", _scenario(), _result(deterministic_failures=[]))

    assert result.score == 0.9
    assert result.passed is True
    assert result.verdict == "pass"


@pytest.mark.asyncio
async def test_judge_scenario_result_deterministic_failure_overrides_pass():
    provider = _FakeJudgeProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="1",
                    name="evaluate_run",
                    arguments={
                        "score": 0.9,
                        "passed": True,
                        "verdict": "pass",
                        "reasoning": "Looks good.",
                    },
                )
            ],
            usage={},
        )
    )

    result = await judge_scenario_result(provider, "test-model", _scenario(), _result(deterministic_failures=["Check failed: pytest-cart"]))

    assert result.passed is False
    assert result.verdict == "fail"
    assert "Deterministic failures override judge pass" in result.reasoning


@pytest.mark.asyncio
async def test_judge_scenario_result_falls_back_on_exception():
    provider = _FakeJudgeProvider(RuntimeError("boom"))

    result = await judge_scenario_result(provider, "test-model", _scenario(), _result(deterministic_failures=[]))

    assert result.passed is False
    assert result.verdict == "judge_error"
    assert "boom" in result.reasoning
