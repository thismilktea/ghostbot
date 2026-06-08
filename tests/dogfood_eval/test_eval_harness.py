from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from ghostbot.eval.harness import load_suite, EvalHarness
from ghostbot.eval.schema import EvalCheck
from ghostbot.providers.base import LLMResponse, ToolCallRequest


class _FakeProvider:
    def __init__(self, response: LLMResponse):
        self.response = response
        self.calls: list[dict] = []

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeLoopResponse:
    def __init__(self, content: str):
        self.content = content
        self.metadata = {"stop_reason": "completed"}


class _FakeLoop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._extra_hooks = []

    async def process_direct(self, prompt: str, session_key: str):
        workspace = Path(self.kwargs["workspace"])
        cart_file = workspace / "shop" / "cart.py"
        cart_file.write_text(
            "def add_to_cart(cart: dict[str, int], item: str, quantity: int) -> dict[str, int]:\n"
            "    if quantity < 0:\n"
            '        raise ValueError("quantity must be non-negative")\n'
            "    updated = dict(cart)\n"
            "    updated[item] = updated.get(item, 0) + quantity\n"
            "    return updated\n",
            encoding="utf-8",
        )
        for hook in self._extra_hooks:
            hook.iterations.append(
                {
                    "iteration": 1,
                    "usage": {"total_tokens": 12},
                    "tool_calls": [{"id": "1", "name": "Read", "arguments": {"file": "shop/cart.py"}}],
                    "tool_events": [],
                    "final_content": "已修复负数数量问题并运行 pytest tests/test_cart.py",
                    "stop_reason": "completed",
                    "error": None,
                }
            )
        return _FakeLoopResponse("已修复负数数量问题并运行 pytest tests/test_cart.py")


class _CaptureLoop:
    kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs
        self._extra_hooks = []

    async def process_direct(self, prompt: str, session_key: str):
        return _FakeLoopResponse("计划")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        agents=SimpleNamespace(
            defaults=SimpleNamespace(
                model="test-model",
                max_iterations=4,
                context_window_tokens=4096,
                context_block_limit=32,
                approved_plan_context_block_limit=32,
                max_tool_result_chars=20000,
                provider_retry_mode="fail_fast",
                timezone="UTC",
                unified_session=False,
                disabled_skills=[],
                session_ttl_minutes=60,
                planning=SimpleNamespace(),
            )
        ),
        tools=SimpleNamespace(
            web=SimpleNamespace(),
            exec=SimpleNamespace(),
            restrict_to_workspace=True,
            mcp_servers={},
        ),
    )


def test_load_suite_validates_json():
    suite = load_suite(Path("tests/dogfood_eval/suites/smoke.json"))

    assert suite.name == "ghostbot-eval-smoke"
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].id == "plan.empty.001"


def test_snapshot_diff_detects_added_changed_deleted(tmp_path):
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()

    (before_dir / "same.txt").write_text("same", encoding="utf-8")
    (before_dir / "changed.txt").write_text("before", encoding="utf-8")
    (before_dir / "deleted.txt").write_text("gone", encoding="utf-8")

    (after_dir / "same.txt").write_text("same", encoding="utf-8")
    (after_dir / "changed.txt").write_text("after", encoding="utf-8")
    (after_dir / "added.txt").write_text("new", encoding="utf-8")

    before = EvalHarness._snapshot_files(before_dir)
    after = EvalHarness._snapshot_files(after_dir)

    changed, added, deleted = EvalHarness._diff_snapshots(before, after)

    assert changed == ["changed.txt"]
    assert added == ["added.txt"]
    assert deleted == ["deleted.txt"]


def test_snapshot_ignores_internal_project_state(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "projects" / "eval_case").mkdir(parents=True)
    (workspace / "projects" / "eval_case" / "state.json").write_text("{}", encoding="utf-8")
    (workspace / "projects" / "eval_case" / "history.jsonl").write_text("[]", encoding="utf-8")
    (workspace / "README.md").write_text("fixture", encoding="utf-8")

    snapshot = EvalHarness._snapshot_files(workspace)

    assert "README.md" in snapshot
    assert "projects/eval_case/state.json" not in snapshot
    assert "projects/eval_case/history.jsonl" not in snapshot


def test_resolve_check_command_uses_current_python_for_module_invocations():
    command = EvalHarness._resolve_check_command("python -m pytest tests/test_cart.py")

    assert command.startswith('"') or command.startswith("'") or command.startswith(sys.executable)
    assert " -m pytest tests/test_cart.py" in command


def test_run_deterministic_checks_reports_scope_and_failed_checks():
    suite = load_suite(Path("tests/dogfood_eval/suites/smoke.json"))
    scenario = suite.scenarios[0].model_copy(deep=True)
    scenario.expected.allowed_changed_files = ["allowed.txt"]
    scenario.expected.forbidden_changed_files = ["secret.txt"]
    scenario.expected.required_changed_files = ["required.txt"]
    scenario.expected.max_changed_files = 1
    scenario.expected.max_tool_calls = 1
    scenario.expected.checks = [EvalCheck(name="pytest", command="pytest")]

    failures, scope_violations = EvalHarness._run_deterministic_checks(
        scenario=scenario,
        final_content="这里只有计划",
        changed_files=["secret.txt", "other.txt"],
        tool_call_count=2,
        check_results=[],
    )

    assert "Scenario must not modify files" in failures
    assert "Tool call budget exceeded" in failures
    assert "Changed file budget exceeded" in failures
    assert "Missing required changed file: required.txt" in failures
    assert "Modified forbidden file: secret.txt" in failures
    assert "Modified file outside allowed scope: secret.txt" in failures
    assert "Modified file outside allowed scope: other.txt" in failures
    assert scope_violations == ["other.txt", "secret.txt"]


@pytest.mark.asyncio
async def test_run_scenario_disables_mcp_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("ghostbot.eval.harness._make_provider", lambda config: object())
    monkeypatch.setattr("ghostbot.eval.harness.AgentLoop", _CaptureLoop)

    async def _fake_run_checks(workspace, checks):
        return []

    monkeypatch.setattr("ghostbot.eval.harness.EvalHarness._run_checks", staticmethod(_fake_run_checks))

    harness = EvalHarness(_config(), Path("tests/dogfood_eval/suites/smoke.json"), tmp_path)
    scenario = harness.suite.scenarios[0]

    await harness._run_scenario("run-1", tmp_path / "run", scenario)

    assert _CaptureLoop.kwargs is not None
    assert _CaptureLoop.kwargs["mcp_servers"] == {}


@pytest.mark.asyncio
async def test_run_scenario_populates_judge_when_configured(tmp_path, monkeypatch):
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="judge-1",
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

    monkeypatch.setattr("ghostbot.eval.harness._make_provider", lambda config: provider)
    monkeypatch.setattr("ghostbot.eval.harness.AgentLoop", _FakeLoop)

    original_process_direct = _FakeLoop.process_direct

    async def _process_direct_with_fixture(self, prompt: str, session_key: str):
        result = await original_process_direct(self, prompt, session_key)
        tests_file = Path(self.kwargs["workspace"]) / "tests" / "test_cart.py"
        tests_file.write_text(
            (
                "from shop.cart import add_to_cart\n\n\n"
                "def test_add_to_cart_accepts_positive_quantity():\n"
                "    cart = {}\n"
                "    updated = add_to_cart(cart, \"apple\", 2)\n"
                "    assert updated[\"apple\"] == 2\n\n\n"
                "def test_add_to_cart_accumulates_quantity():\n"
                "    cart = {\"apple\": 1}\n"
                "    updated = add_to_cart(cart, \"apple\", 3)\n"
                "    assert updated[\"apple\"] == 4\n\n\n"
                "def test_add_to_cart_rejects_negative_quantity():\n"
                "    cart = {}\n"
                "    try:\n"
                "        add_to_cart(cart, \"apple\", -1)\n"
                "    except ValueError:\n"
                "        return\n"
                "    assert False, \"expected ValueError\"\n"
            ),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(_FakeLoop, "process_direct", _process_direct_with_fixture)

    async def _fake_run_checks(workspace, checks):
        return []

    monkeypatch.setattr("ghostbot.eval.harness.EvalHarness._run_checks", staticmethod(_fake_run_checks))

    harness = EvalHarness(_config(), Path("tests/dogfood_eval/suites/core.json"), tmp_path)
    scenario = next(item for item in harness.suite.scenarios if item.id == "bugfix.cart_negative_quantity.001")
    scenario = scenario.model_copy(deep=True)
    scenario.judging.rubric = "requirement_completion"

    result = await harness._run_scenario("run-1", tmp_path / "run", scenario)

    assert result.judge.score == 0.9
    assert result.judge.passed is True
    assert result.judge.verdict == "pass"
    assert result.status == "pass"
    assert len(provider.calls) == 1
