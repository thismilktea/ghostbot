"""Schema models for GhostBot dogfooding evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvalDefaults(BaseModel):
    max_iterations: int = Field(default=30, ge=1)
    timeout_seconds: int = Field(default=900, ge=1)
    temperature: float = 0.0
    judge_models: list[str] = Field(default_factory=list)
    disable_mcp: bool = True


class EvalFixture(BaseModel):
    type: Literal["copy", "empty"] = "copy"
    source: str | None = None
    reset: Literal["fresh_copy", "fresh"] = "fresh_copy"


class EvalCheck(BaseModel):
    name: str
    command: str
    cwd: str | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)


class EvalSetup(BaseModel):
    commands: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    memory_seed: list[str] = Field(default_factory=list)


class EvalExpected(BaseModel):
    must_not_modify_files: bool = False
    allowed_changed_files: list[str] = Field(default_factory=list)
    forbidden_changed_files: list[str] = Field(default_factory=list)
    required_changed_files: list[str] = Field(default_factory=list)
    required_phrases: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    checks: list[EvalCheck] = Field(default_factory=list)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_changed_files: int | None = Field(default=None, ge=0)


class EvalJudging(BaseModel):
    rubric: str | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    pairwise_baseline: bool = False
    required_capabilities: list[str] = Field(default_factory=list)


class EvalTurn(BaseModel):
    input: str
    expect_approval_required: bool = False


class EvalScenario(BaseModel):
    id: str
    category: str
    fixture: str
    mode: Literal["plan_only", "direct_execute", "plan_approve_execute", "multi_turn"] = "direct_execute"
    prompt: str | None = None
    turns: list[EvalTurn] = Field(default_factory=list)
    setup: EvalSetup = Field(default_factory=EvalSetup)
    expected: EvalExpected = Field(default_factory=EvalExpected)
    judging: EvalJudging = Field(default_factory=EvalJudging)
    tags: list[str] = Field(default_factory=list)


class EvalSuite(BaseModel):
    schema_version: int = 1
    name: str
    description: str | None = None
    defaults: EvalDefaults = Field(default_factory=EvalDefaults)
    fixtures: dict[str, EvalFixture] = Field(default_factory=dict)
    scenarios: list[EvalScenario] = Field(default_factory=list)


class DeterministicCheckResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)


class EvalJudgeResult(BaseModel):
    score: float | None = None
    passed: bool | None = None
    verdict: str | None = None
    reasoning: str | None = None


class EvalToolSummary(BaseModel):
    tools_used: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    redundant_tool_call_count: int = 0
    failed_tool_call_count: int = 0
    redundant_calls: list[str] = Field(default_factory=list)


class EvalFileSummary(BaseModel):
    changed_files: list[str] = Field(default_factory=list)
    added_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    scope_violations: list[str] = Field(default_factory=list)


class EvalTiming(BaseModel):
    started_at: datetime
    ended_at: datetime
    latency_ms: int


class EvalCommandResult(BaseModel):
    name: str
    command: str
    cwd: str | None = None
    exit_code: int
    passed: bool
    stdout: str = ""
    stderr: str = ""


class EvalMetricSummary(BaseModel):
    requirement_hit_rate: float = 0.0
    self_correction_rate: float = 0.0
    recovery_cost: int = 0
    check_pass_rate: float = 0.0


class EvalAgentSummary(BaseModel):
    stop_reason: str = "completed"
    iterations: int = 0
    had_injections: bool = False
    pending_approval: dict[str, Any] | None = None
    final_content: str | None = None
    error: str | None = None


class EvalScenarioResult(BaseModel):
    schema_version: int = 1
    run_id: str
    scenario_id: str
    category: str
    fixture: str
    model: str
    status: Literal["pass", "fail", "error"]
    timing: EvalTiming
    agent: EvalAgentSummary
    usage: dict[str, int] = Field(default_factory=dict)
    tools: EvalToolSummary = Field(default_factory=EvalToolSummary)
    files: EvalFileSummary = Field(default_factory=EvalFileSummary)
    checks: list[EvalCommandResult] = Field(default_factory=list)
    metrics: EvalMetricSummary = Field(default_factory=EvalMetricSummary)
    deterministic: DeterministicCheckResult = Field(default_factory=lambda: DeterministicCheckResult(passed=True))
    judge: EvalJudgeResult = Field(default_factory=EvalJudgeResult)
    trace_file: str | None = None


class EvalRunSummary(BaseModel):
    schema_version: int = 1
    run_id: str
    suite_name: str
    generated_at: datetime
    results: list[EvalScenarioResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
