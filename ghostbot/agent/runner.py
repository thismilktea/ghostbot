"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any

from loguru import logger

from ghostbot.agent.hook import AgentHook, AgentHookContext
from ghostbot.agent.policy import PolicyAction, PolicyContext, PolicyEngine
from ghostbot.agent.tools.base import ToolResult
from ghostbot.utils.prompt_templates import render_template
from ghostbot.agent.tools.registry import ToolRegistry
from ghostbot.providers.base import LLMProvider, ToolCallRequest
from ghostbot.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    maybe_persist_tool_result,
    truncate_text,
)
from ghostbot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_finalization_retry_message,
    build_length_recovery_message,
    ensure_nonempty_tool_result,
    is_blank_text,
    repeated_external_lookup_error,
)

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_MAX_INJECTIONS_PER_TURN = 3
_MAX_INJECTION_CYCLES = 5
_SNIP_SAFETY_BUFFER = 1024
_MICROCOMPACT_KEEP_RECENT = 10
_MICROCOMPACT_NOISY_KEEP_RECENT = 5
_MICROCOMPACT_MIN_CHARS = 500
_COMPACTABLE_TOOLS = frozenset({
    "read_file", "exec", "grep", "glob",
    "web_search", "web_fetch", "list_dir",
})
_WRITE_TOOL_NAMES = frozenset({"write_file", "edit_file", "notebook_edit"})
_EXEC_TOOL_NAMES = frozenset({"exec"})
_WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
_BACKGROUND_TOOL_NAMES = frozenset({"spawn", "cron"})
_REPEATED_LOOKUP_TOOLS = frozenset({"read_file", "grep", "glob"})
_BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    max_tool_result_chars: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    workspace: Path | None = None
    project_key: str | None = None
    context_window_tokens: int | None = None
    context_block_limit: int | None = None
    provider_retry_mode: str = "standard"
    progress_callback: Any | None = None
    checkpoint_callback: Any | None = None
    injection_callback: Any | None = None
    mode: str = "general"
    approval_mode: str = "auto"
    require_approval_for_write: bool = False
    require_approval_for_exec: bool = False
    allow_background_tools: bool = True
    allow_external_tools: bool = True
    allowed_tool_names: set[str] | None = None
    blocked_tool_prefixes: tuple[str, ...] = ()
    policy_engine: PolicyEngine | None = None
    policy_context: PolicyContext | None = None
@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False
    pending_approval: dict[str, Any] | None = None


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                    for item in value
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    @classmethod
    def _append_injected_messages(
        cls,
        messages: list[dict[str, Any]],
        injections: list[dict[str, Any]],
    ) -> None:
        """Append injected user messages while preserving role alternation."""
        for injection in injections:
            if (
                messages
                and injection.get("role") == "user"
                and messages[-1].get("role") == "user"
            ):
                merged = dict(messages[-1])
                merged["content"] = cls._merge_message_content(
                    merged.get("content"),
                    injection.get("content"),
                )
                messages[-1] = merged
                continue
            messages.append(injection)


    @staticmethod
    def _tool_policy_error(spec: AgentRunSpec, tool_name: str) -> str | None:
        if spec.allowed_tool_names is not None and tool_name not in spec.allowed_tool_names:
            return f"Error: Tool '{tool_name}' is disabled in {spec.mode} mode"
        if any(tool_name.startswith(prefix) for prefix in spec.blocked_tool_prefixes):
            return f"Error: Tool '{tool_name}' is disabled in {spec.mode} mode"
        if tool_name in _WRITE_TOOL_NAMES and spec.require_approval_for_write:
            return (
                f"Error: Tool '{tool_name}' requires approval in {spec.mode} mode "
                f"(approval_mode={spec.approval_mode})"
            )
        if tool_name in _EXEC_TOOL_NAMES and spec.require_approval_for_exec:
            return (
                f"Error: Tool '{tool_name}' requires approval in {spec.mode} mode "
                f"(approval_mode={spec.approval_mode})"
            )
        if tool_name in _WEB_TOOL_NAMES and not spec.allow_external_tools:
            return f"Error: Tool '{tool_name}' is disabled in {spec.mode} mode"
        if tool_name in _BACKGROUND_TOOL_NAMES and not spec.allow_background_tools:
            return f"Error: Tool '{tool_name}' is disabled in {spec.mode} mode"
        return None

    async def _drain_injections(self, spec: AgentRunSpec) -> list[dict[str, Any]]:
        """Drain pending user messages via the injection callback.

        Returns normalized user messages (capped by
        ``_MAX_INJECTIONS_PER_TURN``), or an empty list when there is
        nothing to inject. Messages beyond the cap are logged so they
        are not silently lost.
        """
        if spec.injection_callback is None:
            return []
        try:
            signature = inspect.signature(spec.injection_callback)
            accepts_limit = (
                "limit" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_limit:
                items = await spec.injection_callback(limit=_MAX_INJECTIONS_PER_TURN)
            else:
                items = await spec.injection_callback()
        except Exception:
            logger.exception("injection_callback failed")
            return []
        if not items:
            return []
        injected_messages: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict) and item.get("role") == "user" and "content" in item:
                injected_messages.append(item)
                continue
            text = getattr(item, "content", str(item))
            if text.strip():
                injected_messages.append({"role": "user", "content": text})
        if len(injected_messages) > _MAX_INJECTIONS_PER_TURN:
            dropped = len(injected_messages) - _MAX_INJECTIONS_PER_TURN
            logger.warning(
                "Injection callback returned {} messages, capping to {} ({} dropped)",
                len(injected_messages), _MAX_INJECTIONS_PER_TURN, dropped,
            )
            injected_messages = injected_messages[:_MAX_INJECTIONS_PER_TURN]
        return injected_messages

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        repeated_lookup_keys: set[tuple[str, str]] = set()
        empty_content_retries = 0
        length_recovery_count = 0
        had_injections = False
        injection_cycles = 0
        pending_approval: dict[str, Any] | None = None

        for iteration in range(spec.max_iterations):
            try:
                messages_for_model = self._drop_orphan_tool_results(messages)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                messages_for_model = self._microcompact(spec, messages_for_model)
                messages_for_model = self._apply_tool_result_budget(spec, messages_for_model)
                messages_for_model = self._snip_history(spec, messages_for_model)
                messages_for_model = self._drop_orphan_tool_results(messages_for_model)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
            except Exception as exc:
                logger.warning(
                    "Context governance failed on turn {} for {}: {}; applying minimal repair",
                    iteration,
                    spec.project_key or "default",
                    exc,
                )
                try:
                    messages_for_model = self._drop_orphan_tool_results(messages)
                    messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                except Exception:
                    messages_for_model = messages
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            response = await self._request_model(spec, messages_for_model, hook, context)
            raw_usage = self._usage_dict(response.usage)
            context.response = response
            context.usage = dict(raw_usage)
            context.tool_calls = list(response.tool_calls)
            self._accumulate_usage(usage, raw_usage)

            if response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                messages.append(assistant_message)
                tools_used.extend(tc.name for tc in response.tool_calls)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )

                await hook.before_execute_tools(context)

                results, new_events, fatal_error, pending_approval = await self._execute_tools(
                    spec,
                    response.tool_calls,
                    external_lookup_counts,
                    repeated_lookup_keys,
                    assistant_message,
                )
                tool_events.extend(new_events)
                if pending_approval is not None:
                    final_content = self._format_pending_approval_prompt(pending_approval)
                    stop_reason = "approval_required"
                    context.final_content = final_content
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                completed_tool_results: list[dict[str, Any]] = []
                for tool_call, result in zip(response.tool_calls, results):
                    result_content = self._tool_result_content(result)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": self._normalize_tool_result(
                            spec,
                            tool_call.id,
                            tool_call.name,
                            result_content,
                        ),
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    final_content = error
                    stop_reason = "tool_error"
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                if new_events and all(event.get("policy_decision") in {"deny", "require_approval"} for event in new_events):
                    final_content = str(results[-1]) if results else "Tool blocked by policy."
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_content_retries = 0
                length_recovery_count = 0
                if injection_cycles < _MAX_INJECTION_CYCLES:
                    injections = await self._drain_injections(spec)
                    if injections:
                        had_injections = True
                        injection_cycles += 1
                        self._append_injected_messages(messages, injections)
                        logger.info(
                            "Injected {} follow-up message(s) after tool execution ({}/{})",
                            len(injections), injection_cycles, _MAX_INJECTION_CYCLES,
                        )
                await hook.after_iteration(context)
                continue

            clean = hook.finalize_content(context, response.content)
            if response.finish_reason != "error" and is_blank_text(clean):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.project_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.project_key or "default",
                    empty_content_retries,
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)
                response = await self._request_finalization_retry(spec, messages_for_model)
                retry_usage = self._usage_dict(response.usage)
                self._accumulate_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = dict(raw_usage)
                context.tool_calls = list(response.tool_calls)
                clean = hook.finalize_content(context, response.content)

            if response.finish_reason == "length" and not is_blank_text(clean):
                length_recovery_count += 1
                if length_recovery_count <= _MAX_LENGTH_RECOVERIES:
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.project_key or "default",
                        length_recovery_count,
                        _MAX_LENGTH_RECOVERIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                    messages.append(build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))
                    messages.append(build_length_recovery_message())
                    await hook.after_iteration(context)
                    continue

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

            _injected_after_final = False
            if injection_cycles < _MAX_INJECTION_CYCLES:
                injections = await self._drain_injections(spec)
                if injections:
                    had_injections = True
                    injection_cycles += 1
                    _injected_after_final = True
                    if assistant_message is not None:
                        messages.append(assistant_message)
                        await self._emit_checkpoint(
                            spec,
                            {
                                "phase": "final_response",
                                "iteration": iteration,
                                "model": spec.model,
                                "assistant_message": assistant_message,
                                "completed_tool_results": [],
                                "pending_tool_calls": [],
                            },
                        )
                    self._append_injected_messages(messages, injections)
                    logger.info(
                        "Injected {} follow-up message(s) after final response ({}/{})",
                        len(injections), injection_cycles, _MAX_INJECTION_CYCLES,
                    )

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=_injected_after_final)

            if _injected_after_final:
                await hook.after_iteration(context)
                continue

            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                self._append_model_error_placeholder(messages)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break
            if is_blank_text(clean):
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            messages.append(assistant_message or build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "aborted_by_circuit_breaker"
            logger.error(
                f"🚨 [系统熔断] Agent 陷入死循环！"
                f"已连续执行 {spec.max_iterations} 次工具调用未能结束任务。(Project: {spec.project_key})"
            )
            abort_message = (
                "❌ **[系统强行干预：死循环熔断]**\n\n"
                f"系统检测到你已经连续进行了 {spec.max_iterations} 次工具调用（Read/Edit/Search），但依然没有完成任务。"
                "这通常意味着你陷入了逻辑死循环、正则匹配错误，或者遇到了无法自动修复的 Bug。\n\n"
                "**为了保护 API 额度，系统已强制剥夺你的自动执行权限，终止本次运行。**\n"
                "请在下一次回复时，**立刻向人类开发者总结你目前遇到的困境，并请求人工接管排查。**"
            )
            if spec.max_iterations_message:
                final_content = spec.max_iterations_message.format(
                    max_iterations=spec.max_iterations,
                )
            else:
                final_content = abort_message
            self._append_final_message(messages, final_content)
            error = f"CircuitBreakerError: Reached max iterations ({spec.max_iterations})"

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            pending_approval=pending_approval,
        )

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": [dict(message) for message in messages],
            "tools": tools,
            "model": spec.model,
            "retry_mode": spec.provider_retry_mode,
            "on_retry_wait": spec.progress_callback,
        }
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            kwargs["max_tokens"] = spec.max_tokens
        if spec.reasoning_effort is not None:
            kwargs["reasoning_effort"] = spec.reasoning_effort
        return kwargs

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
    ):
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=spec.tools.get_definitions(),
        )
        if hook.wants_streaming():
            async def _stream(delta: str) -> None:
                await hook.on_stream(context, delta)

            return await self.provider.chat_stream_with_retry(
                **kwargs,
                on_content_delta=_stream,
            )
        return await self.provider.chat_with_retry(**kwargs)

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ):
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        kwargs = self._build_request_kwargs(spec, retry_messages, tools=None)
        return await self.provider.chat_with_retry(**kwargs)

    @staticmethod
    def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        merged = dict(left)
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        external_lookup_counts: dict[str, int],
        repeated_lookup_keys: set[tuple[str, str]],
        assistant_message: dict[str, Any],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None, dict[str, Any] | None]:
        batches = self._partition_tool_batches(spec, tool_calls)
        tool_results: list[tuple[Any, dict[str, str], BaseException | None, dict[str, Any] | None]] = []
        for batch in batches:
            for tool_call in batch:
                result = await self._run_tool(
                    spec,
                    tool_call,
                    external_lookup_counts,
                    repeated_lookup_keys,
                    assistant_message,
                )
                tool_results.append(result)
                if result[3] is not None:
                    break
            if tool_results and tool_results[-1][3] is not None:
                break
            if spec.concurrent_tools and len(batch) > 1:
                continue

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        pending_approval: dict[str, Any] | None = None
        for result, event, error, approval in tool_results:
            if approval is not None:
                pending_approval = approval
                events.append(event)
                break
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error, pending_approval

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        external_lookup_counts: dict[str, int],
        repeated_lookup_keys: set[tuple[str, str]],
        assistant_message: dict[str, Any],
    ) -> tuple[Any, dict[str, str], BaseException | None, dict[str, Any] | None]:
        _HINT = "\n\n[Analyze the error above and try a different approach.]"
        policy_error = self._tool_policy_error(spec, tool_call.name)
        if policy_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": policy_error,
            }
            err = RuntimeError(policy_error)
            return policy_error + _HINT, event, err, None
        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        if lookup_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + _HINT, event, RuntimeError(lookup_error), None
            return lookup_error + _HINT, event, None, None
        repeated_key = self._repeated_lookup_key(tool_call.name, tool_call.arguments)
        if repeated_key is not None and repeated_key in repeated_lookup_keys:
            message = f"[Repeated {tool_call.name} omitted: same parameters were already used during this run and no write invalidated them.]"
            return message, {"name": tool_call.name, "status": "ok", "detail": "repeated lookup omitted"}, None, None
        prepare_call = getattr(spec.tools, "prepare_call", None)
        tool, params, prep_error = None, tool_call.arguments, None
        if callable(prepare_call):
            try:
                prepared = prepare_call(tool_call.name, tool_call.arguments)
                if isinstance(prepared, tuple) and len(prepared) == 3:
                    tool, params, prep_error = prepared
            except Exception:
                pass
        if prep_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
            }
            return prep_error + _HINT, event, RuntimeError(prep_error) if spec.fail_on_tool_error else None, None
        if spec.policy_engine is not None:
            decision = spec.policy_engine.evaluate(PolicyAction(
                tool_name=tool_call.name,
                params=params,
                workspace=spec.workspace,
                mode=spec.mode,
                session_key=spec.project_key,
                tool_read_only=bool(getattr(tool, "read_only", False)),
                side_effect_level=str(getattr(tool, "side_effect_level", "unknown")),
                risk_tags=frozenset(getattr(tool, "risk_tags", frozenset())),
                context=spec.policy_context,
            ))
            if decision.kind != "allow":
                message = f"Error: Tool '{tool_call.name}' {decision.kind}: {decision.reason or 'blocked by policy'}"
                event = {
                    "name": tool_call.name,
                    "status": "error",
                    "detail": message,
                    "policy_decision": decision.kind,
                    "policy_reason": decision.reason,
                }
                if decision.kind == "require_approval":
                    approval = {
                        "id": tool_call.id,
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "arguments": dict(tool_call.arguments or {}),
                        "tool_call": tool_call.to_openai_tool_call(),
                        "reason": decision.reason or "Tool requires approval",
                        "assistant_message": dict(assistant_message),
                    }
                    return message, event, None, approval
                return message, event, RuntimeError(message) if spec.fail_on_tool_error else None, None
        try:
            if tool is not None:
                result = await tool.execute(**params)
            else:
                result = await spec.tools.execute(tool_call.name, params)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc, None
            return f"Error: {type(exc).__name__}: {exc}", event, None, None

        if isinstance(result, str) and result.startswith("Error"):
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            if spec.fail_on_tool_error:
                return result + _HINT, event, RuntimeError(result), None
            return result + _HINT, event, None, None

        if tool_call.name in _WRITE_TOOL_NAMES:
            repeated_lookup_keys.clear()
        elif repeated_key is not None:
            repeated_lookup_keys.add(repeated_key)

        event = {
            "name": tool_call.name,
            "status": "ok",
        }
        metadata = self._tool_result_metadata(result)
        change_summary = metadata.get("change_summary")
        if isinstance(change_summary, dict):
            event["change_summary"] = change_summary
            formatted = change_summary.get("formatted")
            if formatted and spec.progress_callback is not None:
                await spec.progress_callback(str(formatted), change_summary=True)

        detail = "" if self._tool_result_content(result) is None else str(self._tool_result_content(result))
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        event["detail"] = detail
        return result, event, None, None

    @staticmethod
    def _tool_result_content(result: Any) -> Any:
        return result.content if isinstance(result, ToolResult) else result

    @staticmethod
    def _tool_result_metadata(result: Any) -> dict[str, Any]:
        return dict(result.metadata) if isinstance(result, ToolResult) else {}

    @staticmethod
    def _repeated_lookup_key(tool_name: str, arguments: dict[str, Any] | None) -> tuple[str, str] | None:
        if tool_name not in _REPEATED_LOOKUP_TOOLS:
            return None
        args = dict(arguments or {})
        if tool_name == "read_file":
            selected = {
                "path": args.get("path"),
                "offset": args.get("offset"),
                "limit": args.get("limit"),
                "pages": args.get("pages") or "",
            }
        elif tool_name == "glob":
            selected = {
                "pattern": args.get("pattern"),
                "path": args.get("path") or "",
                "entry_type": args.get("entry_type") or "files",
                "head_limit": args.get("head_limit") or args.get("max_results"),
                "offset": args.get("offset") or 0,
            }
        else:
            selected = {
                key: args.get(key)
                for key in (
                    "pattern", "path", "glob", "type", "output_mode",
                    "head_limit", "max_results", "max_matches", "offset",
                    "context", "context_before", "context_after", "case_insensitive",
                    "fixed_strings",
                )
            }
        try:
            payload = json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str)
        except TypeError:
            payload = str(sorted(selected.items()))
        return tool_name, payload

    @staticmethod
    def _format_pending_approval_prompt(pending_approval: dict[str, Any]) -> str:
        tool_name = pending_approval.get("tool_name") or "tool"
        reason = pending_approval.get("reason") or "Tool requires approval"
        return (
            "Tool approval required.\n\n"
            f"Tool: {tool_name}\n"
            f"Reason: {reason}\n\n"
            "Reply `approve` to run it or `deny` to skip it."
        )

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict[str, Any],
    ) -> None:
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = build_assistant_message(content)
            return
        messages.append(build_assistant_message(content))

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
        messages.append(build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))

    def _normalize_tool_result(
        self,
        spec: AgentRunSpec,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        result = ensure_nonempty_tool_result(tool_name, result)
        try:
            content = maybe_persist_tool_result(
                spec.workspace,
                spec.project_key,
                tool_call_id,
                result,
                max_chars=spec.max_tool_result_chars,
            )
        except Exception as exc:
            logger.warning(
                "Tool result persist failed for {} in {}: {}; using raw result",
                tool_call_id,
                spec.project_key or "default",
                exc,
            )
            content = result
        if isinstance(content, str) and len(content) > spec.max_tool_result_chars:
            return truncate_text(content, spec.max_tool_result_chars)
        return content

    @staticmethod
    def _drop_orphan_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop tool results that have no matching assistant tool_call earlier in the history."""
        declared: set[str] = set()
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            if role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]
                    continue
            if updated is not None:
                updated.append(dict(msg))

        if updated is None:
            return messages
        return updated

    @staticmethod
    def _backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert synthetic error results for orphaned tool_use blocks."""
        declared: list[tuple[int, str, str]] = []  # (assistant_idx, call_id, name)
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        name = ""
                        func = tc.get("function")
                        if isinstance(func, dict):
                            name = func.get("name", "")
                        declared.append((idx, str(tc["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": _BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    def _microcompact(
            self,
            spec: AgentRunSpec,
            messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace old compactable tool results with virtual memory pointers (Page Out)."""
        compactable_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool" and msg.get("name") in _COMPACTABLE_TOOLS:
                compactable_indices.append(idx)

        stale: list[int] = []
        for idx, msg_idx in enumerate(compactable_indices):
            name = str(messages[msg_idx].get("name") or "tool")
            keep_recent = _MICROCOMPACT_NOISY_KEEP_RECENT if name in {"read_file", "grep", "glob", "list_dir", "exec"} else _MICROCOMPACT_KEEP_RECENT
            newer_same_tool = sum(
                1
                for later in compactable_indices[idx + 1:]
                if messages[later].get("name") == name
            )
            if newer_same_tool >= keep_recent:
                stale.append(msg_idx)

        if not stale:
            return messages
        updated: list[dict[str, Any]] | None = None

        # 🔥 架构融合：临时实例化 MemoryStore 仅用于 Swap 功能
        from ghostbot.agent.memory import MemoryStore
        memory_store = MemoryStore(spec.workspace)

        for idx in stale:
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < _MICROCOMPACT_MIN_CHARS:
                continue

            name = msg.get("name", "tool")

            pointer = memory_store.page_out(content)
            compact_name = str(name)
            summary = f"[{compact_name} result compacted: approx_chars={len(content)}, pointer={pointer}]"

            if updated is None:
                updated = [dict(m) for m in messages]
            updated[idx]["content"] = summary

        return updated if updated is not None else messages

    def _apply_tool_result_budget(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self._normalize_tool_result(
                spec,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def _snip_history(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages or not spec.context_window_tokens:
            return messages

        provider_max_tokens = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        max_output = spec.max_tokens if isinstance(spec.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )
        budget = spec.context_block_limit or (
            spec.context_window_tokens - max_output - _SNIP_SAFETY_BUFFER
        )
        if budget <= 0:
            return messages

        estimate, _ = estimate_prompt_tokens_chain(
            self.provider,
            spec.model,
            messages,
            spec.tools.get_definitions(),
        )
        if estimate <= budget:
            return messages

        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not non_system:
            return messages

        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        remaining_budget = max(128, budget - system_tokens)
        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for message in reversed(non_system):
            msg_tokens = estimate_message_tokens(message)
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        kept.reverse()

        if kept:
            for i, message in enumerate(kept):
                if message.get("role") == "user":
                    kept = kept[i:]
                    break
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        if not kept:
            kept = non_system[-min(len(non_system), 4) :]
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        return system_messages + kept

    def _partition_tool_batches(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        if not spec.concurrent_tools:
            return [[tool_call] for tool_call in tool_calls]

        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            get_tool = getattr(spec.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            can_batch = bool(tool and tool.concurrency_safe)
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches

