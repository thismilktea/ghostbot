# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and common commands

- Install in editable mode: `pip install -e .`
- Install dev dependencies: `pip install -e '.[dev]'`
- Alternative environment sync: `uv sync`
- Run the CLI locally: `ghostbot agent`
- Run a one-shot prompt: `ghostbot agent -m "summarize this repo"`
- Initialize local config/workspace: `ghostbot onboard` or `ghostbot onboard --wizard`
- Show version: `ghostbot --version`

## Tests and lint

- Run all tests: `python -m pytest`
- Run a single test file: `python -m pytest tests/agent/test_planning_flow.py`
- Run a single test: `python -m pytest tests/agent/test_planning_flow.py -k concise`
- Run eval-harness tests: `python -m pytest tests/dogfood_eval/test_eval_harness.py`
- Lint: `python -m ruff check .`

## Architecture overview

GhostBot is a local coding-agent runtime centered on **project-scoped state**, **approval-gated planning**, and a **tool-executing agent loop**.

### Entry points and user surfaces

- `ghostbot/cli/commands.py` is the main Typer CLI entrypoint. It sets up logging, interactive terminal UX, history, and routes commands like `ghostbot agent` and onboarding.
- `ghostbot/api/server.py` exposes a small OpenAI-compatible HTTP surface (`/v1/chat/completions`, `/v1/models`, `/health`) that routes every request into one persistent GhostBot session.
- Slash commands are dispatched through `ghostbot/command/router.py`, which supports priority commands, exact matches, prefix handlers, and interceptors.

### Agent runtime

- `ghostbot/agent/runner.py` contains the shared LLM execution loop. It owns iterative model calls, tool-call execution, policy gating, history compaction/snipping, checkpointing, and approval handoff when a tool action needs confirmation.
- `ghostbot/agent/tools/` contains the built-in tool implementations. `ghostbot/agent/tools/registry.py` is the normalization and validation layer for dynamic tool registration/execution.
- The runtime is intentionally structured so the model does not directly mutate the world; writes, exec, web access, background work, and approvals are mediated through tool/policy boundaries.

### Planning and controlled execution

- `ghostbot/agent/planning.py` implements the approval-first planning workflow. Plans are structured artifacts with required sections, checklist extraction, quality validation, history/archive support, and execution-mode tracking.
- Planning state is durable: active plans are stored under workspace memory/history rather than living only in the chat transcript.
- The planning config in `ghostbot/config/schema.py` controls when planning triggers, whether approval is required, and how much exploration happens before a plan is shown.

### Project-scoped memory and session state

- `ghostbot/project/manager.py` is the core project-state layer. Each project keeps its own `state.json`, `history.jsonl`, metadata, and recently active files.
- This is a key design choice in the repo: GhostBot organizes long-lived context around a **project**, not just a transient chat session.
- The workspace directory is therefore part of the product architecture, not just a cache.

### Config and provider model

- `ghostbot/config/schema.py` defines nearly all runtime behavior: agent defaults, planning, cron/dream jobs, tool restrictions, coding mode, MCP servers, API settings, and web search settings.
- `ghostbot/config/loader.py` loads config, resolves `${ENV_VAR}` placeholders, applies small migrations, and wires SSRF whitelist settings into the network-security layer.
- `ghostbot/providers/registry.py` is the single source of truth for provider metadata. Provider matching is data-driven: adding a provider means extending the registry plus schema rather than scattering logic.
- Providers are split by backend style: native Anthropic, OpenAI-compatible endpoints, Azure OpenAI, OAuth-backed Codex/Copilot, and gateway-style providers such as OpenRouter.

### Scheduling and background services

- `ghostbot/cron/service.py` manages durable scheduled jobs. Jobs are persisted to disk, merged across instances through an action log, and can run one-shot, interval, or cron schedules.
- The config schema also defines “dream”/heartbeat style background behavior, so recurring maintenance and memory consolidation are first-class runtime concepts.

### Retrieval and repository understanding

- `ghostbot/agent/adaptive_retrieval.py` shows the repo’s retrieval philosophy: use a lightweight intent router to decide when local code/history retrieval is needed instead of always injecting more context.
- The README describes the broader intended architecture here: project memory + structured repository understanding + context buckets, with AST/symbol-aware tooling layered on top.

### Evaluation and dogfooding

- `ghostbot/eval/` is not auxiliary; it is part of the product loop. The project includes a dogfooding harness that runs scenarios against fixture repos and scores requirement coverage, tool efficiency, recovery, and latency.
- `ghostbot/eval/metrics.py` summarizes scenario-level outcomes, while tests under `tests/dogfood_eval/` validate the harness behavior itself.
- When changing planning, tool policy, or runtime orchestration, check whether dogfood/eval expectations also need updates.

## Repo-specific working notes

- This repo targets Python 3.11+ and uses Ruff plus Pytest from `pyproject.toml`.
- Tests are configured from `pyproject.toml`; there is no separate `pytest.ini`.
- Logging writes to `~/.ghostbot/logs/ghostbot_runtime.log` in normal CLI use.
- Many behaviors are workspace-backed (`~/.ghostbot/workspace` by default), so bugs around plans, sessions, cron, or project switching often involve on-disk state as well as in-memory state.
- There is currently no checked-in `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` file in this repository.
