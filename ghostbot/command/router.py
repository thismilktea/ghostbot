"""Minimal command routing table for slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from ghostbot.bus.events import InboundMessage, OutboundMessage
    from ghostbot.project import ProjectState

Handler = Callable[["CommandContext"], Awaitable["OutboundMessage | None"]]


@dataclass(init=False)
class CommandContext:
    """Everything a command handler needs to produce a response."""

    msg: InboundMessage
    project: ProjectState | None
    project_key: str
    raw: str
    args: str
    loop: Any

    def __init__(
        self,
        *,
        msg: InboundMessage,
        raw: str,
        project: ProjectState | None = None,
        project_key: str | None = None,
        session: ProjectState | None = None,
        key: str | None = None,
        args: str = "",
        loop: Any = None,
    ) -> None:
        self.msg = msg
        self.project = project if project is not None else session
        self.project_key = project_key or key or msg.session_key
        self.raw = raw
        self.args = args
        self.loop = loop

    @property
    def session(self) -> ProjectState | None:
        return self.project

    @property
    def key(self) -> str:
        return self.project_key


class CommandRouter:
    """Pure dict-based command dispatch.

    Three tiers checked in order:
      1. *priority* — exact-match commands handled before the dispatch lock
         (e.g. /stop, /restart).
      2. *exact* — exact-match commands handled inside the dispatch lock.
      3. *prefix* — longest-prefix-first match (e.g. "/team ").
      4. *interceptors* — fallback predicates (e.g. team-mode active check).
    """

    def __init__(self) -> None:
        self._priority: dict[str, Handler] = {}
        self._exact: dict[str, Handler] = {}
        self._prefix: list[tuple[str, Handler]] = []
        self._interceptors: list[Handler] = []

    def priority(self, cmd: str, handler: Handler) -> None:
        self._priority[cmd] = handler

    def exact(self, cmd: str, handler: Handler) -> None:
        self._exact[cmd] = handler

    def prefix(self, pfx: str, handler: Handler) -> None:
        self._prefix.append((pfx, handler))
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def intercept(self, handler: Handler) -> None:
        self._interceptors.append(handler)

    def is_priority(self, text: str) -> bool:
        return text.strip().lower() in self._priority

    async def dispatch_priority(self, ctx: CommandContext) -> OutboundMessage | None:
        """Dispatch a priority command. Called from run() without the lock."""
        handler = self._priority.get(ctx.raw.lower())
        if handler:
            return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> OutboundMessage | None:
        """Try exact, prefix, then interceptors. Returns None if unhandled."""
        cmd = ctx.raw.lower()

        if handler := self._exact.get(cmd):
            return await handler(ctx)

        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        for interceptor in self._interceptors:
            result = await interceptor(ctx)
            if result is not None:
                return result

        return None
