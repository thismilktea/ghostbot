"""Auto compact: proactive compression of idle sessions to reduce token cost and latency."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger
from ghostbot.project import ProjectManager, ProjectState

if TYPE_CHECKING:
    from ghostbot.agent.memory import Consolidator


@dataclass
class CheckpointSummary:
    text: str
    last_active: datetime
    source: str = "idle"


class AutoCompact:
    _RECENT_SUFFIX_MESSAGES = 8
    _MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, sessions: ProjectManager, consolidator: Consolidator,
                 session_ttl_minutes: int = 0):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set[str] = set()
        self._summaries: dict[str, CheckpointSummary] = {}
        self._failure_counts: dict[str, int] = {}

    def _is_expired(self, ts: datetime | str | None) -> bool:
        if self._ttl <= 0 or not ts:
            return False
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return (datetime.now() - ts).total_seconds() >= self._ttl * 60

    @classmethod
    def _format_summary(cls, summary: CheckpointSummary) -> str:
        idle_min = int((datetime.now() - summary.last_active).total_seconds() / 60)
        return (
            f"Inactive for {idle_min} minutes.\n"
            f"Previous conversation summary:\n{summary.text}"
        )

    def _split_unconsolidated(
        self, session: ProjectState,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split live session tail into archiveable prefix and retained recent suffix."""
        tail = list(session.messages[session.last_consolidated:])
        if not tail:
            return [], []

        probe = ProjectState(
            key=session.key,
            messages=tail.copy(),
            created_at=session.created_at,
            updated_at=session.updated_at,
            metadata={},
            last_consolidated=0,
        )
        probe.retain_recent_legal_suffix(self._RECENT_SUFFIX_MESSAGES)
        kept = probe.messages
        cut = len(tail) - len(kept)
        return tail[:cut], kept

    def _record_failure(self, key: str) -> bool:
        count = self._failure_counts.get(key, 0) + 1
        self._failure_counts[key] = count
        return count >= self._MAX_CONSECUTIVE_FAILURES

    def _reset_failures(self, key: str) -> None:
        self._failure_counts.pop(key, None)

    def check_expired(self, schedule_background: Callable[[Coroutine], None]) -> None:
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            if not key or key in self._archiving:
                continue
            if self._failure_counts.get(key, 0) >= self._MAX_CONSECUTIVE_FAILURES:
                continue
            if self._is_expired(info.get("updated_at")):
                self._archiving.add(key)
                logger.debug("Auto-compact: scheduling archival for {} (idle > {} min)", key, self._ttl)
                schedule_background(self._archive(key))

    async def _archive(self, key: str) -> None:
        try:
            self.sessions.invalidate(key)
            session = self.sessions.get_or_create(key)
            archive_msgs, kept_msgs = self._split_unconsolidated(session)
            if not archive_msgs and not kept_msgs:
                logger.debug("Auto-compact: skipping {}, no un-consolidated messages", key)
                session.updated_at = datetime.now()
                self.sessions.save(session)
                self._reset_failures(key)
                return

            last_active = session.updated_at
            summary = ""
            if archive_msgs:
                summary = await self.consolidator.archive(archive_msgs) or ""
            if summary and summary != "(nothing)":
                checkpoint = CheckpointSummary(text=summary, last_active=last_active)
                self._summaries[key] = checkpoint
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat(),
                    "source": checkpoint.source,
                }
            session.messages = kept_msgs
            session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)
            self._reset_failures(key)
            logger.info(
                "Auto-compact: archived {} (archived={}, kept={}, summary={})",
                key,
                len(archive_msgs),
                len(kept_msgs),
                bool(summary),
            )
        except Exception:
            tripped = self._record_failure(key)
            if tripped:
                logger.error("Auto-compact fuse tripped for {} after {} failures", key, self._failure_counts[key])
            logger.exception("Auto-compact: failed for {}", key)
        finally:
            self._archiving.discard(key)

    def prepare_session(self, session: ProjectState, key: str) -> tuple[ProjectState, CheckpointSummary | None]:
        if key in self._archiving or self._is_expired(session.updated_at):
            logger.info("Auto-compact: reloading session {} (archiving={})", key, key in self._archiving)
            session = self.sessions.get_or_create(key)
        entry = self._summaries.pop(key, None)
        if entry:
            session.metadata.pop("_last_summary", None)
            return session, entry
        if "_last_summary" in session.metadata:
            meta = session.metadata.pop("_last_summary")
            self.sessions.save(session)
            return session, CheckpointSummary(
                text=meta["text"],
                last_active=datetime.fromisoformat(meta["last_active"]),
                source=str(meta.get("source") or "idle"),
            )
        return session, None
