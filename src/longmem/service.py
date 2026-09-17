"""Service wiring: working + durable stores, LLM, reranker, Settings."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from .config import Settings
from .context import assemble
from .formation import (
    consolidate,
    format_formation_event,
    recent_turns,
    should_trigger,
)
from .llm import Embedder, LLMClient, Reranker
from .retrieval import retrieve
from .schemas import MemoryContext, Operation, ScoredMemory, Turn
from .store import DurableMemoryStore, WorkingMemoryStore


class MemoryService:
    """Thin wire-up over the Phase 02/03 primitives (mocks in this phase)."""

    def __init__(
        self,
        durable: DurableMemoryStore,
        working: WorkingMemoryStore,
        llm: LLMClient,
        reranker: Reranker | None,
        settings: Settings,
        embedder: Embedder | None = None,
    ) -> None:
        self.durable = durable
        self.working = working
        self.llm = llm
        self.reranker = reranker
        self.settings = settings
        self.embedder = embedder
        self._buffers: Dict[str, List[Turn]] = {}

    def ingest_turn(
        self, session_id: str, turn: Turn | dict
    ) -> Operation | None:
        """Append one turn, then run judge+consolidate on N-boundaries.

        The boundary follows the working store count (the persistent
        truth), not the in-process buffer length, so a restart over a
        persisted working store keeps triggering on schedule. The buffer
        only supplies judge context; after a restart it holds post-restart
        turns until callers re-feed history (full rehydrate needs turn
        retrieval on the working store — a Phase 04 concern).
        """
        item = turn if isinstance(turn, Turn) else Turn(**turn)
        self.working.append_turn(session_id, item)
        buf = self._buffers.setdefault(session_id, [])
        buf.append(item)
        if not should_trigger(self.working.turn_count(session_id), self.settings):
            return None
        try:
            candidate = self.llm.judge(recent_turns(buf, self.settings))
        except Exception as exc:
            print(format_formation_event(session_id, "ERROR", None, exc))
            return "ERROR"
        op = consolidate(
            candidate,
            session_id,
            self.durable,
            self.llm,
            self.settings,
            self.embedder,
            message_idx=self.working.turn_count(session_id) - 1,
        )
        print(format_formation_event(session_id, op, candidate))
        return op

    def flush_session(self, session_id: str) -> Operation | None:
        """Judge the trailing window when a session ends mid-interval.

        ``ingest_session`` calls this itself. Callers that drive
        ``ingest_turn`` one at a time must call it too, or the last
        ``turns % N`` turns of every session are never judged — a silent
        coverage hole, because nothing downstream can tell that they were
        skipped. Returns None when there is no tail to judge.
        """
        buf = self._buffers.get(session_id, [])
        n = self.settings.memory.write_interval_turns
        if not (len(buf) > n and len(buf) % n != 0):
            return None
        try:
            candidate = self.llm.judge(recent_turns(buf, self.settings))
        except Exception as exc:
            print(format_formation_event(session_id, "ERROR", None, exc))
            return "ERROR"
        op = consolidate(
            candidate,
            session_id,
            self.durable,
            self.llm,
            self.settings,
            self.embedder,
            message_idx=len(buf) - 1,
        )
        print(format_formation_event(session_id, op, candidate))
        return op

    def ingest_session(
        self, session_id: str, turns: List[Turn | dict]
    ) -> List[Operation | None]:
        """Sequential per-turn ingest, plus a tail flush.

        Per-turn N-checks leave a session tail (``len % N != 0``) unjudged
        forever — and answers often live there. The closing judge over the
        trailing window costs one extra call per long session; short sessions
        that never triggered stay free.
        """
        ops: List[Operation | None] = [
            self.ingest_turn(session_id, t) for t in turns
        ]
        tail = self.flush_session(session_id)
        if tail is not None:
            ops.append(tail)
        return ops

    def answer_context(
        self,
        query: str,
        question_time: datetime | None = None,
        include_expired: bool = False,
    ) -> tuple[MemoryContext, List[ScoredMemory]]:
        """Retrieve Top-K and assemble context; returns both for metrics."""
        scored = retrieve(
            query,
            self.durable,
            self.llm,
            self.reranker,
            self.settings,
            question_time=question_time,
            include_expired=include_expired,
        )
        return assemble(scored, self.durable), scored

    def answer(
        self,
        query: str,
        question_time: datetime | None = None,
        include_expired: bool = False,
    ) -> tuple[str, MemoryContext, List[ScoredMemory]]:
        """Full QA loop: retrieve, assemble, then generate the answer text."""
        from .llm import build_answer_prompt

        ctx, scored = self.answer_context(
            query,
            question_time=question_time,
            include_expired=include_expired,
        )
        chat_text = getattr(self.llm, "chat_text", None)
        if chat_text is None:
            raise TypeError(
                f"{type(self.llm).__name__} cannot generate answers "
                "(no chat_text); use a full LLM client"
            )
        return (
            chat_text(build_answer_prompt(query, ctx.to_prompt())),
            ctx,
            scored,
        )
