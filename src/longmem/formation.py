"""Write path: trigger, enrich, consolidate. LLM judges; code decides the rest."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import List

from .config import Settings
from .llm import Embedder, LLMClient
from .schemas import (
    Memory,
    MemoryCandidate,
    MemorySource,
    MemoryStatus,
    Operation,
    StructuredQuery,
    Turn,
)
from .store import DurableMemoryStore

_CONSOLIDATE_SEARCH_LIMIT = 5
_EXCERPT_CHARS = 300
_KEYWORD_TERMS = 8


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def should_trigger(turn_count: int, settings: Settings) -> bool:
    """True exactly on every N-turn boundary (N from Settings)."""
    n = settings.memory.write_interval_turns
    return turn_count > 0 and turn_count % n == 0


def enrich(
    candidate: MemoryCandidate,
    session_id: str,
    message_idx: int | None,
    settings: Settings,
) -> tuple[Memory, MemorySource]:
    """Attach IDs, timestamps, Settings-driven confidence, and provenance.

    No LLM, no DB. Confidence comes from Settings based on whether the
    basis starts with explicit; excerpt is a plain prefix of the content.
    """
    now = _utcnow_naive()
    basis = candidate.confidence_basis or ""
    if basis.lower().startswith("explicit"):
        confidence = settings.scoring.confidence_explicit
    else:
        confidence = settings.scoring.confidence_default
    memory_id = f"mem_{uuid.uuid4().hex[:12]}"
    memory = Memory(
        memory_id=memory_id,
        content=candidate.content,
        embedding=None,
        type=candidate.type,
        scope_type=candidate.scope.type,
        scope_id=candidate.scope.id,
        confidence=confidence,
        created_at=now,
        valid_from=candidate.valid_from or now,
        valid_until=None,
        last_retrieved_at=None,
        last_updated_at=None,
        status=MemoryStatus.active,
        version=1,
        supersedes_id=None,
    )
    source = MemorySource(
        memory_id=memory_id,
        session_id=session_id,
        message_idx=message_idx,
        excerpt=candidate.content[:_EXCERPT_CHARS],
    )
    return memory, source


def consolidate(
    candidate: MemoryCandidate,
    session_id: str,
    durable: DurableMemoryStore,
    llm: LLMClient,
    settings: Settings,
    embedder: Embedder | None = None,
    message_idx: int | None = None,
) -> Operation:
    """Fold one candidate into the store; return the op taken.

    LOCKED: consolidation is scope-siloed — the related search uses the
    candidate's own scope with the exact-match rule, so a user-scope
    candidate never sees or merges a duplicate global-scope memory (and
    vice versa). This mirrors the retrieval hard-filter rule; cross-scope
    duplicates are kept as separate memories by design.
    MERGE keeps the existing row (content and embedding unchanged — the
    fact itself did not change) and only refreshes confidence/version plus
    provenance; a repeated merge from the same session is idempotent on
    sources (first row kept) so long sessions hitting two N-boundaries
    cannot violate the source PK.
    """
    if not candidate.should_store:
        return "DISCARD"

    query = StructuredQuery(
        rewritten_query=candidate.content,
        scope=candidate.scope,
        keywords=candidate.content.split()[:_KEYWORD_TERMS],
    )
    hits = durable.hybrid_search(
        query, limit=_CONSOLIDATE_SEARCH_LIMIT, statuses=("active",)
    )
    related: List[Memory] = [m for m, _, _ in hits]
    by_id = {m.memory_id: m for m in related}

    memory, source = enrich(candidate, session_id, message_idx, settings)

    if not related:
        if embedder is not None:
            vecs = embedder.embed([memory.content])
            memory = memory.model_copy(update={"embedding": vecs[0]})
            source = source.model_copy(update={"memory_id": memory.memory_id})
        durable.insert(memory, [source])
        return "CREATE"

    try:
        relationship, related_id = llm.classify(candidate, related)
    except Exception:
        # Unreadable classifier output: keep the new memory rather than
        # lose data or follow an unknown id (caller retries/discards).
        relationship, related_id = "no_relation", None

    if relationship == "similar" and related_id in by_id:
        target = by_id[related_id]
        merged_confidence = min(
            settings.scoring.confidence_max,
            target.confidence + settings.scoring.merge_bump,
        )
        evidence = source.model_copy(update={"memory_id": target.memory_id})
        durable.merge(target.memory_id, evidence, merged_confidence)
        return "MERGE"

    if relationship == "contradiction" and related_id in by_id:
        if embedder is not None:
            vecs = embedder.embed([memory.content])
            memory = memory.model_copy(update={"embedding": vecs[0]})
            source = source.model_copy(update={"memory_id": memory.memory_id})
        durable.supersede(related_id, memory, [source])
        return "SUPERSEDE"

    if embedder is not None:
        vecs = embedder.embed([memory.content])
        memory = memory.model_copy(update={"embedding": vecs[0]})
        source = source.model_copy(update={"memory_id": memory.memory_id})
    durable.insert(memory, [source])
    return "CREATE"


def format_formation_event(
    session_id: str,
    op: Operation | str | None,
    candidate: MemoryCandidate | None = None,
    error: Exception | None = None,
) -> str:
    """One JSON audit line per judged boundary (pure, shared by callers).

    Exception paths log too (with the error, without content) so a dead
    boundary is visible instead of silent.
    """
    return json.dumps(
        {
            "event": "formation",
            "session_id": session_id,
            "op": op,
            "should_store": candidate.should_store if candidate is not None else False,
            "content": candidate.content[:120] if candidate is not None else "",
            "error": None
            if error is None
            else f"{type(error).__name__}: {str(error)[:200]}",
        }
    )


def recent_turns(turns: List[Turn], settings: Settings) -> List[Turn]:
    """Trailing window the judge actually sees (bounded prompts).

    Full-prefix judging grows prompts with session length until calls
    fail; the window keeps each verdict cheap and parseable while the
    durable store carries long-range history instead.
    """
    window = settings.memory.judge_window_turns
    return turns[-window:] if len(turns) > window else list(turns)


def maybe_judge_and_consolidate(
    turns: List[Turn],
    session_id: str,
    durable: DurableMemoryStore,
    llm: LLMClient,
    settings: Settings,
    embedder: Embedder | None = None,
) -> Operation | None:
    """Run judge+consolidate on N-turn boundaries; None off-boundary.

    An unreadable judge verdict discards (store untouched, never crash).
    Every boundary logs one JSON line with the distilled candidate, so a
    later question can be traced to exactly what the judge saw — without
    re-running paid calls. The judge sees the trailing window, never the
    full prefix; message_idx still points at the true global turn.
    """
    if not should_trigger(len(turns), settings):
        return None
    try:
        candidate = llm.judge(recent_turns(turns, settings))
    except Exception as exc:
        print(format_formation_event(session_id, "DISCARD", None, exc))
        return "DISCARD"
    op = consolidate(
        candidate,
        session_id,
        durable,
        llm,
        settings,
        embedder,
        message_idx=len(turns) - 1,
    )
    print(format_formation_event(session_id, op, candidate))
    return op
