"""Write path: trigger, enrich, consolidate. LLM judges; code decides the rest."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import List

from .config import Settings
from .llm import Embedder, LLMClient, cosine
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

    Because MERGE never writes the candidate's wording, a "similar" verdict is
    only honoured when the candidate actually restates the target
    (``restatement_evidence``). If the classifier is wrong about that — two
    different assertions about the same subject — the merge is vetoed and the
    candidate is created as its own memory instead, so no information is lost
    to a mislabel. The veto is logged as its own event.
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
        evidence = restatement_evidence(candidate.content, target, embedder, settings)
        if evidence["is_restatement"]:
            merged_confidence = min(
                settings.scoring.confidence_max,
                target.confidence + settings.scoring.merge_bump,
            )
            merged_evidence = source.model_copy(
                update={"memory_id": target.memory_id}
            )
            durable.merge(target.memory_id, merged_evidence, merged_confidence)
            return "MERGE"
        print(
            json.dumps(
                {
                    "event": "merge_vetoed",
                    "session_id": session_id,
                    "target_id": target.memory_id,
                    "candidate": candidate.content[:120],
                    "rule": evidence["rule"],
                    "score": round(float(evidence["score"]), 4),
                    "threshold": evidence["threshold"],
                }
            )
        )

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


_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
        "with", "by", "is", "are", "was", "were", "be", "been", "it", "its",
        "this", "that", "these", "those", "i", "my", "me", "you", "your",
        "their", "they", "he", "she", "we", "our", "as", "from", "but", "if",
        "then", "than", "so", "do", "does", "did", "has", "have", "had",
        "not", "no", "will", "would", "can", "could", "about", "into", "over",
    }
)
_TERM_RE = re.compile(r"[a-z0-9]+")


def content_terms(text: str) -> set:
    """Significant words in a memory, used by the lexical fallback rule."""
    return {
        term for term in _TERM_RE.findall((text or "").lower())
        if term not in _STOPWORDS
    }


def lexical_containment(candidate: str, target: str) -> float:
    """Share of the candidate's content terms already present in the target.

    Catches the same failure as the embedding rule but needs no model: 1.0
    means every word the candidate says is already in the target (a restatement),
    low means it is asserting something the target does not.
    """
    terms = content_terms(candidate)
    if not terms:
        return 1.0
    return len(terms & content_terms(target)) / len(terms)


def restatement_evidence(
    candidate_content: str,
    target: "Memory",
    embedder: Embedder | None,
    settings: Settings,
) -> dict:
    """Decide whether MERGE would preserve everything the candidate says.
    MERGE keeps the target row's wording and only raises its confidence, so a
    candidate that is *not* a restatement loses its content silently. The
    classifier is an LLM and can label same-topic-different-assertion pairs as
    ``similar``; this is the deterministic check that keeps that mistake from
    deleting information.

    Embedding cosine when an embedder is available (measured: paraphrases
    0.91-1.00, different assertions about the same subject 0.70-0.78), lexical
    containment otherwise. Both are deliberately biased toward refusing the
    merge — a duplicate memory is recoverable, deleted content is not.
    """
    scores = settings.scoring
    if embedder is not None:
        vectors = embedder.embed([candidate_content, target.content])
        if len(vectors) == 2:
            similarity = cosine(vectors[0], vectors[1])
            return {
                "rule": "embedding",
                "score": similarity,
                "threshold": scores.merge_similarity_min,
                "is_restatement": similarity >= scores.merge_similarity_min,
            }
    overlap = lexical_containment(candidate_content, target.content)
    return {
        "rule": "lexical",
        "score": overlap,
        "threshold": scores.merge_overlap_min,
        "is_restatement": overlap >= scores.merge_overlap_min,
    }


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

    An unreadable judge verdict returns ERROR (store untouched, never crash).
    That is deliberately distinct from DISCARD: "the judge said nothing here is
    worth keeping" and "the judge could not be read" are different events, and
    collapsing them hid real losses behind what looked like a decision.

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
        print(format_formation_event(session_id, "ERROR", None, exc))
        return "ERROR"
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
