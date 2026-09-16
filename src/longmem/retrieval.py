"""Read path: RRF fusion, hard filtering, additive gated scoring, retrieve.

Locked model: staleness is an additive, temporally-gated penalty — never a
multiplicative kill-switch. All decay math goes through ``lifecycle`` (this
module holds no local exponential); the lifecycle-only conf-times-decay
helper is for expiry/archival ranking and must never be used as a
retrieval score here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from . import lifecycle
from .config import Settings
from .llm import LLMClient, Reranker, year_heuristic
from .schemas import Memory, ScoredMemory, StructuredQuery
from .store import DurableMemoryStore

_RECENCY_DAYS = 30.0
_TEMPORAL_OUTSIDE = 0.2


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fuse_and_filter(hits, sq, settings, include_expired):
    """Rank both channels, fuse, and apply the hard gate."""
    by_id = {m.memory_id: m for m, _, _ in hits}
    vec_sorted = sorted(hits, key=lambda t: (-t[1], t[0].memory_id))
    lex_sorted = sorted(hits, key=lambda t: (-t[2], t[0].memory_id))
    v_rank = {m.memory_id: i + 1 for i, (m, _, _) in enumerate(vec_sorted)}
    l_rank = {m.memory_id: i + 1 for i, (m, _, _) in enumerate(lex_sorted)}
    fused = rrf_fuse(v_rank, l_rank, k=settings.retrieval.rrf_k)
    kept = [
        mid
        for mid in fused
        if mid in by_id and hard_filter(by_id[mid], sq, include_expired)
    ]
    return by_id, fused, kept


def rrf_fuse(
    v_rank: Dict[str, int], l_rank: Dict[str, int], k: int = 60
) -> Dict[str, float]:
    """Reciprocal-rank fuse two 1-based rank maps; missing channel adds 0."""
    fused: Dict[str, float] = {}
    for mid, rank in v_rank.items():
        fused[mid] = fused.get(mid, 0.0) + 1.0 / (k + rank)
    for mid, rank in l_rank.items():
        fused[mid] = fused.get(mid, 0.0) + 1.0 / (k + rank)
    return fused


def hard_filter(
    m: Memory, q: StructuredQuery, include_expired: bool = False
) -> bool:
    """Hard gate before scoring: expired/type/scope; superseded is kept."""
    status = m.status.value if hasattr(m.status, "value") else str(m.status)
    if status == "expired" and not include_expired:
        return False
    if q.memory_types:
        wanted = {t.value if hasattr(t, "value") else str(t) for t in q.memory_types}
        mtype = m.type.value if hasattr(m.type, "value") else str(m.type)
        if mtype not in wanted:
            return False
    if q.scope is not None:
        q_scope = q.scope.type
        q_val = q_scope.value if hasattr(q_scope, "value") else str(q_scope)
        m_val = (
            m.scope_type.value
            if hasattr(m.scope_type, "value")
            else str(m.scope_type)
        )
        if m_val != q_val:
            return False
        if q.scope.id is not None and m.scope_id != q.scope.id:
            return False
    return True


def _scope_is_default(sq: StructuredQuery) -> bool:
    """True when the query carries no scope signal (bare user default).

    ``scope=None`` means already unconstrained (no filtering happened);
    an explicit type or id means deliberate targeting that stays strict.
    """
    if sq.scope is None:
        return False
    scope_type = sq.scope.type
    type_val = scope_type.value if hasattr(scope_type, "value") else str(scope_type)
    return type_val == "user" and sq.scope.id is None


def _temporal_score(m: Memory, question_time: datetime | None) -> float:
    if question_time is None:
        return 1.0
    if m.valid_from is not None and m.valid_until is not None:
        if m.valid_from <= question_time <= m.valid_until:
            return 1.0
        return _TEMPORAL_OUTSIDE
    if m.valid_from is not None:
        if question_time >= m.valid_from:
            return 1.0
        return _TEMPORAL_OUTSIDE
    if m.valid_until is not None:
        if question_time <= m.valid_until:
            return 1.0
        return _TEMPORAL_OUTSIDE
    return 1.0


def score_candidates(
    fused: Dict[str, float],
    by_id: Dict[str, Memory],
    q: StructuredQuery,
    settings: Settings,
    now: datetime | None = None,
    question_time: datetime | None = None,
) -> List[ScoredMemory]:
    """Additive gated scoring, desc-sorted; decay via ``lifecycle`` only."""
    moment = now or _utcnow_naive()
    lam = settings.lifecycle.decay_lambda
    out: List[ScoredMemory] = []
    for mid, mem in by_id.items():
        rel = fused.get(mid, 0.0)
        inact = lifecycle.inactive_days(
            moment, mem.last_retrieved_at, mem.last_updated_at, mem.created_at
        )
        decay = lifecycle.decay_factor(inact, lam)
        penalty = lifecycle.decay_penalty(inact, lam)
        w_decay_eff = settings.scoring.w_decay * (1.0 - q.temporal_confidence)
        recency = 1.0 / (1.0 + inact / _RECENCY_DAYS)
        temporal = _temporal_score(mem, question_time)
        final = (
            settings.scoring.w_relevance * rel
            + settings.scoring.w_confidence * mem.confidence
            + settings.scoring.w_temporal * temporal
            + settings.scoring.w_recency * recency
            - w_decay_eff * penalty
        )
        out.append(
            ScoredMemory(
                memory=mem,
                scores={
                    "relevance": rel,
                    "confidence": mem.confidence,
                    "temporal": temporal,
                    "recency": recency,
                    "decay_factor": decay,
                    "decay_penalty": penalty,
                    "w_decay_eff": w_decay_eff,
                },
                final_score=final,
                temporal_confidence=q.temporal_confidence,
            )
        )
    out.sort(key=lambda s: (-s.final_score, s.memory.memory_id))
    return out


def retrieve(
    query: str,
    durable: DurableMemoryStore,
    llm: LLMClient,
    reranker: Reranker | None,
    settings: Settings,
    question_time: datetime | None = None,
    include_expired: bool = False,
    now: datetime | None = None,
) -> List[ScoredMemory]:
    """Full read path: understand -> hybrid -> RRF -> filter -> score -> rerank.

    LOCKED: an explicitly-passed ``question_time`` never overrides a
    successful ``understand`` — model output wins for the gate, and the
    1.0 fallback applies only when ``understand`` itself fails. Forcing
    the gate from the caller would erase the calibrated model signal that
    ablations compare against.

    Misconfiguration note: ``load_settings`` rejects ``reranker_top_n``
    below ``final_memory_count`` (except the 0 bypass), but Settings built
    by hand bypass that check — then retrieval silently returns at most
    ``reranker_top_n`` rows. Keep the invariant at construction.

    Type fallback: writer and reader LLM calls share no ontology, so a
    hard AND on their type guesses fails silently (empty list, no signal).
    When the gate empties a typed query, retrieval retries once without
    the type dimension. Scope and status stay hard — only the fuzzy
    ontology degrades.

    Thin-result backfill: when a typed query keeps some rows but fewer
    than requested, the shortfall is filled from a typeless re-search,
    APPENDED after the same-type rows regardless of score. Cross-type
    rows can fill empty seats but never outrank same-type rows, so the
    hard gate keeps its precision meaning while recall degrades loudly
    (visible rows) instead of silently (empty list).

    Scope backfill (last tier): writer and reader also share no scope
    ontology (a personal chat judged project-scoped is invisible to the
    default user query). When seats remain empty, one scopeless re-search
    appends cross-scope rows — again ranked after, never promoted. Only
    DEFAULT scopes (user with no id: absence of signal) degrade this way;
    explicit scopes stay strict, and consolidation targeting is untouched.
    Single-tenant assumption: cross-scope rows surface only into
    otherwise-empty seats; multi-tenant deployments should scope queries
    with ids instead of relying on the default.
    """
    if not query or not query.strip():
        return []
    try:
        sq = llm.understand(query)
    except Exception:
        fallback_tc = 1.0 if question_time is not None else year_heuristic(query)
        sq = StructuredQuery(
            rewritten_query=query,
            keywords=query.split()[:8],
            temporal_confidence=fallback_tc,
        )
    statuses = ("active", "superseded")
    if include_expired:
        statuses = ("active", "superseded", "expired")
    hits = durable.hybrid_search(
        sq,
        limit=settings.retrieval.candidate_count,
        include_expired=include_expired,
        statuses=statuses,
    )
    if not hits and sq.memory_types:
        # Search-level type cut (both backends filter types up front):
        # retry once without the type dimension. Scope/status stay hard.
        sq = sq.model_copy(update={"memory_types": []})
        hits = durable.hybrid_search(
            sq,
            limit=settings.retrieval.candidate_count,
            include_expired=include_expired,
            statuses=statuses,
        )
    if not hits and _scope_is_default(sq):
        # Nothing same-scope matched at all: retry once fully open
        # (typeless + scopeless). Explicit scopes return [] instead.
        sq = sq.model_copy(update={"memory_types": [], "scope": None})
        hits = durable.hybrid_search(
            sq,
            limit=settings.retrieval.candidate_count,
            include_expired=include_expired,
            statuses=statuses,
        )
    if not hits:
        return []
    by_id, fused, kept = _fuse_and_filter(hits, sq, settings, include_expired)
    if not kept and sq.memory_types:
        # Post-filter cut (backends returning untyped rows): same retry.
        # Fires at most once per call — the retry already cleared types.
        sq = sq.model_copy(update={"memory_types": []})
        hits = durable.hybrid_search(
            sq,
            limit=settings.retrieval.candidate_count,
            include_expired=include_expired,
            statuses=statuses,
        )
        by_id, fused, kept = (
            _fuse_and_filter(hits, sq, settings, include_expired) if hits else ({}, {}, [])
        )
    scored = score_candidates(
        {mid: fused[mid] for mid in kept},
        {mid: by_id[mid] for mid in kept},
        sq,
        settings,
        now=now,
        question_time=question_time,
    )
    if reranker is None or settings.retrieval.reranker_top_n <= 0:
        final_same = scored[: settings.retrieval.final_memory_count]
    else:
        pre = scored[: settings.retrieval.reranker_top_n]
        want = min(len(pre), settings.retrieval.final_memory_count * 2)
        reranked = reranker.rerank(
            sq.rewritten_query, [s.memory for s in pre], top_n=want
        )
        pos = {m.memory_id: i for i, m in enumerate(reranked)}
        orig = {s.memory.memory_id: i for i, s in enumerate(pre)}
        ordered = sorted(
            pre,
            key=lambda s: (0, pos[s.memory.memory_id])
            if s.memory.memory_id in pos
            else (1, orig[s.memory.memory_id]),
        )
        final_same = ordered[: settings.retrieval.final_memory_count]
    final = final_same
    if sq.memory_types and len(final_same) < settings.retrieval.final_memory_count:
        open_sq = sq.model_copy(update={"memory_types": []})
        open_hits = durable.hybrid_search(
            open_sq,
            limit=settings.retrieval.candidate_count,
            include_expired=include_expired,
            statuses=statuses,
        )
        if open_hits:
            open_by, open_fused, open_kept = _fuse_and_filter(
                open_hits, open_sq, settings, include_expired
            )
            seen = {s.memory.memory_id for s in final_same}
            fresh = [mid for mid in open_kept if mid not in seen]
            if fresh:
                extra = score_candidates(
                    {mid: open_fused[mid] for mid in fresh},
                    {mid: open_by[mid] for mid in fresh},
                    open_sq,
                    settings,
                    now=now,
                    question_time=question_time,
                )
                final = (final_same + extra)[: settings.retrieval.final_memory_count]
    if _scope_is_default(sq) and len(final) < settings.retrieval.final_memory_count:
        # Scope backfill, last tier: same default-scope query, one fully
        # open re-search (scope=None is honored as "no constraint" by both
        # backends and hard_filter). Rank-preserving append only.
        noscope_sq = sq.model_copy(update={"memory_types": [], "scope": None})
        noscope_hits = durable.hybrid_search(
            noscope_sq,
            limit=settings.retrieval.candidate_count,
            include_expired=include_expired,
            statuses=statuses,
        )
        if noscope_hits:
            noscope_by, noscope_fused, noscope_kept = _fuse_and_filter(
                noscope_hits, noscope_sq, settings, include_expired
            )
            seen = {s.memory.memory_id for s in final}
            fresh = [mid for mid in noscope_kept if mid not in seen]
            if fresh:
                extra = score_candidates(
                    {mid: noscope_fused[mid] for mid in fresh},
                    {mid: noscope_by[mid] for mid in fresh},
                    noscope_sq,
                    settings,
                    now=now,
                    question_time=question_time,
                )
                final = (final + extra)[: settings.retrieval.final_memory_count]
    durable.touch_retrieved([s.memory.memory_id for s in final])
    return final
