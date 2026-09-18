"""Retrieval quality: did we find the right memories, ranked well?

Ranked id lists in, numbers out — no ``Memory`` objects, so the same metrics
score the fake store, Postgres, and any future backend, and the caller picks
the granularity (memory ids or session provenance ids).

Three metrics, because each is blind to something the others catch: recall
misses nothing but cannot see ordering or noise, precision sees the noise but
not the omission, and rank position is invisible to both.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def _unique(ids: Iterable[str]) -> List[str]:
    """Keep rank order, drop repeats (a repeated id is not a second hit)."""
    seen: Set[str] = set()
    out: List[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of gold ids present in the top-k.

    Catches: the retrieval miss — the evidence never came back. Everything
    downstream (reasoning, answering) is bounded by this, so a drop here
    explains most answer drops and is the first thing to check.
    """
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = set(_unique(retrieved)[:k])
    return len(top & gold_set) / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of the top-k slots that are gold.

    Catches: buying recall with noise. A system that pads its context with
    near-misses raises recall without helping the answer, and dilutes the
    attention the real evidence gets. Denominator is ``k`` rather than the
    number returned, so under-filled seats stay visible instead of being
    silently forgiven.
    """
    if k <= 0:
        return 0.0
    top = _unique(retrieved)[:k]
    if not top:
        return 0.0
    gold_set = set(gold)
    return sum(1 for i in top if i in gold_set) / k


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first gold item; 0.0 when no gold item was retrieved.

    Catches: the buried-evidence failure. Recall and precision are both
    position-blind — a hit at rank 8 counts the same as a hit at rank 1 — but
    the answer model attends most to the top of its context, so where the
    evidence lands changes whether it gets used.
    """
    gold_set = set(gold)
    for rank, mid in enumerate(_unique(retrieved), start=1):
        if mid in gold_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
) -> float:
    """Suite-level MRR: the mean of per-query reciprocal ranks.

    Catches: the same thing as ``reciprocal_rank`` but at report level. Kept as
    a separate function so a single query's RR is never confused with the
    aggregate — averaging single-query RR values is what makes MRR meaningful.
    """
    if not pairs:
        return 0.0
    return sum(reciprocal_rank(r, g) for r, g in pairs) / len(pairs)


def retrieval_report(
    retrieved: Sequence[str], gold: Sequence[str], k: int
) -> Dict[str, float]:
    """The three metrics for one query, keyed for a report row."""
    return {
        f"recall@{k}": recall_at_k(retrieved, gold, k),
        f"precision@{k}": precision_at_k(retrieved, gold, k),
        "mrr": reciprocal_rank(retrieved, gold),
    }
