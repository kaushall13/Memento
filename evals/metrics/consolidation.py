"""Consolidation drift: is the write path keeping up as a conversation grows?

A single duplicate count after a run answers the wrong question. What matters
is the *shape*: consolidation can look fine on a short session and fall behind
one that runs long, because the classifier only ever sees the candidates
retrieval hands it and the store keeps growing underneath. So the signal is a
history — the duplicate rate sampled at every write boundary — and the metric
that matters is whether that history is rising.

The write path already triggers every N turns (``write_interval_turns``), so
sampling there costs no extra scheduling: snapshot the store's contents at each
boundary and read the rate off it.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, List, Optional, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN.findall((text or "").lower()))


def jaccard(a: set, b: set) -> float:
    """Token-set Jaccard similarity (1.0 identical, 0.0 disjoint)."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def duplicate_rate(
    contents: Sequence[str],
    threshold: float = 0.85,
    window: Optional[int] = None,
) -> Dict[str, object]:
    """Share of stored memories that restate another one, at one checkpoint.

    Catches: the consolidation miss. MERGE and SUPERSEDE exist to keep one row
    per fact; memories that restate each other mean the relation classifier is
    missing them, and every context window pays for it in redundant tokens.

    ``window`` limits the comparison to the most recent N memories, which is
    what a monitoring run wants on a large store — pairwise comparison is
    quadratic, so an unbounded sweep at every boundary does not scale. Left
    ``None`` for eval-scale snapshots where full coverage is affordable.

    Returns the rate, a count, and the offending index pairs (index into
    ``contents``) so a failure can be inspected rather than just counted.
    """
    total = len(contents)
    if total == 0:
        return {"duplicate_rate": 0.0, "duplicate_items": 0, "pairs": [], "n": 0}

    start = 0
    if window is not None and window > 0 and total > window:
        start = total - window
    indices = list(range(start, total))
    token_sets = {i: _tokens(contents[i]) for i in indices}

    duplicates = set()
    pairs: List[str] = []
    for i, j in combinations(indices, 2):
        if jaccard(token_sets[i], token_sets[j]) >= threshold:
            duplicates.add(j)
            pairs.append(f"{i}:{j}")
    return {
        "duplicate_rate": len(duplicates) / total,
        "duplicate_items": len(duplicates),
        "pairs": pairs,
        "n": total,
    }


def duplicate_rate_history(
    snapshots: Sequence[Sequence[str]],
    threshold: float = 0.85,
    window: Optional[int] = None,
    turns: Optional[Sequence[int]] = None,
    include_pairs: bool = False,
) -> List[Dict[str, object]]:
    """Duplicate rate at each write boundary, oldest first.

    ``snapshots`` is the store's memory contents captured after each N-turn
    boundary. ``turns`` optionally labels each snapshot with the turn count it
    was taken at, so the history can be plotted against conversation length
    rather than sample index.

    Catches: the drift that a single end-of-run number averages away — a rate
    that is low early and climbing late is a different problem (consolidation
    falling behind) from one that is uniformly high (classifier miscalibrated),
    and only the history distinguishes them.

    ``include_pairs`` is off by default because a long history of quadratic
    pair lists is the one thing here that can grow unbounded.
    """
    if turns is not None and len(turns) != len(snapshots):
        raise ValueError("turns and snapshots must be parallel")
    history: List[Dict[str, object]] = []
    for index, contents in enumerate(snapshots):
        step = duplicate_rate(contents, threshold, window)
        entry: Dict[str, object] = {
            "sample": index,
            "duplicate_rate": step["duplicate_rate"],
            "duplicate_items": step["duplicate_items"],
            "n": step["n"],
        }
        if turns is not None:
            entry["turn"] = turns[index]
        if include_pairs:
            entry["pairs"] = step["pairs"]
        history.append(entry)
    return history


def duplicate_rate_trend(history: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Summarise a duplicate-rate history: where it started, and is it rising.

    Catches: the direction of the drift, which is the actionable part. ``delta``
    is the raw fact (last minus first); ``rising`` is the boolean a gate can be
    put on. ``peak`` is reported because a spike that recovers still cost
    context slots while it lasted.
    """
    rates = [float(entry["duplicate_rate"]) for entry in history]
    if not rates:
        return {
            "first": 0.0,
            "last": 0.0,
            "delta": 0.0,
            "mean": 0.0,
            "peak": 0.0,
            "rising": 0.0,
            "samples": 0.0,
        }
    first, last = rates[0], rates[-1]
    return {
        "first": first,
        "last": last,
        "delta": last - first,
        "mean": sum(rates) / len(rates),
        "peak": max(rates),
        "rising": 1.0 if last > first else 0.0,
        "samples": float(len(rates)),
    }
