"""Pure freshness math. Single source of truth for decay.

All datetimes are naive UTC. No imports from store/llm — pure by design.
"""

from __future__ import annotations

import math
from datetime import datetime


def activity_at(
    retr: datetime | None, upd: datetime | None
) -> datetime | None:
    """Most recent activity, or None when neither is present."""
    if retr is not None and upd is not None:
        return retr if retr >= upd else upd
    if retr is not None:
        return retr
    return upd


def inactive_days(
    now: datetime,
    retr: datetime | None,
    upd: datetime | None,
    created: datetime | None = None,
) -> float:
    """Days since last activity, floored at 0.

    Falls back to ``created`` so brand-new memories score 0 inactivity.
    When nothing is known, returns 0.0 (fresh) rather than infinity.
    """
    base = activity_at(retr, upd)
    if base is None:
        base = created
    if base is None:
        return 0.0
    delta = (now - base).total_seconds() / 86400
    return max(0.0, delta)


def decay_factor(days: float, lamb: float) -> float:
    """Exponential decay factor exp(-lamb * days). Only exp() in the repo.

    Negative ``days`` are floored to 0 (fresh) so the factor never exceeds 1;
    normal callers pass ``inactive_days`` output, which already floors.
    """
    if days < 0:
        days = 0.0
    return math.exp(-lamb * days)


def decay_penalty(days: float, lamb: float) -> float:
    """Staleness penalty 1 - decay. Fresh -> 0, stale -> 1."""
    return 1.0 - decay_factor(days, lamb)


def effective_score(conf: float, days: float, lamb: float) -> float:
    """conf * decay. LIFECYCLE-ONLY (expiry/archival ranking).

    Must NOT be used as the retrieval final_score; retrieval uses the
    additive gated formula in retrieval.py (Phase 03).
    """
    return conf * decay_factor(days, lamb)


def is_expired(days: float, expiry: int) -> bool:
    """Soft-expire gate. Caller flips status; nothing is deleted here."""
    return days >= expiry
