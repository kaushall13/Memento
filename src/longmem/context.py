"""Context assembly: ranked memories -> active/historical/evidence. No LLM."""

from __future__ import annotations

from typing import List

from .schemas import MemoryContext, MemoryStatus, ScoredMemory
from .store import DurableMemoryStore


def assemble(
    scored: List[ScoredMemory], durable: DurableMemoryStore
) -> MemoryContext:
    """Split ranked memories; evidence only for memories in context.

    Expired rows appear in neither list, and their sources are excluded
    from evidence as well (garbage-in cannot leak provenance).
    """
    active = [
        s.memory for s in scored if s.memory.status == MemoryStatus.active
    ]
    historical = [
        s.memory for s in scored if s.memory.status == MemoryStatus.superseded
    ]
    kept_ids = [m.memory_id for m in active + historical]
    evidence = durable.fetch_sources(kept_ids)
    return MemoryContext(active=active, historical=historical, evidence=evidence)
