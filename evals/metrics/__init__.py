"""Metric suite, grouped by the layer each one judges.

Seven modules, one per axis. Nothing here imports the memory system: metrics
judge behaviour from the outside using primitives the harness extracts, which
keeps them honest (a metric that reuses production scoring cannot detect its
own target's bugs) and testable offline.

Final set
---------
- ``retrieval``     — recall@k, precision@k, MRR
- ``answer``        — exact_match, semantic_similarity, abstention_report
- ``temporal``      — temporal_gate_lift
- ``usefulness``    — memory_lift (with paired counts, cost normalisation,
                      ceiling and sham guards)
- ``consolidation`` — duplicate_rate sampled at every write boundary, i.e. a
                      running history plus its trend
"""

from . import (
    answer,
    consolidation,
    retrieval,
    temporal,
    usefulness,
)

__all__ = [
    "answer",
    "consolidation",
    "retrieval",
    "temporal",
    "usefulness",
]
