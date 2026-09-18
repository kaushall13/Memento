"""Temporal metrics: is the temporal gate doing real work?

One metric, because the temporal axis in this system is a single mechanism: the
query's ``temporal_confidence`` scales how much staleness penalty applies, so
that a question about 2023 is not decided by which memory is freshest. That
mechanism is either earning its keep or it is dead weight, and the only way to
know is to force it on and off and compare.

The comparison is reported per question type because that is where the claim
lives: the gate should lift historical questions and leave current questions
alone. A single averaged lift could be produced by a gate that helps history
while *hurting* current-state questions, which would be a different system
altogether.
"""

from __future__ import annotations

from typing import Dict


def temporal_gate_lift(gated_metric: float, ungated_metric: float) -> float:
    """Change in a metric when the temporal gate is forced on versus off.

    Catches: dormant machinery. If forcing ``temporal_confidence`` to 1.0
    instead of letting the query decide moves nothing, either the mechanism is
    unwired or it has no effect — both of which make the design decision
    unfalsifiable. Signed, so a gate that backfires reads negative.
    """
    return float(gated_metric) - float(ungated_metric)


def temporal_gate_lift_by_type(
    historical_gated: float,
    historical_ungated: float,
    current_gated: float,
    current_ungated: float,
) -> Dict[str, float]:
    """Gate lift split into historical and current questions.

    Catches: the average that hides a trade. The gate is supposed to help
    historical questions and be neutral for current ones; a net-positive
    average is still a defect if it is bought by degrading current-state
    answers. Splitting the two makes that visible instead of netting it out.
    """
    return {
        "historical_lift": temporal_gate_lift(
            historical_gated, historical_ungated
        ),
        "current_lift": temporal_gate_lift(current_gated, current_ungated),
    }
