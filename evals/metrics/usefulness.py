"""Usefulness: does the memory layer actually make answers better, at what cost?

Every other axis measures the system against itself. This one measures the
claim the system exists to make — that answering *with* memory beats answering
*without* it — and it is the only axis that can come back negative.

The headline number is ``memory_lift``. It comes with two guards that keep it
from being misread, because a bare lift is easy to trust for the wrong reason:

- ``ceiling_check`` flags an eval set the model already answers without memory,
  where a low lift is a *dataset* defect rather than a system one.
- ``sham_scores`` controls for the fact that adding *any* text to a prompt
  changes behaviour, by comparing against unrelated memories of similar size.

Three rules make the number trustworthy:

1. **Paired, one variable.** The two score series must come from identical
   questions, model, and decoding; any other difference would be measured here
   instead of memory.
2. **Wins and losses, not just the mean.** A mean lift of zero can hide a
   system that helps five questions and hurts five, and regression is the
   honest cost of using memory — extra context distracts, stale facts mislead.
3. **Cost-adjusted.** Long-context lift is bought with tokens, so the
   interesting quantity is lift per extra token. Without it, "memory helps" is
   a claim about prompt length, not about memory.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

_EPS = 1e-9


def memory_lift(
    with_scores: Sequence[float], without_scores: Sequence[float]
) -> float:
    """Mean difference in score between the two conditions.

    Catches: the headline question — is memory worth it at all. Signed, so a
    harmful memory layer reads negative rather than being hidden by abs().
    """
    if len(with_scores) != len(without_scores):
        raise ValueError("with and without scores must be paired")
    if not with_scores:
        return 0.0
    paired = [
        float(a) - float(b) for a, b in zip(with_scores, without_scores)
    ]
    return sum(paired) / len(paired)


def paired_comparison(
    with_scores: Sequence[float], without_scores: Sequence[float]
) -> Dict[str, float]:
    """Wins, losses, ties, and lift over paired questions.

    Catches: distribution blindness in the mean. Two systems with identical
    lift are very different if one is uniformly +0.1 and the other is +0.5 on
    half the questions and -0.5 on the rest — the second is actively harmful on
    part of the workload. ``net_gain`` (wins minus losses) is a robustness
    readout that a mean cannot provide.
    """
    if len(with_scores) != len(without_scores):
        raise ValueError("with and without scores must be paired")
    n = len(with_scores)
    if n == 0:
        return {
            "n": 0.0,
            "mean_with_memory": 0.0,
            "mean_without_memory": 0.0,
            "absolute_lift": 0.0,
            "relative_lift": 0.0,
            "relative_lift_defined": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "ties": 0.0,
            "win_rate": 0.0,
            "regression_rate": 0.0,
            "net_gain": 0.0,
        }
    wins = losses = ties = 0
    for a, b in zip(with_scores, without_scores):
        delta = float(a) - float(b)
        if delta > _EPS:
            wins += 1
        elif delta < -_EPS:
            losses += 1
        else:
            ties += 1
    mean_with = sum(float(a) for a in with_scores) / n
    mean_without = sum(float(b) for b in without_scores) / n
    absolute = mean_with - mean_without
    defined = mean_without > _EPS
    return {
        "n": float(n),
        "mean_with_memory": mean_with,
        "mean_without_memory": mean_without,
        "absolute_lift": absolute,
        "relative_lift": (absolute / mean_without) if defined else 0.0,
        "relative_lift_defined": 1.0 if defined else 0.0,
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
        "win_rate": wins / n,
        "regression_rate": losses / n,
        "net_gain": float(wins - losses),
    }


def usefulness_per_token(
    absolute_lift: float,
    tokens_with_memory: int,
    tokens_without_memory: int,
) -> Optional[float]:
    """Lift per 1000 extra prompt tokens; None when memory costs nothing extra.

    Catches: the unfair comparison. Long-context retrieval scores well by
    spending tokens; if memory's lift is bought with the same token budget, the
    win is not memory's. None (rather than 0.0 or inf) when the memory
    condition is not more expensive, because the ratio is genuinely undefined
    there and silently returning 0 would read as "no benefit per token".
    """
    extra = int(tokens_with_memory) - int(tokens_without_memory)
    if extra <= 0:
        return None
    return float(absolute_lift) / (extra / 1000.0)


def ceiling_check(
    without_scores: Sequence[float], threshold: float = 0.8
) -> Dict[str, object]:
    """Is the eval set even capable of showing a memory benefit?

    Catches: the uninformative benchmark. If the model already answers most
    questions with no memory (general-knowledge questions mislabelled as memory
    questions), lift will be near zero and the result says nothing about the
    memory layer. Flags it instead of reporting a misleadingly small lift.
    """
    if not without_scores:
        return {"informative": False, "mean_without_memory": 0.0, "n": 0.0}
    mean_without = sum(float(s) for s in without_scores) / len(without_scores)
    return {
        "informative": mean_without < threshold,
        "mean_without_memory": mean_without,
        "threshold": float(threshold),
        "n": float(len(without_scores)),
    }


def usefulness_report(
    with_scores: Sequence[float],
    without_scores: Sequence[float],
    tokens_with_memory: Optional[int] = None,
    tokens_without_memory: Optional[int] = None,
    sham_scores: Optional[Sequence[float]] = None,
    ceiling_threshold: float = 0.8,
) -> Dict[str, object]:
    """The full usefulness report for one run.

    Bundles the paired comparison with the cost normalisation, the
    informativeness guard, and (when supplied) the sham-control lift. Kept as
    one dict so a report row cannot show lift without also showing what it cost
    and whether the question set could detect it.
    """
    report: Dict[str, object] = dict(
        paired_comparison(with_scores, without_scores)
    )
    report["baseline_check"] = ceiling_check(without_scores, ceiling_threshold)
    if tokens_with_memory is not None and tokens_without_memory is not None:
        report["lift_per_1k_extra_tokens"] = usefulness_per_token(
            float(report["absolute_lift"]),
            tokens_with_memory,
            tokens_without_memory,
        )
        report["extra_tokens"] = float(
            int(tokens_with_memory) - int(tokens_without_memory)
        )
    if sham_scores is not None:
        if len(sham_scores) != len(with_scores):
            raise ValueError("sham scores must be paired with with_scores")
        report["lift_vs_sham"] = memory_lift(with_scores, sham_scores)
        report["mean_sham"] = (
            sum(float(s) for s in sham_scores) / len(sham_scores)
            if sham_scores
            else 0.0
        )
    return report
