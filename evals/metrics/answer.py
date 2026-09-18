"""Answer quality: did the final text use the retrieved memory correctly, and
did it know when memory had nothing?

Three metrics, chosen to fail in different ways rather than to overlap:

- ``exact_match`` is the judge-free floor — no interpretation at all, so when
  it moves you know the system moved, not the scorer.
- ``semantic_similarity`` forgives paraphrase. Necessary because this system
  answers in its own words, so a lexical-only readout systematically
  under-reports a correct system.
- ``abstention_report`` scores the questions that have no answer in memory,
  where the right behaviour is to say so. Every other metric here is silent
  about those questions, which is how a confidently hallucinating system can
  look perfect.

Deliberately excluded: token-F1 and substring overlap. Both were tried and cut
— they add a middle grade between exact and semantic without isolating a
failure that either end misses.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Sequence

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]")

# "I have no information about that" phrasings. Narrow on purpose: a false
# positive would mark a real answer as an abstention and corrupt the
# hallucination rate, which is the metric this module exists to expose.
_ABSTENTION_PATTERNS = (
    r"\bno information\b",
    r"\bnot (?:mentioned|provided|available|in the (?:provided )?(?:chat|memories|context))\b",
    r"\b(?:don't|do not|doesn't|does not|cannot|can't|can not)\b[^.]{0,40}\b(?:have|find|know|determine|tell|recall|say)\b",
    r"\bthere (?:is|are) no\b",
    r"\bunable to\b",
    r"\bmemories do not\b",
    r"\bnot enough information\b",
)
_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), re.IGNORECASE)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Shared so every text metric agrees on what "the same words" means.
    """
    return _WS.sub(" ", _PUNCT.sub("", (s or "").lower())).strip()


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized equality between prediction and gold.

    Catches: nothing subtle — that is the point. Use it on questions whose
    answer is a literal value (a date, a shift, a count), where any deviation
    is a real failure and no judgement should be involved.
    """
    return bool(gold) and normalize_text(prediction) == normalize_text(gold)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors; 0.0 if either is degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(prediction: str, gold: str, embedder) -> float:
    """Cosine between prediction and gold embeddings.

    Catches: the correct-but-paraphrased answer that a lexical metric scores as
    wrong. This is the metric that keeps the report honest when the system is
    right without quoting.

    ``embedder`` is injected — any object with ``embed(list[str]) -> list[vec]``
    — so scoring stays offline-testable and no vendor is baked in.
    """
    if not prediction or not gold:
        return 0.0
    vectors = embedder.embed([prediction, gold])
    if len(vectors) != 2:
        return 0.0
    return cosine(vectors[0], vectors[1])


def is_abstention(text: str) -> bool:
    """Heuristic: does this text decline for lack of information?"""
    return bool(_ABSTENTION_RE.search(text or ""))


def abstention_report(
    predictions: Sequence[str], answerable: Sequence[bool]
) -> Dict[str, float]:
    """Score the questions where memory had (or lacked) the answer.

    ``answerable[i]`` says whether the evidence existed in memory for question
    i. Treating "abstained" as the predicted-positive class:

    - ``hallucination_rate`` — answered while the evidence was absent. The one
      failure no other metric in this suite can see, because those questions
      have no gold answer to compare against.
    - ``omission_rate`` — abstained while the evidence was present. The price
      of caution, and the mirror failure; a system tuned to never hallucinate
      usually pays here.
    - ``abstention_precision``/``abstention_recall`` — the same confusion as
      classification quality of the abstain decision.

    The two rates are never averaged together: hallucination and omission have
    different severities, and a single number would hide which one moved.
    """
    if len(predictions) != len(answerable):
        raise ValueError("predictions and answerable must be parallel")
    n = len(predictions)
    if n == 0:
        return {
            "hallucination_rate": 0.0,
            "omission_rate": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "n": 0.0,
        }

    abstained = [is_abstention(p) for p in predictions]
    unanswerable = [not flag for flag in answerable]

    # abstain-as-positive confusion
    true_positive = sum(1 for a, u in zip(abstained, unanswerable) if a and u)
    false_positive = sum(1 for a, u in zip(abstained, unanswerable) if a and not u)
    false_negative = sum(1 for a, u in zip(abstained, unanswerable) if not a and u)

    n_unanswerable = sum(1 for u in unanswerable if u)
    n_answerable = n - n_unanswerable
    n_abstained = sum(1 for a in abstained if a)

    return {
        "hallucination_rate": false_negative / n_unanswerable
        if n_unanswerable
        else 0.0,
        "omission_rate": false_positive / n_answerable if n_answerable else 0.0,
        "abstention_precision": true_positive / n_abstained if n_abstained else 0.0,
        "abstention_recall": true_positive / n_unanswerable
        if n_unanswerable
        else 0.0,
        "n": float(n),
    }
