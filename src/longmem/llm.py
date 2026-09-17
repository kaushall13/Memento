"""LLM contracts, pure prompt builders/parsers, and offline mocks.

Pure functions (build_*/parse_*) are testable without any API key.
Real API clients are deferred to Phase 04; Phases 02-03 run on the mocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime
from typing import Dict, List, Protocol, Tuple

from .schemas import (
    Memory,
    MemoryCandidate,
    MemoryType,
    Scope,
    ScopeType,
    StructuredQuery,
    Turn,
)

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors; 0.0 if either is degenerate.

    Lives here because it is embedder math used by the write path (deciding
    whether a candidate restates an existing memory). The eval suite keeps its
    own copy on purpose: a metric that imports the system cannot judge it.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def year_heuristic(query: str) -> float:
    """Query-level time signal: 1.0 when the text names a year, else 0.0.

    Shared by the mock ``understand`` and the retrieval fallback path so
    both agree on the gate when no model output exists.
    """
    return 1.0 if _YEAR_RE.search(query) else 0.0


class LLMClient(Protocol):
    def judge(self, turns: List[Turn]) -> MemoryCandidate: ...
    def classify(
        self, candidate: MemoryCandidate, related: List[Memory]
    ) -> Tuple[str, str | None]: ...
    def understand(self, query: str) -> StructuredQuery: ...


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, memories: List[Memory], top_n: int
    ) -> List[Memory]: ...


def _format_turns(turns: List[Turn]) -> str:
    lines: List[str] = []
    for t in turns:
        role = t.role if isinstance(t, Turn) else t.get("role", "user")
        content = t.content if isinstance(t, Turn) else t.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_judge_prompt(turns: List[Turn]) -> str:
    """Pure judge prompt for the last-N turns.

    The rules are worded around one distinction that a window's *topic* does
    not capture: whether the user revealed something durable about themselves.
    A window can be mostly the assistant explaining general options and still
    contain the user's own facts, stated as an aside — and those asides are
    exactly what a long conversation later gets asked about. An earlier wording
    ("general-knowledge questions do NOT count") was read as "this window is
    instructional, so store nothing", and a durable location fact was lost with
    it.
    """
    convo = _format_turns(turns)
    return (
        "You are a memory judge. Decide whether the recent conversation "
        "contains durable information about the user, agent, or project that "
        "is worth keeping in long-term memory.\n"
        "\n"
        "Conversation (last turns):\n"
        f"{convo}\n"
        "\n"
        "Rules:\n"
        "- Store distilled facts, preferences, and decisions "
        "(e.g. likes, home city, project choices).\n"
        "- Judge the USER's own statements, not the topic of the window. A "
        "window can be mostly the assistant explaining general options and "
        "still contain a durable fact about the user.\n"
        "- Details stated in passing count, even as an aside. For example, "
        "\"I'm thinking of buying new sandals... by the way, I need to "
        "organize my closet and get rid of my old sneakers in a shoe rack\" "
        "states where the sneakers are kept and must be stored, even though "
        "the window is mostly shopping advice.\n"
        "- Only requests for general explanations with nothing about the user "
        "count as nothing to store. For example, \"Can you provide the "
        "technical details of how a rocket operates in space?\" carries no "
        "durable fact about the user, so it yields should_store=false.\n"
        "- When in doubt about a concrete detail the user stated about "
        "themselves, store it rather than skipping it; a missed fact cannot "
        "be recovered later, while an unnecessary memory only costs a slot.\n"
        "- confidence_basis must start with explicit when the user stated "
        "the fact directly (explicit_user_statement), otherwise describe "
        "the inference (inferred_from_context).\n"
        "- valid_from: ISO date the fact became true "
        "(e.g. 2023-05-30) when the conversation states one, otherwise "
        "null for facts that hold generally. Temporal questions depend on "
        "this, so never invent a date.\n"
        "- scope_id: the project/agent identifier when scope_type is "
        "project or agent, otherwise null.\n"
        "\n"
        "Output strict JSON only, no prose, with exactly these keys:\n"
        '{"should_store": true/false, "content": "distilled fact or empty", '
        '"type": "semantic|episodic|procedural|preference", '
        '"scope_type": "user|agent|project|global", '
        '"scope_id": "identifier or null", '
        '"valid_from": "ISO date or null", '
        '"confidence_basis": "explicit_user_statement|inferred_from_context"}'
    )


def build_classify_prompt(
    candidate: MemoryCandidate, related: List[Memory]
) -> str:
    """Pure classifier prompt over the active-head related set.

    The rule that matters most is the boundary between *similar* and
    *no_relation*. MERGE replaces nothing: it keeps the existing row's text and
    raises its confidence, so calling two different assertions "similar" silently
    deletes the newer one. Sharing a topic is therefore explicitly not enough —
    the examples that pin this are the cobbler/shoe-rack pair (same subject,
    different assertion) and the Italian-food pair (same assertion, reworded).
    """
    lines: List[str] = [
        "You are a relationship classifier. Compare the new candidate memory "
        "against the existing memories and decide how they relate.",
        "",
        "Candidate (new memory):",
        candidate.content,
        "",
        "Existing memories (all candidates below are the current active "
        "head — pick among these only, never invent an older id):",
    ]
    for i, m in enumerate(related, start=1):
        lines.append(f"{i}. [{m.memory_id}] {m.content}")
    lines.extend(
        [
            "",
            "Rules:",
            "- similar means the SAME fact, restated. Merging keeps the existing "
            "wording, so anything the candidate says that the existing memory "
            "does not say would be lost. Only choose similar when the candidate "
            "adds nothing new.",
            "- Sharing a topic is not enough. Two memories about the same "
            "subject that assert different things are no_relation, not similar.",
            "- contradiction means the candidate asserts something incompatible "
            "with the existing memory (an update, a reversal, a denial), so the "
            "old state must be kept as history.",
            "",
            "Examples:",
            '- Similar: "User likes Italian food." vs '
            '"User enjoys Italian restaurants." -> similar '
            "(same fact, reworded, nothing new).",
            '- No relation (same topic, different assertion): "User plans to '
            'drop old sneakers at a cobbler." vs "User keeps old sneakers in a '
            'shoe rack." -> no_relation (both are about old sneakers, but each '
            "states a different fact; merging would erase one).",
            '- Contradiction: "User prefers concise answers." vs '
            '"User prefers detailed explanations." -> contradiction '
            "(new fact replaces the old one as history, never overwrite).",
            '- No relation: "User prefers concise answers." vs '
            '"User lives in Bangalore." -> no_relation.',
            "",
            "Output strict JSON only, no prose, with exactly these keys:",
            '{"relationship": "no_relation|similar|contradiction", '
            '"related_id": "memory id from the list above or null"}',
        ]
    )
    return "\n".join(lines)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.splitlines()
        parts = parts[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        text = "\n".join(parts).strip()
    return text


def _coerce_enum(value, enum_cls, default):
    """Map an unusable enum value onto the documented default.

    Catches: the silent loss this was written for. A small model emitted
    ``"type": ""`` for an otherwise perfect memory, the whole candidate failed
    validation, and the write path turned that into DISCARD — the fact was gone
    forever with no trace of why. Content is the valuable part of a candidate;
    an unreadable category is not worth losing it over.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            return default
    return default


def _coerce_datetime(value):
    """Return a parsed datetime, or None for anything unusable.

    Same reasoning as ``_coerce_enum``: ``valid_from`` is optional, so a
    malformed date should drop the field, not the memory that carries it.
    """
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def parse_judge_json(raw: str) -> MemoryCandidate:
    """Parse LLM judge output into a MemoryCandidate or raise ValueError.

    Raises only when the reply is not JSON, is not an object, or is missing
    ``should_store`` / contradicts it with empty content. Optional fields with
    unusable values are coerced to their defaults instead, so one bad enum
    cannot discard an otherwise valid memory.
    """
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("judge output must be a JSON object")
    data = dict(data)
    if "scope" not in data and ("scope_type" in data or "scope_id" in data):
        scope = {"type": data.pop("scope_type", "user")}
        if "scope_id" in data:
            scope["id"] = data.pop("scope_id")
        data["scope"] = scope
    elif (
        isinstance(data.get("scope"), dict)
        and data["scope"].get("id") is None
        and "scope_id" in data
    ):
        scope = dict(data["scope"])
        scope["id"] = data.pop("scope_id")
        data["scope"] = scope
    else:
        data.pop("scope_id", None)

    data["type"] = _coerce_enum(data.get("type"), MemoryType, MemoryType.semantic)
    scope = data.get("scope")
    if isinstance(scope, dict):
        scope = dict(scope)
        scope["type"] = _coerce_enum(
            scope.get("type"), ScopeType, ScopeType.user
        )
        data["scope"] = scope
    else:
        data["scope"] = {"type": ScopeType.user}
    if "valid_from" in data:
        data["valid_from"] = _coerce_datetime(data["valid_from"])
    if not isinstance(data.get("confidence_basis", ""), str):
        data["confidence_basis"] = ""
    if data.get("content") is None:
        data["content"] = ""
    try:
        return MemoryCandidate(**data)
    except Exception as exc:
        raise ValueError(f"judge output failed schema validation: {exc}") from exc


_ALLOWED_RELATIONSHIPS = frozenset({"no_relation", "similar", "contradiction"})


def parse_classify_json(raw: str) -> Tuple[str, str | None]:
    """Parse LLM classify output into (relationship, related_id)."""
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"classify output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("classify output must be a JSON object")
    rel = data.get("relationship")
    if rel not in _ALLOWED_RELATIONSHIPS:
        raise ValueError(f"classify output has bad relationship: {rel!r}")
    rid = data.get("related_id")
    if rid is not None and not isinstance(rid, str):
        raise ValueError(f"classify output has bad related_id: {rid!r}")
    return rel, rid


def build_understand_prompt(query: str) -> str:
    """Pure query-understanding prompt. No question-time extraction."""
    return (
        "You rewrite a user query for memory retrieval. Resolve pronouns "
        "and ambiguity, then describe what to look for.\n"
        "\n"
        f"Query: {query}\n"
        "\n"
        "Rules:\n"
        "- memory_types is a subset of"
        " [semantic, episodic, procedural, preference] (empty means all).\n"
        "- scope is {\"type\": user|agent|project|global} plus \"id\" only"
        " for project/agent scopes.\n"
        "- temporal_confidence in [0, 1] records how strongly the query"
        " constrains time: 1.0 for an explicit era"
        " (\"Where did I live in 2023?\"), 0.0 for current-state questions"
        " (\"Where do I live?\").\n"
        "- Never include a question-time or date field: the caller supplies"
        " the era separately; you only score how much time matters.\n"
        "\n"
        "Output strict JSON only, no prose, with exactly these keys:\n"
        '{"rewritten_query": "...", "intent": "...", '
        '"entities": [], "memory_types": [], '
        '"scope": {"type": "user"}, "keywords": [], '
        '"temporal_confidence": 0.0}'
    )


def build_answer_prompt(query: str, context_text: str) -> str:
    """Pure answer prompt: grounded, extractive, abstains when thin."""
    return (
        "Answer the user question using ONLY the memories below. Quote"
        " names, shifts, and dates exactly as written. If the memories"
        " do not contain the answer, say so plainly instead of guessing.\n"
        "\n"
        f"Question: {query}\n"
        "\n"
        "Memories:\n"
        f"{context_text}\n"
        "\n"
        "Answer in one or two sentences:"
    )


def parse_understand_json(raw: str) -> StructuredQuery:
    """Parse LLM understand output; clamp gate, default it, drop unknowns."""
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"understand output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("understand output must be a JSON object")
    rewritten = data.get("rewritten_query")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("understand output is missing rewritten_query")
    intent = data.get("intent", "")
    entities = [e for e in data.get("entities", []) if isinstance(e, str)]
    keywords = [k for k in data.get("keywords", []) if isinstance(k, str)]
    kept_types = []
    for t in data.get("memory_types", []):
        try:
            kept_types.append(MemoryType(t))
        except ValueError:
            continue
    scope_raw = data.get("scope", {})
    if isinstance(scope_raw, dict):
        stype = scope_raw.get("type", "user")
        sid = scope_raw.get("id")
    else:
        stype, sid = "user", None
    if "scope" not in data:
        if "scope_type" in data:
            stype = data["scope_type"]
        if "scope_id" in data:
            sid = data["scope_id"]
    try:
        scope = Scope(type=stype, id=sid)
    except Exception as exc:
        raise ValueError(f"understand output has bad scope: {exc}") from exc
    tc_raw = data.get("temporal_confidence", 0.0)
    try:
        tc = float(tc_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"understand output has bad temporal_confidence: {tc_raw!r}"
        ) from exc
    tc = min(1.0, max(0.0, tc))
    try:
        return StructuredQuery(
            rewritten_query=rewritten,
            intent=intent if isinstance(intent, str) else "",
            entities=entities,
            memory_types=kept_types,
            scope=scope,
            keywords=keywords,
            temporal_confidence=tc,
        )
    except Exception as exc:
        raise ValueError(
            f"understand output failed schema validation: {exc}"
        ) from exc


class MockLLM:
    """Heuristic offline stand-in: short chatter is not stored."""

    def judge(self, turns: List[Turn]) -> MemoryCandidate:
        parts: List[str] = []
        for t in turns:
            if isinstance(t, Turn):
                parts.append(t.content)
            else:
                parts.append(str(t.get("content", "")))
        text = "\n".join(parts)
        if len(text) < 20:
            return MemoryCandidate(should_store=False, content="")
        return MemoryCandidate(
            should_store=True,
            content=text[:500],
            confidence_basis="mock_inferred",
        )

    def classify(
        self, candidate: MemoryCandidate, related: List[Memory]
    ) -> Tuple[str, str | None]:
        return ("no_relation", None)

    def chat_text(self, prompt: str) -> str:
        """Offline stand-in answer (never used for quality claims)."""
        _ = prompt
        return "mock answer (no real LLM)"

    def understand(self, query: str) -> StructuredQuery:
        return StructuredQuery(
            rewritten_query=query,
            keywords=query.split()[:8],
            temporal_confidence=year_heuristic(query),
        )


class FixedLLM:
    """Deterministic stub for op-level tests."""

    def __init__(
        self,
        judge_result: MemoryCandidate | None = None,
        classify_result: Tuple[str, str | None] | None = None,
        understand_result: StructuredQuery | None = None,
        answer_result: str | None = None,
    ) -> None:
        self._judge_result = judge_result or MemoryCandidate(
            should_store=False, content=""
        )
        self._classify_result = classify_result or ("no_relation", None)
        self._understand_result = understand_result
        self._answer_result = answer_result or "fixed answer"

    def judge(self, turns: List[Turn]) -> MemoryCandidate:
        return self._judge_result

    def classify(
        self, candidate: MemoryCandidate, related: List[Memory]
    ) -> Tuple[str, str | None]:
        return self._classify_result

    def understand(self, query: str) -> StructuredQuery:
        if self._understand_result is not None:
            return self._understand_result
        return StructuredQuery(
            rewritten_query=query,
            keywords=query.split()[:8],
            temporal_confidence=year_heuristic(query),
        )

    def chat_text(self, prompt: str) -> str:
        _ = prompt
        return self._answer_result


class MockEmbedder:
    """Deterministic 4-dim vectors from an md5 hash (offline only)."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs: List[List[float]] = []
        for t in texts:
            digest = hashlib.md5(t.encode("utf-8")).digest()
            vecs.append([b / 255 for b in digest[:4]])
        return vecs


class MockReranker:
    """Identity slice: keep score order, cut to top_n."""

    def rerank(
        self, query: str, memories: List[Memory], top_n: int
    ) -> List[Memory]:
        return list(memories[:top_n])


# ---------------------------------------------------------------------------
# Production clients (Phase 04). Same protocols as the mocks above.
# ---------------------------------------------------------------------------

_REPAIR_SUFFIX = (
    " Your previous reply was not valid JSON."
    " Reply with return valid JSON only, no prose."
)


def _with_repair(chat_json, model: str, prompt: str, parse):
    """One repair retry on unparseable output, then raise (shared)."""
    try:
        return parse(chat_json(prompt, model))
    except ValueError:
        return parse(chat_json(prompt + _REPAIR_SUFFIX, model))


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """Remove <think> reasoning spans local models emit before answers."""
    return _THINK_RE.sub("", text).strip()


class OpenAICompatLLM:
    """Chat-backed judge/classify/understand over any OpenAI-style API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        judge_model: str = "gpt-4o-mini",
        classifier_model: str = "gpt-4o-mini",
        understand_model: str = "gpt-4o-mini",
        timeout_s: float = 30,
        max_retries: int = 2,
        extra_body: dict | None = None,
    ) -> None:
        from openai import OpenAI

        options: dict = {
            "api_key": api_key,
            "timeout": timeout_s,
            "max_retries": max_retries,
        }
        if base_url:
            options["base_url"] = base_url
        self._client = OpenAI(**options)
        self.judge_model = judge_model
        self.classifier_model = classifier_model
        self.understand_model = understand_model
        self.extra_body = extra_body
        self.calls: List[Dict] = []

    def chat_json(self, prompt: str, model: str) -> str:
        """POST one chat turn requesting a JSON object; log usage.

        Logs tokens + latency per call. USD cost is intentionally not
        computed here (no price table at this layer); aggregate
        ``prompt_tokens``/``completion_tokens`` downstream for costing.
        """
        started = time.time()
        create_kw: dict = {"extra_body": self.extra_body} if self.extra_body else {}
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                **create_kw,
            )
        except Exception as exc:
            text = str(exc).lower()
            if "response_format" in text or "response format" in text:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    **create_kw,
                )
            else:
                raise
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or 0
        self.calls.append(
            {"model": model, "in": prompt_tokens, "out": completion_tokens}
        )
        print(
            json.dumps(
                {
                    "event": "llm_call",
                    "model": model,
                    "elapsed_ms": round((time.time() - started) * 1000, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
        )
        return content

    def _once(self, model: str, prompt: str, parse):
        return _with_repair(self.chat_json, model, prompt, parse)

    def chat_text(self, prompt: str, model: str | None = None) -> str:
        """POST one plain-text chat turn; log usage like chat_json."""
        started = time.time()
        create_kw: dict = {"extra_body": self.extra_body} if self.extra_body else {}
        resp = self._client.chat.completions.create(
            model=model or self.understand_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **create_kw,
        )
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or 0
        self.calls.append(
            {
                "model": model or self.understand_model,
                "in": prompt_tokens,
                "out": completion_tokens,
            }
        )
        print(
            json.dumps(
                {
                    "event": "llm_call",
                    "model": model or self.understand_model,
                    "elapsed_ms": round((time.time() - started) * 1000, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
        )
        return content

    def judge(self, turns: List[Turn]) -> MemoryCandidate:
        prompt = build_judge_prompt(turns)
        return self._once(self.judge_model, prompt, parse_judge_json)

    def classify(
        self, candidate: MemoryCandidate, related: List[Memory]
    ) -> Tuple[str, str | None]:
        prompt = build_classify_prompt(candidate, related)
        return self._once(self.classifier_model, prompt, parse_classify_json)

    def understand(self, query: str) -> StructuredQuery:
        prompt = build_understand_prompt(query)
        return self._once(self.understand_model, prompt, parse_understand_json)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenRouterLLM(OpenAICompatLLM):
    """Single-model OpenRouter backend with reasoning enabled.

    Same wire format as the provider docs: chat completions plus
    ``reasoning: {"enabled": True}`` in the request body.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30,
        max_retries: int = 2,
        reasoning: bool = True,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            judge_model=model,
            classifier_model=model,
            understand_model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            extra_body={"reasoning": {"enabled": True}} if reasoning else None,
        )


class GroqLLM(OpenAICompatLLM):
    """Single-model Groq backend (OpenAI-compatible endpoint, no extras)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            judge_model=model,
            classifier_model=model,
            understand_model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )


class OpenAIEmbedder:
    """Truncate, batch, dimension-check, and L2-normalize for cosine.

    Kept for API-metered embeddings; the default wiring is local
    (SentenceTransformerEmbedder, free) — see scripts/_lib.py.
    """

    def __init__(
        self,
        model: str,
        dim: int,
        batch_size: int = 64,
        api_key: str = "",
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        options: dict = {"api_key": api_key or None}
        if base_url:
            options["base_url"] = base_url
        self._client = OpenAI(**options)
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.calls: List[Dict] = []

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:8000] for t in texts[start : start + self.batch_size]]
            if not batch:
                continue
            resp = self._client.embeddings.create(model=self.model, input=batch)
            usage = getattr(resp, "usage", None)
            self.calls.append(
                {
                    "model": self.model,
                    "in": getattr(usage, "prompt_tokens", None) or 0,
                    "out": 0,
                }
            )
            for item in sorted(resp.data, key=lambda d: d.index):
                vec = [float(x) for x in item.embedding]
                if len(vec) != self.dim:
                    raise ValueError(
                        f"embedder dim mismatch: got {len(vec)}, want {self.dim}"
                    )
                norm = math.sqrt(sum(x * x for x in vec))
                out.append([x / norm for x in vec] if norm > 0 else vec)
        return out


class LLMReranker:
    """Pairwise 0-10 scoring reranker over the post-filter slice only."""

    def __init__(self, client: OpenAICompatLLM, model: str) -> None:
        self._client = client
        self.model = model

    def _score(self, query: str, memory: Memory) -> float:
        prompt = (
            "Rate how relevant this memory is to the query on a 0-10 scale."
            " Output strict JSON only: {\"score\": N}.\n"
            f"\nQuery: {query}\nMemory: {memory.content}"
        )
        try:
            data = json.loads(
                _strip_fences(self._client.chat_json(prompt, self.model))
            )
            return float(data.get("score", 0.0))
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def rerank(
        self, query: str, memories: List[Memory], top_n: int
    ) -> List[Memory]:
        scored = [(self._score(query, m), i, m) for i, m in enumerate(memories)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [m for _, _, m in scored[:top_n]]


class SentenceTransformerEmbedder:
    """Local, free embeddings (default: Qwen/Qwen3-Embedding-0.6B @ 1024 dim).

    Heavy imports stay inside the constructor so importing this module
    never pays the torch load cost. Vectors are L2-normalized here for
    cosine search regardless of backend flags.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        dim: int = 1024,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings"
            ) from exc
        kwargs: dict = {}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(model, **kwargs)
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as _np

        if not texts:
            return []
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = _np.asarray(vecs, dtype=_np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"embedder dim mismatch: got {arr.shape}, want (*, {self.dim})"
            )
        norms = _np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()


class LlamaCppLLM:
    """Local GGUF backend (llama.cpp): free, private, no quotas.

    Plain decoding only: grammar-guided JSON stalls pathologically on
    long prompts, so discipline comes from the prompt plus the shared
    repair retry, with think spans stripped before parsing. Usage tokens
    feed the standard per-question cost log (priced as zero downstream).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for local GGUF models"
            ) from exc
        from pathlib import Path as _Path

        options: dict = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "verbose": verbose,
        }
        if n_threads:
            options["n_threads"] = n_threads
        self._llm = Llama(**options)
        self.model_id = _Path(model_path).name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls: List[Dict] = []

    def chat_json(self, prompt: str, model: str = "local") -> str:
        """One plain chat turn; grammar constraint disabled (it stalls).

        llama.cpp grammar-guided decoding goes pathological on long
        prompts (25+ min for a 2k-token window that plain decoding does
        in ~3 min), so JSON discipline comes from the prompt plus the
        shared repair retry; think spans are stripped before parsing.
        """
        started = time.time()
        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = _strip_think_blocks(
            resp["choices"][0]["message"]["content"] or ""
        )
        usage = resp.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        self.calls.append(
            {"model": self.model_id, "in": prompt_tokens, "out": completion_tokens}
        )
        print(
            json.dumps(
                {
                    "event": "llm_call",
                    "model": self.model_id,
                    "elapsed_ms": round((time.time() - started) * 1000, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
        )
        return content

    def _once(self, model: str, prompt: str, parse):
        return _with_repair(self.chat_json, model, prompt, parse)

    def chat_text(self, prompt: str, model: str = "local") -> str:
        """One plain-text chat turn (answer generation); log usage."""
        started = time.time()
        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = _strip_think_blocks(
            resp["choices"][0]["message"]["content"] or ""
        )
        usage = resp.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        self.calls.append(
            {"model": self.model_id, "in": prompt_tokens, "out": completion_tokens}
        )
        print(
            json.dumps(
                {
                    "event": "llm_call",
                    "model": self.model_id,
                    "elapsed_ms": round((time.time() - started) * 1000, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
        )
        return content

    def judge(self, turns: List[Turn]) -> MemoryCandidate:
        return self._once("local", build_judge_prompt(turns), parse_judge_json)

    def classify(
        self, candidate: MemoryCandidate, related: List[Memory]
    ) -> Tuple[str, str | None]:
        return self._once(
            "local", build_classify_prompt(candidate, related), parse_classify_json
        )

    def understand(self, query: str) -> StructuredQuery:
        return self._once("local", build_understand_prompt(query), parse_understand_json)
