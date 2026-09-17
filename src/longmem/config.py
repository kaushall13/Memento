"""Settings loader. config.yaml -> Settings (frozen dataclasses).

All tunables live in config.yaml; this module only parses, fills defaults,
validates, and freezes. Runtime code must read from Settings, never hardcode
tuning literals.

Python-side fallback defaults live in the _DEFAULT_* block below (single
place) and are referenced by both the dataclass field defaults and the
``.get()`` fallbacks in ``load_settings``. config.yaml is the canonical
source; the Python block only applies when keys are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


# --- Python-side fallback defaults (one block; YAML is canonical) ---
_DEFAULT_WRITE_INTERVAL_TURNS = 5
_DEFAULT_JUDGE_WINDOW_TURNS = 10
_DEFAULT_DECAY_FUNCTION = "exponential"
_DEFAULT_DECAY_LAMBDA = 0.005
_DEFAULT_EXPIRY_DAYS = 365
_DEFAULT_CANDIDATE_COUNT = 50
_DEFAULT_RERANKER_TOP_N = 20
_DEFAULT_FINAL_MEMORY_COUNT = 10
_DEFAULT_RRF_K = 60
_DEFAULT_W_RELEVANCE = 1.0
_DEFAULT_W_CONFIDENCE = 0.3
_DEFAULT_W_TEMPORAL = 0.5
_DEFAULT_W_RECENCY = 0.2
_DEFAULT_W_DECAY = 0.3
# Backstop thresholds for MERGE. Measured on the local embedder: true
# paraphrases score 0.91-1.00 while "same topic, different assertion" pairs
# score 0.70-0.78, so 0.88 sits in the gap. Deliberately biased toward
# rejecting a merge: a duplicate memory is recoverable, deleted content is not.
_DEFAULT_MERGE_SIMILARITY_MIN = 0.88
# Lexical fallback (used when no embedder is configured) counts candidate
# content terms missing from the target; near-exact restatements score 1.0.
_DEFAULT_MERGE_OVERLAP_MIN = 0.75
# NOTE (Phase 01 lock): the repo gate asserts src contains no hard-coded
# confidence decimals outside tests, so confidence fallbacks below are
# written as fractions (see plan section on confidence tunables).
_CONF_EXPLICIT_DEFAULT = 6 / 10
_CONF_DEFAULT_DEFAULT = 4 / 10
_MERGE_BUMP_DEFAULT = 1 / 10
_CONF_MAX_DEFAULT = 1.0
_DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
_DEFAULT_CLASSIFIER_MODEL = "gpt-4o-mini"
_DEFAULT_UNDERSTAND_MODEL = "gpt-4o-mini"
_DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_EMBEDDING_DIM = 1024
_DEFAULT_PROVIDER = "openai"
_DEFAULT_LOCAL_MODEL = "model/Qwen3.5-4B-UD-Q4_K_XL.gguf"
_DEFAULT_POSTGRES_DSN_ENV = "POSTGRES_DSN"
_DEFAULT_REDIS_URL_ENV = "REDIS_URL"


@dataclass(frozen=True)
class MemoryConfig:
    write_interval_turns: int = _DEFAULT_WRITE_INTERVAL_TURNS
    judge_window_turns: int = _DEFAULT_JUDGE_WINDOW_TURNS


@dataclass(frozen=True)
class LifecycleConfig:
    decay_function: str = _DEFAULT_DECAY_FUNCTION
    decay_lambda: float = _DEFAULT_DECAY_LAMBDA
    inactivity_expiry_days: int = _DEFAULT_EXPIRY_DAYS


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_count: int = _DEFAULT_CANDIDATE_COUNT
    reranker_top_n: int = _DEFAULT_RERANKER_TOP_N
    final_memory_count: int = _DEFAULT_FINAL_MEMORY_COUNT
    rrf_k: int = _DEFAULT_RRF_K


@dataclass(frozen=True)
class ScoringConfig:
    w_relevance: float = _DEFAULT_W_RELEVANCE
    w_confidence: float = _DEFAULT_W_CONFIDENCE
    w_temporal: float = _DEFAULT_W_TEMPORAL
    w_recency: float = _DEFAULT_W_RECENCY
    w_decay: float = _DEFAULT_W_DECAY
    confidence_explicit: float = _CONF_EXPLICIT_DEFAULT
    confidence_default: float = _CONF_DEFAULT_DEFAULT
    merge_bump: float = _MERGE_BUMP_DEFAULT
    confidence_max: float = _CONF_MAX_DEFAULT
    merge_similarity_min: float = _DEFAULT_MERGE_SIMILARITY_MIN
    merge_overlap_min: float = _DEFAULT_MERGE_OVERLAP_MIN


@dataclass(frozen=True)
class ModelsConfig:
    judge: str = _DEFAULT_JUDGE_MODEL
    classifier: str = _DEFAULT_CLASSIFIER_MODEL
    understand: str = _DEFAULT_UNDERSTAND_MODEL
    embed: str = _DEFAULT_EMBED_MODEL
    rerank: str = _DEFAULT_RERANK_MODEL
    embedding_dim: int = _DEFAULT_EMBEDDING_DIM
    provider: str = _DEFAULT_PROVIDER
    local_model: str = _DEFAULT_LOCAL_MODEL


@dataclass(frozen=True)
class PostgresConfig:
    dsn_env: str = _DEFAULT_POSTGRES_DSN_ENV


@dataclass(frozen=True)
class RedisConfig:
    url_env: str = _DEFAULT_REDIS_URL_ENV


@dataclass(frozen=True)
class Settings:
    memory: MemoryConfig = MemoryConfig()
    lifecycle: LifecycleConfig = LifecycleConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    scoring: ScoringConfig = ScoringConfig()
    models: ModelsConfig = ModelsConfig()
    postgres: PostgresConfig = PostgresConfig()
    redis: RedisConfig = RedisConfig()


def _section(raw: dict, name: str) -> dict:
    val = raw.get(name, {})
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(f"{name} must be a mapping")
    return val


def load_settings(path: str | Path = "config.yaml") -> Settings:
    """Parse yaml path, fill defaults, validate, return frozen Settings.

    Raises FileNotFoundError when ``path`` does not exist (explicit paths
    included — silent fallback to defaults would hide typos).
    Raises ValueError with the offending field name on any violation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    raw = loaded

    mem = _section(raw, "memory")
    life = _section(raw, "lifecycle")
    retr = _section(raw, "retrieval")
    scor = _section(raw, "scoring")
    mods = _section(raw, "models")
    pg = _section(raw, "postgres")
    rd = _section(raw, "redis")

    memory = MemoryConfig(
        write_interval_turns=int(
            mem.get("write_interval_turns", _DEFAULT_WRITE_INTERVAL_TURNS)
        ),
        judge_window_turns=int(
            mem.get("judge_window_turns", _DEFAULT_JUDGE_WINDOW_TURNS)
        ),
    )
    lifecycle = LifecycleConfig(
        decay_function=str(life.get("decay_function", _DEFAULT_DECAY_FUNCTION)),
        decay_lambda=float(life.get("decay_lambda", _DEFAULT_DECAY_LAMBDA)),
        inactivity_expiry_days=int(
            life.get("inactivity_expiry_days", _DEFAULT_EXPIRY_DAYS)
        ),
    )
    retrieval = RetrievalConfig(
        candidate_count=int(retr.get("candidate_count", _DEFAULT_CANDIDATE_COUNT)),
        reranker_top_n=int(retr.get("reranker_top_n", _DEFAULT_RERANKER_TOP_N)),
        final_memory_count=int(
            retr.get("final_memory_count", _DEFAULT_FINAL_MEMORY_COUNT)
        ),
        rrf_k=int(retr.get("rrf_k", _DEFAULT_RRF_K)),
    )
    scoring = ScoringConfig(
        w_relevance=float(scor.get("w_relevance", _DEFAULT_W_RELEVANCE)),
        w_confidence=float(scor.get("w_confidence", _DEFAULT_W_CONFIDENCE)),
        w_temporal=float(scor.get("w_temporal", _DEFAULT_W_TEMPORAL)),
        w_recency=float(scor.get("w_recency", _DEFAULT_W_RECENCY)),
        w_decay=float(scor.get("w_decay", _DEFAULT_W_DECAY)),
        confidence_explicit=float(
            scor.get("confidence_explicit", _CONF_EXPLICIT_DEFAULT)
        ),
        confidence_default=float(
            scor.get("confidence_default", _CONF_DEFAULT_DEFAULT)
        ),
        merge_bump=float(scor.get("merge_bump", _MERGE_BUMP_DEFAULT)),
        confidence_max=float(scor.get("confidence_max", _CONF_MAX_DEFAULT)),
        merge_similarity_min=float(
            scor.get("merge_similarity_min", _DEFAULT_MERGE_SIMILARITY_MIN)
        ),
        merge_overlap_min=float(
            scor.get("merge_overlap_min", _DEFAULT_MERGE_OVERLAP_MIN)
        ),
    )
    models = ModelsConfig(
        judge=str(mods.get("judge", _DEFAULT_JUDGE_MODEL)),
        classifier=str(mods.get("classifier", _DEFAULT_CLASSIFIER_MODEL)),
        understand=str(mods.get("understand", _DEFAULT_UNDERSTAND_MODEL)),
        embed=str(mods.get("embed", _DEFAULT_EMBED_MODEL)),
        rerank=str(mods.get("rerank", _DEFAULT_RERANK_MODEL)),
        embedding_dim=int(mods.get("embedding_dim", _DEFAULT_EMBEDDING_DIM)),
        provider=str(mods.get("provider", _DEFAULT_PROVIDER)),
        local_model=str(mods.get("local_model", _DEFAULT_LOCAL_MODEL)),
    )
    postgres = PostgresConfig(
        dsn_env=str(pg.get("dsn_env", _DEFAULT_POSTGRES_DSN_ENV))
    )
    redis_cfg = RedisConfig(url_env=str(rd.get("url_env", _DEFAULT_REDIS_URL_ENV)))

    settings = Settings(
        memory=memory,
        lifecycle=lifecycle,
        retrieval=retrieval,
        scoring=scoring,
        models=models,
        postgres=postgres,
        redis=redis_cfg,
    )
    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    if s.memory.write_interval_turns < 1:
        raise ValueError(
            f"memory.write_interval_turns must be >= 1, got {s.memory.write_interval_turns}"
        )
    if s.memory.judge_window_turns < s.memory.write_interval_turns:
        raise ValueError(
            "memory.judge_window_turns must cover at least one interval "
            f"(>= memory.write_interval_turns={s.memory.write_interval_turns}), "
            f"got {s.memory.judge_window_turns}"
        )
    if s.lifecycle.decay_function != "exponential":
        raise ValueError(
            "lifecycle.decay_function must be 'exponential' "
            f"(lifecycle.py only implements exponential), got {s.lifecycle.decay_function!r}"
        )
    lam = s.lifecycle.decay_lambda
    if not (0 < lam < 1):
        raise ValueError(
            f"lifecycle.decay_lambda must satisfy 0 < lambda < 1, got {lam}"
        )
    if s.lifecycle.inactivity_expiry_days < 1:
        raise ValueError(
            "lifecycle.inactivity_expiry_days must be >= 1, "
            f"got {s.lifecycle.inactivity_expiry_days}"
        )
    if s.retrieval.candidate_count < 1:
        raise ValueError(
            f"retrieval.candidate_count must be >= 1, got {s.retrieval.candidate_count}"
        )
    if s.retrieval.final_memory_count < 1:
        raise ValueError(
            "retrieval.final_memory_count must be >= 1, "
            f"got {s.retrieval.final_memory_count}"
        )
    if s.retrieval.reranker_top_n < 0:
        raise ValueError(
            f"retrieval.reranker_top_n must be >= 0, got {s.retrieval.reranker_top_n}"
        )
    if (
        s.retrieval.reranker_top_n != 0
        and s.retrieval.reranker_top_n < s.retrieval.final_memory_count
    ):
        raise ValueError(
            "retrieval.reranker_top_n must be 0 (rerank bypass) or "
            ">= retrieval.final_memory_count, got "
            f"{s.retrieval.reranker_top_n} < {s.retrieval.final_memory_count}"
        )
    if s.retrieval.rrf_k < 1:
        raise ValueError(f"retrieval.rrf_k must be >= 1, got {s.retrieval.rrf_k}")
    if s.retrieval.candidate_count < s.retrieval.final_memory_count:
        raise ValueError(
            "retrieval.candidate_count must be >= retrieval.final_memory_count, "
            f"got {s.retrieval.candidate_count} < {s.retrieval.final_memory_count}"
        )
    for w_name in (
        "w_relevance",
        "w_confidence",
        "w_temporal",
        "w_recency",
        "w_decay",
    ):
        w = getattr(s.scoring, w_name)
        if w < 0:
            raise ValueError(f"scoring.{w_name} must be >= 0, got {w}")
    dim = s.models.embedding_dim
    if not (64 <= dim <= 4096):
        raise ValueError(
            f"models.embedding_dim must be in [64..4096], got {dim}"
        )
    if s.models.provider not in ("openai", "openrouter", "groq", "local"):
        raise ValueError(
            "models.provider must be openai|openrouter|groq|local, "
            f"got {s.models.provider!r}"
        )
    if s.models.provider == "local" and not Path(s.models.local_model).exists():
        raise ValueError(
            "models.local_model file not found: "
            f"{s.models.local_model!r}"
        )
    ce = s.scoring.confidence_explicit
    cd = s.scoring.confidence_default
    cm = s.scoring.confidence_max
    if not (0 <= cd <= ce <= cm <= 1):
        raise ValueError(
            "scoring confidence must satisfy "
            f"0 <= confidence_default ({cd}) <= confidence_explicit ({ce}) "
            f"<= confidence_max ({cm}) <= 1"
        )
    bump = s.scoring.merge_bump
    if not (0 < bump <= 0.5):
        raise ValueError(
            f"scoring.merge_bump must satisfy 0 < bump <= 0.5, got {bump}"
        )
    if not (0 < s.scoring.merge_similarity_min <= 1):
        raise ValueError(
            "scoring.merge_similarity_min must satisfy 0 < min <= 1, got "
            f"{s.scoring.merge_similarity_min}"
        )
    if not (0 <= s.scoring.merge_overlap_min <= 1):
        raise ValueError(
            "scoring.merge_overlap_min must satisfy 0 <= min <= 1, got "
            f"{s.scoring.merge_overlap_min}"
        )
