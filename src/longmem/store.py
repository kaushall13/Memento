"""Storage contracts (Protocols), DDL, and InMemory fakes.

Protocols first so Phase 04 PG swap changes zero call sites.
Fakes mirror Postgres op semantics exactly; they never compute tuning
values (confidence bump / initial values come from formation via Settings).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Protocol, Tuple

import psycopg
import redis

from .schemas import Memory, MemorySource, MemoryStatus, StructuredQuery, Turn

_DEFAULT_REDIS_TTL_S = 86400 * 7


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DDL: str = """-- longmem durable schema (Postgres + pgvector + FTS)
-- NOTE: to_tsvector uses an explicit 'english' regconfig for determinism
-- (plan text shows the single-arg form; behavior is equivalent on an
-- english-default cluster, explicit form wins on other locales).
-- Python-side datetimes are naive UTC (see lifecycle.py); the PG columns
-- below are TIMESTAMPTZ, so the Phase 04 adapter must attach UTC on write
-- and strip (or keep aware) on read consistently.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
  memory_id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector({embedding_dim}),
  type TEXT NOT NULL,
  scope_type TEXT NOT NULL,
  scope_id TEXT,
  confidence DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  valid_from TIMESTAMPTZ,
  valid_until TIMESTAMPTZ,
  last_retrieved_at TIMESTAMPTZ,
  last_updated_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  supersedes_id TEXT REFERENCES memories(memory_id),
  search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_search_tsv ON memories USING GIN (search_tsv);
-- HNSW over embeddings is created once embedding_dim is known; see render_ddl():
-- CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS memory_sources (
  memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  session_id TEXT NOT NULL,
  message_idx INTEGER,
  excerpt TEXT NOT NULL,
  PRIMARY KEY (memory_id, session_id)
);
"""


def render_ddl(embedding_dim: int) -> str:
    """Fill the embedding dimension into DDL plus the HNSW index line."""
    base = DDL.format(embedding_dim=embedding_dim)
    hnsw = (
        "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
        "ON memories USING hnsw (embedding vector_cosine_ops);"
    )
    return base + "\n" + hnsw + "\n"


def split_ddl_statements(ddl: str) -> List[str]:
    """Split DDL into executable statements.

    Strips ``--`` line comments first, so semicolons inside comments can
    never become bare-comment fragments (naive ``split(";")`` produced
    those). No block-comment or dollar-quote handling: the schema DDL
    uses neither.
    """
    code_lines = []
    for line in ddl.splitlines():
        cut = line.find("--")
        code_lines.append(line[:cut] if cut != -1 else line)
    return [
        stmt.strip()
        for stmt in "\n".join(code_lines).split(";")
        if stmt.strip()
    ]


_ALLOWED_STATUSES = frozenset({"active", "superseded", "expired"})


class WorkingMemoryStore(Protocol):
    def append_turn(self, session_id: str, turn: Turn | dict) -> None: ...
    def get_state(self, session_id: str) -> dict: ...
    def set_state(self, session_id: str, state: dict) -> None: ...
    def turn_count(self, session_id: str) -> int: ...


class DurableMemoryStore(Protocol):
    def insert(self, memory: Memory, sources: List[MemorySource]) -> None: ...
    def merge(
        self, memory_id: str, new_evidence: MemorySource, confidence: float
    ) -> None: ...
    def supersede(
        self, old_id: str, new: Memory, sources: List[MemorySource]
    ) -> None: ...
    def hybrid_search(
        self,
        query: StructuredQuery,
        limit: int,
        include_expired: bool = False,
        statuses: Tuple[str, ...] = ("active", "superseded"),
    ) -> List[Tuple[Memory, float, float]]: ...
    def touch_retrieved(self, ids: List[str]) -> None: ...
    def fetch_sources(self, ids: List[str]) -> List[MemorySource]: ...


class InMemoryWorkingStore:
    """Dict+list fake for working memory. Same op semantics as Redis."""

    def __init__(self) -> None:
        self._turns: Dict[str, list] = {}
        self._states: Dict[str, dict] = {}

    def append_turn(self, session_id: str, turn: Turn | dict) -> None:
        if isinstance(turn, dict):
            turn = Turn(**turn)
        self._turns.setdefault(session_id, []).append(turn)

    def get_state(self, session_id: str) -> dict:
        return dict(self._states.get(session_id, {}))

    def set_state(self, session_id: str, state: dict) -> None:
        self._states[session_id] = dict(state)

    def turn_count(self, session_id: str) -> int:
        return len(self._turns.get(session_id, []))

    def delete_session(self, session_id: str) -> None:
        """Drop one session's turns + state (eval isolation)."""
        self._turns.pop(session_id, None)
        self._states.pop(session_id, None)


class InMemoryDurableStore:
    """Dict+list fake for durable memory. Same op semantics as Postgres.

    PK parity with DDL: ``memories.memory_id`` is unique and
    ``memory_sources(memory_id, session_id)`` is unique. Duplicates raise
    ``ValueError`` here just as PG would raise a PK violation — silent
    overwrite would hide consolidation bugs — with one locked exception:
    ``merge()`` treats a repeated source key from the same session as an
    idempotent re-assertion (keeps the first source row, still refreshes
    confidence/version). A long session hitting two N-boundaries with
    similar content otherwise 500s on the second merge; Phase 04 must
    replicate with ``ON CONFLICT (memory_id, session_id) DO NOTHING``.
    """

    def __init__(self) -> None:
        self.memories: Dict[str, Memory] = {}
        self.sources: List[MemorySource] = []
        self._source_keys: set[tuple[str, str]] = set()

    def _register_sources(self, sources: List[MemorySource]) -> None:
        seen_in_batch: set[tuple[str, str]] = set()
        for s in sources:
            key = (s.memory_id, s.session_id)
            if key in self._source_keys or key in seen_in_batch:
                raise ValueError(f"duplicate memory_source key: {key}")
            seen_in_batch.add(key)
        for s in sources:
            self._source_keys.add((s.memory_id, s.session_id))
        self.sources.extend(sources)

    # -- writes ---------------------------------------------------------
    def insert(self, memory: Memory, sources: List[MemorySource]) -> None:
        if memory.memory_id in self.memories:
            raise ValueError(f"duplicate memory_id: {memory.memory_id}")
        self.memories[memory.memory_id] = memory
        self._register_sources(sources)

    def merge(
        self, memory_id: str, new_evidence: MemorySource, confidence: float
    ) -> None:
        old = self.memories.get(memory_id)
        if old is None:
            raise KeyError(f"unknown memory_id: {memory_id}")
        updated = old.model_copy(
            update={
                "confidence": confidence,
                "last_updated_at": _utcnow_naive(),
                "version": old.version + 1,
            }
        )
        self.memories[memory_id] = updated
        key = (new_evidence.memory_id, new_evidence.session_id)
        if key not in self._source_keys:
            self._register_sources([new_evidence])
        # else: same-session re-assertion of the same memory — keep the
        # first source row (PG parity: ON CONFLICT DO NOTHING) while still
        # applying the caller-computed confidence/version refresh above.

    def supersede(
        self, old_id: str, new: Memory, sources: List[MemorySource]
    ) -> None:
        old = self.memories.get(old_id)
        if old is None:
            raise KeyError(f"unknown memory_id: {old_id}")
        if new.memory_id in self.memories:
            raise ValueError(f"duplicate memory_id: {new.memory_id}")
        now = _utcnow_naive()
        closed = old.model_copy(
            update={"status": MemoryStatus.superseded, "valid_until": now}
        )
        self.memories[old_id] = closed
        newcomer = new.model_copy(update={"supersedes_id": old_id})
        self.memories[newcomer.memory_id] = newcomer
        self._register_sources(sources)

    # -- reads ----------------------------------------------------------
    def hybrid_search(
        self,
        query: StructuredQuery,
        limit: int,
        include_expired: bool = False,
        statuses: Tuple[str, ...] = ("active", "superseded"),
    ) -> List[Tuple[Memory, float, float]]:
        """Keyword-overlap fake for hybrid search (no vectors in Phase 01).

        LOCKED scope semantics (Phase 01 decision, Phase 04 PG must replicate):
        exact match on ``scope_type`` plus ``scope_id`` equality when the
        query sets ``scope.id``. In particular a default ``user``-scoped query
        does NOT match ``global``-scoped memories; callers needing global
        memories must query with a global scope. This mirrors the Phase 03
        hard-filter rule (reject when scope types differ) so fake and SQL
        stay identical.
        """
        for st in statuses:
            if st not in _ALLOWED_STATUSES:
                raise ValueError(f"unknown status in statuses filter: {st}")
        allowed = set(statuses)
        if include_expired:
            allowed.add("expired")

        keywords = [k.lower() for k in (query.keywords or []) if k.strip()]
        qwords = [
            w.strip(".,!?;:\"'()[]").lower()
            for w in query.rewritten_query.split()
            if w.strip()
        ]
        wanted_types = {t.value if hasattr(t, "value") else str(t) for t in (query.memory_types or [])}

        scored: List[Tuple[Memory, float, float]] = []
        for m in self.memories.values():
            status_val = m.status.value if hasattr(m.status, "value") else str(m.status)
            if status_val not in allowed:
                continue
            if wanted_types:
                mtype = m.type.value if hasattr(m.type, "value") else str(m.type)
                if mtype not in wanted_types:
                    continue
            if query.scope is not None:
                q_scope_type = query.scope.type
                q_scope_val = (
                    q_scope_type.value if hasattr(q_scope_type, "value") else str(q_scope_type)
                )
                m_scope_val = (
                    m.scope_type.value
                    if hasattr(m.scope_type, "value")
                    else str(m.scope_type)
                )
                if m_scope_val != q_scope_val:
                    continue
                if query.scope.id is not None and m.scope_id != query.scope.id:
                    continue
            text = m.content.lower()
            lex = float(sum(1 for k in keywords if k and k in text))
            vec = float(sum(1 for w in qwords if w and w in text))
            scored.append((m, vec, lex))

        scored.sort(key=lambda t: (t[1] + t[2], t[0].memory_id), reverse=True)
        if limit is not None:
            scored = scored[:limit]
        return scored

    def touch_retrieved(self, ids: List[str]) -> None:
        now = _utcnow_naive()
        for mid in ids:
            m = self.memories.get(mid)
            if m is None:
                continue
            self.memories[mid] = m.model_copy(update={"last_retrieved_at": now})

    def fetch_sources(self, ids: List[str]) -> List[MemorySource]:
        wanted = set(ids)
        return [s for s in self.sources if s.memory_id in wanted]

    def reset(self) -> None:
        """Clear everything (eval isolation on throwaway stores)."""
        self.memories.clear()
        self.sources.clear()
        self._source_keys.clear()


# ---------------------------------------------------------------------------
# Production backends (Phase 04). Same op contracts as the fakes above.
# ---------------------------------------------------------------------------

_MEM_COLS = (
    "memory_id, content, embedding, type, scope_type, scope_id, confidence,"
    " created_at, valid_from, valid_until, last_retrieved_at, last_updated_at,"
    " status, version, supersedes_id"
)


def _vec_literal(vec: List[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _naive(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _parse_vec(raw) -> List[float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return None
    return [float(x) for x in text.split(",")]


def _row_to_memory(row: tuple) -> Memory:
    (
        memory_id, content, embedding, mtype, scope_type, scope_id,
        confidence, created_at, valid_from, valid_until,
        last_retrieved_at, last_updated_at, status, version, supersedes_id,
    ) = row
    return Memory(
        memory_id=memory_id,
        content=content,
        embedding=_parse_vec(embedding),
        type=mtype,
        scope_type=scope_type,
        scope_id=scope_id,
        confidence=confidence,
        created_at=_naive(created_at),
        valid_from=_naive(valid_from),
        valid_until=_naive(valid_until),
        last_retrieved_at=_naive(last_retrieved_at),
        last_updated_at=_naive(last_updated_at),
        status=status,
        version=version,
        supersedes_id=supersedes_id,
    )


class PostgresDurableStore:
    """Postgres + pgvector + FTS durable store. Same contract as the fake.

    ``embedder`` is optional: when set, hybrid search embeds the rewritten
    query for the vector channel; when None, every ``vs`` is 0.0 and
    ranking is lex-driven (the SQL shape is unchanged either way).
    """

    def __init__(self, dsn: str, embedding_dim: int, embedder=None) -> None:
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self._embedder = embedder

    def _connect(self):
        dsn = self.dsn
        if "connect_timeout" not in dsn:
            sep = "&" if "?" in dsn else "?"
            dsn = f"{dsn}{sep}connect_timeout=5"
        return psycopg.connect(dsn)

    def init_schema(self) -> None:
        ddl = render_ddl(self.embedding_dim)
        with self._connect() as conn:
            with conn.cursor() as cur:
                for stmt in split_ddl_statements(ddl):
                    cur.execute(stmt)

    def insert(self, memory: Memory, sources: List[MemorySource]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO memories (memory_id, content, embedding, type,"
                        " scope_type, scope_id, confidence, created_at, valid_from,"
                        " valid_until, last_retrieved_at, last_updated_at, status,"
                        " version, supersedes_id) VALUES"
                        " (%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (
                            memory.memory_id, memory.content,
                            _vec_literal(memory.embedding),
                            memory.type.value, memory.scope_type.value,
                            memory.scope_id, memory.confidence,
                            memory.created_at, memory.valid_from,
                            memory.valid_until, memory.last_retrieved_at,
                            memory.last_updated_at, memory.status.value,
                            memory.version, memory.supersedes_id,
                        ),
                    )
                except psycopg.errors.UniqueViolation as exc:
                    raise ValueError(
                        f"duplicate memory_id: {memory.memory_id}"
                    ) from exc
                for s in sources:
                    try:
                        cur.execute(
                            "INSERT INTO memory_sources"
                            " (memory_id, session_id, message_idx, excerpt)"
                            " VALUES (%s,%s,%s,%s)",
                            (s.memory_id, s.session_id, s.message_idx, s.excerpt),
                        )
                    except psycopg.errors.UniqueViolation as exc:
                        raise ValueError(
                            "duplicate memory_source key:"
                            f" {(s.memory_id, s.session_id)}"
                        ) from exc

    def merge(
        self, memory_id: str, new_evidence: MemorySource, confidence: float
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET confidence=%s,"
                    " last_updated_at=now(), version=version+1"
                    " WHERE memory_id=%s",
                    (confidence, memory_id),
                )
                if cur.rowcount == 0:
                    raise KeyError(f"unknown memory_id: {memory_id}")
                cur.execute(
                    "INSERT INTO memory_sources"
                    " (memory_id, session_id, message_idx, excerpt)"
                    " VALUES (%s,%s,%s,%s)"
                    " ON CONFLICT (memory_id, session_id) DO NOTHING",
                    (
                        new_evidence.memory_id, new_evidence.session_id,
                        new_evidence.message_idx, new_evidence.excerpt,
                    ),
                )

    def supersede(
        self, old_id: str, new: Memory, sources: List[MemorySource]
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET status='superseded',"
                    " valid_until=now() WHERE memory_id=%s",
                    (old_id,),
                )
                if cur.rowcount == 0:
                    raise KeyError(f"unknown memory_id: {old_id}")
                try:
                    cur.execute(
                        "INSERT INTO memories (memory_id, content, embedding, type,"
                        " scope_type, scope_id, confidence, created_at, valid_from,"
                        " valid_until, last_retrieved_at, last_updated_at, status,"
                        " version, supersedes_id) VALUES"
                        " (%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (
                            new.memory_id, new.content,
                            _vec_literal(new.embedding),
                            new.type.value, new.scope_type.value,
                            new.scope_id, new.confidence,
                            new.created_at, new.valid_from,
                            new.valid_until, new.last_retrieved_at,
                            new.last_updated_at, new.status.value,
                            new.version, old_id,
                        ),
                    )
                except psycopg.errors.UniqueViolation as exc:
                    raise ValueError(
                        f"duplicate memory_id: {new.memory_id}"
                    ) from exc
                for s in sources:
                    try:
                        cur.execute(
                            "INSERT INTO memory_sources"
                            " (memory_id, session_id, message_idx, excerpt)"
                            " VALUES (%s,%s,%s,%s)",
                            (s.memory_id, s.session_id, s.message_idx, s.excerpt),
                        )
                    except psycopg.errors.UniqueViolation as exc:
                        raise ValueError(
                            "duplicate memory_source key:"
                            f" {(s.memory_id, s.session_id)}"
                        ) from exc

    def hybrid_search(
        self,
        query: StructuredQuery,
        limit: int,
        include_expired: bool = False,
        statuses: Tuple[str, ...] = ("active", "superseded"),
    ) -> List[Tuple[Memory, float, float]]:
        unknown = set(statuses) - _ALLOWED_STATUSES
        if unknown:
            raise ValueError(f"unknown status in statuses filter: {sorted(unknown)}")
        allowed = list(statuses)
        if include_expired and "expired" not in allowed:
            allowed.append("expired")
        qvec = None
        if self._embedder is not None and query.rewritten_query.strip():
            vecs = self._embedder.embed([query.rewritten_query])
            qvec = _vec_literal(vecs[0]) if vecs else None
        keywords_text = " ".join(query.keywords or [])
        params: dict = {
            "has_vec": qvec is not None,
            "qvec": qvec,
            "kw": keywords_text,
            "statuses": allowed,
            "limit": limit,
        }
        sql = (
            "SELECT * FROM (SELECT memory_id, content, embedding::text, type, scope_type,"
            " scope_id, confidence, created_at, valid_from, valid_until,"
            " last_retrieved_at, last_updated_at, status, version, supersedes_id,"
            " COALESCE(CASE WHEN %(has_vec)s"
            " THEN 1 - (embedding <=> %(qvec)s::vector) ELSE 0.0 END, 0.0) AS vs,"
            " COALESCE(ts_rank(search_tsv,"
            " plainto_tsquery('english', %(kw)s)), 0.0) AS ls"
            " FROM memories WHERE status = ANY(%(statuses)s)"
        )
        if query.memory_types:
            params["types"] = [
                t.value if hasattr(t, "value") else str(t)
                for t in query.memory_types
            ]
            sql += " AND type = ANY(%(types)s)"
        if query.scope is not None:
            q_scope = query.scope.type
            params["scope_type"] = (
                q_scope.value if hasattr(q_scope, "value") else str(q_scope)
            )
            sql += " AND scope_type = %(scope_type)s"
            if query.scope.id is not None:
                params["scope_id"] = query.scope.id
                sql += " AND scope_id = %(scope_id)s"
        sql += " ) s ORDER BY (s.vs + s.ls) DESC, s.memory_id LIMIT %(limit)s"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            (_row_to_memory(row[:15]), float(row[15]), float(row[16]))
            for row in rows
        ]

    def touch_retrieved(self, ids: List[str]) -> None:
        if not ids:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET last_retrieved_at=now()"
                    " WHERE memory_id = ANY(%s)",
                    (list(ids),),
                )

    def fetch_memory(self, memory_id: str) -> Memory | None:
        """Single-row read for tests/debug (not on the hot path)."""
        cols = _MEM_COLS.replace("embedding", "embedding::text")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT " + cols + " FROM memories WHERE memory_id=%s",
                    (memory_id,),
                )
                row = cur.fetchone()
        return _row_to_memory(row) if row is not None else None

    def fetch_sources(self, ids: List[str]) -> List[MemorySource]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT memory_id, session_id, message_idx, excerpt"
                    " FROM memory_sources WHERE memory_id = ANY(%s)"
                    " ORDER BY memory_id, session_id",
                    (list(ids),),
                )
                rows = cur.fetchall()
        return [
            MemorySource(
                memory_id=row[0], session_id=row[1],
                message_idx=row[2], excerpt=row[3],
            )
            for row in rows
        ]

    def reset(self) -> None:
        """Delete all rows (eval isolation; production uses fresh DBs)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory_sources")
                cur.execute("DELETE FROM memories")


class RedisWorkingStore:
    """Redis working store: ordered turn lists plus session state."""

    def __init__(
        self,
        url: str,
        ttl_s: int = _DEFAULT_REDIS_TTL_S,
        socket_connect_timeout: int = 5,
    ) -> None:
        self._client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=socket_connect_timeout,
        )
        self.ttl_s = ttl_s

    def _turns_key(self, session_id: str) -> str:
        return f"wm:turns:{session_id}"

    def _state_key(self, session_id: str) -> str:
        return f"wm:state:{session_id}"

    def append_turn(self, session_id: str, turn: Turn | dict) -> None:
        if isinstance(turn, Turn):
            payload = turn.model_dump_json()
        else:
            payload = json.dumps(
                {"role": turn.get("role"), "content": turn.get("content")}
            )
        key = self._turns_key(session_id)
        self._client.rpush(key, payload)
        self._client.expire(key, self.ttl_s)

    def get_state(self, session_id: str) -> dict:
        raw = self._client.get(self._state_key(session_id))
        if raw is None:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def set_state(self, session_id: str, state: dict) -> None:
        key = self._state_key(session_id)
        self._client.set(key, json.dumps(dict(state)))
        self._client.expire(key, self.ttl_s)

    def turn_count(self, session_id: str) -> int:
        return int(self._client.llen(self._turns_key(session_id)))

    def delete_session(self, session_id: str) -> None:
        """Drop one session's turns + state (eval isolation)."""
        self._client.delete(self._turns_key(session_id), self._state_key(session_id))
