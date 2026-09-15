"""Pydantic contracts. No I/O — raw dicts in, validated models out."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryType(str, Enum):
    semantic = "semantic"
    episodic = "episodic"
    procedural = "procedural"
    preference = "preference"


class ScopeType(str, Enum):
    user = "user"
    agent = "agent"
    project = "project"
    global_ = "global"

    @classmethod
    def _missing_(cls, value):  # allow "global" string for global_ member
        if value == "global":
            return cls.global_
        return None


class MemoryStatus(str, Enum):
    active = "active"
    superseded = "superseded"
    expired = "expired"


class Relationship(str, Enum):
    no_relation = "no_relation"
    similar = "similar"
    contradiction = "contradiction"


class Scope(BaseModel):
    type: ScopeType = ScopeType.user
    id: Optional[str] = None


class MemoryCandidate(BaseModel):
    should_store: bool
    content: str = ""
    type: MemoryType = MemoryType.semantic
    scope: Scope = Field(default_factory=Scope)
    confidence_basis: str = ""
    valid_from: Optional[datetime] = None

    @model_validator(mode="after")
    def _content_required_when_storing(self):
        if self.should_store and not self.content.strip():
            raise ValueError("content must be non-empty when should_store=true")
        return self


class Memory(BaseModel):
    memory_id: str
    content: str = Field(min_length=1)
    embedding: Optional[List[float]] = None
    type: MemoryType
    scope_type: ScopeType
    scope_id: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    created_at: datetime
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    last_retrieved_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    status: MemoryStatus = MemoryStatus.active
    version: int = Field(default=1, ge=1)
    supersedes_id: Optional[str] = None


class MemorySource(BaseModel):
    memory_id: str
    session_id: str
    message_idx: Optional[int] = None
    excerpt: str


class StructuredQuery(BaseModel):
    """Retrieval query. No temporal value, no question_time here by design.

    temporal_confidence in [0, 1] records how strongly the query constrains
    time. question_time itself is supplied externally to retrieve()/scoring.
    """

    rewritten_query: str
    intent: str = ""
    entities: List[str] = Field(default_factory=list)
    memory_types: List[MemoryType] = Field(default_factory=list)
    scope: Scope = Field(default_factory=Scope)
    keywords: List[str] = Field(default_factory=list)
    temporal_confidence: float = Field(default=0.0, ge=0, le=1)


class ScoredMemory(BaseModel):
    memory: Memory
    scores: Dict[str, float] = Field(default_factory=dict)
    final_score: float = 0.0
    temporal_confidence: float = Field(default=0.0, ge=0, le=1)


class MemoryContext(BaseModel):
    active: List[Memory] = Field(default_factory=list)
    historical: List[Memory] = Field(default_factory=list)
    evidence: List[MemorySource] = Field(default_factory=list)

    def to_prompt(self) -> str:
        lines: List[str] = ["## Active Memories"]
        if self.active:
            for m in self.active:
                lines.append(f"- [{m.memory_id}] {m.content}")
        else:
            lines.append("(none)")
        lines.append("## Historical Memories")
        if self.historical:
            for m in self.historical:
                lines.append(f"- [{m.memory_id}] {m.content}")
        else:
            lines.append("(none)")
        lines.append("## Evidence")
        if self.evidence:
            for s in self.evidence:
                lines.append(f"- [{s.memory_id}/{s.session_id}] {s.excerpt}")
        else:
            lines.append("(none)")
        return "\n".join(lines)


class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


Operation = Literal["CREATE", "MERGE", "SUPERSEDE", "DISCARD"]
