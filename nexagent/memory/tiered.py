"""Tiered memory model: working / episodic / semantic / procedural.

Retrieval priority:
  1. Working memory  — in-RAM, current session
  2. Episodic memory — SQLite, recent turns across sessions
  3. Semantic memory — vector search for relevance
  4. Procedural memory — named skill definitions

All tiers share a common MemoryItem schema to make retrieval results
composable regardless of tier.

Usage::

    mem = TieredMemory(base_path=Path("~/.nexagent/memory"))
    mem.working.set("last_task", {"title": "Buy groceries"})

    # Retrieve from the most relevant tier automatically
    results = await mem.retrieve("What was my last task?", embedding=embed_fn)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

import numpy as np

from nexagent.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[list[float]]]


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryItem:
    id: str
    tier: MemoryTier
    content: Any
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Working Memory — in-process dict with optional TTL
# ---------------------------------------------------------------------------


class WorkingMemory:
    """Fast in-RAM key/value store for the current session.

    Items are stored as (value, expiry_ts). Pass ttl=None for no expiry.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expiry = time.monotonic() + ttl if ttl is not None else None
        self._store[key] = (value, expiry)

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and time.monotonic() > expiry:
            del self._store[key]
            return None
        return value

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def all_keys(self) -> list[str]:
        now = time.monotonic()
        return [k for k, (_, exp) in self._store.items() if exp is None or now <= exp]

    def search(self, query: str) -> list[MemoryItem]:
        """Simple substring search over working memory keys and string values."""
        q = query.lower()
        results = []
        for key in self.all_keys():
            val = self.get(key)
            if val is None:
                continue
            haystack = f"{key} {val}".lower()
            if q in haystack:
                results.append(
                    MemoryItem(
                        id=key,
                        tier=MemoryTier.WORKING,
                        content=val,
                        score=1.0,
                        metadata={"key": key},
                    )
                )
        return results

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Episodic Memory — SQLite-backed conversation turns
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """SQLite-backed storage for conversation turns and events.

    Uses WAL mode for better concurrent read performance.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    TEXT DEFAULT '{}',
                    created_at  REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(created_at DESC)"
            )
            conn.commit()

    @contextmanager
    def _connect(self):  # type: ignore[return]
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def store(
        self,
        episode_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                    (id, session_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    session_id,
                    role,
                    content,
                    json.dumps(metadata or {}),
                    time.time(),
                ),
            )
            conn.commit()

    def recent(self, session_id: str | None = None, limit: int = 50) -> list[MemoryItem]:
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM episodes WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        return [
            MemoryItem(
                id=row["id"],
                tier=MemoryTier.EPISODIC,
                content=row["content"],
                score=1.0,
                metadata={
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "created_at": row["created_at"],
                    **json.loads(row["metadata"]),
                },
            )
            for row in rows
        ]

    def search_text(self, query: str, limit: int = 20) -> list[MemoryItem]:
        """Full-text substring search (adequate for small datasets)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                tier=MemoryTier.EPISODIC,
                content=row["content"],
                score=0.8,
                metadata={"session_id": row["session_id"], "role": row["role"]},
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Semantic Memory — vector-search tier
# ---------------------------------------------------------------------------


class SemanticMemory:
    """Vector-search tier backed by VectorStore.

    Requires an embedding function to be passed at query time. The store is
    lazily loaded from disk on first use.
    """

    EMBEDDING_DIM = 768

    def __init__(self, store_path: Path, dim: int = EMBEDDING_DIM) -> None:
        self._path = store_path
        self._dim = dim
        self._store: VectorStore | None = None

    def _get_store(self) -> VectorStore:
        if self._store is None:
            npz = self._path.with_suffix(".npz")
            if npz.exists():
                self._store = VectorStore.load(self._path)
            else:
                self._store = VectorStore(dim=self._dim)
        return self._store

    async def store(
        self,
        doc_id: str,
        text: str,
        embed_fn: EmbedFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        embedding = await embed_fn(text)
        vec = np.asarray(embedding, dtype=np.float32)
        self._get_store().add(doc_id, vec, {"text": text, **(metadata or {})})

    async def search(
        self,
        query: str,
        embed_fn: EmbedFn,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[MemoryItem]:
        embedding = await embed_fn(query)
        vec = np.asarray(embedding, dtype=np.float32)
        raw = self._get_store().search(vec, top_k=top_k, min_score=min_score)
        return [
            MemoryItem(
                id=r.id,
                tier=MemoryTier.SEMANTIC,
                content=r.metadata.get("text", ""),
                score=r.score,
                metadata=r.metadata,
            )
            for r in raw
        ]

    def persist(self) -> None:
        if self._store and self._store.size > 0:
            self._store.save(self._path)


# ---------------------------------------------------------------------------
# Procedural Memory — named skill definitions
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    name: str
    description: str
    trigger_patterns: list[str]
    steps: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    active: bool = True


class ProceduralMemory:
    """JSON-file-backed store for named skill definitions.

    Skills describe repeatable multi-step procedures. They are matched
    by trigger pattern at query time.
    """

    def __init__(self, skills_path: Path) -> None:
        self._path = skills_path
        self._skills: dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with self._path.open() as f:
                raw = json.load(f)
            for name, data in raw.items():
                self._skills[name] = Skill(**data)
            logger.debug("Loaded %d skills from %s", len(self._skills), self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            json.dump(
                {name: vars(skill) for name, skill in self._skills.items()},
                f,
                indent=2,
            )

    def add_skill(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        self._save()

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def match(self, query: str) -> list[MemoryItem]:
        """Find skills whose trigger patterns match the query."""
        q = query.lower()
        results = []
        for skill in self._skills.values():
            if not skill.active:
                continue
            for pattern in skill.trigger_patterns:
                if pattern.lower() in q:
                    results.append(
                        MemoryItem(
                            id=skill.name,
                            tier=MemoryTier.PROCEDURAL,
                            content=skill,
                            score=0.9,
                            metadata={"description": skill.description},
                        )
                    )
                    break
        return results

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())


# ---------------------------------------------------------------------------
# TieredMemory — unified facade
# ---------------------------------------------------------------------------


class TieredMemory:
    """Unified facade over all four memory tiers.

    Parameters
    ----------
    base_path:
        Root directory for persistent memory. Defaults to ~/.nexagent/memory.
    embedding_dim:
        Dimension of vectors for the semantic tier.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        embedding_dim: int = SemanticMemory.EMBEDDING_DIM,
    ) -> None:
        if base_path is None:
            base_path = Path.home() / ".nexagent" / "memory"
        base_path.mkdir(parents=True, exist_ok=True)

        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path=base_path / "episodes.sqlite")
        self.semantic = SemanticMemory(
            store_path=base_path / "semantic", dim=embedding_dim
        )
        self.procedural = ProceduralMemory(skills_path=base_path / "skills.json")
        self._base_path = base_path

    async def retrieve(
        self,
        query: str,
        embed_fn: EmbedFn | None = None,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """Query all tiers and return a merged, ranked list of results.

        Tiers are queried in priority order; duplicates (by id) are removed.
        """
        seen_ids: set[str] = set()
        results: list[MemoryItem] = []

        # 1. Working memory
        for item in self.working.search(query):
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                results.append(item)

        # 2. Episodic memory
        for item in self.episodic.search_text(query):
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                results.append(item)

        # 3. Semantic memory (only if we have an embedding function)
        if embed_fn is not None:
            for item in await self.semantic.search(query, embed_fn=embed_fn, top_k=top_k):
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    results.append(item)

        # 4. Procedural memory
        for item in self.procedural.match(query):
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                results.append(item)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def persist_all(self) -> None:
        """Flush all dirty in-memory state to disk."""
        self.semantic.persist()
