"""Tests for TieredMemory: WorkingMemory, EpisodicMemory, SemanticMemory."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from nexagent.memory.tiered import (
    EpisodicMemory,
    MemoryTier,
    SemanticMemory,
    TieredMemory,
    WorkingMemory,
)

# Small dimension for fast tests — overrides the default 768.
_DIM = 8


@pytest.fixture
def fake_embed():
    """Deterministic embedding: each unique text maps to a reproducible unit vector."""

    async def _embed(text: str) -> list[float]:
        vec = np.zeros(_DIM, dtype=np.float32)
        for i, ch in enumerate(text):
            vec[i % _DIM] += ord(ch)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    return _embed


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_set_and_get_roundtrip(self) -> None:
        mem = WorkingMemory()
        mem.set("task", {"title": "Buy groceries"})
        assert mem.get("task") == {"title": "Buy groceries"}

    def test_get_missing_key_returns_none(self) -> None:
        assert WorkingMemory().get("nonexistent") is None

    def test_ttl_item_available_before_expiry(self) -> None:
        mem = WorkingMemory()
        mem.set("temp", "value", ttl=10.0)
        assert mem.get("temp") == "value"

    def test_ttl_item_expires_after_deadline(self) -> None:
        mem = WorkingMemory()
        mem.set("temp", "value", ttl=0.02)
        assert mem.get("temp") == "value"
        time.sleep(0.05)
        assert mem.get("temp") is None

    def test_no_ttl_item_persists_across_time(self) -> None:
        mem = WorkingMemory()
        mem.set("permanent", "here forever", ttl=None)
        time.sleep(0.05)
        assert mem.get("permanent") == "here forever"

    def test_delete_returns_true_and_removes_key(self) -> None:
        mem = WorkingMemory()
        mem.set("k", "v")
        assert mem.delete("k") is True
        assert mem.get("k") is None

    def test_delete_missing_key_returns_false(self) -> None:
        assert WorkingMemory().delete("not_here") is False

    def test_all_keys_excludes_expired_entries(self) -> None:
        mem = WorkingMemory()
        mem.set("alive", 1)
        mem.set("dying", 2, ttl=0.02)
        time.sleep(0.05)
        keys = mem.all_keys()
        assert "alive" in keys
        assert "dying" not in keys

    def test_clear_empties_all_entries(self) -> None:
        mem = WorkingMemory()
        mem.set("x", 1)
        mem.set("y", 2)
        mem.clear()
        assert mem.all_keys() == []
        assert mem.get("x") is None

    def test_clear_simulates_new_session(self) -> None:
        """WorkingMemory.clear() is how the caller resets state between sessions."""
        mem = WorkingMemory()
        mem.set("session_context", "user is editing auth.py")
        mem.clear()
        assert mem.get("session_context") is None
        assert mem.all_keys() == []

    def test_search_finds_matching_value(self) -> None:
        mem = WorkingMemory()
        mem.set("current_task", "deploy kubernetes cluster")
        mem.set("note", "water plants")
        results = mem.search("kubernetes")
        assert len(results) == 1
        assert results[0].content == "deploy kubernetes cluster"
        assert results[0].tier == MemoryTier.WORKING

    def test_search_matches_on_key_name(self) -> None:
        mem = WorkingMemory()
        mem.set("kubernetes_context", "some value")
        results = mem.search("kubernetes")
        assert len(results) == 1

    def test_search_no_match_returns_empty_list(self) -> None:
        mem = WorkingMemory()
        mem.set("note", "nothing relevant here")
        assert mem.search("kubernetes") == []

    def test_overwrite_replaces_value(self) -> None:
        mem = WorkingMemory()
        mem.set("key", "original")
        mem.set("key", "updated")
        assert mem.get("key") == "updated"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    def test_store_and_retrieve_basic(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-1", "sess-A", "user", "What is the weather today?")
        items = db.recent(session_id="sess-A")
        assert len(items) == 1
        assert items[0].content == "What is the weather today?"
        assert items[0].tier == MemoryTier.EPISODIC

    def test_stored_event_has_created_at_timestamp(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        t_before = time.time()
        db.store("ep-2", "sess-B", "assistant", "It is sunny today!")
        t_after = time.time()
        items = db.recent(session_id="sess-B")
        created_at = items[0].metadata["created_at"]
        assert t_before <= created_at <= t_after

    def test_later_events_have_larger_timestamps(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-first", "sess-C", "user", "First event")
        time.sleep(0.02)
        db.store("ep-second", "sess-C", "user", "Second event")
        items = db.recent(session_id="sess-C")
        # recent() returns most-recent first
        ts_second = items[0].metadata["created_at"]
        ts_first = items[1].metadata["created_at"]
        assert ts_second > ts_first

    def test_persists_across_separate_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ep.sqlite"
        EpisodicMemory(db_path=db_path).store("ep-3", "sess-D", "user", "Remember this!")
        items = EpisodicMemory(db_path=db_path).recent(session_id="sess-D")
        assert len(items) == 1
        assert items[0].content == "Remember this!"

    def test_recent_returns_most_recent_first(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-old", "sess-E", "user", "Old message")
        time.sleep(0.02)
        db.store("ep-new", "sess-E", "user", "New message")
        items = db.recent(session_id="sess-E")
        assert items[0].content == "New message"
        assert items[1].content == "Old message"

    def test_search_text_finds_matching_content(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-k8s", "sess-F", "user", "Deploy the Kubernetes cluster")
        db.store("ep-pg", "sess-F", "user", "Check the postgres database")
        results = db.search_text("Kubernetes")
        assert len(results) == 1
        assert "Kubernetes" in results[0].content

    def test_search_text_no_match_returns_empty(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-x", "sess-G", "user", "nothing relevant here")
        assert db.search_text("kubernetes") == []

    def test_session_isolation(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-p", "sess-P", "user", "Private message for P")
        db.store("ep-q", "sess-Q", "user", "Private message for Q")
        p_items = db.recent(session_id="sess-P")
        assert len(p_items) == 1
        assert p_items[0].content == "Private message for P"

    def test_metadata_stored_and_retrieved(self, tmp_path: Path) -> None:
        db = EpisodicMemory(db_path=tmp_path / "ep.sqlite")
        db.store("ep-m", "sess-M", "user", "content", metadata={"tag": "important"})
        items = db.recent(session_id="sess-M")
        assert items[0].metadata.get("tag") == "important"


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    @pytest.mark.asyncio
    async def test_store_and_search_returns_results(self, tmp_path: Path, fake_embed) -> None:
        sem = SemanticMemory(store_path=tmp_path / "sem", dim=_DIM)
        await sem.store("doc-1", "machine learning tutorial", embed_fn=fake_embed)
        results = await sem.search("machine learning", embed_fn=fake_embed, top_k=1, min_score=0.0)
        assert len(results) == 1
        assert results[0].id == "doc-1"
        assert results[0].tier == MemoryTier.SEMANTIC

    @pytest.mark.asyncio
    async def test_search_returns_most_similar_document(self, tmp_path: Path, fake_embed) -> None:
        """The document with the closest embedding to the query is ranked first."""
        sem = SemanticMemory(store_path=tmp_path / "sem", dim=_DIM)
        await sem.store("py", "python programming language", embed_fn=fake_embed)
        await sem.store("bk", "sourdough bread baking recipe", embed_fn=fake_embed)
        # Exact-match query — "py" must be the top hit
        results = await sem.search(
            "python programming language", embed_fn=fake_embed, top_k=1, min_score=0.0
        )
        assert results[0].id == "py"
        assert results[0].content == "python programming language"

    @pytest.mark.asyncio
    async def test_min_score_filters_low_scoring_results(self, tmp_path: Path, fake_embed) -> None:
        sem = SemanticMemory(store_path=tmp_path / "sem", dim=_DIM)
        await sem.store("unrelated", "xyzzy qwerty plugh zork abc", embed_fn=fake_embed)
        results = await sem.search(
            "python programming", embed_fn=fake_embed, top_k=5, min_score=0.99
        )
        for r in results:
            assert r.score >= 0.99

    @pytest.mark.asyncio
    async def test_empty_store_returns_no_results(self, tmp_path: Path, fake_embed) -> None:
        sem = SemanticMemory(store_path=tmp_path / "sem", dim=_DIM)
        results = await sem.search("anything at all", embed_fn=fake_embed, top_k=5, min_score=0.0)
        assert results == []


# ---------------------------------------------------------------------------
# Memory tier independence
# ---------------------------------------------------------------------------


class TestTierIndependence:
    def test_working_and_episodic_stores_are_separate(self, tmp_path: Path) -> None:
        mem = TieredMemory(base_path=tmp_path)
        mem.working.set("working_only", "in RAM")
        mem.episodic.store("ep-only", "sess-Z", "user", "in SQLite")

        # Working memory has no knowledge of episodic entries
        assert mem.working.get("ep-only") is None
        # Episodic has no knowledge of working entries
        episodic_contents = [i.content for i in mem.episodic.recent()]
        assert "in RAM" not in episodic_contents

    def test_clear_working_does_not_delete_episodic(self, tmp_path: Path) -> None:
        mem = TieredMemory(base_path=tmp_path)
        mem.episodic.store("ep-persist", "sess-P", "user", "durable event")
        mem.working.set("transient", "gone soon")

        mem.working.clear()

        # Episodic data survives the working-memory clear
        items = mem.episodic.recent(session_id="sess-P")
        assert len(items) == 1
        assert items[0].content == "durable event"
        assert mem.working.get("transient") is None

    @pytest.mark.asyncio
    async def test_retrieve_surfaces_working_memory_hits(self, tmp_path: Path) -> None:
        mem = TieredMemory(base_path=tmp_path)
        mem.working.set("current_plan", "refactor the authentication module")
        results = await mem.retrieve("authentication module")
        assert any(item.tier == MemoryTier.WORKING for item in results)

    @pytest.mark.asyncio
    async def test_retrieve_surfaces_episodic_memory_hits(self, tmp_path: Path) -> None:
        mem = TieredMemory(base_path=tmp_path)
        mem.episodic.store("ep-r", "sess-R", "user", "deploy redis cluster to production")
        results = await mem.retrieve("redis cluster")
        assert any(item.tier == MemoryTier.EPISODIC for item in results)

    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_by_id(self, tmp_path: Path) -> None:
        """The same id from two tiers must appear at most once in merged results."""
        mem = TieredMemory(base_path=tmp_path)
        # "alpha" exists as a working-memory key AND as an episodic episode_id
        mem.working.set("alpha", "kubernetes content")
        mem.episodic.store("alpha", "sess-dup", "user", "kubernetes deployment steps")
        results = await mem.retrieve("kubernetes")
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate IDs in retrieve() results"

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, tmp_path: Path) -> None:
        mem = TieredMemory(base_path=tmp_path)
        for i in range(10):
            mem.working.set(f"key_{i}", "kubernetes cluster node")
        results = await mem.retrieve("kubernetes", top_k=3)
        assert len(results) <= 3
