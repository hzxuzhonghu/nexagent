"""Task success/failure tracking and skill patch proposals.

The tracker records every task outcome and periodically analyses failure
patterns. When a recurring failure pattern is identified, it proposes a
skill patch — a structured update to the procedural memory tier.

Skill patches require SYSTEM-level approval before activation.

Usage::

    tracker = ImprovementTracker(memory=mem)
    await tracker.record(outcome=TaskOutcome(
        task_id="task-1",
        prompt="Find the latest research on quantum computing",
        success=True,
        tool_calls=[{"name": "web_search", "args": {...}}],
        duration_ms=1200,
        tokens_used=500,
        cost_usd=0.003,
    ))
    proposals = tracker.analyse_failures()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TRACKER_PATH = Path.home() / ".nexagent" / "improvement.sqlite"
MIN_FAILURES_FOR_PROPOSAL = 3


@dataclass
class TaskOutcome:
    """Record of a single task execution."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    success: bool = True
    partial: bool = False
    error_message: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    session_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillPatchProposal:
    """A proposed update to the procedural memory tier."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skill_name: str = ""
    description: str = ""
    trigger_patterns: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    rationale: str = ""
    failure_examples: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | approved | rejected
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImprovementTracker:
    """Tracks task outcomes and proposes skill patches for recurring failures.

    Parameters
    ----------
    db_path:
        SQLite database for storing outcomes and proposals.
    min_failures:
        Minimum number of failures with a shared pattern before a patch
        is proposed.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        min_failures: int = MIN_FAILURES_FOR_PROPOSAL,
    ) -> None:
        self._db_path = db_path or DEFAULT_TRACKER_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._min_failures = min_failures
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id         TEXT UNIQUE NOT NULL,
                    prompt          TEXT NOT NULL,
                    success         INTEGER NOT NULL,
                    partial         INTEGER NOT NULL DEFAULT 0,
                    error_message   TEXT,
                    tool_calls_json TEXT DEFAULT '[]',
                    duration_ms     REAL DEFAULT 0,
                    tokens_used     INTEGER DEFAULT 0,
                    cost_usd        REAL DEFAULT 0,
                    session_id      TEXT,
                    timestamp       REAL NOT NULL,
                    metadata_json   TEXT DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id     TEXT PRIMARY KEY,
                    skill_name      TEXT NOT NULL,
                    description     TEXT,
                    proposal_json   TEXT NOT NULL,
                    status          TEXT DEFAULT 'pending',
                    created_at      REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcomes_ts ON outcomes(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcomes_success ON outcomes(success)"
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

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record(self, outcome: TaskOutcome) -> None:
        """Persist a task outcome."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes
                    (task_id, prompt, success, partial, error_message,
                     tool_calls_json, duration_ms, tokens_used, cost_usd,
                     session_id, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.task_id,
                    outcome.prompt,
                    int(outcome.success),
                    int(outcome.partial),
                    outcome.error_message,
                    json.dumps(outcome.tool_calls),
                    outcome.duration_ms,
                    outcome.tokens_used,
                    outcome.cost_usd,
                    outcome.session_id,
                    outcome.timestamp,
                    json.dumps(outcome.metadata),
                ),
            )
            conn.commit()
        logger.debug(
            "Recorded outcome task=%s success=%s", outcome.task_id, outcome.success
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def recent_failures(self, limit: int = 100) -> list[TaskOutcome]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM outcomes WHERE success=0 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_outcome(r) for r in rows]

    def success_rate(self, last_n: int = 100) -> float:
        """Return the success rate over the last n outcomes."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(success) as successes
                FROM (
                    SELECT success FROM outcomes ORDER BY timestamp DESC LIMIT ?
                )
                """,
                (last_n,),
            ).fetchone()
        if not row or row["total"] == 0:
            return 1.0
        return row["successes"] / row["total"]

    def analyse_failures(self) -> list[SkillPatchProposal]:
        """Identify recurring failure patterns and propose skill patches.

        Simple pattern detection: group failures by shared keywords in the
        prompt. For real deployments, replace with semantic clustering.
        """
        failures = self.recent_failures(limit=200)
        if len(failures) < self._min_failures:
            return []

        # Group by first significant noun/verb phrase (heuristic: first 3 words)
        groups: dict[str, list[TaskOutcome]] = {}
        for f in failures:
            key = " ".join(f.prompt.lower().split()[:3])
            groups.setdefault(key, []).append(f)

        proposals = []
        for pattern_key, group in groups.items():
            if len(group) < self._min_failures:
                continue

            # Check we haven't already proposed for this pattern
            if self._proposal_exists_for_pattern(pattern_key):
                continue

            proposal = SkillPatchProposal(
                skill_name=f"fix_{pattern_key.replace(' ', '_')}",
                description=f"Proposed fix for recurring failures matching: '{pattern_key}'",
                trigger_patterns=[pattern_key],
                steps=[
                    f"Detected {len(group)} failures matching pattern '{pattern_key}'.",
                    "Review the error messages below and revise the approach.",
                    *(
                        f"Failure example: {f.error_message or f.prompt[:80]}"
                        for f in group[:3]
                    ),
                ],
                rationale=(
                    f"Pattern '{pattern_key}' failed {len(group)} times. "
                    "A skill definition may improve reliability."
                ),
                failure_examples=[f.task_id for f in group[:5]],
            )
            self._save_proposal(proposal)
            proposals.append(proposal)
            logger.info(
                "Skill patch proposed: %s (based on %d failures)",
                proposal.skill_name,
                len(group),
            )

        return proposals

    def pending_proposals(self) -> list[SkillPatchProposal]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT proposal_json FROM proposals WHERE status='pending' ORDER BY created_at DESC"
            ).fetchall()
        return [SkillPatchProposal(**json.loads(r["proposal_json"])) for r in rows]

    def approve_proposal(self, proposal_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE proposals SET status='approved' WHERE proposal_id=? AND status='pending'",
                (proposal_id,),
            )
            conn.commit()
        approved = cursor.rowcount > 0
        if approved:
            logger.info("Skill patch proposal approved: %s", proposal_id)
        return approved

    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE proposals SET status='rejected' WHERE proposal_id=? AND status='pending'",
                (proposal_id,),
            )
            conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) as total, SUM(success) as successes, "
                "SUM(cost_usd) as total_cost, SUM(tokens_used) as total_tokens "
                "FROM outcomes"
            ).fetchone()
            proposal_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM proposals WHERE status='pending'"
            ).fetchone()

        total = totals["total"] or 0
        successes = totals["successes"] or 0
        return {
            "total_tasks": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total else 1.0,
            "total_cost_usd": round(totals["total_cost"] or 0, 4),
            "total_tokens": totals["total_tokens"] or 0,
            "pending_proposals": proposal_count["cnt"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_proposal(self, proposal: SkillPatchProposal) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO proposals
                    (proposal_id, skill_name, description, proposal_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.skill_name,
                    proposal.description,
                    json.dumps(proposal.to_dict()),
                    proposal.status,
                    proposal.created_at,
                ),
            )
            conn.commit()

    def _proposal_exists_for_pattern(self, pattern: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM proposals WHERE skill_name LIKE ? AND status='pending' LIMIT 1",
                (f"%{pattern.replace(' ', '_')}%",),
            ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_outcome(row: sqlite3.Row) -> TaskOutcome:
        return TaskOutcome(
            task_id=row["task_id"],
            prompt=row["prompt"],
            success=bool(row["success"]),
            partial=bool(row["partial"]),
            error_message=row["error_message"],
            tool_calls=json.loads(row["tool_calls_json"] or "[]"),
            duration_ms=row["duration_ms"],
            tokens_used=row["tokens_used"],
            cost_usd=row["cost_usd"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
