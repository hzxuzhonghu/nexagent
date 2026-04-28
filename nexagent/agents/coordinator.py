"""Multi-agent coordinator with stateful DAG and checkpoints.

The coordinator manages a directed acyclic graph (DAG) of agent tasks. It:
  - Executes ready nodes concurrently using anyio task groups
  - Writes a checkpoint after each node completes
  - Resumes from the last checkpoint on failure
  - Emits a final aggregated result

Usage::

    coordinator = AgentCoordinator(registry=registry, policy=policy)

    run_id = await coordinator.submit(
        graph=TaskGraph(nodes=[
            TaskNode(id="search", agent_class=SearchAgent, task=AgentTask(prompt="Find X")),
            TaskNode(id="summarise", agent_class=SummaryAgent,
                     task=AgentTask(prompt="Summarise the search"), depends_on=["search"]),
        ])
    )
    result = await coordinator.wait(run_id)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import anyio

from nexagent.agents.generic import GenericAgent
from nexagent.agents.registry import AgentConfig
from nexagent.agents.subagent import AgentOutput, AgentTask, SubAgent
from nexagent.memory.tiered import TieredMemory
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path.home() / ".nexagent" / "checkpoints.sqlite"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """A single node in the task graph."""

    id: str
    agent_class: type[SubAgent]
    task: AgentTask
    depends_on: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    output: AgentOutput | None = None
    error: str | None = None
    retries: int = 0
    max_retries: int = 2
    agent_config: AgentConfig | None = None


@dataclass
class TaskGraph:
    """DAG of TaskNodes for a single coordinator run."""

    nodes: list[TaskNode]
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, TaskNode]:
        return {n.id: n for n in self.nodes}

    def ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all DONE and are themselves PENDING."""
        done_ids = {n.id for n in self.nodes if n.status == NodeStatus.DONE}
        return [
            n
            for n in self.nodes
            if n.status == NodeStatus.PENDING and all(d in done_ids for d in n.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(
            n.status in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.SKIPPED)
            for n in self.nodes
        )

    def failed_nodes(self) -> list[TaskNode]:
        return [n for n in self.nodes if n.status == NodeStatus.FAILED]


# ---------------------------------------------------------------------------
# Checkpoint store (SQLite-backed)
# ---------------------------------------------------------------------------


class CheckpointStore:
    """Stores node status and outputs across runs.

    Uses SQLite in WAL mode. The schema is append-oriented — we insert new
    checkpoint rows rather than updating in place, preserving audit history.
    """

    def __init__(self, db_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT NOT NULL,
                    node_id     TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    output_json TEXT,
                    error       TEXT,
                    created_at  REAL NOT NULL DEFAULT (unixepoch('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_run ON checkpoints(run_id, node_id)"
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

    def save(self, run_id: str, node: TaskNode) -> None:
        output_json = None
        if node.output:
            output_json = json.dumps(asdict(node.output))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (run_id, node_id, status, output_json, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, node.id, node.status.value, output_json, node.error),
            )
            conn.commit()

    def load_latest(self, run_id: str) -> dict[str, tuple[NodeStatus, AgentOutput | None, str | None]]:
        """Return the most recent checkpoint for each node in a run."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT node_id, status, output_json, error
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY id DESC
                """,
                (run_id,),
            ).fetchall()

        result: dict[str, tuple[NodeStatus, AgentOutput | None, str | None]] = {}
        for row in rows:
            nid = row["node_id"]
            if nid not in result:
                output = None
                if row["output_json"]:
                    try:
                        output = AgentOutput(**json.loads(row["output_json"]))
                    except Exception:
                        pass
                result[nid] = (NodeStatus(row["status"]), output, row["error"])
        return result


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorResult:
    run_id: str
    outputs: dict[str, AgentOutput]
    failed_nodes: list[str]
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentCoordinator:
    """Orchestrates a DAG of subagent tasks with checkpointing and concurrency.

    Parameters
    ----------
    registry:
        Tool registry passed to each subagent.
    policy:
        Trust policy.
    memory:
        Shared memory tier (optional; subagents may have their own).
    checkpoint_store:
        Where to persist node checkpoints.
    max_concurrent:
        Maximum number of nodes running concurrently.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory | None = None,
        checkpoint_store: CheckpointStore | None = None,
        max_concurrent: int = 4,
        model_pool: Any | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._memory = memory or TieredMemory()
        self._checkpoints = checkpoint_store or CheckpointStore()
        self._max_concurrent = max_concurrent
        self._model_pool = model_pool
        self._dynamic_nodes: list[TaskNode] = []
        self._dynamic_lock = anyio.Lock()

    async def run(
        self,
        graph: TaskGraph,
        workflow_context: "WorkflowContext" | None = None,
    ) -> CoordinatorResult:
        """Execute the task graph, blocking until all nodes are terminal.

        Resumes from checkpoints if any nodes were previously completed.

        Parameters
        ----------
        graph:
            The task graph to execute.
        workflow_context:
            Optional shared context for structured data passing between nodes.
            Node outputs are stored here after completion.
        """
        # Import here to avoid circular import at module load time
        from nexagent.agents.workflow import interpolate_template

        self._restore_from_checkpoints(graph)
        logger.info("Starting coordinator run %s (%d nodes)", graph.run_id, len(graph.nodes))

        # Pre-render node prompts via template interpolation
        for node in graph.nodes:
            if node.status == NodeStatus.PENDING and workflow_context:
                node.task.prompt = interpolate_template(node.task.prompt, workflow_context)

        semaphore = anyio.Semaphore(self._max_concurrent)

        while not graph.is_complete():
            # Merge any dynamically added nodes
            async with self._dynamic_lock:
                if self._dynamic_nodes:
                    new_nodes = self._dynamic_nodes
                    self._dynamic_nodes = []
                    graph.nodes.extend(new_nodes)
                    logger.info(
                        "Coordinator merged %d dynamic nodes into run %s",
                        len(new_nodes),
                        graph.run_id,
                    )

            ready = graph.ready_nodes()
            if not ready:
                if any(n.status == NodeStatus.RUNNING for n in graph.nodes):
                    await anyio.sleep(0.05)
                    continue
                # No ready nodes and nothing running = deadlock or all done
                logger.warning(
                    "No ready nodes and nothing running — possible cycle or all failed"
                )
                break

            async with anyio.create_task_group() as tg:
                for node in ready:
                    node.status = NodeStatus.RUNNING
                    tg.start_soon(self._run_node, node, graph.run_id, semaphore, workflow_context)

        outputs = {
            n.id: n.output
            for n in graph.nodes
            if n.status == NodeStatus.DONE and n.output is not None
        }
        failed = [n.id for n in graph.failed_nodes()]
        success = len(failed) == 0

        logger.info(
            "Coordinator run %s complete — success=%s failed=%s",
            graph.run_id,
            success,
            failed,
        )
        return CoordinatorResult(
            run_id=graph.run_id,
            outputs=outputs,
            failed_nodes=failed,
            success=success,
        )

    async def _run_node(
        self,
        node: TaskNode,
        run_id: str,
        semaphore: anyio.Semaphore,
        workflow_context: "WorkflowContext" | None = None,
    ) -> None:
        async with semaphore:
            logger.debug("Running node %s (run=%s)", node.id, run_id)
            try:
                # Re-interpolate prompt at runtime so downstream nodes
                # can resolve upstream outputs (e.g. {{research.content}})
                if workflow_context:
                    from nexagent.agents.workflow import interpolate_template

                    node.task.prompt = interpolate_template(node.task.prompt, workflow_context)

                if node.agent_class is GenericAgent and node.agent_config:
                    agent = node.agent_config.instantiate(
                        registry=self._registry,
                        policy=self._policy,
                        memory=self._memory,
                        model_pool=self._model_pool,
                    )
                else:
                    agent = node.agent_class(
                        registry=self._registry,
                        policy=self._policy,
                        memory=self._memory,
                    )
                output = await agent.run(node.task)
                node.output = output
                node.status = NodeStatus.DONE if output.success else NodeStatus.FAILED
                if not output.success:
                    node.error = output.error
            except Exception as exc:
                logger.error("Node %s failed with exception: %s", node.id, exc, exc_info=True)
                node.status = NodeStatus.FAILED
                node.error = str(exc)
                if node.retries < node.max_retries:
                    node.retries += 1
                    node.status = NodeStatus.PENDING
                    logger.info("Retrying node %s (attempt %d)", node.id, node.retries)
                    return

            # Store output in workflow context if available
            if workflow_context:
                workflow_context.set_node_output(node.id, node.output)

            self._checkpoints.save(run_id, node)

    def _restore_from_checkpoints(self, graph: TaskGraph) -> None:
        """Apply previously-saved checkpoints to the graph, skipping completed nodes."""
        saved = self._checkpoints.load_latest(graph.run_id)
        node_map = graph.node_map()
        for node_id, (status, output, error) in saved.items():
            if node_id in node_map:
                node = node_map[node_id]
                if status == NodeStatus.DONE:
                    node.status = status
                    node.output = output
                    logger.debug("Restored checkpoint for node %s (status=%s)", node_id, status)

    async def add_nodes(self, nodes: list[TaskNode]) -> None:
        """Dynamically add nodes to the running task graph.

        Safe to call from within a node's execution context (e.g., from a
        SupervisorAgent). The nodes are queued for merging at the next
        coordinator loop iteration.
        """
        async with self._dynamic_lock:
            self._dynamic_nodes.extend(nodes)
            logger.debug(
                "Queued %d dynamic nodes (total queued: %d)",
                len(nodes),
                len(self._dynamic_nodes),
            )

    async def submit_and_wait(self, graph: TaskGraph) -> CoordinatorResult:
        """Convenience method: submit a graph and await its result."""
        return await self.run(graph)
