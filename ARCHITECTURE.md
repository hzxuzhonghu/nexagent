# NexAgent — Architecture Decision Record

> This document captures the key design decisions behind NexAgent. Each decision includes context, the chosen approach, alternatives considered, and trade-offs.

---

## 1. Tiered Memory Model

### Context
Agents need different kinds of memory with different latency, capacity, and persistence characteristics. A single flat context window is insufficient for long-running personal assistants.

### Decision
Four distinct tiers managed by `memory/tiered.py`:

| Tier | Storage | Latency | Capacity | Persistence |
|---|---|---|---|---|
| **Working** | In-process dict | ~0 μs | ~100 items | Session only |
| **Episodic** | SQLite (WAL mode) | ~1 ms | Unlimited | Disk |
| **Semantic** | Numpy vector store | ~10 ms | ~100K items | Disk |
| **Procedural** | JSON file | ~5 ms | ~1K skills | Disk |

**Retrieval flow:** Working → Episodic (recency) → Semantic (relevance) → Procedural (skill lookup).

### Alternatives Considered
- **Redis for working memory** — overkill for single-user deployments; adds an operational dependency
- **ChromaDB / Weaviate** — excellent but introduce container dependencies; we keep semantic search pure-numpy for zero-dep local operation
- **Single SQLite for everything** — loses the latency benefits of in-RAM working memory and the vector-native access of the semantic tier

### Trade-offs
The numpy vector store is O(n) at query time. For >500K vectors, a proper ANN index (HNSW, FAISS) should replace it. The architecture is designed so `vector_store.py` is the only file that needs changing — the `TieredMemory` interface is stable.

---

## 2. MCP-Native Tool Registry

### Context
The Model Context Protocol (MCP) is emerging as the standard for tool description and invocation across LLM frameworks. Building against it future-proofs integrations.

### Decision
`tools/registry.py` implements:
- **JSON Schema tool descriptors** compatible with MCP tool definitions
- **Auto-discovery** via Python entry-points (`nexagent.tools` group)
- **Async invoke** for all tools — no blocking calls on the agent thread

Tools are plain Python callables decorated with `@tool(name, description, schema)`. The registry validates arguments against the schema before invocation.

### Alternatives Considered
- **LangChain tool format** — popular but tightly coupled to LangChain's abstractions
- **OpenAI function calling format** — limited to OpenAI; MCP is model-agnostic
- **Custom binary protocol** — unnecessary complexity; JSON Schema is sufficient and human-readable

### Trade-offs
Schema validation adds ~0.1 ms per call. This is intentional — it prevents malformed arguments from reaching external services.

---

## 3. Checkpoint Strategy for Multi-Agent Coordination

### Context
Multi-agent workflows can take minutes. Network failures, model timeouts, and process restarts must not lose progress.

### Decision
`agents/coordinator.py` implements a DAG-based coordinator with:
- **Node-level checkpoints** written to SQLite after each node completes
- **Idempotent node execution** — each node receives a stable `run_id` so retries are safe
- **Checkpoint replay** — on startup, incomplete runs are resumed from the last successful checkpoint
- **Topological sort** — parallelism is expressed in the DAG; the coordinator executes ready nodes concurrently using `anyio.create_task_group`

### Alternatives Considered
- **LangGraph / Temporal** — excellent but heavyweight; we want a zero-dependency option
- **In-memory graph only** — brittle; one crash loses the entire workflow
- **Event sourcing** — powerful but complex; checkpoints are sufficient for our reliability target

### Trade-offs
The checkpoint store is SQLite, which has a single-writer limitation. For distributed multi-agent deployments (multiple processes), the checkpoint store should be swapped for PostgreSQL or a distributed KV store. The `CheckpointStore` abstract base class makes this straightforward.

---

## 4. Two-Stage Inference Routing

### Context
Frontier model calls are expensive (~$0.01–0.10 per complex query). Many agent operations (intent classification, slot filling, simple lookups) don't need GPT-4-class reasoning.

### Decision
`inference/router.py` implements a two-stage router:

1. **Local stage**: Regex patterns + heuristic keyword scoring classify intent. If confidence ≥ `NEXAGENT_LOCAL_THRESHOLD` (default 0.7), the request is handled locally.
2. **Frontier stage**: Everything else goes to the configured OpenAI-compatible model via `httpx`.

The local classifier handles: simple Q&A, memory lookups, task listing, arithmetic, date/time queries.

### Alternatives Considered
- **Always frontier** — simple but expensive; untenable for high-frequency personal assistant use
- **Local SLM (Ollama, llama.cpp)** — ideal future state; the router interface is designed to plug this in
- **Separate classifier model** — adds complexity; the regex + heuristic approach handles ~60% of queries at zero latency and zero cost

### Trade-offs
The local classifier has false positives (routes complex queries locally) and false negatives (sends simple queries to frontier). The threshold is configurable. The improvement tracker captures misrouting patterns for refinement.

---

## 5. Trust Model and Autonomy Dials

### Context
An autonomous agent that can call tools and take actions needs a trust model. Not all callers should have the same capability grants.

### Decision
`trust/policy.py` defines:

**Channel trust levels** (highest to lowest):
```
SYSTEM > API > UI > CLI > PUBLIC
```

**Autonomy dials** per trust level:

| Trust | Can Act | Can Exfiltrate | Max Tools | Requires Confirmation |
|---|---|---|---|---|
| SYSTEM | ✅ | ✅ | Unlimited | Never |
| API | ✅ | ❌ | 20 | Never |
| UI | ✅ | ❌ | 10 | On destructive |
| CLI | ✅ | ❌ | 5 | Always |
| PUBLIC | ❌ | ❌ | 0 | N/A |

### Alternatives Considered
- **OAuth scopes** — standard but session-granular; we need request-granular control
- **Capability tokens** — cryptographically verifiable but adds key management complexity
- **Simple boolean allow/deny** — too coarse for a personal assistant with diverse callers

### Trade-offs
The trust model is enforced in `tools/sandbox.py` at invocation time. A misconfigured trust level can grant unintended capabilities. The audit log provides the forensic trail for post-hoc review.

---

## 6. Prompt Injection Detection

### Context
When tool outputs are re-injected into the prompt, adversarial content in tool results can redirect the agent.

### Decision
`tools/sandbox.py` implements pattern-based prompt injection detection applied to:
- All tool arguments before invocation
- All tool results before inclusion in the prompt

Detection patterns include: role-switching instructions (`"ignore previous instructions"`), jailbreak prefixes, base64-encoded instructions, and Unicode homoglyph attacks.

Detected injections are logged to the audit trail and the argument/result is sanitised (injection-suspect content replaced with a sentinel).

### Alternatives Considered
- **LLM-based detection** — most accurate but adds latency and cost; used as a second pass for high-trust channels only
- **No detection** — unacceptable for a tool-calling agent in a production environment
- **Blocklist only** — too easy to bypass; we use heuristic pattern matching + structural analysis

---

## 7. Observability Architecture

### Context
Debugging agent failures requires understanding the full reasoning chain: what was inferred, which tools were called, what they returned, how much it cost.

### Decision
`observability/tracer.py` wraps every agent operation in an OpenTelemetry span:
- `nexagent.loop.step` — one span per agent loop iteration
- `nexagent.tool.invoke` — one span per tool call with input/output attributes
- `nexagent.inference.route` — one span per inference decision with routing metadata
- `nexagent.memory.retrieve` — one span per memory tier queried

`observability/cost.py` accumulates token counts per model and converts to USD using a configurable price table. Cost events are emitted as OTEL metric events.

### Alternatives Considered
- **Custom logging only** — not structured enough for distributed tracing
- **Datadog/Honeycomb-specific** — vendor lock-in; OTEL exports to any backend
- **No observability** — unacceptable; blind autonomous agents are dangerous

---

## 8. Proactive Reasoning Loop

### Context
A personal assistant should not only react to queries but proactively surface relevant information (upcoming deadlines, anomalies, opportunities).

### Decision
`proactive/loop.py` runs as a background `anyio` task:
- Polls registered **trigger handlers** at configurable intervals
- Each trigger handler returns a `ProactiveTrigger` with a prompt, priority, and optional TTL
- Triggers are deduplicated by content hash to avoid spam
- High-priority triggers bypass the queue and go directly to the agent loop

### Design Constraints
- The proactive loop must never block the main agent loop
- Triggers must be idempotent — the same underlying condition should not fire repeatedly
- The loop is suspended when the agent is handling a user request (back-pressure)

---

## 9. Improvement and Self-Modification

### Context
The agent makes mistakes. Those mistakes should inform future behaviour without requiring manual engineering effort.

### Decision
`improvement/tracker.py` records every task outcome with:
- Task description, expected outcome, actual outcome
- Tool calls made and their results
- Time taken, tokens used, cost

Failure analysis runs periodically and:
1. Groups failures by pattern (using the semantic memory tier)
2. Proposes **skill patches** — structured diffs to the procedural memory tier
3. Skill patches require SYSTEM-level trust approval before activation

### Constraints
- The agent cannot modify its own `trust/policy.py` or `tools/sandbox.py` — these are protected
- Skill patches are stored as pending proposals; humans approve them
- Improvement data is local only — never exfiltrated

---

## Dependency Philosophy

| Category | Choice | Reason |
|---|---|---|
| HTTP | `httpx` | Async-first, clean API, no Requests dependency |
| Async | `anyio` | Backend-agnostic (asyncio/trio) |
| Numerics | `numpy` | Vector store; no external vector DB |
| Validation | `pydantic>=2` | Fast, typed, great error messages |
| CLI/UI | `rich` | Beautiful terminal output with zero effort |
| Tracing | `opentelemetry-sdk` | Vendor-neutral; exports to any OTEL-compatible backend |
| Workflow DSL | `pyyaml>=6.0` | Standard YAML parsing; minimal dependency |

**No LLM framework dependency** (no LangChain, no LlamaIndex). This is intentional: frameworks evolve faster than their semver suggests. NexAgent's abstractions are thin enough to adapt to any model API.

---

## 10. GenericAgent and AgentConfig Registry

### Context
The coordinator originally required users to write Python subclasses of `SubAgent` for every agent type. This is verbose and prevents non-programmers from defining new agents or workflows.

### Decision
`GenericAgent` is a `SubAgent` subclass configured entirely at runtime with a system prompt, tool whitelist, and max_steps. It wraps the existing `AgentLoop` internally, so no new inference logic is introduced. `AgentConfig` is a Pydantic model that serializes agent definitions, and `AgentRegistry` manages named configurations. Agents can be loaded from YAML:

```yaml
agents:
  - name: researcher
    system_prompt: "You are a research expert..."
    tools: [web_search, read_url]
    max_steps: 15
```

The coordinator's `TaskNode` gains an `agent_config` field. When `agent_class is GenericAgent and agent_config`, the coordinator uses `agent_config.instantiate()` instead of the raw constructor.

### Alternatives Considered
- **Agent subclasses only** — original approach; requires code changes for every new agent
- **JSON-only config** — YAML is more readable and supports comments
- **LLM-based agent definition** — too expensive for a configuration problem

### Trade-offs
`GenericAgent` uses `ToolRegistry.subset()` to create a filtered tool view, since `AgentLoop` creates its own `Sandbox` internally. This is clean but means the subset is evaluated at instantiation time, not at invoke time.

---

## 11. Workflow DSL

### Context
Users need to define multi-agent workflows without writing Python code. The coordinator's `TaskGraph` API is programmatic but not user-facing.

### Decision
`WorkflowParser` converts YAML workflow definitions into `(TaskGraph, WorkflowContext)` tuples. The YAML syntax supports:

- `workflow` name
- `variables` section for template interpolation
- `nodes` list with `id`, `agent`, `prompt`, `depends_on`, `max_retries`
- Agent names are resolved against the `AgentRegistry`

The parser applies template interpolation to prompts that reference `variables` at parse time. Node output references (e.g., `{{search.content}}`) are left unrendered at parse time and resolved by the coordinator at execution time.

### Alternatives Considered
- **JSON DSL** — YAML is more human-readable and supports comments
- **Graphical editor** — useful future addition but out of scope for v0
- **Full Jinja2 templates** — adds a dependency; simple regex interpolation covers 90% of use cases

### Trade-offs
The regex-based `{{var}}` interpolation is O(n) per prompt and doesn't support conditionals or loops. For complex workflows, users can still use the programmatic `TaskGraph` API directly.

---

## 12. WorkflowContext and Structured Data Passing

### Context
Originally, coordinator nodes communicated only through raw string outputs. Downstream nodes received `AgentOutput.content` but had no way to access structured data, metadata, or artifacts.

### Decision
`WorkflowContext` is a shared dictionary passed to `coordinator.run()`. It tracks:
- Arbitrary key-value variables (populated from workflow `variables`)
- Node outputs by node ID (set after each node completes)

The coordinator pre-renders node prompts via `interpolate_template()` before execution and stores `AgentOutput` objects in the context after completion. This enables downstream nodes to reference upstream results via `{{node_id.content}}`.

### Alternatives Considered
- **Pass AgentOutput through task.context** — works but downstream nodes must know the dataclass structure
- **Shared SQLite for inter-node data** — overkill for in-process communication
- **Message queue** — unnecessary complexity for synchronous execution

### Trade-offs
`WorkflowContext` is in-memory only. If the coordinator process crashes, intermediate context data is lost (checkpoints only save `AgentOutput` per node, not the full context).

---

## 13. Dynamic Graph Mutation (Supervisor and FanOut Patterns)

### Context
The coordinator's DAG model is static: all nodes are defined before execution begins. Some workflows require runtime decisions — a supervisor agent that creates sub-tasks, or a fan-out pattern that splits work dynamically.

### Decision
The coordinator gains an `add_nodes()` async method and a `_dynamic_nodes` queue protected by `anyio.Lock`. On each loop iteration, pending dynamic nodes are merged into the graph before checking for ready nodes. This allows agents running inside the coordinator to extend the workflow at runtime.

`SupervisorAgent` is a `SubAgent` that receives an `_add_nodes_fn` callback via its task context. `FanOutBuilder` constructs parallel worker nodes with an optional collector.

### Alternatives Considered
- **Pre-computed DAG with all branches** — wasteful when only some branches are needed
- **Separate coordinator per sub-workflow** — fragments the checkpoint trail
- **Event-driven agent spawning** — more flexible but much harder to reason about

### Trade-offs
Dynamic node merging happens synchronously in the coordinator loop (with a lock). If many nodes are added rapidly, the lock contention could slow the loop. The `_dynamic_nodes` list is drained atomically, so nodes added during the current iteration are processed in the next.
