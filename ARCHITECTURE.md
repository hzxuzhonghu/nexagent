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

---

## 14. CLI Entry Point and Interactive REPL

### Context
The framework is entirely programmatic — users must write Python to define agents, workflows, and execute them. This creates a high barrier to entry and prevents non-programmers from using the system.

### Decision
`__main__.py` implements three subcommands:

- **`nexagent run <workflow.yaml>`** — one-shot workflow execution from YAML. Accepts `--agents` to preload agent definitions.
- **`nexagent agents list/import`** — manage and inspect agent definitions.
- **`nexagent chat`** — interactive REPL that maintains an in-memory `AgentRegistry` and workflow graph. Users can register agents, add nodes, set variables, and run — all without writing Python.

The CLI uses `argparse` for subcommand parsing and `rich` for formatted tables and markdown output. The interactive session is a stateful loop around `asyncio.run()`.

### Alternatives Considered
- **Click / Typer** — richer CLI features but adds a dependency; `argparse` is stdlib
- **REST API server** — enables remote use but adds deployment complexity; CLI is sufficient for local single-user operation
- **Web UI** — ideal for non-technical users but out of scope; the CLI REPL provides a middle ground

### Trade-offs
The REPL session is in-memory only — exiting loses the session state. Workflow definitions can be exported via the YAML serializer (`_dict_to_yaml`), but there's no `save` command yet. For persistent sessions, users should use YAML files directly.

---

## 15. Model Pool and Multi-Provider Support

### Context
The inference layer originally used a single global model configured via `NEXAGENT_MODEL`, `NEXAGENT_API_BASE`, and `NEXAGENT_API_KEY` environment variables. This prevents per-agent model selection, multi-provider setups, and cost-aware model routing.

### Decision
`inference/models.py` introduces a `ModelPool` abstraction:

- **`ProviderConfig`** — name, api_key, base_url, headers for a single inference provider
- **`ModelConfig`** — id, provider reference, capabilities list (e.g. `["text", "image"]`), `ModelCost` (per-million token pricing), context_window, max_tokens, reasoning flag
- **`ModelPool`** — maps provider names → ProviderConfigs and model ids → ModelConfigs. Provides:
  - `get_provider(model_id)` → resolves provider credentials for a model
  - `select_by_capability(caps)` → first model supporting all requested capabilities
  - `select_cheapest(caps?)` → cost-optimised model selection
  - `from_yaml(path)` → load from YAML with `${ENV_VAR}` interpolation
  - `from_env()` → backward-compatible pool from existing NEXAGENT_* env vars

The inference router (`call_frontier()` and `InferenceRouter`) accepts optional `model_pool` + `model_id` params. When provided, provider credentials are resolved from the pool instead of environment variables.

**AgentConfig integration:** New fields `model`, `model_capabilities`, and deprecated `model_override`. Resolution order: explicit `model` → `model_override` → capability-based selection from pool → env var fallback.

### Alternatives Considered
- **Single env var per model** — doesn't scale; multi-model setups require coordinating multiple env vars
- **Pydantic-settings** — adds a dependency; our dataclass approach is lighter and sufficient
- **Model registry service** — overkill for single-user deployment; YAML file is simple and auditable

### Trade-offs
The model pool is loaded once at startup. Dynamic model hot-reloading is not supported — changing models requires restarting the agent process.

---

## 16. Agent Workspace Dirs and Persona

### Context
Agents were defined as flat `AgentConfig` with only `system_prompt`. There was no persistent agent identity, no personality, and no environment-specific memory beyond the session.

### Decision
`agents/workspace.py` introduces a workspace directory model inspired by OpenClaw:

- **`AgentPersona`** — data class holding identity fields (name, creature, vibe, emoji), user context, soul rules, tool notes, and long-term memory
- **`AgentWorkspace`** — manages a directory at a configurable path. `ensure()` creates the directory and seeds template files (`IDENTITY.md`, `USER.md`, `SOUL.md`, `TOOLS.md`, `MEMORY.md`). `load_persona()` / `save_persona()` read/write persona data. `compose_system_prompt(base_prompt)` merges the base system prompt with identity and soul rules.

**AgentConfig integration:** `workspace_dir` field. When set, `GenericAgent.__init__()` composes the effective system prompt via `workspace.compose_system_prompt()`.

### Alternatives Considered
- **Single AGENTS.md for everything** — simpler but conflates personality, memory, and tool config
- **Database-backed persona** — adds latency; file-based is accessible, diffable, and editable
- **No workspace — everything in env vars** — works for simple cases but doesn't support evolving agent identity

### Trade-offs
Workspace files grow over time. `compose_system_prompt()` truncates MEMORY.md to 20 lines to stay within reasonable prompt length. For very long memory, the semantic memory tier should be used instead.

---

## 17. Tool Discovery via Markdown Frontmatter

### Context
Tools could only be registered via Python decorators (`@registry.tool()`). Users needed to write Python code to add tools, creating a barrier for non-programmers.

### Decision
`tools/discovery.py` implements a discovery system for tools defined in `.md` files with YAML frontmatter:

- **`ToolSpec`** — parsed representation: name, description, JSON Schema parameters, source path, implementation code (optional)
- **`ToolDiscovery`** — scans directories for `.md` files, parses YAML frontmatter for metadata, extracts Python implementation from fenced code blocks, compiles and registers functions

Format:
```
---
name: current_date
description: Return the current date
schema:
  type: object
  properties:
    format: {type: string}
---

```python
from datetime import datetime
async def current_date(format: str = "%Y-%m-%d") -> str:
    return datetime.now().strftime(format)
```
```

Tools can be metadata-only (no implementation) — useful for documenting external tools that are registered elsewhere.

### Alternatives Considered
- **JSON config files** — valid but YAML is more readable and supports comments
- **TOML-based tools** — pyproject-compatible but less familiar to non-Python users
- **Inline shell scripts** — fragile; Python ensures cross-platform compatibility

### Trade-offs
`exec()` is used to compile tool implementations from markdown. This is safe because tool files are local and trusted. For untrusted tool sources, module-path references should be used instead of inline code.
