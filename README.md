# NexAgent

> **Next-generation agentic personal assistant** — autonomous, multi-model, cost-aware, and built for production.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Vision

NexAgent is a composable, observable, and trustworthy autonomous agent framework. It bridges local intent classification with frontier model reasoning, orchestrates specialized subagents via a stateful graph, and tracks every decision through an immutable audit trail.

**Design principles:**
- **Tiered cognition** — fast local inference for routine tasks, expensive frontier calls only when needed
- **Memory as first-class** — working, episodic, semantic, and procedural memory form a coherent knowledge layer
- **MCP-native tooling** — all tools are registered via the Model Context Protocol with capability grants per task
- **Observable by default** — every reasoning→tool→result chain is traced with OpenTelemetry
- **Trust before autonomy** — channel-level trust policies and autonomy dials gate what the agent can do unsupervised

---

## Architecture Overview

```
User / CLI / API Channel
           │
           ▼
     ┌─────────────┐
     │ Trust Policy│  ← channel trust, autonomy dials
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │  Agent Loop │  ← async tool-calling loop (runtime/)
     └──────┬──────┘
            │
      ┌─────┴──────┐
      │  Inference  │  ← local classifier → frontier model (inference/)
      │   Router   │
      └─────┬──────┘
            │
     ┌──────▼──────────┐      ┌─────────────────┐
     │   Coordinator   │◄────►│  Tiered Memory  │
     │   + Workflow    │      │  (memory/)      │
     │   DSL (agents/) │      └─────────────────┘
     └──────┬──────────┘
            │
     ┌──────▼──────┐      ┌─────────────────┐
     │Tool Registry│◄────►│  Audit + Sandbox │
     │  (tools/)   │      │  (tools/)        │
     └─────────────┘      └─────────────────┘
            │
     ┌──────▼──────┐
     │Observability│  ← traces, cost tracking (observability/)
     └─────────────┘
```

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### CLI — Three modes of operation

**1. Run a YAML workflow:**

```bash
# Define agents and a workflow in YAML, then execute
nexagent run examples/workflow.yaml -a examples/agents.yaml
```

**2. Interactive chat — build workflows on the fly:**

```bash
nexagent chat
```

In the REPL you can register agents, add workflow nodes, set variables, and run — all dynamically:

```
nexagent> agents add       # Define a new agent interactively
nexagent> node add         # Add a node to the workflow graph
nexagent> vars set topic AI
nexagent> run              # Execute the workflow
```

**3. Manage agent definitions:**

```bash
nexagent agents list -f examples/agents.yaml
nexagent agents import examples/agents.yaml
```

### Programmatic API

**Single-agent mode:**

```python
import asyncio
from nexagent.runtime.agent_loop import AgentLoop
from nexagent.runtime.context import SessionContext
from nexagent.trust.policy import TrustPolicy, Channel

async def main():
    ctx = SessionContext.new(channel=Channel.API)
    policy = TrustPolicy.default()
    loop = AgentLoop(context=ctx, policy=policy)
    result = await loop.run("Summarise the top 3 items in my task list.")
    print(result.output)

asyncio.run(main())
```

**Multi-agent workflow from YAML:**

```python
import asyncio
from nexagent.agents import AgentConfig, AgentRegistry, WorkflowParser
from nexagent.agents import AgentCoordinator
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

async def main():
    registry = ToolRegistry()
    policy = TrustPolicy.default()
    agent_registry = AgentRegistry(tool_registry=registry, policy=policy)

    # Register agents
    agent_registry.register(AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
    ))
    agent_registry.register(AgentConfig(
        name="writer",
        system_prompt="You write reports.",
    ))

    # Parse workflow
    parser = WorkflowParser(agent_registry=agent_registry)
    graph, ctx = parser.parse_yaml("""
workflow: research
variables:
  topic: AI
nodes:
  - id: search
    agent: researcher
    prompt: "Research {{topic}}"
  - id: write
    agent: writer
    prompt: "Write report from {{search.content}}"
    depends_on: [search]
""")

    # Execute
    coordinator = AgentCoordinator(registry=registry, policy=policy)
    result = await coordinator.run(graph, ctx)
    for node_id, output in result.outputs.items():
        print(f"{node_id}: {output.content}")

asyncio.run(main())
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Module Reference

### `nexagent/runtime/`

| File | Purpose |
|---|---|
| `agent_loop.py` | Core async agent loop — receives a prompt, calls the inference router, dispatches tool calls, loops until a terminal `finish` action or step limit. |
| `context.py` | Immutable-ish session context: message history, active tools, metadata. Pydantic-based for serialisation. |

### `nexagent/memory/`

| File | Purpose |
|---|---|
| `tiered.py` | Four-tier memory model: **working** (in-RAM dict), **episodic** (recent turns, SQLite), **semantic** (vector search), **procedural** (skill definitions). |
| `vector_store.py` | Pure-numpy cosine-similarity vector store. No external vector DB required. Supports add, search, persist, load. |

### `nexagent/tools/`

| File | Purpose |
|---|---|
| `registry.py` | MCP-native tool registry. Tools self-register with JSON Schema. Supports auto-discovery via entry-points. `subset()` for tool filtering. |
| `discovery.py` | **Tool Discovery** — loads tools from ``.md`` files with YAML frontmatter. Parse name/description/schema from frontmatter, Python implementation from fenced code blocks. |
| `sandbox.py` | Per-task capability grants. Blocks tools not in the granted set. Detects prompt-injection patterns in tool arguments. |
| `audit.py` | Append-only audit log for every tool invocation: who called it, with what args, what was returned, at what cost. |

### `nexagent/agents/`

| File | Purpose |
|---|---|
| `coordinator.py` | Multi-agent coordinator with a directed acyclic task graph, checkpoint/replay, dynamic node injection via `add_nodes()`, and `WorkflowContext` for structured data passing. |
| `subagent.py` | Base class for specialised subagents. Provides tool execution, memory access, and structured output emission. |
| `generic.py` | `GenericAgent` — runtime-configured SubAgent wrapping AgentLoop. Supports `model_pool`, `model_id`, and `workspace` for per-agent model selection and persona enrichment. |
| `registry.py` | `AgentConfig` (Pydantic) + `AgentRegistry` for runtime agent definitions. Config fields: `model`, `workspace_dir`, `model_capabilities`. Load from YAML. |
| `workspace.py` | **Agent Workspace** — `AgentPersona` (identity, soul, memory) and `AgentWorkspace` (directory management, persona I/O, system prompt composition). |
| `workflow.py` | `WorkflowParser` (YAML → TaskGraph), `WorkflowContext` (shared state), template interpolation. |
| `patterns.py` | `FanOutBuilder` (parallel workers + collector), `SupervisorAgent` (dynamic graph mutation). |

### `nexagent/tools/`

| File | Purpose |
|---|---|
| `registry.py` | MCP-native tool registry. Tools self-register with JSON Schema. Supports auto-discovery via entry-points. `subset()` for tool filtering. |
| `sandbox.py` | Per-task capability grants. Blocks tools not in the granted set. Detects prompt-injection patterns in tool arguments. |
| `audit.py` | Append-only audit log for every tool invocation: who called it, with what args, what was returned, at what cost. |

### `nexagent/__main__.py`

CLI entry point. `nexagent run`, `nexagent chat`, `nexagent agents`.

### `nexagent/inference/`

| File | Purpose |
|---|---|
| `router.py` | Two-stage routing: a lightweight local intent classifier (regex + heuristic scoring) gates calls to the frontier model. Supports both single-model and ModelPool-backed inference. |
| `models.py` | **Model Pool** — `ModelConfig` (id, provider, capabilities, cost, context), `ProviderConfig` (api_key, base_url), `ModelPool` (from_yaml, from_env, capability/cost selection). |

### `nexagent/proactive/`

| File | Purpose |
|---|---|
| `loop.py` | Background anyio task that polls for triggers (schedule, anomaly, threshold) and enqueues proactive reasoning tasks. |

### `nexagent/observability/`

| File | Purpose |
|---|---|
| `tracer.py` | OpenTelemetry-based tracer. Wraps every reasoning step, tool call, and model invocation in a span with semantic attributes. |
| `cost.py` | Per-session cost accumulator. Tracks prompt/completion tokens per model, maps to USD, emits cost events. |

### `nexagent/trust/`

| File | Purpose |
|---|---|
| `policy.py` | Channel-level trust levels (SYSTEM > API > UI > CLI > PUBLIC). Autonomy dials control whether the agent acts, asks, or refuses. |

### `nexagent/improvement/`

| File | Purpose |
|---|---|
| `tracker.py` | Records task outcomes (success/failure/partial). Identifies recurring failure patterns and proposes skill patches as structured diffs. |

---

## Workflow YAML

Define agents and workflows in YAML — no Python code required.

**Agents file (`agents.yaml`):**

```yaml
agents:
  - name: researcher
    description: "Researches topics"
    system_prompt: "You are a research expert..."
    tools: [web_search]
    max_steps: 10
```

**Workflow file (`workflow.yaml`):**

```yaml
workflow: research_report
variables:
  topic: "autonomous agents"
nodes:
  - id: research
    agent: researcher
    prompt: "Research {{topic}}"
  - id: write
    agent: writer
    prompt: "Write report from {{research.content}}"
    depends_on: [research]
```

Run with: `nexagent run workflow.yaml -a agents.yaml`

---

## Configuration

NexAgent is 12-factor compliant. All configuration via environment variables or YAML config files:

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEXAGENT_MODEL` | `gpt-4o` | Frontier model identifier (fallback if no models.yaml) |
| `NEXAGENT_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL (fallback) |
| `NEXAGENT_API_KEY` | *(required)* | API key — never hardcoded (fallback) |
| `NEXAGENT_LOCAL_THRESHOLD` | `0.7` | Intent confidence threshold for local handling |
| `NEXAGENT_MAX_STEPS` | `20` | Maximum agent loop iterations |
| `NEXAGENT_MEMORY_PATH` | `~/.nexagent/memory` | Persistent memory directory |
| `NEXAGENT_AUDIT_PATH` | `~/.nexagent/audit.jsonl` | Audit log file |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry collector |

### Model Pool (models.yaml)

For multi-provider, multi-model setups, define a `models.yaml`:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    base_url: https://api.anthropic.com/v1

models:
  gpt-4o:
    provider: openai
    capabilities: [text, image]
    cost: {input_per_m: 2.50, output_per_m: 10.00}
    context_window: 128000
  gpt-4o-mini:
    provider: openai
    capabilities: [text]
    cost: {input_per_m: 0.15, output_per_m: 0.60}
  claude-sonnet-4-6:
    provider: anthropic
    capabilities: [text, image]
    cost: {input_per_m: 3.00, output_per_m: 15.00}
    context_window: 200000
```

Load it:

```python
from nexagent.inference.models import ModelPool

pool = ModelPool.from_yaml("models.yaml")
router = InferenceRouter(model_pool=pool, model_id="claude-sonnet-4-6")
```

### Agent Workspaces

Each agent can have a persistent workspace directory with persona files (SOUL.md, MEMORY.md, IDENTITY.md, etc.):

```
~/.nexagent/agents/researcher/
  AGENTS.md     # operational manual
  SOUL.md       # personality and behavioral rules
  USER.md       # user profile
  IDENTITY.md   # agent name, vibe, emoji
  TOOLS.md      # environment-specific notes
  MEMORY.md     # long-term curated memory
```

Configure per agent:

```yaml
agents:
  - name: researcher
    system_prompt: "You research topics."
    model: gpt-4o
    workspace_dir: ~/.nexagent/agents/researcher
```

The workspace enriches the system prompt with persona data automatically. See `nexagent/agents/workspace.py`.

---

## Contributing

1. Fork, branch (`feat/your-feature`), implement, test
2. `ruff check . && mypy nexagent/` must pass
3. All new capabilities require a test in `tests/`
4. No secrets in code — ever

---

## License

MIT © NexAgent Authors
