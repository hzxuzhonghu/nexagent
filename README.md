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
User / API Channel
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
  ┌──────▼──────┐      ┌─────────────────┐
  │ Coordinator │◄────►│  Tiered Memory  │
  │  (agents/)  │      │  (memory/)      │
  └──────┬──────┘      └─────────────────┘
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

### Run

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
| `registry.py` | MCP-native tool registry. Tools self-register with JSON Schema. Supports auto-discovery via entry-points. |
| `sandbox.py` | Per-task capability grants. Blocks tools not in the granted set. Detects prompt-injection patterns in tool arguments. |
| `audit.py` | Append-only audit log for every tool invocation: who called it, with what args, what was returned, at what cost. |

### `nexagent/agents/`

| File | Purpose |
|---|---|
| `coordinator.py` | Multi-agent coordinator with a directed acyclic task graph. Checkpoints state after each node. Replays from checkpoint on failure. |
| `subagent.py` | Base class for specialised subagents. Provides tool execution, memory access, and structured output emission. |

### `nexagent/inference/`

| File | Purpose |
|---|---|
| `router.py` | Two-stage routing: a lightweight local intent classifier (regex + heuristic scoring) gates calls to the frontier model. Low-complexity tasks never leave the process. |

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

## Configuration

NexAgent is 12-factor compliant. All configuration via environment variables:

| Variable | Default | Description |
|---|---|---|
| `NEXAGENT_MODEL` | `gpt-4o` | Frontier model identifier |
| `NEXAGENT_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `NEXAGENT_API_KEY` | *(required)* | API key — never hardcoded |
| `NEXAGENT_LOCAL_THRESHOLD` | `0.7` | Intent confidence threshold for local handling |
| `NEXAGENT_MAX_STEPS` | `20` | Maximum agent loop iterations |
| `NEXAGENT_MEMORY_PATH` | `~/.nexagent/memory` | Persistent memory directory |
| `NEXAGENT_AUDIT_PATH` | `~/.nexagent/audit.jsonl` | Audit log file |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry collector |

---

## Contributing

1. Fork, branch (`feat/your-feature`), implement, test
2. `ruff check . && mypy nexagent/` must pass
3. All new capabilities require a test in `tests/`
4. No secrets in code — ever

---

## License

MIT © NexAgent Authors
