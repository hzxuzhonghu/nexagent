# NexAgent User Guide

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Quick Start](#quick-start)
4. [Single-Agent Mode](#single-agent-mode)
5. [Defining Agents](#defining-agents)
6. [Workflow YAML](#workflow-yaml)
7. [CLI Usage](#cli-usage)
8. [Interactive REPL](#interactive-repl)
9. [Structured Data Passing](#structured-data-passing)
10. [Advanced Patterns](#advanced-patterns)
11. [Programmatic API](#programmatic-api)
12. [Trust and Autonomy](#trust-and-autonomy)
13. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install -e ".[dev]"
```

Verify:

```bash
nexagent --help
```

---

## Configuration

### Environment Variables (Quick Start)

For simple single-model setups, use environment variables:

| Variable | Default | Description |
|---|---|---|
| `NEXAGENT_MODEL` | `gpt-4o` | Frontier model identifier |
| `NEXAGENT_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `NEXAGENT_API_KEY` | *(required)* | Your API key — never commit this |
| `NEXAGENT_LOCAL_THRESHOLD` | `0.7` | Confidence threshold for local routing |
| `NEXAGENT_MAX_STEPS` | `20` | Default max loop iterations |
| `NEXAGENT_MEMORY_PATH` | `~/.nexagent/memory` | Persistent memory directory |
| `NEXAGENT_AUDIT_PATH` | `~/.nexagent/audit.jsonl` | Audit log location |

### Model Pool (models.yaml)

For multi-provider, multi-model, or cost-aware setups, define a `models.yaml`:

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

Load in code:

```python
from nexagent.inference.models import ModelPool
from nexagent.inference.router import InferenceRouter

pool = ModelPool.from_yaml("models.yaml")
router = InferenceRouter(model_pool=pool, model_id="claude-sonnet-4-6")
```

### Using Non-OpenAI Models

NexAgent works with any OpenAI-compatible API. Set `NEXAGENT_API_BASE` accordingly:

```bash
# Anthropic (via OpenAI-compatible endpoint)
export NEXAGENT_MODEL="claude-3.5-sonnet-20241022"
export NEXAGENT_API_BASE="https://api.anthropic.com/v1"

# Ollama (local models)
export NEXAGENT_MODEL="llama3.1"
export NEXAGENT_API_BASE="http://localhost:11434/v1"

# DeepSeek
export NEXAGENT_MODEL="deepseek-chat"
export NEXAGENT_API_BASE="https://api.deepseek.com/v1"
```

Or with `models.yaml`, just configure the provider and reference it from your models.

---

## Quick Start

The fastest way to try NexAgent is the interactive REPL:

```bash
nexagent chat
```

This starts a prompt where you can define agents, build workflows, and run them — no files or code required.

---

## Single-Agent Mode

The simplest way to use NexAgent programmatically:

```python
import asyncio
from nexagent.runtime.agent_loop import AgentLoop
from nexagent.runtime.context import SessionContext
from nexagent.trust.policy import TrustPolicy, Channel

async def main():
    ctx = SessionContext.new(channel=Channel.API)
    policy = TrustPolicy.default()
    loop = AgentLoop(context=ctx, policy=policy)
    result = await loop.run("What is 2 + 2?")
    print(result.output)

asyncio.run(main())
```

The agent loop follows a ReAct pattern: think → act → observe → repeat until the agent finishes or hits the step limit.

---

## Defining Agents

Agents can be defined in three ways: YAML, programmatic config, or interactive REPL.

### YAML Definition

Create `agents.yaml`:

```yaml
agents:
  - name: researcher
    description: "Researches topics and gathers information"
    system_prompt: "You are a research assistant. Analyze the given topic and provide a detailed summary."
    model: gpt-4o                     # model from pool
    tools: []
    max_steps: 10
    workspace_dir: ~/.nexagent/agents/researcher  # persistent workspace

  - name: writer
    description: "Writes structured reports"
    system_prompt: "You are a technical writer. Take the provided research and write a clear, concise report."
    model: gpt-4o-mini                # cheaper model for simple tasks
    tools: []
    max_steps: 10
```

Field reference:

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Unique identifier for the agent |
| `description` | No | `""` | Human-readable description |
| `system_prompt` | Yes | — | The system prompt that guides the agent's behavior |
| `tools` | No | `[]` (all) | Whitelist of tool names. Empty means all available tools |
| `max_steps` | No | `20` | Maximum reasoning loop iterations for this agent |
| `model` | No | `None` | Model id from a `ModelPool` (preferred over `model_override`) |
| `model_override` | No | `None` | *(deprecated)* Use `model` instead |
| `workspace_dir` | No | `None` | Path to agent workspace for persona enrichment |
| `model_capabilities` | No | `[]` | Auto-select a model from the pool by capabilities (e.g. `["text", "image"]`) |

Load from the CLI:

```bash
nexagent agents import agents.yaml
```

Or programmatically:

```python
from pathlib import Path
from nexagent.agents import AgentRegistry, AgentConfig

registry = AgentRegistry(tool_registry=tools, policy=policy)
registry.load_yaml(Path("agents.yaml"))
```

### Programmatic Definition

```python
from nexagent.agents import AgentConfig

config = AgentConfig(
    name="coder",
    description="Writes and debugs code",
    system_prompt="You are an expert Python developer...",
    tools=["file_read", "file_write", "shell"],
    max_steps=15,
    model="gpt-4o",                         # per-agent model selection
    workspace_dir="~/.nexagent/agents/coder",  # persistent persona
)
registry.register(config)
```

### Agent Workspaces

When `workspace_dir` is set, the agent's system prompt is enriched with persona data:

```
~/.nexagent/agents/coder/
  IDENTITY.md   # - **Name:** Coder   - **Vibe:** sharp
  SOUL.md       # - Be concise. - Favor functional style.
  USER.md       # - **Name:** Alice   - **Timezone:** UTC+8
  MEMORY.md     # - User prefers pytest over unittest
  TOOLS.md      # - Local cluster: k3s @ 192.168.1.50
```

To create a workspace:

```bash
mkdir -p ~/.nexagent/agents/coder
```

Or use the Python API:

```python
from nexagent.agents.workspace import AgentWorkspace

workspace = AgentWorkspace("~/.nexagent/agents/coder")
workspace.ensure()  # creates dir + seeds template files

persona = workspace.load_persona()
persona.identity["name"] = "Coder"
persona.identity["vibe"] = "sharp"
workspace.save_persona(persona)
```

The workspace is optional. Without it, agents use only their `system_prompt`.

### Interactive REPL

```
nexagent> agents add
Agent name: coder
System prompt: You are an expert Python developer...
Tool names (comma-separated, or blank for all): file_read, file_write
Max steps: 15
Description (optional): Writes and debugs code
Agent 'coder' registered.
```

---

## Workflow YAML

Workflows are defined as directed acyclic graphs (DAGs) where nodes represent agent tasks and edges represent dependencies.

### Basic Workflow

Create `workflow.yaml`:

```yaml
workflow: research_report

variables:
  topic: "autonomous agents"

nodes:
  - id: research
    agent: researcher
    prompt: "Research the topic: {{topic}}. Provide a detailed summary."

  - id: report
    agent: writer
    prompt: "Write a report based on: {{research.content}}"
    depends_on: [research]

  - id: review
    agent: reviewer
    prompt: "Review this report: {{report.content}}"
    depends_on: [report]
```

Run:

```bash
nexagent run workflow.yaml -a agents.yaml
```

### Template Variables

Workflows support `{{var}}` interpolation in prompts:

- `{{topic}}` — resolved from the `variables` section
- `{{research.content}}` — resolved from the output of node `research`
- `{{research.task_id}}` — resolved from the output metadata of node `research`

### Dependency Graph

Nodes run in parallel when they have no dependency on each other:

```yaml
nodes:
  - id: search_a
    agent: researcher
    prompt: "Research topic A"

  - id: search_b
    agent: researcher
    prompt: "Research topic B"

  - id: compare
    agent: writer
    prompt: "Compare: {{search_a.content}} vs {{search_b.content}}"
    depends_on: [search_a, search_b]
```

Here `search_a` and `search_b` run concurrently. `compare` waits for both.

---

## CLI Usage

### `nexagent run` — Execute a workflow

```bash
nexagent run workflow.yaml -a agents.yaml
```

Flags:
- `--agents`, `-a` — Path to agents YAML file

### `nexagent chat` — Interactive REPL

```bash
nexagent chat
```

Available commands in the REPL:

| Command | Description |
|---|---|
| `agents list` | Show registered agents |
| `agents add` | Add an agent interactively |
| `agents import <file>` | Load agents from YAML |
| `node add` | Add a workflow node |
| `node list` | Show workflow nodes |
| `vars set <key> <value>` | Set a workflow variable |
| `vars list` | Show workflow variables |
| `run` | Execute the workflow |
| `clear` | Reset nodes and variables (keeps agents) |
| `quit` | Exit |

### `nexagent agents` — Manage agent definitions

```bash
# List agents (optionally from a file)
nexagent agents list --from agents.yaml

# Import agents from YAML
nexagent agents import agents.yaml
```

---

## Interactive REPL

The REPL lets you build workflows without writing files. Example session:

```
nexagent> agents add
Agent name: researcher
System prompt: You are a research expert...
Max steps: 10
Agent 'researcher' registered.

nexagent> agents add
Agent name: writer
System prompt: You write reports...
Max steps: 10
Agent 'writer' registered.

nexagent> vars set topic "machine learning"
Variable 'topic' = 'machine learning'

nexagent> node add
  Agent name (from list above): researcher
  Node ID: research
  Prompt: Research {{topic}}
  Dependencies (comma-separated, or blank):
Node 'research' added.

nexagent> node add
  Agent name (from list above): writer
  Node ID: report
  Prompt: Write a report from {{research.content}}
  Dependencies (comma-separated, or blank): research
Node 'report' added.

nexagent> node list
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ ID     ┃ Agent      ┃ Prompt                   ┃ Depends On ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ research │ researcher │ Research {{topic}}       │ (none)     │
│ report   │ writer     │ Write a report from...   │ research   │
└────────┴────────────┴──────────────────────────┴────────────┘

nexagent> run
Running workflow with 2 node(s)...
...
```

---

## Structured Data Passing

Nodes communicate through `WorkflowContext`, a shared dictionary:

- Each node's output is stored by node ID
- Downstream nodes access upstream results via `{{node_id.field}}`

### Available Fields

After a node completes, its output has these fields:

| Field | Description |
|---|---|
| `content` | The agent's text output |
| `agent` | The agent name that produced the output |
| `task_id` | Unique task identifier |
| `success` | Boolean — whether the agent succeeded |
| `tool_calls_made` | List of tools that were called |
| `metadata.steps` | Number of reasoning steps taken |
| `metadata.finish_reason` | Why the agent stopped (`stop` or `max_steps`) |
| `metadata.cost_usd` | Estimated cost of the agent's inference |

### Example

```yaml
nodes:
  - id: analysis
    agent: analyst
    prompt: "Analyze this data and return a JSON summary."

  - id: report
    agent: writer
    prompt: >
      Write a report from the analysis.
      Steps taken: {{analysis.metadata.steps}}
      Cost: ${{analysis.metadata.cost_usd}}
    depends_on: [analysis]
```

---

## Advanced Patterns

### Fan-Out / Fan-In

Run multiple workers in parallel, then collect results:

```python
from nexagent.agents.patterns import FanOutBuilder

builder = FanOutBuilder(
    worker_agent="researcher",
    worker_prompts=[
        "Research AI trends in 2024",
        "Research AI trends in 2025",
        "Research AI regulation",
    ],
    collector_agent="writer",
    collector_prompt="Synthesize these findings: {{worker_0.content}}, {{worker_1.content}}, {{worker_2.content}}",
)
nodes = builder.build_nodes()
```

This creates:
- 3 parallel `researcher` nodes (`worker_0`, `worker_1`, `worker_2`)
- 1 `writer` collector that depends on all three

### Supervisor with Dynamic Graph Mutation

A supervisor agent can create new nodes at runtime:

```python
from nexagent.agents.patterns import SupervisorAgent

supervisor = SupervisorAgent(
    name="supervisor",
    registry=registry,
    policy=policy,
)
```

The supervisor receives an `_add_nodes_fn` callback through its task context. During execution, it can call:

```python
add_nodes_fn([
    TaskNode(id="followup", agent_class=GenericAgent, ...),
])
```

The coordinator merges these nodes into the graph on the next loop iteration.

### Combining with YAML

You can mix static YAML nodes with dynamic supervisor nodes:

```yaml
nodes:
  - id: initial_research
    agent: researcher
    prompt: "Research {{topic}}"

  - id: supervisor
    agent: supervisor
    prompt: "Review the research. If gaps exist, create follow-up tasks."
    depends_on: [initial_research]
```

The supervisor can then spawn additional nodes that weren't in the original YAML.

---

## Programmatic API

### Full Workflow from Code

```python
import asyncio
from nexagent.agents import AgentConfig, AgentRegistry, WorkflowParser
from nexagent.agents.coordinator import AgentCoordinator
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

async def main():
    # 1. Set up infrastructure
    registry = ToolRegistry()
    policy = TrustPolicy.default()

    # 2. Register agents
    agent_registry = AgentRegistry(
        tool_registry=registry,
        policy=policy,
    )
    agent_registry.register(AgentConfig(
        name="researcher",
        system_prompt="You research topics.",
    ))
    agent_registry.register(AgentConfig(
        name="writer",
        system_prompt="You write reports.",
    ))

    # 3. Parse workflow
    parser = WorkflowParser(agent_registry=agent_registry)
    graph, ctx = parser.parse_yaml("""
workflow: my_workflow
variables:
  topic: AI
nodes:
  - id: search
    agent: researcher
    prompt: "Research {{topic}}"
  - id: write
    agent: writer
    prompt: "Report: {{search.content}}"
    depends_on: [search]
""")

    # 4. Execute
    coordinator = AgentCoordinator(registry=registry, policy=policy)
    result = await coordinator.run(graph, ctx)

    for node_id, output in result.outputs.items():
        print(f"{node_id}: {output.content[:200]}")

asyncio.run(main())
```

### Tool Registration

Tools can be registered via Python decorator or markdown files.

**Decorator approach:**

```python
from nexagent.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.tool(
    name="web_search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
async def web_search(query: str) -> str:
    # ... implementation
    return f"Results for: {query}"
```

**Markdown-based tool discovery:**

Tools can be defined as `.md` files with YAML frontmatter — no Python imports required. Create `~/.nexagent/tools/date.md`:

```markdown
---
name: current_date
description: Return the current date and time
schema:
  type: object
  properties:
    format:
      type: string
      description: "strftime format, default '%Y-%m-%d'"
---

```python
from datetime import datetime
async def current_date(format: str = "%Y-%m-%d") -> str:
    return datetime.now().strftime(format)
```
```

Load all tools from a directory:

```python
from nexagent.tools.discovery import ToolDiscovery

discovery = ToolDiscovery(registry)
count = discovery.scan_directory("~/.nexagent/tools")
```

Each `.md` file defines:
- **Frontmatter:** `name`, `description`, `schema` (JSON Schema parameters), optional `tags`/`version`
- **Code block:** Python async function matching the tool name

---

## Trust and Autonomy

NexAgent enforces trust at the channel level. Each agent runs in a trust context determined by its channel.

### Trust Levels (highest to lowest)

```
SYSTEM > API > UI > CLI > PUBLIC
```

### Autonomy Dials

| Trust Level | Can Act | Max Tools | Requires Confirmation |
|---|---|---|---|
| SYSTEM | Yes | Unlimited | Never |
| API | Yes | 20 | Never |
| UI | Yes | 10 | On destructive ops |
| CLI | Yes | 5 | Always |
| PUBLIC | No | 0 | Always |

### Creating a Custom Trust Policy

```python
from nexagent.trust.policy import TrustPolicy, TrustLevel, Channel

policy = TrustPolicy(
    trust_levels={
        Channel.API: TrustLevel.API,
    },
    max_tools={
        TrustLevel.API: 50,
    },
    require_confirmation={
        TrustLevel.API: False,
    },
)
```

---

## Troubleshooting

### "NEXAGENT_API_KEY is not set"

The API key is required for frontier model inference. Set it before running:

```bash
export NEXAGENT_API_KEY="sk-..."
```

### "Agent 'X' is not registered"

The agent name in your workflow YAML does not match any registered agent. Check spelling:

```bash
nexagent agents list -a agents.yaml
```

### "Workflow file not found"

The path to your YAML file is incorrect. Use absolute paths or verify the working directory:

```bash
nexagent run /full/path/to/workflow.yaml -a /full/path/to/agents.yaml
```

### "Unknown agent in workflow"

This happens when a workflow YAML references an agent name that hasn't been loaded. Ensure you pass `-a agents.yaml` with your agent definitions.

### Agent loop runs too many steps

Reduce `max_steps` per agent:

```yaml
agents:
  - name: researcher
    max_steps: 5
    system_prompt: "Keep it brief."
```

### High cost from model calls

Use cheaper models for simpler agents. Set the `model` field per agent:

```yaml
agents:
  - name: researcher
    model: gpt-4o          # expensive — complex reasoning
    system_prompt: "Research topics thoroughly."
  - name: summarizer
    model: gpt-4o-mini     # cheap — simple summarization
    system_prompt: "Summarize concisely."
```

Or use `model_capabilities` to auto-select from the pool:

```yaml
agents:
  - name: image_analyst
    model_capabilities: [text, image]
    system_prompt: "Analyze images."
```

You can also route simple tasks through the local classifier by lowering the threshold:

```bash
export NEXAGENT_LOCAL_THRESHOLD=0.5
```

### Prompt injection warnings

If you see prompt injection warnings in the audit log, review the tool arguments and results. The sandbox detects patterns like role-switching instructions and sanitises them automatically.

### Session state lost after REPL exit

The interactive REPL is in-memory only. For reproducible workflows, export your definitions to YAML:

```
nexagent> node list   # Note your nodes
```

Then create a `workflow.yaml` file manually, or save your agent definitions via:

```
nexagent> agents import agents.yaml
```

For persistent sessions, use YAML files directly with `nexagent run`.
