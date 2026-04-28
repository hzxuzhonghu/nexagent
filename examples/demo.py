"""Self-contained NexAgent demo — no API key required.

Runs a multi-agent workflow where all agents use mocked AgentLoop
so you can see the full orchestration pipeline in action.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from nexagent.agents import AgentConfig, AgentRegistry, WorkflowParser
from nexagent.agents.coordinator import AgentCoordinator
from nexagent.memory.tiered import TieredMemory
from nexagent.runtime.agent_loop import AgentResult
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy
from rich.console import Console
from rich.table import Table

console = Console()


async def main() -> None:
    console.print("\n[bold cyan]NexAgent Demo — Multi-Agent Workflow[/bold cyan]\n")

    # 1. Setup infrastructure
    registry = ToolRegistry()
    policy = TrustPolicy.default()
    memory = TieredMemory(base_path=Path("/tmp/nexagent-demo"))

    # 2. Define agents
    agent_registry = AgentRegistry(
        tool_registry=registry,
        policy=policy,
        memory=memory,
    )

    agent_registry.register(AgentConfig(
        name="researcher",
        description="Research assistant",
        system_prompt="You research topics thoroughly.",
        max_steps=3,
    ))

    agent_registry.register(AgentConfig(
        name="writer",
        description="Report writer",
        system_prompt="You write polished reports.",
        max_steps=3,
    ))

    agent_registry.register(AgentConfig(
        name="reviewer",
        description="Critical reviewer",
        system_prompt="You review and critique content.",
        max_steps=3,
    ))

    console.print("[green]Registered 3 agents: researcher, writer, reviewer[/green]\n")

    # 3. Parse the workflow
    parser = WorkflowParser(agent_registry=agent_registry)
    graph, ctx = parser.parse_yaml("""
workflow: demo_workflow
variables:
  topic: "the future of AI agents"
nodes:
  - id: research
    agent: researcher
    prompt: "Research: {{topic}}"
  - id: write
    agent: writer
    prompt: "Write report from: {{research.content}}"
    depends_on: [research]
  - id: review
    agent: reviewer
    prompt: "Review: {{write.content}}"
    depends_on: [write]
""")

    console.print("[cyan]Workflow DAG: research -> write -> review[/cyan]")
    console.print(f"[cyan]Variables: topic = 'the future of AI agents'[/cyan]\n")

    # 4. Mock AgentLoop.run to return simulated outputs (no real API calls)
    mock_outputs = {
        "research": AgentResult(
            output=(
                "Key findings on the future of AI agents:\n"
                "1. Agents are becoming more autonomous and capable\n"
                "2. Multi-agent collaboration is an emerging pattern\n"
                "3. Cost-efficiency will drive adoption of local classifiers\n"
                "4. Memory and tool-use are differentiating capabilities"
            ),
            session_id="demo-research",
            steps_taken=2,
            tool_calls_made=1,
            finish_reason="stop",
            cost_usd=0.003,
        ),
        "write": AgentResult(
            output=(
                "# Report: The Future of AI Agents\n\n"
                "AI agents are rapidly evolving toward greater autonomy. "
                "Key trends include multi-agent collaboration, tiered inference "
                "routing to reduce costs, and persistent memory. "
                "Adoption will accelerate as frameworks mature."
            ),
            session_id="demo-write",
            steps_taken=2,
            tool_calls_made=0,
            finish_reason="stop",
            cost_usd=0.002,
        ),
        "review": AgentResult(
            output=(
                "Review: The report is well-structured and covers the key points. "
                "Suggestion: add concrete adoption timelines and cite specific "
                "frameworks. Overall grade: B+"
            ),
            session_id="demo-review",
            steps_taken=1,
            tool_calls_made=0,
            finish_reason="stop",
            cost_usd=0.001,
        ),
    }

    call_count = {"n": 0}

    async def fake_run(self_prompt: str) -> AgentResult:
        idx = call_count["n"]
        call_count["n"] += 1
        keys = list(mock_outputs.keys())
        if idx < len(keys):
            console.print(f"  [dim]AgentLoop.run() called — returning mock for '{keys[idx]}'[/dim]")
            return mock_outputs[keys[idx]]
        return AgentResult(output="done", session_id="x", steps_taken=0,
                          tool_calls_made=0, finish_reason="stop")

    # 5. Execute the workflow (with mocked agent loops)
    console.print("[bold]Executing workflow...[/bold]\n")

    with patch("nexagent.runtime.agent_loop.AgentLoop.run", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = fake_run

        coordinator = AgentCoordinator(
            registry=registry,
            policy=policy,
            memory=memory,
        )
        result = await coordinator.run(graph, ctx)

    # 6. Display results
    table = Table(show_header=True, header_style="bold magenta", title="Workflow Results")
    table.add_column("Node")
    table.add_column("Agent")
    table.add_column("Output", max_width=70)
    table.add_column("Cost")

    for node_id in ("research", "write", "review"):
        if node_id in result.outputs:
            output = result.outputs[node_id]
            cost_str = f"${output.metadata.get('cost_usd', 0):.4f}"
            table.add_row(node_id, output.agent, output.content[:120], cost_str)

    console.print(table)

    total_cost = sum(
        o.metadata.get("cost_usd", 0) for o in result.outputs.values()
    )
    console.print(f"\n[green]Workflow complete![/green]")
    console.print(f"  Success: {result.success}")
    console.print(f"  Total cost: ${total_cost:.4f}")
    console.print(f"  Run ID: {result.run_id}")


if __name__ == "__main__":
    asyncio.run(main())
