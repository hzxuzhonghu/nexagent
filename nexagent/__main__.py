"""NexAgent CLI entry point.

Provides two modes:
  - ``nexagent run <workflow.yaml>`` — execute a YAML-defined workflow
  - ``nexagent chat`` — interactive REPL for dynamic agent/workflow building
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from nexagent.agents.coordinator import AgentCoordinator
from nexagent.agents.registry import AgentConfig, AgentRegistry
from nexagent.agents.subagent import AgentTask
from nexagent.agents.workflow import WorkflowContext, WorkflowParser
from nexagent.runtime.context import SessionContext
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy, Channel

console = Console()


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def make_tool_registry() -> ToolRegistry:
    """Create a tool registry with built-in tools."""
    registry = ToolRegistry()

    @registry.tool(
        name="echo",
        description="Echo the input text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    async def echo(text: str) -> str:
        return text

    return registry


def make_policy() -> TrustPolicy:
    return TrustPolicy.default()


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


async def cmd_run(workflow_path: Path, agents_path: Path | None) -> int:
    """Execute a YAML workflow definition."""
    if not workflow_path.exists():
        console.print(f"[red]Workflow file not found: {workflow_path}[/red]")
        return 1

    tool_registry = make_tool_registry()
    policy = make_policy()
    agent_registry = AgentRegistry(
        tool_registry=tool_registry,
        policy=policy,
    )

    # Optionally load agent definitions from a separate file
    if agents_path and agents_path.exists():
        count = agent_registry.load_yaml(agents_path)
        console.print(f"[green]Loaded {count} agent(s) from {agents_path}[/green]")

    console.print(f"[cyan]Loading workflow from {workflow_path}...[/cyan]")
    parser = WorkflowParser(agent_registry=agent_registry)

    try:
        graph, ctx = parser.parse_file(workflow_path)
    except KeyError as e:
        console.print(f"[red]Unknown agent in workflow: {e}[/red]")
        console.print(f"[yellow]Available agents: {agent_registry.names}[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Failed to parse workflow: {e}[/red]")
        return 1

    console.print(f"[green]Workflow '{graph.metadata.get('name', 'unnamed')}': {len(graph.nodes)} node(s)[/green]")
    console.print(f"  Nodes: {', '.join(n.id for n in graph.nodes)}")
    console.print()

    coordinator = AgentCoordinator(
        registry=tool_registry,
        policy=policy,
    )
    result = await coordinator.run(graph, ctx)

    _print_run_result(result, ctx)
    return 0 if result.success else 1


def _print_run_result(result: Any, ctx: WorkflowContext) -> None:
    """Display coordinator outputs in a nice format."""
    table = Table(show_header=True, header_style="bold cyan", title="Workflow Results")
    table.add_column("Node")
    table.add_column("Agent")
    table.add_column("Output")
    table.add_column("Status")

    for node_id, output in result.outputs.items():
        status = "[green]Done[/green]" if output.success else "[red]Failed[/red]"
        table.add_row(node_id, output.agent, output.content[:120], status)

    console.print(table)

    if result.failed_nodes:
        console.print(f"\n[red]Failed nodes: {', '.join(result.failed_nodes)}[/red]")

    console.print(f"\nRun ID: {result.run_id} | Success: {result.success}")


# ---------------------------------------------------------------------------
# Subcommand: chat (interactive REPL)
# ---------------------------------------------------------------------------


class InteractiveSession:
    """Manages the interactive REPL session for dynamic agent/workflow building."""

    def __init__(self) -> None:
        self.tool_registry = make_tool_registry()
        self.policy = make_policy()
        self.agent_registry = AgentRegistry(
            tool_registry=self.tool_registry,
            policy=self.policy,
        )
        self.nodes: list[dict[str, Any]] = []
        self.variables: dict[str, Any] = {}

    async def loop(self) -> int:
        """Run the interactive REPL."""
        console.print(Panel(
            Markdown(self._help_text()),
            title="NexAgent Interactive Mode",
            border_style="blue",
        ))

        while True:
            try:
                cmd = Prompt.ask("\n[cyan]nexagent>[/cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Exiting interactive mode.[/yellow]")
                return 0

            if not cmd:
                continue

            parts = cmd.split(None, 1)
            action = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if action in ("q", "quit", "exit"):
                console.print("[yellow]Goodbye![/yellow]")
                return 0
            elif action in ("h", "help"):
                console.print(Markdown(self._help_text()))
            elif action in ("agents", "agent") and args.startswith("list"):
                self._cmd_list_agents()
            elif action in ("agents", "agent") and args.startswith("add"):
                await self._cmd_add_agent()
            elif action in ("agents", "agent") and args.startswith("import"):
                await self._cmd_import_agents(args.split(None, 1)[1] if len(args.split(None, 1)) > 1 else None)
            elif action in ("node", "nodes") and args.startswith("add"):
                await self._cmd_add_node()
            elif action in ("node", "nodes") and args.startswith("list"):
                self._cmd_list_nodes()
            elif action in ("var", "vars", "variables") and args.startswith("set"):
                self._cmd_set_var(args.split(None, 2)[1] if len(args.split(None, 2)) > 1 else None,
                                  args.split(None, 2)[2] if len(args.split(None, 2)) > 2 else None)
            elif action in ("var", "vars", "variables") and args.startswith("list"):
                self._cmd_list_vars()
            elif action in ("run", "execute"):
                await self._cmd_run()
            elif action in ("clear", "reset"):
                self._cmd_clear()
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                console.print("Type [bold]help[/bold] for available commands.")

        return 0

    def _help_text(self) -> str:
        return """
**Commands:**
- `agents list` — Show registered agents
- `agents add` — Add a new agent interactively
- `agents import <file.yaml>` — Load agents from YAML file
- `node add` — Add a workflow node
- `node list` — Show workflow nodes
- `vars set <key> <value>` — Set a workflow variable
- `vars list` — Show workflow variables
- `run` — Execute the current workflow
- `clear` — Reset everything
- `quit` — Exit
"""

    def _cmd_list_agents(self) -> None:
        if not self.agent_registry.names:
            console.print("[yellow]No agents registered.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan", title="Registered Agents")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Tools")
        table.add_column("Max Steps")

        for name in self.agent_registry.names:
            config = self.agent_registry.get(name)
            tools = ", ".join(config.tools) if config.tools else "(all)"
            table.add_row(name, config.description, tools, str(config.max_steps))

        console.print(table)

    async def _cmd_add_agent(self) -> None:
        console.print("[cyan]Adding a new agent:[/cyan]")

        name = Prompt.ask("  Agent name")
        if not name:
            return

        system_prompt = Prompt.ask("  System prompt")
        if not system_prompt:
            console.print("[red]System prompt is required.[/red]")
            return

        tools_str = Prompt.ask("  Tool names (comma-separated, or blank for all)", default="")
        tools = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else []

        max_steps_str = Prompt.ask("  Max steps", default="20")
        try:
            max_steps = int(max_steps_str)
        except ValueError:
            max_steps = 20

        description = Prompt.ask("  Description (optional)", default="")

        config = AgentConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=max_steps,
        )
        self.agent_registry.register(config)
        console.print(f"[green]Agent '{name}' registered.[/green]")

    async def _cmd_import_agents(self, path: str | None) -> None:
        if not path:
            path = Prompt.ask("  Path to agents YAML file")

        yaml_path = Path(path)
        if not yaml_path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            return

        count = self.agent_registry.load_yaml(yaml_path)
        console.print(f"[green]Imported {count} agent(s).[/green]")

    async def _cmd_add_node(self) -> None:
        if not self.agent_registry.names:
            console.print("[yellow]No agents registered yet. Use 'agents add' first.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#")
        table.add_column("Available Agents")
        for i, name in enumerate(self.agent_registry.names, 1):
            table.add_row(str(i), name)
        console.print(table)

        agent_name = Prompt.ask("  Agent name (from list above)")
        if agent_name not in self.agent_registry.names:
            console.print(f"[red]Unknown agent: {agent_name}[/red]")
            return

        node_id = Prompt.ask("  Node ID")
        prompt_text = Prompt.ask("  Prompt")
        depends_on_str = Prompt.ask("  Dependencies (comma-separated node IDs, or blank)", default="")
        depends_on = [d.strip() for d in depends_on_str.split(",") if d.strip()] if depends_on_str else []

        self.nodes.append({
            "id": node_id,
            "agent": agent_name,
            "prompt": prompt_text,
            "depends_on": depends_on,
        })
        console.print(f"[green]Node '{node_id}' added.[/green]")

    def _cmd_list_nodes(self) -> None:
        if not self.nodes:
            console.print("[yellow]No workflow nodes defined.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan", title="Workflow Nodes")
        table.add_column("ID")
        table.add_column("Agent")
        table.add_column("Prompt")
        table.add_column("Depends On")

        for node in self.nodes:
            deps = ", ".join(node["depends_on"]) if node["depends_on"] else "(none)"
            table.add_row(node["id"], node["agent"], node["prompt"][:60], deps)

        console.print(table)

    def _cmd_set_var(self, key: str | None, value: str | None) -> None:
        if key is None:
            key = Prompt.ask("  Variable name")
        if value is None:
            value = Prompt.ask("  Value")
        self.variables[key] = value
        console.print(f"[green]Variable '{key}' = '{value}'[/green]")

    def _cmd_list_vars(self) -> None:
        if not self.variables:
            console.print("[yellow]No variables set.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan", title="Variables")
        table.add_column("Key")
        table.add_column("Value")
        for k, v in self.variables.items():
            table.add_row(k, str(v))
        console.print(table)

    async def _cmd_run(self) -> None:
        if not self.nodes:
            console.print("[yellow]No nodes to run. Use 'node add' first.[/yellow]")
            return

        # Build workflow YAML dynamically
        workflow_data = {
            "workflow": "interactive",
            "variables": dict(self.variables),
            "nodes": self.nodes,
        }
        yaml_text = _dict_to_yaml(workflow_data)

        console.print(f"[cyan]Running workflow with {len(self.nodes)} node(s)...[/cyan]")

        parser = WorkflowParser(agent_registry=self.agent_registry)
        try:
            graph, ctx = parser.parse_yaml(yaml_text)
        except KeyError as e:
            console.print(f"[red]Unknown agent: {e}[/red]")
            return

        coordinator = AgentCoordinator(
            registry=self.tool_registry,
            policy=self.policy,
        )
        result = await coordinator.run(graph, ctx)
        _print_run_result(result, ctx)

    def _cmd_clear(self) -> None:
        self.nodes.clear()
        self.variables.clear()
        console.print("[yellow]Cleared all nodes and variables. Agents remain.[/yellow]")


def _dict_to_yaml(data: dict[str, Any]) -> str:
    """Minimal dict-to-YAML converter (avoids yaml.dump formatting quirks)."""
    lines: list[str] = []
    lines.append(f"workflow: {data.get('workflow', 'unnamed')}")

    if data.get("variables"):
        lines.append("variables:")
        for k, v in data["variables"].items():
            lines.append(f"  {k}: {v}")

    lines.append("nodes:")
    for node in data.get("nodes", []):
        lines.append(f"  - id: {node['id']}")
        lines.append(f"    agent: {node['agent']}")
        lines.append(f"    prompt: \"{node['prompt']}\"")
        if node.get("depends_on"):
            deps = ", ".join(node["depends_on"])
            lines.append(f"    depends_on: [{deps}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand: agents
# ---------------------------------------------------------------------------


async def cmd_agents(action: str, path: Path | None) -> int:
    """Manage agent definitions."""
    tool_registry = make_tool_registry()
    policy = make_policy()
    agent_registry = AgentRegistry(
        tool_registry=tool_registry,
        policy=policy,
    )

    if action == "list":
        if path and path.exists():
            agent_registry.load_yaml(path)

        if not agent_registry.names:
            console.print("[yellow]No agents registered.[/yellow]")
            return 0

        table = Table(show_header=True, header_style="bold cyan", title="Agents")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Tools")
        table.add_column("Max Steps")

        for name in agent_registry.names:
            config = agent_registry.get(name)
            tools = ", ".join(config.tools) if config.tools else "(all)"
            table.add_row(name, config.description, tools, str(config.max_steps))

        console.print(table)
        return 0

    elif action == "import" and path:
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            return 1

        count = agent_registry.load_yaml(path)
        console.print(f"[green]Imported {count} agent(s) from {path}[/green]")

        table = Table(show_header=True, header_style="bold cyan", title="Loaded Agents")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Tools")
        for name in agent_registry.names:
            config = agent_registry.get(name)
            tools = ", ".join(config.tools) if config.tools else "(all)"
            table.add_row(name, config.description, tools)
        console.print(table)
        return 0

    else:
        console.print("[red]Usage: nexagent agents list [--from file.yaml][/red]")
        console.print("[red]       nexagent agents import <file.yaml>[/red]")
        return 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nexagent",
        description="NexAgent — multi-agent orchestration CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Execute a YAML workflow")
    run_parser.add_argument("workflow", type=Path, help="Path to workflow YAML file")
    run_parser.add_argument("--agents", "-a", type=Path, help="Path to agents YAML file")

    # chat
    subparsers.add_parser("chat", help="Interactive multi-agent session")

    # agents
    agents_parser = subparsers.add_parser("agents", help="Manage agent definitions")
    agents_parser.add_argument("action", choices=["list", "import"], help="Action to perform")
    agents_parser.add_argument("path", nargs="?", type=Path, help="Path to agents YAML file")
    agents_parser.add_argument("--from", "-f", dest="from_path", type=Path, help="Path to agents YAML file (for list)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        exit_code = asyncio.run(cmd_run(args.workflow, getattr(args, "agents", None)))
        sys.exit(exit_code)
    elif args.command == "chat":
        session = InteractiveSession()
        exit_code = asyncio.run(session.loop())
        sys.exit(exit_code)
    elif args.command == "agents":
        path = args.from_path or getattr(args, "path", None)
        exit_code = asyncio.run(cmd_agents(args.action, path))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
