"""Agent configuration and registry for runtime-defined agents.

Agents can be defined through configuration (YAML or programmatic) rather than
requiring Python subclasses. An AgentConfig specifies a name, system prompt,
tool whitelist, and other parameters, which are resolved to GenericAgent
instances at execution time.

Usage::

    registry = AgentRegistry(tool_registry=tools, policy=policy)
    registry.register(AgentConfig(
        name="researcher",
        system_prompt="You are a research expert...",
        tools=["web_search", "read_url"],
    ))
    agent = registry.create_agent("researcher")

Or load from YAML::

    registry.load_yaml("agents.yaml")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from nexagent.agents.generic import GenericAgent
from nexagent.inference.router import InferenceRouter
from nexagent.memory.tiered import TieredMemory
from nexagent.tools.registry import ToolRegistry
from nexagent.trust.policy import TrustPolicy

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for a runtime-defined agent."""

    name: str
    description: str = ""
    system_prompt: str
    tools: list[str] = []
    max_steps: int = 20
    model_override: str | None = None          # DEPRECATED — use ``model`` instead
    model: str | None = None                    # Model id from a ModelPool
    workspace_dir: str | None = None            # Path to agent workspace directory
    model_capabilities: list[str] = []          # Capability requirements for auto-select

    def instantiate(
        self,
        registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory | None = None,
        router: InferenceRouter | None = None,
        model_pool: Any | None = None,
    ) -> GenericAgent:
        """Create a GenericAgent from this configuration.

        Resolution order for model selection:
        1. Explicit ``model`` field (preferred)
        2. Deprecated ``model_override`` field (backward compat)
        3. Auto-select from ``model_pool`` using ``model_capabilities``
        4. Fall back to the global NEXAGENT_MODEL env var
        """
        from nexagent.agents.workspace import AgentWorkspace

        # Resolve effective model id: prefer model, then model_override
        model_id = self.model or self.model_override

        # If model_pool is available but no explicit model, use capabilities to select
        if model_pool and not model_id and self.model_capabilities:
            try:
                selected = model_pool.select_by_capability(self.model_capabilities)
                model_id = selected.id
            except KeyError:
                logger.warning(
                    "No model in pool supports capabilities %s for agent '%s', "
                    "falling back to default",
                    self.model_capabilities,
                    self.name,
                )

        # Build router if a specific model is targeted and no router provided
        effective_router = router
        if effective_router is None and model_id:
            effective_router = InferenceRouter(
                model_pool=model_pool,
                model_id=model_id,
            )

        # Build workspace if configured
        workspace: AgentWorkspace | None = None
        if self.workspace_dir:
            workspace = AgentWorkspace(self.workspace_dir)

        return GenericAgent(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            tools=self.tools,
            max_steps=self.max_steps,
            registry=registry,
            policy=policy,
            memory=memory,
            router=effective_router,
            model_pool=model_pool if effective_router is None else None,
            model_id=model_id if effective_router is None else None,
            workspace=workspace,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Construct from a plain dict."""
        return cls(**data)


class AgentRegistry:
    """Registry of agent configurations.

    Maps agent names to AgentConfig instances, which can be resolved to
    GenericAgent instances at execution time.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        policy: TrustPolicy,
        memory: TieredMemory | None = None,
        router: InferenceRouter | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._policy = policy
        self._memory = memory
        self._router = router
        self._agents: dict[str, AgentConfig] = {}

    def register(self, config: AgentConfig) -> None:
        """Register an agent configuration."""
        if config.name in self._agents:
            logger.warning("Agent '%s' is being overwritten in the registry.", config.name)
        self._agents[config.name] = config
        logger.debug("Registered agent: %s", config.name)

    def get(self, name: str) -> AgentConfig:
        """Get an agent configuration by name."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered")
        return self._agents[name]

    def create_agent(self, name: str) -> GenericAgent:
        """Instantiate a GenericAgent from a registered configuration."""
        config = self.get(name)
        return config.instantiate(
            registry=self._tool_registry,
            policy=self._policy,
            memory=self._memory,
            router=self._router,
        )

    def load_yaml(self, path: Path) -> int:
        """Load agent definitions from a YAML file.

        Expected format::

            agents:
              - name: researcher
                description: "Research assistant"
                system_prompt: "You are a research expert..."
                tools: [web_search, read_url]
                max_steps: 15
                model_override: "gpt-4o"   # optional; overrides NEXAGENT_MODEL

        Returns the number of agents loaded.
        """
        text = Path(path).read_text()
        data = yaml.safe_load(text)
        if not data or "agents" not in data:
            return 0

        count = 0
        for item in data["agents"]:
            config = AgentConfig(**item)
            self.register(config)
            count += 1

        logger.info("Loaded %d agent(s) from %s", count, path)
        return count

    @property
    def names(self) -> list[str]:
        """Return registered agent names."""
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={list(self._agents.keys())})"
