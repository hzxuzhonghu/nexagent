"""MCP-native tool registry with auto-discovery.

Tools are plain Python async callables decorated with @tool. The registry
validates arguments against JSON Schema before invocation and supports
auto-discovery via Python entry-points (group: "nexagent.tools").

Usage::

    registry = ToolRegistry()

    @registry.tool(
        name="get_weather",
        description="Return current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    async def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    # MCP-compatible schema for all registered tools
    schemas = registry.schemas()

    # Invoke
    result = await registry.invoke("get_weather", {"city": "London"})
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@dataclass
class ToolDefinition:
    """Complete definition of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[Any]]
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"

    def to_schema(self) -> dict[str, Any]:
        """Return an MCP/OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class SchemaValidationError(Exception):
    """Raised when tool arguments fail JSON Schema validation."""


class ToolNotFoundError(KeyError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    """Registry of available tools.

    Thread-safe for read operations. Not designed for concurrent registration
    from multiple threads (registration happens at startup).
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Awaitable[Any]],
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> None:
        """Register a tool directly."""
        if name in self._tools:
            logger.warning("Tool '%s' is being overwritten in the registry.", name)
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
            tags=tags or [],
            version=version,
        )
        logger.debug("Registered tool: %s", name)

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> Callable[[F], F]:
        """Decorator to register an async function as a tool."""

        def decorator(fn: F) -> F:
            self.register(
                name=name,
                description=description,
                parameters=parameters,
                fn=fn,
                tags=tags,
                version=version,
            )
            return fn

        return decorator

    def deregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def auto_discover(self, entry_point_group: str = "nexagent.tools") -> int:
        """Load tools from installed entry-points.

        Entry-point callables should be module-level functions that accept a
        ToolRegistry and register their tools:

            def register(registry: ToolRegistry) -> None:
                registry.register(...)

        Returns the number of tools newly registered.
        """
        count_before = len(self._tools)
        try:
            eps = importlib.metadata.entry_points(group=entry_point_group)
        except Exception:
            eps = []

        for ep in eps:
            try:
                register_fn = ep.load()
                register_fn(self)
                logger.info("Auto-discovered tools from entry-point: %s", ep.name)
            except Exception as exc:
                logger.error(
                    "Failed to load tools from entry-point '%s': %s", ep.name, exc
                )
        return len(self._tools) - count_before

    def load_module(self, module_path: str) -> int:
        """Import a Python module and trigger its @tool registrations.

        The module must import this registry instance or accept a registry
        parameter. For simple cases where the module registers on a global
        registry, just importing it is sufficient.
        """
        count_before = len(self._tools)
        importlib.import_module(module_path)
        return len(self._tools) - count_before

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return MCP-compatible schemas for registered tools.

        If names is provided, only those tools are included.
        """
        tools = (
            [self._tools[n] for n in names if n in self._tools]
            if names
            else list(self._tools.values())
        )
        return [t.to_schema() for t in tools]

    def subset(self, names: list[str]) -> ToolRegistry:
        """Return a new registry containing only the named tools."""
        new = ToolRegistry()
        for name in names:
            if name in self._tools:
                new._tools[name] = self._tools[name]
        return new

    def filter_by_tag(self, tag: str) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if tag in t.tags]

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def invoke(
        self,
        name: str,
        args: dict[str, Any],
        validate: bool = True,
    ) -> Any:
        """Invoke a registered tool with the given arguments.

        Parameters
        ----------
        name:
            Tool name to invoke.
        args:
            Arguments dict; validated against JSON Schema if validate=True.
        validate:
            Run schema validation before invocation. Disable only in tests.
        """
        defn = self.get(name)

        if validate:
            self._validate_args(defn, args)

        sig = inspect.signature(defn.fn)
        if sig.parameters:
            return await defn.fn(**args)
        return await defn.fn()

    def _validate_args(self, defn: ToolDefinition, args: dict[str, Any]) -> None:
        """Lightweight JSON Schema validation (required fields + basic types)."""
        schema = defn.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for req_field in required:
            if req_field not in args:
                raise SchemaValidationError(
                    f"Tool '{defn.name}' missing required argument: '{req_field}'"
                )

        for arg_name, arg_value in args.items():
            if arg_name not in properties:
                continue
            expected_type = properties[arg_name].get("type")
            if expected_type and not self._check_type(arg_value, expected_type):
                raise SchemaValidationError(
                    f"Tool '{defn.name}' argument '{arg_name}': "
                    f"expected {expected_type}, got {type(arg_value).__name__}"
                )

    @staticmethod
    def _check_type(value: Any, json_type: str) -> bool:
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_map.get(json_type)
        if expected is None:
            return True
        # JSON number can be int; bool is a subtype of int in Python — handle carefully
        if json_type == "integer" and isinstance(value, bool):
            return False
        return isinstance(value, expected)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# ---------------------------------------------------------------------------
# Module-level default registry (convenience for decorator usage)
# ---------------------------------------------------------------------------

_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    tags: list[str] | None = None,
) -> Callable[[F], F]:
    """Convenience decorator that registers on the default registry."""
    return get_default_registry().tool(
        name=name, description=description, parameters=parameters, tags=tags
    )
