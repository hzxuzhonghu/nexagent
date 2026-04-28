"""Discover and register tools from markdown files with YAML frontmatter.

Tools can be defined as ``.md`` files with a YAML header specifying name,
description, and JSON Schema parameters, followed by a Python implementation
in a fenced code block.

Usage::

    discovery = ToolDiscovery(registry)
    count = discovery.scan_directory(Path("~/.nexagent/tools"))
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nexagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Parsed tool specification from a markdown file."""

    name: str
    description: str
    schema: dict[str, Any]
    source_path: Path
    implementation: str | None = None
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class ToolDiscovery:
    """Discover and load tools from markdown files with YAML frontmatter.

    Parameters
    ----------
    registry:
        The ToolRegistry to register discovered tools with.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def scan_directory(self, path: Path) -> int:
        """Scan a directory for ``.md`` tool files and register them.

        Returns the number of tools successfully registered.
        """
        path = Path(path).expanduser()
        if not path.is_dir():
            logger.warning("Tool directory does not exist: %s", path)
            return 0

        count = 0
        for md_file in sorted(path.glob("*.md")):
            try:
                spec = self.load_from_file(md_file)
                if spec:
                    self.register_tool(spec)
                    count += 1
            except Exception as exc:
                logger.error("Failed to load tool from %s: %s", md_file, exc)

        logger.info("Discovered and registered %d tool(s) from %s", count, path)
        return count

    def load_from_file(self, path: Path) -> ToolSpec | None:
        """Parse a single markdown file into a ToolSpec.

        Returns None if the file does not contain valid frontmatter.
        """
        text = path.read_text()

        frontmatter_m = _FRONTMATTER_RE.match(text)
        if not frontmatter_m:
            logger.debug("No YAML frontmatter in %s, skipping", path)
            return None

        try:
            data = yaml.safe_load(frontmatter_m.group(1)) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML frontmatter in {path}: {exc}") from exc

        name = data.get("name")
        if not name:
            raise ValueError(f"Tool file {path} missing 'name' in frontmatter")

        description = data.get("description", "")
        schema = data.get("schema", {"type": "object", "properties": {}})
        tags = data.get("tags", [])
        version = data.get("version", "1.0.0")

        # Extract Python implementation from code block
        code_m = _CODE_BLOCK_RE.search(text)
        implementation = code_m.group(1) if code_m else None

        return ToolSpec(
            name=name,
            description=description,
            schema=schema,
            source_path=path,
            implementation=implementation,
            tags=tags,
            version=version,
        )

    def register_tool(self, spec: ToolSpec) -> None:
        """Register a tool from its specification.

        If the spec has an implementation, it is compiled into a function
        and registered. If not, the tool metadata is logged for reference.
        """
        if spec.implementation:
            fn = _compile_tool_function(spec)
            self._registry.register(
                name=spec.name,
                description=spec.description,
                parameters=spec.schema,
                fn=fn,
                tags=spec.tags,
                version=spec.version,
            )
            logger.info("Registered tool '%s' from %s", spec.name, spec.source_path)
        else:
            logger.info(
                "Tool '%s' from %s has no implementation — metadata only",
                spec.name,
                spec.source_path,
            )

    def __repr__(self) -> str:
        return f"ToolDiscovery(registry={self._registry!r})"


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------


def _compile_tool_function(spec: ToolSpec):
    """Compile tool implementation source code into an async callable.

    The source is executed in an isolated namespace. The function named
    after ``spec.name`` is extracted and returned.
    """
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(spec.implementation, namespace)  # noqa: S102
    except Exception as exc:
        raise ValueError(
            f"Failed to compile tool '{spec.name}' from {spec.source_path}: {exc}"
        ) from exc

    fn = namespace.get(spec.name)
    if fn is None:
        raise ValueError(
            f"Tool '{spec.name}' implementation must define a function named '{spec.name}'"
        )
    return fn


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def discover_from_directory(
    path: Path,
    registry: ToolRegistry | None = None,
) -> int:
    """Convenience: discover tools from a directory using the default registry.

    If no registry is provided, uses ToolRegistry's default.
    """
    if registry is None:
        registry = ToolRegistry()
    disco = ToolDiscovery(registry)
    return disco.scan_directory(path)
