"""Agent workspace management for persistent agent identity and persona.

Each agent can have a workspace directory containing personality and memory
files inspired by the OpenClaw multi-agent model. When a workspace is
provided, the system prompt is enriched with persona data.

Usage::

    workspace = AgentWorkspace(Path("~/.nexagent/agents/researcher"))
    workspace.ensure()  # create dirs + seed templates if missing
    persona = workspace.load_persona()
    enriched_prompt = workspace.compose_system_prompt("You are a researcher.")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona data class
# ---------------------------------------------------------------------------


@dataclass
class AgentPersona:
    """Personality and context loaded from an agent's workspace."""

    name: str = ""
    identity: dict[str, str] = field(default_factory=dict)
    user_context: dict[str, str] = field(default_factory=dict)
    soul: str = ""
    tool_notes: str = ""
    memory: str = ""

    @property
    def has_soul(self) -> bool:
        return bool(self.soul.strip())

    @property
    def has_memory(self) -> bool:
        return bool(self.memory.strip())


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

# Template content seeded for new workspaces
_IDENTITY_TEMPLATE = """# IDENTITY.md - Who Am I?

- **Name:** _(pick something you like)_
- **Creature:** _(AI? robot? familiar? something weirder?)_
- **Vibe:** _(sharp? warm? chaotic? calm?)_
- **Emoji:** _(your signature)_
"""

_USER_TEMPLATE = """# USER.md - About Your Human

- **Name:**
- **What to call them:**
- **Timezone:**
- **Notes:**
"""

_SOUL_TEMPLATE = """# SOUL.md - Who You Are

## Core Truths

- Be genuinely helpful, not performatively helpful.
- Be resourceful before asking.
- Earn trust through competence.
- When in doubt, ask before acting externally.
"""

_TOOLS_TEMPLATE = """# TOOLS.md - Local Notes

Environment-specific notes for this agent go here.
"""

_MEMORY_TEMPLATE = """# MEMORY.md - Long-Term Memory

_(Distilled learnings and important context go here)_
"""


class AgentWorkspace:
    """Manages an agent's persistent workspace directory.

    Parameters
    ----------
    path:
        Path to the workspace directory.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).expanduser()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure(self) -> None:
        """Create the workspace directory and seed template files if missing."""
        self._path.mkdir(parents=True, exist_ok=True)
        self._seed_if_missing("IDENTITY.md", _IDENTITY_TEMPLATE)
        self._seed_if_missing("USER.md", _USER_TEMPLATE)
        self._seed_if_missing("SOUL.md", _SOUL_TEMPLATE)
        self._seed_if_missing("TOOLS.md", _TOOLS_TEMPLATE)
        self._seed_if_missing("MEMORY.md", _MEMORY_TEMPLATE)
        logger.debug("Ensured workspace at %s", self._path)

    def _seed_if_missing(self, filename: str, content: str) -> None:
        target = self._path / filename
        if not target.exists():
            target.write_text(content)
            logger.debug("Seeded %s in workspace %s", filename, self._path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_persona(self) -> AgentPersona:
        """Load persona data from workspace files."""
        persona = AgentPersona()

        # IDENTITY.md — parse key-value pairs from markdown
        identity_path = self._path / "IDENTITY.md"
        if identity_path.exists():
            persona.identity = _parse_key_values(identity_path.read_text())
            persona.name = persona.identity.get("name", "")

        # USER.md
        user_path = self._path / "USER.md"
        if user_path.exists():
            persona.user_context = _parse_key_values(user_path.read_text())

        # SOUL.md
        soul_path = self._path / "SOUL.md"
        if soul_path.exists():
            persona.soul = soul_path.read_text()

        # TOOLS.md
        tools_path = self._path / "TOOLS.md"
        if tools_path.exists():
            persona.tool_notes = tools_path.read_text()

        # MEMORY.md
        memory_path = self._path / "MEMORY.md"
        if memory_path.exists():
            persona.memory = memory_path.read_text()

        return persona

    def save_persona(self, persona: AgentPersona) -> None:
        """Save persona data to workspace files."""
        self.ensure()
        if persona.identity:
            _write_key_values(self._path / "IDENTITY.md", persona.identity, "IDENTITY.md - Who Am I?")
        if persona.user_context:
            _write_key_values(self._path / "USER.md", persona.user_context, "USER.md - About Your Human")
        if persona.soul:
            (self._path / "SOUL.md").write_text(persona.soul)
        if persona.tool_notes:
            (self._path / "TOOLS.md").write_text(persona.tool_notes)
        if persona.memory:
            (self._path / "MEMORY.md").write_text(persona.memory)

    # ------------------------------------------------------------------
    # System prompt composition
    # ------------------------------------------------------------------

    def compose_system_prompt(self, base_prompt: str) -> str:
        """Compose an enriched system prompt from base + persona.

        Merges the base system prompt with SOUL.md personality rules
        and identity information.
        """
        persona = self.load_persona()
        parts = [base_prompt]

        if persona.name:
            parts.append(f"\nYour name is '{persona.name}'.")

        if persona.identity:
            vibe = persona.identity.get("vibe", "")
            creature = persona.identity.get("creature", "")
            if vibe or creature:
                descriptor = ", ".join(d for d in [creature, vibe] if d)
                parts.append(f"\nYou are a {descriptor} assistant.")

        if persona.has_soul:
            # Include non-empty soul rules, skipping markdown headers and comments
            soul_lines = [
                line for line in persona.soul.split("\n")
                if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("_( ")
            ]
            if soul_lines:
                parts.append("\n\n## Operating Rules\n")
                parts.append("\n".join(soul_lines))

        if persona.has_memory:
            memory_lines = [
                line for line in persona.memory.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            if memory_lines:
                parts.append("\n\n## Context\n")
                # Only include first 20 lines of memory to keep prompt manageable
                parts.append("\n".join(memory_lines[:20]))

        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"AgentWorkspace(path={self._path!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_KV_RE = re.compile(r"-\s*\*\*(.+?)\*\*\s*[:=]?\s*(?:_(.+?)_|(.+?))\s*$")


def _parse_key_values(text: str) -> dict[str, str]:
    """Parse markdown key-value pairs like `- **Name:** value` or `- **Name:** _(placeholder)_`."""
    result: dict[str, str] = {}
    for line in text.split("\n"):
        m = _KV_RE.match(line.strip())
        if m:
            key = m.group(1).strip().rstrip(":").lower().replace(" ", "_")
            # Group 2 is _(value), group 3 is plain value
            value = (m.group(2) or m.group(3) or "").strip()
            if value and not value.startswith("("):
                result[key] = value
    return result


def _write_key_values(path: Path, data: dict[str, str], title: str) -> None:
    """Write key-value pairs as markdown list."""
    lines = [f"# {title}", ""]
    for key, value in data.items():
        display_key = key.replace("_", " ").title()
        lines.append(f"- **{display_key}:** {value}")
    lines.append("")
    path.write_text("\n".join(lines))
