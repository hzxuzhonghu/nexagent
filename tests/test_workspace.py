"""Tests for AgentWorkspace and AgentPersona."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexagent.agents.workspace import AgentPersona, AgentWorkspace


# ---------------------------------------------------------------------------
# AgentWorkspace lifecycle
# ---------------------------------------------------------------------------


class TestAgentWorkspaceLifecycle:
    def test_ensure_creates_directory(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "agent1")
        workspace.ensure()
        assert workspace.path.is_dir()

    def test_ensure_seeds_template_files(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "agent2")
        workspace.ensure()
        for name in ("IDENTITY.md", "USER.md", "SOUL.md", "TOOLS.md", "MEMORY.md"):
            assert (workspace.path / name).exists()

    def test_ensure_idempotent(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "agent3")
        workspace.ensure()
        soul_content = (workspace.path / "SOUL.md").read_text()
        workspace.ensure()
        # Content should not be overwritten
        assert (workspace.path / "SOUL.md").read_text() == soul_content

    def test_path_expands_tilde(self) -> None:
        workspace = AgentWorkspace("~/myagent")
        assert "~" not in str(workspace.path)

    def test_repr(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path)
        assert "AgentWorkspace" in repr(workspace)


# ---------------------------------------------------------------------------
# AgentWorkspace persona loading
# ---------------------------------------------------------------------------


class TestAgentWorkspacePersona:
    def test_load_persona_empty_workspace(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path)
        workspace.ensure()
        persona = workspace.load_persona()
        # Template files exist but have placeholder text, not real values
        assert isinstance(persona, AgentPersona)

    def test_load_persona_nonexistent_workspace(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "nonexistent")
        persona = workspace.load_persona()
        assert persona.name == ""
        assert not persona.has_soul
        assert not persona.has_memory

    def test_load_persona_with_identity(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws1")
        workspace.ensure()
        (workspace.path / "IDENTITY.md").write_text(
            "- **Name:** Researcher\n"
            "- **Creature:** AI assistant\n"
            "- **Vibe:** sharp and analytical\n"
            "- **Emoji:** 🧪\n"
        )
        persona = workspace.load_persona()
        assert persona.name == "Researcher"
        assert "creature" in persona.identity

    def test_load_persona_with_soul(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws2")
        workspace.ensure()
        (workspace.path / "SOUL.md").write_text(
            "# Soul\n\n- Always be helpful\n- Think before responding\n"
        )
        persona = workspace.load_persona()
        assert persona.has_soul
        assert "helpful" in persona.soul

    def test_load_persona_with_memory(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws3")
        workspace.ensure()
        (workspace.path / "MEMORY.md").write_text(
            "# Memory\n\n- User prefers Python 3.12\n- Project uses ruff for linting\n"
        )
        persona = workspace.load_persona()
        assert persona.has_memory
        assert "Python" in persona.memory


# ---------------------------------------------------------------------------
# AgentWorkspace persona saving
# ---------------------------------------------------------------------------


class TestAgentWorkspaceSavePersona:
    def test_save_and_reload_persona(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws4")
        workspace.ensure()

        persona = AgentPersona(
            name="Writer",
            identity={"name": "Writer", "vibe": "creative"},
            soul="Be creative and thorough.",
            tool_notes="Use the local file editor.",
            memory="User writes fiction.",
        )
        workspace.save_persona(persona)

        reloaded = workspace.load_persona()
        assert reloaded.name == "Writer"
        assert reloaded.has_soul
        assert reloaded.has_memory


# ---------------------------------------------------------------------------
# System prompt composition
# ---------------------------------------------------------------------------


class TestComposeSystemPrompt:
    def test_base_prompt_only_no_workspace_files(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws5")
        prompt = workspace.compose_system_prompt("You are a helper.")
        assert "You are a helper." in prompt

    def test_base_prompt_with_identity(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws6")
        workspace.ensure()
        (workspace.path / "IDENTITY.md").write_text(
            "- **Name:** Bot\n"
            "- **Creature:** AI\n"
            "- **Vibe:** friendly\n"
        )
        prompt = workspace.compose_system_prompt("You are a helper.")
        assert "Bot" in prompt

    def test_base_prompt_with_soul(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws7")
        workspace.ensure()
        (workspace.path / "SOUL.md").write_text(
            "# Rules\n\n- Be honest.\n- Be concise.\n"
        )
        prompt = workspace.compose_system_prompt("You are a helper.")
        assert "Operating Rules" in prompt
        assert "Be honest" in prompt

    def test_base_prompt_with_memory(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws8")
        workspace.ensure()
        (workspace.path / "MEMORY.md").write_text(
            "# Memory\n\n- User prefers short answers.\n"
        )
        prompt = workspace.compose_system_prompt("You are a helper.")
        assert "Context" in prompt
        assert "short answers" in prompt

    def test_composed_prompt_includes_all_parts(self, tmp_path: Path) -> None:
        workspace = AgentWorkspace(tmp_path / "ws9")
        workspace.ensure()
        (workspace.path / "IDENTITY.md").write_text(
            "- **Name:** Assistant\n- **Creature:** AI\n- **Vibe:** warm\n"
        )
        (workspace.path / "SOUL.md").write_text("- Be kind.\n")
        (workspace.path / "MEMORY.md").write_text("- User likes tea.\n")

        prompt = workspace.compose_system_prompt("You research topics.")
        assert "You research topics." in prompt
        assert "Assistant" in prompt
        assert "warm" in prompt
        assert "Be kind" in prompt
        assert "tea" in prompt
