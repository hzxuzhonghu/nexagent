"""Tests for ToolDiscovery from markdown files with YAML frontmatter."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexagent.tools.discovery import ToolDiscovery, ToolSpec
from nexagent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# ToolDiscovery from file
# ---------------------------------------------------------------------------


class TestToolDiscoveryFromFile:
    def test_load_valid_tool_file(self, tmp_path: Path) -> None:
        md_file = tmp_path / "echo.md"
        md_file.write_text("""\
---
name: echo_tool
description: Echo the input text
schema:
  type: object
  properties:
    text:
      type: string
  required: [text]
---

```python
async def echo_tool(text: str) -> str:
    return text
```
""")
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        spec = discovery.load_from_file(md_file)

        assert spec is not None
        assert spec.name == "echo_tool"
        assert spec.description == "Echo the input text"
        assert "text" in spec.schema["properties"]
        assert spec.implementation is not None

    def test_load_file_without_frontmatter_returns_none(self, tmp_path: Path) -> None:
        md_file = tmp_path / "no_frontmatter.md"
        md_file.write_text("# Some documentation\nNo YAML header here.")
        discovery = ToolDiscovery(ToolRegistry())
        spec = discovery.load_from_file(md_file)
        assert spec is None

    def test_load_file_missing_name_raises(self, tmp_path: Path) -> None:
        md_file = tmp_path / "no_name.md"
        md_file.write_text("---\ndescription: No name here\n---\n")
        discovery = ToolDiscovery(ToolRegistry())
        with pytest.raises(ValueError, match="missing 'name'"):
            discovery.load_from_file(md_file)

    def test_load_file_with_invalid_yaml_raises(self, tmp_path: Path) -> None:
        md_file = tmp_path / "bad_yaml.md"
        md_file.write_text("---\nname: test\n{invalid: yaml: [}\n---\n")
        discovery = ToolDiscovery(ToolRegistry())
        with pytest.raises(ValueError, match="Invalid YAML"):
            discovery.load_from_file(md_file)

    def test_load_file_without_code_block(self, tmp_path: Path) -> None:
        md_file = tmp_path / "metadata_only.md"
        md_file.write_text("""\
---
name: metadata_only_tool
description: Just metadata, no implementation
schema:
  type: object
  properties: {}
---

Some documentation about the tool but no code.
""")
        discovery = ToolDiscovery(ToolRegistry())
        spec = discovery.load_from_file(md_file)
        assert spec is not None
        assert spec.name == "metadata_only_tool"
        assert spec.implementation is None


# ---------------------------------------------------------------------------
# ToolDiscovery registration
# ---------------------------------------------------------------------------


class TestToolDiscoveryRegistration:
    def test_register_tool_from_spec(self, tmp_path: Path) -> None:
        md_file = tmp_path / "date.md"
        md_file.write_text("""\
---
name: current_date
description: Return the current date
schema:
  type: object
  properties:
    format:
      type: string
---

```python
from datetime import datetime
async def current_date(format: str = "%Y-%m-%d") -> str:
    return datetime.now().strftime(format)
```
""")
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        spec = discovery.load_from_file(md_file)
        discovery.register_tool(spec)

        assert registry.has("current_date")

    def test_register_metadata_only_tool_logs_info(self, tmp_path: Path) -> None:
        md_file = tmp_path / "meta.md"
        md_file.write_text("""\
---
name: placeholder
description: No code
schema:
  type: object
  properties: {}
---

Just docs.
""")
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        spec = discovery.load_from_file(md_file)
        discovery.register_tool(spec)

        # Should not register a callable since there's no implementation
        assert not registry.has("placeholder")


# ---------------------------------------------------------------------------
# ToolDiscovery scan directory
# ---------------------------------------------------------------------------


class TestToolDiscoveryScan:
    def test_scan_directory(self, tmp_path: Path) -> None:
        (tmp_path / "echo.md").write_text("""\
---
name: scan_echo
description: Echo
schema:
  type: object
  properties: {}
---

```python
async def scan_echo() -> str:
    return "echo"
```
""")
        (tmp_path / "date.md").write_text("""\
---
name: scan_date
description: Date
schema:
  type: object
  properties: {}
---

```python
from datetime import datetime
async def scan_date() -> str:
    return datetime.now().isoformat()
```
""")
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        count = discovery.scan_directory(tmp_path)
        assert count == 2
        assert registry.has("scan_echo")
        assert registry.has("scan_date")

    def test_scan_nonexistent_directory_returns_zero(self, tmp_path: Path) -> None:
        discovery = ToolDiscovery(ToolRegistry())
        count = discovery.scan_directory(tmp_path / "nonexistent")
        assert count == 0

    def test_scan_ignores_non_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a tool")
        (tmp_path / "notes.py").write_text("# not a tool")
        discovery = ToolDiscovery(ToolRegistry())
        count = discovery.scan_directory(tmp_path)
        assert count == 0


# ---------------------------------------------------------------------------
# ToolSpec repr
# ---------------------------------------------------------------------------


class TestToolDiscoveryRepr:
    def test_repr(self) -> None:
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        assert "ToolDiscovery" in repr(discovery)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


class TestDiscoverFromDirectory:
    def test_convenience_function(self, tmp_path: Path) -> None:
        (tmp_path / "hello.md").write_text("""\
---
name: hello
description: Say hello
schema:
  type: object
  properties: {}
---

```python
async def hello() -> str:
    return "hello"
```
""")
        from nexagent.tools.discovery import discover_from_directory
        count = discover_from_directory(tmp_path)
        assert count == 1
