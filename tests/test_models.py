"""Tests for ModelPool, ProviderConfig, and ModelConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexagent.inference.models import ModelConfig, ModelCost, ModelPool, ProviderConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pool() -> ModelPool:
    return ModelPool(
        providers={
            "openai": ProviderConfig(
                name="openai",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                api_key="sk-ant-test",
                base_url="https://api.anthropic.com/v1",
            ),
        },
        models={
            "gpt-4o": ModelConfig(
                id="gpt-4o",
                provider="openai",
                capabilities=["text", "image"],
                cost=ModelCost(input_per_m=2.50, output_per_m=10.00),
                context_window=128000,
            ),
            "gpt-4o-mini": ModelConfig(
                id="gpt-4o-mini",
                provider="openai",
                capabilities=["text"],
                cost=ModelCost(input_per_m=0.15, output_per_m=0.60),
            ),
            "claude-sonnet": ModelConfig(
                id="claude-sonnet",
                provider="anthropic",
                capabilities=["text", "image"],
                cost=ModelCost(input_per_m=3.00, output_per_m=15.00),
                context_window=200000,
                reasoning=True,
            ),
        },
    )


# ---------------------------------------------------------------------------
# ModelCost
# ---------------------------------------------------------------------------


class TestModelCost:
    def test_from_dict_all_fields(self) -> None:
        cost = ModelCost.from_dict({
            "input_per_m": 2.5,
            "output_per_m": 10.0,
            "cache_read_per_m": 1.0,
            "cache_write_per_m": 1.25,
        })
        assert cost.input_per_m == 2.5
        assert cost.output_per_m == 10.0

    def test_from_dict_empty_defaults(self) -> None:
        cost = ModelCost.from_dict({})
        assert cost.input_per_m == 0.0
        assert cost.output_per_m == 0.0


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_from_dict(self) -> None:
        cfg = ModelConfig.from_dict("gpt-4o", {
            "provider": "openai",
            "capabilities": ["text", "image"],
            "cost": {"input_per_m": 2.5, "output_per_m": 10.0},
            "context_window": 128000,
        })
        assert cfg.id == "gpt-4o"
        assert cfg.provider == "openai"
        assert "image" in cfg.capabilities
        assert cfg.cost.input_per_m == 2.5

    def test_from_dict_defaults(self) -> None:
        cfg = ModelConfig.from_dict("simple", {"provider": "openai"})
        assert cfg.capabilities == ["text"]
        assert cfg.context_window == 128000
        assert cfg.max_tokens == 4096
        assert cfg.reasoning is False


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_from_dict(self) -> None:
        pc = ProviderConfig.from_dict("openai", {
            "api_key": "sk-test",
            "base_url": "https://api.openai.com/v1",
        })
        assert pc.name == "openai"
        assert pc.api_key == "sk-test"
        assert pc.base_url == "https://api.openai.com/v1"

    def test_from_dict_default_base_url(self) -> None:
        pc = ProviderConfig.from_dict("openai", {"api_key": "sk-test"})
        assert "openai" in pc.base_url


# ---------------------------------------------------------------------------
# ModelPool lookups
# ---------------------------------------------------------------------------


class TestModelPoolLookups:
    def test_get_model(self, sample_pool: ModelPool) -> None:
        model = sample_pool.get_model("gpt-4o")
        assert model.id == "gpt-4o"
        assert model.provider == "openai"

    def test_get_model_missing(self, sample_pool: ModelPool) -> None:
        with pytest.raises(KeyError, match="unknown-model"):
            sample_pool.get_model("unknown-model")

    def test_get_provider(self, sample_pool: ModelPool) -> None:
        provider = sample_pool.get_provider("gpt-4o")
        assert provider.name == "openai"

    def test_get_provider_for_missing_model(self, sample_pool: ModelPool) -> None:
        with pytest.raises(KeyError, match="unknown"):
            sample_pool.get_provider("unknown")

    def test_has_model(self, sample_pool: ModelPool) -> None:
        assert sample_pool.has_model("gpt-4o")
        assert not sample_pool.has_model("gpt-5")

    def test_list_models(self, sample_pool: ModelPool) -> None:
        names = sample_pool.list_models()
        assert len(names) == 3
        assert "gpt-4o-mini" in names


# ---------------------------------------------------------------------------
# ModelPool selection
# ---------------------------------------------------------------------------


class TestModelPoolSelection:
    def test_select_by_capability_image(self, sample_pool: ModelPool) -> None:
        model = sample_pool.select_by_capability(["text", "image"])
        # gpt-4o is first with both capabilities
        assert "image" in model.capabilities

    def test_select_by_capability_text_only(self, sample_pool: ModelPool) -> None:
        model = sample_pool.select_by_capability(["text"])
        assert model is not None

    def test_select_by_capability_unsupported(self, sample_pool: ModelPool) -> None:
        with pytest.raises(KeyError, match="audio"):
            sample_pool.select_by_capability(["audio"])

    def test_select_cheapest(self, sample_pool: ModelPool) -> None:
        model = sample_pool.select_cheapest()
        assert model.id == "gpt-4o-mini"

    def test_select_cheapest_with_capability(self, sample_pool: ModelPool) -> None:
        model = sample_pool.select_cheapest(["text", "image"])
        # gpt-4o-mini doesn't support image, so should be gpt-4o
        assert model.id == "gpt-4o"

    def test_select_cheapest_no_candidates(self) -> None:
        pool = ModelPool()
        with pytest.raises(KeyError):
            pool.select_cheapest()


# ---------------------------------------------------------------------------
# ModelPool from_env
# ---------------------------------------------------------------------------


class TestModelPoolFromEnv:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("NEXAGENT_MODEL", "NEXAGENT_API_BASE", "NEXAGENT_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        pool = ModelPool.from_env()
        assert pool.has_model("gpt-4o")
        assert len(pool.list_models()) == 1

    def test_from_env_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEXAGENT_MODEL", "llama3.1")
        monkeypatch.setenv("NEXAGENT_API_BASE", "http://localhost:11434/v1")
        pool = ModelPool.from_env()
        assert pool.has_model("llama3.1")
        provider = pool.get_provider("llama3.1")
        assert "localhost" in provider.base_url

    def test_from_env_detects_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEXAGENT_MODEL", "claude-3.5")
        monkeypatch.setenv("NEXAGENT_API_BASE", "https://api.anthropic.com/v1")
        pool = ModelPool.from_env()
        provider = pool.get_provider("claude-3.5")
        assert provider.name == "anthropic"


# ---------------------------------------------------------------------------
# ModelPool from_yaml
# ---------------------------------------------------------------------------


class TestModelPoolFromYaml:
    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text("""
providers:
  openai:
    api_key: sk-test-123
    base_url: https://api.openai.com/v1

models:
  gpt-4o:
    provider: openai
    capabilities: [text, image]
    cost: {input_per_m: 2.50, output_per_m: 10.00}
  gpt-4o-mini:
    provider: openai
    capabilities: [text]
    cost: {input_per_m: 0.15, output_per_m: 0.60}
""")
        pool = ModelPool.from_yaml(yaml_file)
        assert pool.has_model("gpt-4o")
        assert pool.has_model("gpt-4o-mini")
        provider = pool.get_provider("gpt-4o")
        assert provider.api_key == "sk-test-123"

    def test_from_yaml_env_var_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "resolved-key")
        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text("""
providers:
  openai:
    api_key: ${MY_API_KEY}
    base_url: https://api.openai.com/v1

models:
  gpt-4o:
    provider: openai
""")
        pool = ModelPool.from_yaml(yaml_file)
        provider = pool.get_provider("gpt-4o")
        assert provider.api_key == "resolved-key"

    def test_repr(self, sample_pool: ModelPool) -> None:
        r = repr(sample_pool)
        assert "gpt-4o" in r
        assert "openai" in r

    def test_len(self, sample_pool: ModelPool) -> None:
        assert len(sample_pool) == 3
