"""Model pool and provider configuration for multi-model, multi-provider inference.

Supports a pool of models from different providers, each with cost metadata
and capability declarations. Agents can select models by name, capability
requirements, or cost optimisation.

Usage::

    pool = ModelPool.from_yaml(Path("models.yaml"))
    provider = pool.get_provider("gpt-4o")
    model = pool.get_model("gpt-4o")

    # Or backward-compatible from env vars:
    pool = ModelPool.from_env()
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelCost:
    """Cost metadata per model."""

    input_per_m: float = 0.0          # $ per million input tokens
    output_per_m: float = 0.0         # $ per million output tokens
    cache_read_per_m: float = 0.0
    cache_write_per_m: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCost:
        return cls(
            input_per_m=data.get("input_per_m", 0.0),
            output_per_m=data.get("output_per_m", 0.0),
            cache_read_per_m=data.get("cache_read_per_m", 0.0),
            cache_write_per_m=data.get("cache_write_per_m", 0.0),
        )


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    id: str
    provider: str
    capabilities: list[str] = field(default_factory=lambda: ["text"])
    cost: ModelCost = field(default_factory=ModelCost)
    context_window: int = 128000
    max_tokens: int = 4096
    reasoning: bool = False

    @classmethod
    def from_dict(cls, model_id: str, data: dict[str, Any]) -> ModelConfig:
        cost_raw = data.get("cost", {})
        return cls(
            id=model_id,
            provider=data["provider"],
            capabilities=data.get("capabilities", ["text"]),
            cost=ModelCost.from_dict(cost_raw) if isinstance(cost_raw, dict) else ModelCost(),
            context_window=data.get("context_window", 128000),
            max_tokens=data.get("max_tokens", 4096),
            reasoning=data.get("reasoning", False),
        )


@dataclass
class ProviderConfig:
    """Configuration for an inference provider."""

    name: str
    api_key: str
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, provider_name: str, data: dict[str, Any]) -> ProviderConfig:
        return cls(
            name=provider_name,
            api_key=data["api_key"],
            base_url=data.get("base_url", f"https://api.{provider_name}.com/v1"),
            headers=data.get("headers", {}),
        )


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------


class ModelPool:
    """Manages a pool of models from multiple providers.

    Parameters
    ----------
    providers:
        Mapping of provider name to ProviderConfig.
    models:
        Mapping of model id to ModelConfig.
    """

    def __init__(
        self,
        providers: dict[str, ProviderConfig] | None = None,
        models: dict[str, ModelConfig] | None = None,
    ) -> None:
        self._providers: dict[str, ProviderConfig] = providers or {}
        self._models: dict[str, ModelConfig] = models or {}

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_provider(self, model_id: str) -> ProviderConfig:
        """Resolve the provider for a given model id."""
        model = self.get_model(model_id)
        provider_name = model.provider
        if provider_name not in self._providers:
            raise KeyError(
                f"Provider '{provider_name}' for model '{model_id}' is not configured"
            )
        return self._providers[provider_name]

    def get_model(self, model_id: str) -> ModelConfig:
        """Get model configuration by id."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' is not in the pool")
        return self._models[model_id]

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def select_by_capability(self, capabilities: list[str]) -> ModelConfig:
        """Select the first model that supports all requested capabilities."""
        for model in self._models.values():
            if all(cap in model.capabilities for cap in capabilities):
                return model
        raise KeyError(
            f"No model supports all capabilities: {capabilities}"
        )

    def select_cheapest(self, capabilities: list[str] | None = None) -> ModelConfig:
        """Select the cheapest model, optionally filtered by capabilities."""
        candidates = self._models.values()
        if capabilities:
            candidates = [
                m for m in candidates
                if all(cap in m.capabilities for cap in capabilities)
            ]
        if not candidates:
            raise KeyError("No models available for selection")
        return min(candidates, key=lambda m: m.cost.output_per_m + m.cost.input_per_m)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> ModelPool:
        """Load model pool from a YAML file.

        Expected format::

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
              gpt-4o-mini:
                provider: openai
                capabilities: [text]
                cost: {input_per_m: 0.15, output_per_m: 0.60}
        """
        text = Path(path).read_text()
        data = yaml.safe_load(text) or {}

        # Resolve environment variable references in provider configs
        providers: dict[str, ProviderConfig] = {}
        for name, pdata in data.get("providers", {}).items():
            resolved: dict[str, Any] = {}
            for k, v in pdata.items():
                if isinstance(v, str):
                    resolved[k] = _resolve_env_var(v)
                else:
                    resolved[k] = v
            providers[name] = ProviderConfig.from_dict(name, resolved)

        models: dict[str, ModelConfig] = {}
        for model_id, mdata in data.get("models", {}).items():
            models[model_id] = ModelConfig.from_dict(model_id, mdata)

        pool = cls(providers=providers, models=models)
        logger.info("Loaded %d model(s) from %d provider(s) via %s", len(models), len(providers), path)
        return pool

    @classmethod
    def from_env(cls) -> ModelPool:
        """Create a minimal pool from environment variables (backward compatible).

        Reads NEXAGENT_MODEL, NEXAGENT_API_BASE, NEXAGENT_API_KEY and creates
        a single-model pool. This ensures existing code works without a models.yaml file.
        """
        model_id = os.environ.get("NEXAGENT_MODEL", "gpt-4o")
        api_base = os.environ.get("NEXAGENT_API_BASE", "https://api.openai.com/v1")
        api_key = os.environ.get("NEXAGENT_API_KEY", "")

        # Derive provider name from base_url
        provider_name = "default"
        if "openai" in api_base:
            provider_name = "openai"
        elif "anthropic" in api_base:
            provider_name = "anthropic"
        elif "localhost" in api_base or "127.0.0.1" in api_base:
            provider_name = "local"

        providers = {
            provider_name: ProviderConfig(
                name=provider_name,
                api_key=api_key,
                base_url=api_base,
            ),
        }
        models = {
            model_id: ModelConfig(
                id=model_id,
                provider=provider_name,
                capabilities=["text"],
            ),
        }
        return cls(providers=providers, models=models)

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"ModelPool(models={list(self._models.keys())}, providers={list(self._providers.keys())})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_var(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def _replace(m: re.Match[str]) -> str:
        var_name = m.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            logger.warning("Environment variable '%s' is not set", var_name)
            return ""
        return env_val
    return _ENV_VAR_RE.sub(_replace, value)
