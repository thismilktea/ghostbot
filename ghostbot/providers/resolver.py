"""Provider resolution helpers shared across CLI and programmatic entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ghostbot.providers.registry import PROVIDERS, ProviderSpec, find_by_name

if TYPE_CHECKING:
    from ghostbot.config.schema import Config, ProviderConfig


@dataclass(frozen=True)
class ResolvedProvider:
    """Resolved provider metadata for a target model."""

    model: str
    config: ProviderConfig | None
    name: str | None
    spec: ProviderSpec | None


def _forced_provider_name(config: Config, forced_provider: str | None = None) -> str:
    if forced_provider is not None:
        return forced_provider
    return config.agents.defaults.provider


def _kw_matches(model_lower: str, model_normalized: str, kw: str) -> bool:
    kw = kw.lower()
    return kw in model_lower or kw.replace("-", "_") in model_normalized


def resolve_provider(
    config: Config,
    model: str | None = None,
    *,
    forced_provider: str | None = None,
) -> ResolvedProvider:
    """Resolve provider config and spec for a model.

    Resolution preserves the existing precedence rules used by ``Config``:
    forced provider, explicit prefix, keyword match, local fallback, then
    generic fallback.
    """
    resolved_model = model or config.agents.defaults.model
    forced = _forced_provider_name(config, forced_provider)

    if forced != "auto":
        spec = find_by_name(forced)
        if spec:
            provider_config = getattr(config.providers, spec.name, None)
            return ResolvedProvider(
                model=resolved_model,
                config=provider_config,
                name=spec.name if provider_config else None,
                spec=spec if provider_config else None,
            )
        return ResolvedProvider(model=resolved_model, config=None, name=None, spec=None)

    model_lower = resolved_model.lower()
    model_normalized = model_lower.replace("-", "_")
    model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
    normalized_prefix = model_prefix.replace("-", "_")

    for spec in PROVIDERS:
        provider_config = getattr(config.providers, spec.name, None)
        if provider_config and model_prefix and normalized_prefix == spec.name:
            if spec.is_oauth or spec.is_local or provider_config.api_key:
                return ResolvedProvider(
                    model=resolved_model,
                    config=provider_config,
                    name=spec.name,
                    spec=spec,
                )

    for spec in PROVIDERS:
        provider_config = getattr(config.providers, spec.name, None)
        if provider_config and any(_kw_matches(model_lower, model_normalized, kw) for kw in spec.keywords):
            if spec.is_oauth or spec.is_local or provider_config.api_key:
                return ResolvedProvider(
                    model=resolved_model,
                    config=provider_config,
                    name=spec.name,
                    spec=spec,
                )

    local_fallback: tuple[ProviderConfig, ProviderSpec] | None = None
    for spec in PROVIDERS:
        if not spec.is_local:
            continue
        provider_config = getattr(config.providers, spec.name, None)
        if not (provider_config and provider_config.api_base):
            continue
        if spec.detect_by_base_keyword and spec.detect_by_base_keyword in provider_config.api_base:
            return ResolvedProvider(
                model=resolved_model,
                config=provider_config,
                name=spec.name,
                spec=spec,
            )
        if local_fallback is None:
            local_fallback = (provider_config, spec)
    if local_fallback is not None:
        provider_config, spec = local_fallback
        return ResolvedProvider(
            model=resolved_model,
            config=provider_config,
            name=spec.name,
            spec=spec,
        )

    for spec in PROVIDERS:
        if spec.is_oauth:
            continue
        provider_config = getattr(config.providers, spec.name, None)
        if provider_config and provider_config.api_key:
            return ResolvedProvider(
                model=resolved_model,
                config=provider_config,
                name=spec.name,
                spec=spec,
            )

    return ResolvedProvider(model=resolved_model, config=None, name=None, spec=None)


def resolve_api_base(
    config: Config,
    model: str | None = None,
    *,
    forced_provider: str | None = None,
) -> str | None:
    """Resolve API base for a model, applying gateway/local defaults."""
    resolved = resolve_provider(config, model, forced_provider=forced_provider)
    if resolved.config and resolved.config.api_base:
        return resolved.config.api_base
    if resolved.spec and (resolved.spec.is_gateway or resolved.spec.is_local) and resolved.spec.default_api_base:
        return resolved.spec.default_api_base
    return None
