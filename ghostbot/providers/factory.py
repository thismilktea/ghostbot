"""Provider construction helpers shared across entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ghostbot.providers.base import GenerationSettings, LLMProvider
from ghostbot.providers.resolver import ResolvedProvider, resolve_api_base, resolve_provider

if TYPE_CHECKING:
    from ghostbot.config.schema import Config


class ProviderConfigurationError(ValueError):
    """Raised when provider configuration is incomplete for the requested backend."""


@dataclass(frozen=True)
class ProviderBuildResult:
    """Built provider plus the resolution metadata used to construct it."""

    provider: LLMProvider
    resolved: ResolvedProvider


def _validate_provider(config: Config, resolved: ResolvedProvider) -> None:
    model = resolved.model
    provider_config = resolved.config
    spec = resolved.spec
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not provider_config or not provider_config.api_key or not provider_config.api_base:
            raise ProviderConfigurationError(
                "Azure OpenAI requires api_key and api_base in config."
            )
        return

    if backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (provider_config and provider_config.api_key)
        exempt = bool(spec and (spec.is_oauth or spec.is_local or spec.is_direct))
        if needs_key and not exempt:
            provider_name = resolved.name or "unknown"
            raise ProviderConfigurationError(
                f"No API key configured for provider '{provider_name}'."
            )


def _build_backend_provider(config: Config, resolved: ResolvedProvider) -> LLMProvider:
    model = resolved.model
    provider_config = resolved.config
    spec = resolved.spec
    backend = spec.backend if spec else "openai_compat"

    if backend == "openai_codex":
        from ghostbot.providers.openai_codex_provider import OpenAICodexProvider

        return OpenAICodexProvider(default_model=model)

    if backend == "github_copilot":
        from ghostbot.providers.github_copilot_provider import GitHubCopilotProvider

        return GitHubCopilotProvider(default_model=model)

    if backend == "azure_openai":
        from ghostbot.providers.azure_openai_provider import AzureOpenAIProvider

        assert provider_config is not None
        return AzureOpenAIProvider(
            api_key=provider_config.api_key,
            api_base=provider_config.api_base,
            default_model=model,
        )

    if backend == "anthropic":
        from ghostbot.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=provider_config.api_key if provider_config and not provider_config.auth_token else None,
            auth_token=provider_config.auth_token if provider_config else None,
            api_base=resolve_api_base(config, model),
            default_model=model,
            extra_headers=provider_config.extra_headers if provider_config else None,
            header_profile=provider_config.header_profile if provider_config else None,
        )

    from ghostbot.providers.openai_compat_provider import OpenAICompatProvider

    return OpenAICompatProvider(
        api_key=provider_config.api_key if provider_config else None,
        api_base=resolve_api_base(config, model),
        default_model=model,
        extra_headers=provider_config.extra_headers if provider_config else None,
        spec=spec,
    )


def apply_generation_defaults(config: Config, provider: LLMProvider) -> LLMProvider:
    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def build_provider(config: Config, model: str | None = None) -> ProviderBuildResult:
    resolved = resolve_provider(config, model)
    _validate_provider(config, resolved)
    provider = _build_backend_provider(config, resolved)
    apply_generation_defaults(config, provider)
    return ProviderBuildResult(provider=provider, resolved=resolved)


def build_provider_for_fast_model(config: Config) -> ProviderBuildResult | None:
    model = config.get_effective_fast_model()
    forced_provider = config.agents.defaults.fast_provider
    resolved = resolve_provider(config, model, forced_provider=forced_provider)
    if resolved.config is None and resolved.spec is None:
        return None
    _validate_provider(config, resolved)
    provider = _build_backend_provider(config, resolved)
    apply_generation_defaults(config, provider)
    return ProviderBuildResult(provider=provider, resolved=resolved)


def build_provider_or_exit(config: Config, *, exit_factory: Any, print_error: Any) -> LLMProvider:
    """CLI adapter: build a provider and translate config errors into exits."""
    try:
        return build_provider(config).provider
    except ProviderConfigurationError as exc:
        message = str(exc)
        if "Azure OpenAI" in message:
            print_error("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            print_error("Set them in ~/.ghostbot/config.json under providers.azure_openai section")
            print_error("Use the model field to specify the deployment name.")
        elif "No API key configured" in message:
            print_error("[red]Error: No API key configured.[/red]")
            print_error("Set one in ~/.ghostbot/config.json under providers section")
        else:
            print_error(f"[red]Error: {message}[/red]")
        raise exit_factory(1)
