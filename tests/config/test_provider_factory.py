from __future__ import annotations

import pytest
import typer

from ghostbot.config.schema import Config
from ghostbot.interface import _make_provider as interface_make_provider
from ghostbot.providers.factory import (
    ProviderConfigurationError,
    build_provider,
    build_provider_for_fast_model,
    build_provider_or_exit,
)
from ghostbot.providers.resolver import resolve_api_base, resolve_provider


def test_resolve_provider_matches_keyword_provider():
    config = Config()
    config.agents.defaults.model = "claude-sonnet-4-20250514"
    config.providers.anthropic.api_key = "sk-ant-test"

    resolved = resolve_provider(config)

    assert resolved.name == "anthropic"
    assert resolved.config is config.providers.anthropic


def test_resolve_provider_prefers_explicit_prefix():
    config = Config()
    config.agents.defaults.model = "github_copilot/gpt-4.1"
    config.providers.github_copilot.auth_token = "token"
    config.providers.openai_codex.auth_token = "codex"

    resolved = resolve_provider(config)

    assert resolved.name == "github_copilot"


def test_resolve_provider_falls_back_to_gateway():
    config = Config()
    config.agents.defaults.model = "some-unknown-model"
    config.providers.openrouter.api_key = "sk-or-test"

    resolved = resolve_provider(config)

    assert resolved.name == "openrouter"
    assert resolve_api_base(config) == "https://openrouter.ai/api/v1"


def test_resolve_provider_prefers_local_base_keyword_fallback():
    config = Config()
    config.agents.defaults.model = "llama3.2"
    config.providers.vllm.api_base = "http://localhost:9999/v1"
    config.providers.ollama.api_base = "http://localhost:11434/v1"

    resolved = resolve_provider(config)

    assert resolved.name == "ollama"
    assert resolve_api_base(config) == "http://localhost:11434/v1"


def test_build_provider_constructs_anthropic_provider():
    config = Config()
    config.agents.defaults.model = "claude-sonnet-4-20250514"
    config.providers.anthropic.api_key = "sk-ant-test"
    config.agents.defaults.temperature = 0.2
    config.agents.defaults.max_tokens = 1234
    config.agents.defaults.reasoning_effort = "high"

    built = build_provider(config)

    assert built.provider.__class__.__name__ == "AnthropicProvider"
    assert built.resolved.name == "anthropic"
    assert built.provider.generation.temperature == 0.2
    assert built.provider.generation.max_tokens == 1234
    assert built.provider.generation.reasoning_effort == "high"


def test_build_provider_constructs_github_copilot_without_api_key():
    config = Config()
    config.agents.defaults.model = "github_copilot/gpt-4.1"
    config.providers.github_copilot.auth_token = "token"

    built = build_provider(config)

    assert built.provider.__class__.__name__ == "GitHubCopilotProvider"
    assert built.resolved.name == "github_copilot"


def test_build_provider_requires_azure_key_and_base():
    config = Config()
    config.agents.defaults.provider = "azure_openai"
    config.agents.defaults.model = "deployment-name"

    with pytest.raises(ProviderConfigurationError, match="Azure OpenAI requires api_key and api_base"):
        build_provider(config)


def test_build_provider_constructs_azure_provider():
    config = Config()
    config.agents.defaults.provider = "azure_openai"
    config.agents.defaults.model = "deployment-name"
    config.providers.azure_openai.api_key = "key"
    config.providers.azure_openai.api_base = "https://example.openai.azure.com"

    built = build_provider(config)

    assert built.provider.__class__.__name__ == "AzureOpenAIProvider"
    assert built.resolved.name == "azure_openai"


def test_build_provider_requires_key_for_openai_compat_provider():
    config = Config()
    config.agents.defaults.provider = "openai"
    config.agents.defaults.model = "gpt-4.1"

    with pytest.raises(ProviderConfigurationError, match="No API key configured"):
        build_provider(config)


def test_build_provider_or_exit_raises_typer_exit_for_cli_errors():
    config = Config()
    config.agents.defaults.provider = "openai"
    config.agents.defaults.model = "gpt-4.1"
    printed: list[str] = []

    with pytest.raises(typer.Exit):
        build_provider_or_exit(
            config,
            exit_factory=typer.Exit,
            print_error=printed.append,
        )

    assert any("No API key configured" in line for line in printed)


def test_build_provider_for_fast_model_uses_forced_fast_provider():
    config = Config()
    config.agents.defaults.model = "claude-sonnet-4-20250514"
    config.providers.anthropic.api_key = "sk-ant-test"
    config.agents.defaults.fast_model = "gpt-4.1"
    config.agents.defaults.fast_provider = "openrouter"
    config.providers.openrouter.api_key = "sk-or-test"

    built = build_provider_for_fast_model(config)

    assert built is not None
    assert built.provider.__class__.__name__ == "OpenAICompatProvider"
    assert built.resolved.name == "openrouter"
    assert built.provider.default_model == "gpt-4.1"


def test_interface_make_provider_matches_shared_factory():
    config = Config()
    config.agents.defaults.model = "claude-sonnet-4-20250514"
    config.providers.anthropic.api_key = "sk-ant-test"

    provider = interface_make_provider(config)
    built = build_provider(config).provider

    assert provider.__class__ is built.__class__
    assert provider.default_model == built.default_model
    assert provider.generation == built.generation
