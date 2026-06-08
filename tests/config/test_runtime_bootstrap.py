from ghostbot.config.schema import PlanningConfig
from ghostbot.providers.base import LLMProvider
from ghostbot.runtime.bootstrap import build_agent_runtime


class _StubProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
        self.default_model = "test-model"

    async def chat(self, *args, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError

    async def chat_with_retry(self, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError

    async def chat_stream_with_retry(self, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError

    def get_default_model(self):
        return self.default_model


def test_planning_config_quality_defaults():
    config = PlanningConfig()

    assert config.force_exploration is True
    assert config.min_exploration_steps == 1
    assert config.max_rewrites == 1
    assert config.history_limit == 10
    assert "fix" in config.trigger_keywords
    assert "delete" in config.risky_keywords


def test_build_agent_runtime_uses_shared_bootstrap(tmp_path):
    from ghostbot.config.schema import Config

    config = Config()
    config.agents.defaults.workspace = str(tmp_path)
    config.agents.defaults.model = "test-model"

    runtime = build_agent_runtime(config, _StubProvider())

    assert runtime.config is config
    assert runtime.provider.get_default_model() == "test-model"
    assert runtime.agent_loop.workspace == tmp_path
    assert runtime.cron_service.store_path == tmp_path / "cron" / "jobs.json"
