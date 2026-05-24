from ghostbot.config.schema import PlanningConfig


def test_planning_config_quality_defaults():
    config = PlanningConfig()

    assert config.force_exploration is True
    assert config.min_exploration_steps == 1
    assert config.max_rewrites == 1
    assert config.history_limit == 10
    assert "fix" in config.trigger_keywords
    assert "delete" in config.risky_keywords
