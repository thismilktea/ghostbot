from __future__ import annotations

from pathlib import Path

from ghostbot.eval.fixtures import FixturePreparer
from ghostbot.eval.harness import load_suite
from ghostbot.eval.schema import EvalFixture


def test_core_suite_loads_expected_scenarios():
    suite = load_suite(Path("tests/dogfood_eval/suites/core.json"))

    assert suite.name == "ghostbot-eval-core"
    assert len(suite.scenarios) == 4
    assert {scenario.id for scenario in suite.scenarios} == {
        "plan.shop_improvement.001",
        "bugfix.cart_negative_quantity.001",
        "tests.cart_regression.001",
        "scaffold.empty_checkout.001",
    }


def test_core_suite_uses_shop_project_fixture():
    suite = load_suite(Path("tests/dogfood_eval/suites/core.json"))
    fixture = suite.fixtures["shop_project"]

    assert fixture.type == "copy"
    assert fixture.source == "fixtures/shop_project"


def test_fixture_preparer_copies_shop_project(tmp_path):
    preparer = FixturePreparer(Path("tests/dogfood_eval"))
    fixture = EvalFixture(type="copy", source="fixtures/shop_project", reset="fresh_copy")

    workspace = preparer.prepare("shop_project", fixture, tmp_path)

    assert (workspace / "shop" / "cart.py").exists()
    assert (workspace / "tests" / "test_cart.py").exists()
    assert (workspace / "README.md").exists()
