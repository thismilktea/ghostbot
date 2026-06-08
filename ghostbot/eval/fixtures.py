"""Fixture preparation for GhostBot evaluation runs."""

from __future__ import annotations

import shutil
from pathlib import Path

from ghostbot.eval.schema import EvalFixture
from ghostbot.utils.helpers import ensure_dir


class FixturePreparer:
    def __init__(self, root: Path):
        self.root = root

    def prepare(self, name: str, fixture: EvalFixture, run_root: Path) -> Path:
        target = ensure_dir(run_root / "fixtures") / name
        if target.exists():
            shutil.rmtree(target)
        if fixture.type == "empty":
            target.mkdir(parents=True, exist_ok=True)
            return target
        if not fixture.source:
            raise ValueError(f"Fixture '{name}' requires a source path")
        source = (self.root / fixture.source).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Fixture source not found: {source}")
        shutil.copytree(source, target)
        return target
