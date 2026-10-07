from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeImage:
    @staticmethod
    def debian_slim(**kwargs):
        return FakeImage()

    def apt_install(self, *args):
        return self

    def pip_install(self, *args, **kwargs):
        return self


class FakeApp:
    def __init__(self, name):
        pass

    def function(self, **kwargs):
        return lambda function: function

    def local_entrypoint(self):
        return lambda function: function


def load_runner(monkeypatch, tmp_path: Path):
    fake = SimpleNamespace(App=FakeApp, Image=FakeImage,
                           Volume=SimpleNamespace(from_name=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "modal", fake)
    spec = importlib.util.spec_from_file_location(
        "modal_seeds_test", Path(__file__).parents[2] / "tools/modal_seeds.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / "tools/modal_seeds.py")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    return module


def completed(job: dict) -> dict:
    run_id = f"seed-comparison-{job['arm']}-seed-{job['seed']}"
    return {"run_id": run_id, "job_seconds": 1.0,
            "checkpoint": {"volume": "test", "path": run_id, "sha256": "a" * 64, "bytes": 1},
            "files": {f"artifacts/results/{run_id}.json": b"{}\n",
                      f"artifacts/manifests/{run_id}.json": b"{}\n"}}


def test_map_failure_preserves_completed_runs_and_resumes_only_missing(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch, tmp_path)

    async def partial(jobs, **kwargs):
        assert kwargs["return_exceptions"]
        yield completed(jobs[0])
        yield RuntimeError("transport interrupted")
        yield RuntimeError("transport interrupted")

    runner.train = SimpleNamespace(map=SimpleNamespace(aio=partial))
    with pytest.raises(RuntimeError, match="incomplete cohort"):
        asyncio.run(runner.main(pilot=True))
    progress = tmp_path / "artifacts/results/seed-comparison-pilot-progress.json"
    saved = json.loads(progress.read_text())
    assert len(saved["runs"]) == 1
    assert len(saved["attempt_errors"]) == 2
    assert saved["source_commit"] == "a" * 40

    async def remainder(jobs, **kwargs):
        assert [job["arm"] for job in jobs] == ["difficulty", "epistemic"]
        assert all(job["commit"] == "a" * 40 for job in jobs)
        for job in jobs:
            yield completed(job)

    runner.train = SimpleNamespace(map=SimpleNamespace(aio=remainder))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "b" * 40)
    asyncio.run(runner.main(pilot=True))
    final = tmp_path / "artifacts/results/seed-comparison-pilot-cloud.json"
    assert len(json.loads(final.read_text())["runs"]) == 3
    assert not progress.exists()


def test_client_deadline_keeps_incomplete_progress(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch, tmp_path)
    runner.WINDOW_MINUTES = 0

    async def unfinished(jobs, **kwargs):
        await asyncio.sleep(1)
        yield completed(jobs[0])

    runner.train = SimpleNamespace(map=SimpleNamespace(aio=unfinished))
    with pytest.raises(TimeoutError):
        asyncio.run(runner.main(pilot=True))
    progress = tmp_path / "artifacts/results/seed-comparison-pilot-progress.json"
    assert json.loads(progress.read_text())["runs"] == []
    assert not (tmp_path / "artifacts/results/seed-comparison-pilot-cloud.json").exists()
    saved = json.loads(progress.read_text())
    saved["window_minutes"] = 5
    saved["client_wall_seconds"] = 300
    progress.write_text(json.dumps(saved))
    runner.WINDOW_MINUTES = 999
    with pytest.raises(TimeoutError, match="original cohort time allowance"):
        asyncio.run(runner.main(pilot=True))
