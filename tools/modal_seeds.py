"""Run the fixed comparison on ephemeral Modal CPU containers.

Install the Modal client separately. Set FAULTLINE_WINDOW_MINUTES to the allowed
app window (default 5); max_containers is always four. Example:
modal run tools/modal_seeds.py --pilot
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import modal

app = modal.App("faultline-seed-comparison")
VOLUME_NAME = "faultline-seed-comparison"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy==2.4.6")
    .pip_install("torch==2.14.0+cpu", index_url="https://download.pytorch.org/whl/cpu")
)
WINDOW_MINUTES = int(os.environ.get("FAULTLINE_WINDOW_MINUTES", "5"))


def write_progress(destination: Path, metadata: dict) -> None:
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


@app.function(image=image, cpu=4, memory=2048, timeout=WINDOW_MINUTES * 60,
              max_containers=4, retries=0, volumes={"/checkpoints": volume})
def train(job: dict) -> dict:
    import hashlib
    import shutil
    import tempfile

    started = perf_counter()
    arm, seed, commit = job["arm"], job["seed"], job["commit"]
    run_id = f"seed-comparison-{arm}-seed-{seed}"
    relative = f"{commit}/{run_id}/policy.pt"
    stored = Path("/checkpoints") / commit / run_id
    volume.reload()
    cached = [stored / name for name in ("policy.pt", "result.json", "manifest.json", "job.json")]
    if all(path.is_file() for path in cached):
        result = json.loads((stored / "result.json").read_text())
        with (stored / "policy.pt").open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        if digest != result["checkpoint"]["sha256"]:
            raise ValueError("retained checkpoint digest mismatch")
        metadata = json.loads((stored / "job.json").read_text())
        metadata["files"] = {
            f"artifacts/{kind}/{run_id}.json": (stored / name).read_bytes()
            for kind, name in (("results", "result.json"), ("manifests", "manifest.json"))
        }
        return metadata
    if any(path.exists() for path in cached):
        raise ValueError(f"incomplete remote artifact set needs inspection: {run_id}")
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory) / "faultline"
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--no-checkout",
             "https://github.com/mottopanikeiku/faultline.git", str(repo)], check=True,
        )
        subprocess.run(["git", "fetch", "--quiet", "origin", commit], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach", commit], cwd=repo, check=True)
        environment = {**os.environ, "PYTHONPATH": str(repo / "src"),
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        subprocess.run(
            ["python", "-c", "from faultline.cli import main; raise SystemExit(main())",
             "train", "--config", "configs/training/seed-comparison.toml",
             "--curriculum", arm, "--seed", str(seed), "--run-id", run_id],
            cwd=repo, env=environment, check=True,
        )
        files = {}
        for kind in ("results", "manifests"):
            path = f"artifacts/{kind}/{run_id}.json"
            files[path] = (repo / path).read_bytes()
        result = json.loads(files[f"artifacts/results/{run_id}.json"])
        checkpoint = repo / result["checkpoint"]["path"]
        with checkpoint.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != result["checkpoint"]["sha256"]:
            raise ValueError("checkpoint digest does not match training result")
        stored.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("rb") as source, (stored / "policy.pt").open("xb") as output:
            shutil.copyfileobj(source, output)
        for kind, name in (("results", "result.json"), ("manifests", "manifest.json")):
            (stored / name).write_bytes(files[f"artifacts/{kind}/{run_id}.json"])
        metadata = {"run_id": run_id, "job_seconds": perf_counter() - started,
                    "checkpoint": {"volume": VOLUME_NAME, "path": relative,
                                   "sha256": digest, "bytes": checkpoint.stat().st_size}}
        (stored / "job.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        volume.commit()
    return {**metadata, "files": files}


@app.local_entrypoint()
async def main(pilot: bool = False, protocol: str = "configs/evaluation/seed-comparison.toml"):
    import tomllib

    started = perf_counter()
    repo = Path(__file__).resolve().parents[1]
    directory = repo / "artifacts/results"
    directory.mkdir(parents=True, exist_ok=True)
    if pilot:
        seeds, arms, label = [199], ["random", "difficulty", "epistemic"], "pilot"
    else:
        with (repo / protocol).open("rb") as source:
            specification = tomllib.load(source)
        seeds, arms, label = specification["training_seeds"], specification["arms"], "cohort"
    progress = directory / f"seed-comparison-{label}-progress.json"
    final = directory / f"seed-comparison-{label}-cloud.json"
    if final.exists():
        raise FileExistsError(f"completed run metadata already exists: {final.name}")
    expected = [f"seed-comparison-{arm}-seed-{seed}" for seed in seeds for arm in arms]
    if progress.exists():
        metadata = json.loads(progress.read_text())
        if metadata["expected_run_ids"] != expected:
            raise ValueError("resume cohort differs from the saved run set")
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        metadata = {"source_commit": commit, "hardware": {"gpu": "none", "cpu_cores": 4,
                    "memory_gib": 2, "max_containers": 4}, "window_minutes": WINDOW_MINUTES,
                    "expected_run_ids": expected, "client_wall_seconds": 0.0,
                    "runs": [], "attempt_errors": []}
    write_progress(progress, metadata)
    allowance = metadata["window_minutes"] * 60 - metadata["client_wall_seconds"]
    if allowance <= 0:
        raise TimeoutError("the original cohort time allowance is exhausted")
    pilot_cloud = directory / "seed-comparison-pilot-cloud.json"
    if not pilot and pilot_cloud.exists() and "pilot_checkpoints_uploaded" not in metadata:
        pilot_metadata = json.loads(pilot_cloud.read_text())
        uploaded = []
        with volume.batch_upload() as upload:
            for run in pilot_metadata["runs"]:
                result = json.loads((directory / f"{run['run_id']}.json").read_text())
                checkpoint = repo / result["checkpoint"]["path"]
                if not checkpoint.exists():
                    raise FileNotFoundError("initial pilot weights must be uploaded once")
                relative = f"{pilot_metadata['source_commit']}/{run['run_id']}/policy.pt"
                upload.put_file(checkpoint, f"/{relative}")
                uploaded.append({"run_id": run["run_id"], "volume": VOLUME_NAME,
                                 "path": relative, "sha256": result["checkpoint"]["sha256"],
                                 "bytes": checkpoint.stat().st_size})
        metadata["pilot_checkpoints_uploaded"] = uploaded
        write_progress(progress, metadata)
    completed_ids = {run["run_id"] for run in metadata["runs"]}
    jobs = [{"arm": arm, "seed": seed, "commit": metadata["source_commit"]}
            for seed in seeds for arm in arms
            if f"seed-comparison-{arm}-seed-{seed}" not in completed_ids]
    try:
        remaining = max(0.0, allowance - (perf_counter() - started))
        if remaining <= 0:
            raise TimeoutError("the original cohort time allowance is exhausted")
        async with asyncio.timeout(remaining):
            async for completed in train.map.aio(jobs, order_outputs=False, return_exceptions=True):
                if isinstance(completed, Exception):
                    metadata["attempt_errors"].append(repr(completed))
                    write_progress(progress, metadata)
                    continue
                for relative, content in completed.pop("files").items():
                    destination = repo / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        if destination.read_bytes() != content:
                            raise ValueError(f"existing local artifact differs: {relative}")
                    else:
                        with destination.open("xb") as output:
                            output.write(content)
                metadata["runs"].append(completed)
                write_progress(progress, metadata)
                print(f"Saved {completed['run_id']}: {completed['job_seconds']:.1f} cloud seconds")
    finally:
        metadata["client_wall_seconds"] += perf_counter() - started
        metadata["runs"].sort(key=lambda run: run["run_id"])
        write_progress(progress, metadata)
    missing = set(expected) - {run["run_id"] for run in metadata["runs"]}
    if missing:
        raise RuntimeError(f"incomplete cohort; rerun the same command for: {sorted(missing)}")
    progress.replace(final)
