"""Run the fixed comparison on ephemeral Modal CPU containers.

Install the Modal client separately. Resource environment variables are
FAULTLINE_WINDOW_MINUTES, FAULTLINE_CPU_CORES, FAULTLINE_MEMORY_MIB and
FAULTLINE_CONTAINERS (defaults: 5 minutes, four cores, 2048 MiB, four containers). Example:
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
CPU_CORES = int(os.environ.get("FAULTLINE_CPU_CORES", "4"))
MEMORY_MIB = int(os.environ.get("FAULTLINE_MEMORY_MIB", "2048"))
MAX_CONTAINERS = int(os.environ.get("FAULTLINE_CONTAINERS", "4"))


def write_progress(destination: Path, metadata: dict) -> None:
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


@app.function(image=image, cpu=CPU_CORES, memory=MEMORY_MIB, timeout=WINDOW_MINUTES * 60,
              max_containers=MAX_CONTAINERS, retries=0)
def startup_probe(index: int) -> dict:
    import socket
    import time

    timestamp = time.time()
    key = os.environ.get("MODAL_TASK_ID", socket.gethostname())
    time.sleep(60)
    return {"index": index, "container_key": key, "started_unix_seconds": timestamp}


@app.function(image=image, cpu=CPU_CORES, memory=MEMORY_MIB, timeout=WINDOW_MINUTES * 60,
              max_containers=MAX_CONTAINERS, retries=0, volumes={"/checkpoints": volume})
def train(job: dict) -> dict:
    import hashlib
    import shutil
    import tempfile

    started = perf_counter()
    arm, seed, commit = job["arm"], job["seed"], job["commit"]
    run_id = job["run_id"]
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
                       "OMP_NUM_THREADS": str(job["threads"]),
                       "MKL_NUM_THREADS": str(job["threads"])}
        subprocess.run(
            ["python", "-c", "from faultline.cli import main; raise SystemExit(main())",
             "train", "--config", job["training_config"],
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
async def main(pilot: bool = False, protocol: str = "configs/evaluation/seed-comparison.toml",
               study: str = "seed-comparison", pilot_seed: int = 199,
               training_config: str = "configs/training/seed-comparison.toml",
               startup_check: bool = False):
    import tomllib

    started = perf_counter()
    repo = Path(__file__).resolve().parents[1]
    directory = repo / "artifacts/results"
    directory.mkdir(parents=True, exist_ok=True)
    if startup_check:
        import time

        started_wall = time.time()
        async with asyncio.timeout(WINDOW_MINUTES * 60):
            rows = [row async for row in startup_probe.map.aio(range(MAX_CONTAINERS))]
        count = len({row["container_key"] for row in rows})
        result = {
            "requested_containers": MAX_CONTAINERS, "distinct_containers": count,
            "hardware": {"cpu_cores": CPU_CORES, "memory_gib": MEMORY_MIB / 1024, "gpu": "none"},
            "window_minutes": WINDOW_MINUTES,
            "startup_span_seconds": max(row["started_unix_seconds"] for row in rows)
            - min(row["started_unix_seconds"] for row in rows),
            "last_start_after_client_seconds": max(row["started_unix_seconds"] for row in rows)
            - started_wall,
            "client_wall_seconds": perf_counter() - started,
            "rows": sorted(rows, key=lambda row: row["index"]),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            ).strip(),
        }
        with (directory / f"{study}-startup-cloud.json").open("x") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"Started {count} distinct containers")
        if count != MAX_CONTAINERS:
            raise RuntimeError("the requested container concurrency was not reached")
        return
    if pilot:
        seeds, arms, label = [pilot_seed], ["random", "difficulty", "epistemic"], "pilot"
        template = f"{study}-{{arm}}-seed-{{seed}}"
    else:
        with (repo / protocol).open("rb") as source:
            specification = tomllib.load(source)
        seeds, arms, label = specification["training_seeds"], specification["arms"], "cohort"
        study = specification["protocol_id"]
        training_config = specification["training_config"]
        template = specification["run_id_template"]
    with (repo / training_config).open("rb") as source:
        settings = tomllib.load(source)
    progress = directory / f"{study}-{label}-progress.json"
    final = directory / f"{study}-{label}-cloud.json"
    if final.exists():
        raise FileExistsError(f"completed run metadata already exists: {final.name}")
    expected = [template.format(arm=arm, seed=seed) for seed in seeds for arm in arms]
    if progress.exists():
        metadata = json.loads(progress.read_text())
        if metadata["expected_run_ids"] != expected:
            raise ValueError("resume cohort differs from the saved run set")
        hardware = {"gpu": "none", "cpu_cores": CPU_CORES,
                    "memory_gib": MEMORY_MIB / 1024, "max_containers": MAX_CONTAINERS}
        if metadata["hardware"] != hardware:
            raise ValueError("resume resource request differs from the saved hardware")
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        metadata = {"source_commit": commit, "hardware": {"gpu": "none", "cpu_cores": CPU_CORES,
                    "memory_gib": MEMORY_MIB / 1024, "max_containers": MAX_CONTAINERS},
                    "window_minutes": WINDOW_MINUTES, "training_config": training_config,
                    "torch_threads": settings["ppo"]["torch_threads"],
                    "expected_run_ids": expected, "client_wall_seconds": 0.0,
                    "runs": [], "attempt_errors": []}
    write_progress(progress, metadata)
    allowance = metadata["window_minutes"] * 60 - metadata["client_wall_seconds"]
    if allowance <= 0:
        raise TimeoutError("the original cohort time allowance is exhausted")
    pilot_cloud = directory / f"{study}-pilot-cloud.json"
    if (not pilot and pilot_cloud.exists() and "pilot_checkpoints_uploaded" not in metadata
            and "checkpoint" not in json.loads(pilot_cloud.read_text())["runs"][0]):
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
    jobs = [{"arm": arm, "seed": seed, "commit": metadata["source_commit"],
             "run_id": template.format(arm=arm, seed=seed),
             "training_config": metadata["training_config"], "threads": metadata["torch_threads"]}
            for seed in seeds for arm in arms
            if template.format(arm=arm, seed=seed) not in completed_ids]
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
