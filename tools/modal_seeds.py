"""Run the fixed curriculum comparison on ephemeral Modal CPU containers.

Install the Modal client separately. Set FAULTLINE_WINDOW_MINUTES to the booked
app window (default 5); max_containers is always four. Example:
modal run tools/modal_seeds.py --pilot
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import modal

app = modal.App("faultline-seed-comparison")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy==2.4.6")
    .pip_install("torch==2.14.0+cpu", index_url="https://download.pytorch.org/whl/cpu")
)
WINDOW_MINUTES = int(os.environ.get("FAULTLINE_WINDOW_MINUTES", "5"))


@app.function(image=image, cpu=4, memory=2048, timeout=WINDOW_MINUTES * 60,
              max_containers=4, retries=0)
def train(job: dict) -> dict:
    import hashlib
    import tempfile

    started = perf_counter()
    arm, seed, commit = job["arm"], job["seed"], job["commit"]
    run_id = f"seed-comparison-{arm}-seed-{seed}"
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory) / "faultline"
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", "night2-seeds",
             "https://github.com/mottopanikeiku/faultline.git", str(repo)], check=True,
        )
        subprocess.run(["git", "fetch", "--quiet", "origin", commit], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach", commit], cwd=repo, check=True)
        environment = {**os.environ, "PYTHONPATH": str(repo / "src"),
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        subprocess.run(
            ["python", "-m", "faultline", "train", "--config",
             "configs/training/seed-comparison.toml", "--curriculum", arm,
             "--seed", str(seed), "--run-id", run_id],
            cwd=repo, env=environment, check=True,
        )
        files = {}
        for relative in (f"artifacts/results/{run_id}.json",
                         f"artifacts/manifests/{run_id}.json",
                         f"artifacts/runs/{run_id}/policy.pt"):
            files[relative] = (repo / relative).read_bytes()
        result = json.loads(files[f"artifacts/results/{run_id}.json"])
        checkpoint = files[f"artifacts/runs/{run_id}/policy.pt"]
        if hashlib.sha256(checkpoint).hexdigest() != result["checkpoint"]["sha256"]:
            raise ValueError("checkpoint digest does not match training result")
    return {"run_id": run_id, "job_seconds": perf_counter() - started, "files": files}


@app.local_entrypoint()
def main(pilot: bool = False, protocol: str = "configs/evaluation/seed-comparison.toml"):
    import tomllib

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if pilot:
        seeds = [199]
        arms = ["random", "difficulty", "epistemic"]
        label = "pilot"
    else:
        with (repo / protocol).open("rb") as source:
            specification = tomllib.load(source)
        seeds, arms = specification["training_seeds"], specification["arms"]
        label = "cohort"
    jobs = [{"arm": arm, "seed": seed, "commit": commit} for seed in seeds for arm in arms]
    metadata = {"source_commit": commit, "hardware": {"gpu": "none", "cpu_cores": 4,
                "memory_gib": 2, "max_containers": 4}, "window_minutes": WINDOW_MINUTES,
                "runs": []}
    started = perf_counter()
    for completed in train.map(jobs, order_outputs=False):
        for relative, content in completed.pop("files").items():
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(content)
        metadata["runs"].append(completed)
        print(f"Saved {completed['run_id']}: {completed['job_seconds']:.1f} cloud seconds")
    metadata["client_wall_seconds"] = perf_counter() - started
    metadata["runs"].sort(key=lambda run: run["run_id"])
    destination = repo / "artifacts/results" / f"seed-comparison-{label}-cloud.json"
    with destination.open("x") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")
