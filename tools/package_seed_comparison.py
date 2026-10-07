"""Archive cohort JSON and record the retained remote checkpoint locations."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from faultline.evaluation.study import load_kill_test_protocol


def package(
    repo: Path, protocol_path: Path = Path("configs/evaluation/seed-comparison.toml"),
) -> None:
    protocol = load_kill_test_protocol(repo / protocol_path)
    directory = repo / "artifacts/results"
    raw_archive = directory / f"{protocol.protocol_id}-runs.tar.gz"
    if raw_archive.exists():
        raise FileExistsError("refusing to replace an existing study archive")
    cloud = json.loads((directory / f"{protocol.protocol_id}-cohort-cloud.json").read_text())
    runs = {run["run_id"]: run for run in cloud["runs"]}
    inventory = []
    with tarfile.open(raw_archive, "w:gz") as raw:
        for seed in protocol.training_seeds:
            for arm in protocol.arms:
                run_id = protocol.run_id_template.format(arm=arm, seed=seed)
                result_path = directory / f"{run_id}.json"
                result = json.loads(result_path.read_text())
                for kind in ("results", "manifests"):
                    relative = f"artifacts/{kind}/{run_id}.json"
                    raw.add(repo / relative, arcname=relative)
                checkpoint = runs[run_id]["checkpoint"]
                if checkpoint["sha256"] != result["checkpoint"]["sha256"]:
                    raise ValueError(f"checkpoint digest mismatch: {run_id}")
                inventory.append({"run_id": run_id, **checkpoint})
    pilots = cloud.get("pilot_checkpoints_uploaded", [])
    pilot_path = directory / f"{protocol.protocol_id}-pilot-cloud.json"
    if not pilots and pilot_path.exists():
        pilots = [
            {"run_id": run["run_id"], **run["checkpoint"]}
            for run in json.loads(pilot_path.read_text())["runs"]
        ]
    summary = {"checkpoints": inventory,
               "checkpoint_count": len(inventory),
               "total_checkpoint_bytes": sum(item["bytes"] for item in inventory),
               "pilot_checkpoints": pilots}
    with (directory / f"{protocol.protocol_id}-checkpoints.json").open("x") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Archived {len(inventory)} runs; raw JSON: {raw_archive.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/seed-comparison.toml"),
    )
    args = parser.parse_args()
    package(args.repo, args.protocol)


if __name__ == "__main__":
    main()
