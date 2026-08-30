#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.common import load_config
from hnsw_experiment.indexing import index_dir
from hnsw_experiment.searching import result_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--build-profile", default="baseline")
    parser.add_argument("--dataset", action="append", choices=("sift10m", "cohere10m"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    datasets = args.dataset or list(cfg["datasets"])
    errors = []
    for dataset in datasets:
        for topology in cfg["topologies"]:
            name = topology["id"]
            idx = index_dir(cfg, dataset, args.build_profile, name)
            out = result_dir(cfg, dataset, args.build_profile, name)
            for complete in (idx / "COMPLETE", out / "COMPLETE"):
                if not complete.exists():
                    errors.append(f"missing {complete}")
            raw = out / "per_query.csv"
            manifest_path = out / "run_manifest.json"
            if not raw.exists() or not manifest_path.exists():
                errors.append(f"missing raw result or manifest: {out}")
                continue
            manifest = json.loads(manifest_path.read_text())
            expected = (
                manifest["measured_query_count"] * manifest["repetitions"]
                * len(manifest["ef_search_values"]) * len(manifest["k_values"])
            )
            with raw.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != expected:
                errors.append(f"{raw}: {len(rows)} rows, expected {expected}")
            for row in rows:
                recall = float(row["recall_at_k"])
                if not 0 <= recall <= 1:
                    errors.append(f"bad recall in {raw}")
                    break
                if int(row["returned_count"]) != int(row["k"]):
                    errors.append(f"bad returned count in {raw}")
                    break
                if int(row["end_to_end_ns"]) < int(row["shard_search_ns"]):
                    errors.append(f"bad timing components in {raw}")
                    break
            print(f"validated {dataset} {name}: {len(rows)} rows")
    if errors:
        raise SystemExit("validation FAILED:\n" + "\n".join(errors))
    print("validation PASS")


if __name__ == "__main__":
    main()

