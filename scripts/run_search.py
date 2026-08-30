#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.common import load_config
from hnsw_experiment.searching import run_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", required=True, choices=("sift10m", "cohere10m"))
    parser.add_argument("--build-profile", default="baseline")
    parser.add_argument("--topology", required=True, choices=("1x10m", "2x5m", "5x2m"))
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = run_topology(
        cfg, args.dataset, args.build_profile, args.topology, args.force,
        args.query_limit, args.repetitions,
    )
    print(f"complete: {path}", flush=True)


if __name__ == "__main__":
    main()

