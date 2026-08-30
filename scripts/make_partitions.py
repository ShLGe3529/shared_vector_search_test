#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.common import load_config
from hnsw_experiment.partition import make_partition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("sift10m", "cohere10m"))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("provide --dataset or --all")
    cfg = load_config(args.config)
    names = list(cfg["datasets"]) if args.all else [args.dataset]
    for name in names:
        manifest = make_partition(cfg, name)
        print(f"partitioned {name}: {manifest['permutation_sha256']}", flush=True)


if __name__ == "__main__":
    main()

