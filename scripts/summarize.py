#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.common import load_config
from hnsw_experiment.reporting import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--build-profile", default="baseline")
    parser.add_argument("--dataset", action="append", choices=("sift10m", "cohere10m"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    datasets = args.dataset or list(cfg["datasets"])
    path = summarize(cfg, datasets, args.build_profile)
    print(f"summary: {path}")


if __name__ == "__main__":
    main()

