#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.common import load_config
from hnsw_experiment.data import finalize_cropped_base, validate_and_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("sift10m", "cohere10m"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("provide --dataset or --all")
    cfg = load_config(args.config)
    names = list(cfg["datasets"]) if args.all else [args.dataset]
    for name in names:
        path = finalize_cropped_base(cfg, name)
        print(f"finalized {name}: {path}", flush=True)
        manifest = validate_and_manifest(cfg, name, not args.skip_hashes)
        print(
            f"validated {name}: {manifest['count']} x {manifest['dimension']}, "
            f"{manifest['hnsw_space']}", flush=True,
        )


if __name__ == "__main__":
    main()

