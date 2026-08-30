#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.common import load_config
from hnsw_experiment.indexing import build_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", required=True, choices=("sift10m", "cohere10m"))
    parser.add_argument("--build-profile", default="baseline")
    parser.add_argument("--topology", choices=("1x10m", "2x5m", "5x2m"))
    parser.add_argument("--all-topologies", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.all_topologies and not args.topology:
        parser.error("provide --topology or --all-topologies")
    cfg = load_config(args.config)
    topologies = [item["id"] for item in cfg["topologies"]]
    if not args.all_topologies:
        topologies = [args.topology]
    for topology in topologies:
        print(f"building {args.dataset} {args.build_profile} {topology}", flush=True)
        manifest = build_topology(
            cfg, args.dataset, args.build_profile, topology, args.force
        )
        print(
            f"complete {topology}: {manifest['total_build_seconds']:.1f}s, "
            f"{manifest['total_index_bytes']} bytes", flush=True,
        )


if __name__ == "__main__":
    main()

