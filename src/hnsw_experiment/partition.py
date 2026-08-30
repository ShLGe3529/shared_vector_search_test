from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import root_path, sha256_array, utc_now, write_json


def partition_dir(cfg: dict[str, Any], dataset: str) -> Path:
    return root_path(cfg, f"data/{dataset}/partitions")


def make_partition(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    n = cfg["datasets"][dataset]["count"]
    seed = cfg["seeds"]["partition_seed"]
    out_dir = partition_dir(cfg, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    permutation_path = out_dir / "permutation.ids.i64.npy"
    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = rng.permutation(n).astype(np.int64, copy=False)
    np.save(permutation_path, permutation)
    topologies = {}
    for topology in cfg["topologies"]:
        shard_count = topology["shard_count"]
        slices = np.array_split(permutation, shard_count)
        topologies[topology["id"]] = {
            "shard_count": shard_count,
            "shards": [
                {
                    "shard_id": shard_id,
                    "start": shard_id * topology["expected_shard_size"],
                    "stop": (shard_id + 1) * topology["expected_shard_size"],
                    "count": len(values),
                    "ordered_labels_sha256": sha256_array(values),
                }
                for shard_id, values in enumerate(slices)
            ],
        }
    manifest = {
        "dataset": dataset,
        "created_at": utc_now(),
        "generator": "numpy.random.Generator(PCG64).permutation",
        "seed": seed,
        "count": n,
        "permutation_file": str(permutation_path.relative_to(Path(cfg["_root"]))),
        "permutation_sha256": sha256_array(permutation),
        "topologies": topologies,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_partition(cfg: dict[str, Any], dataset: str):
    out_dir = partition_dir(cfg, dataset)
    permutation = np.load(out_dir / "permutation.ids.i64.npy", mmap_mode="r")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return permutation, manifest

