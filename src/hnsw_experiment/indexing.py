from __future__ import annotations

import gc
import resource
import time
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np

from .common import environment, root_path, sha256_file, utc_now, write_json
from .data import dataset_arrays
from .partition import load_partition


def topology_config(cfg: dict[str, Any], topology_id: str) -> dict[str, Any]:
    return next(item for item in cfg["topologies"] if item["id"] == topology_id)


def index_dir(cfg: dict[str, Any], dataset: str, build_profile: str,
              topology_id: str) -> Path:
    return root_path(
        cfg,
        f"{cfg['paths']['indexes']}/{cfg['experiment_id']}/{dataset}/"
        f"{build_profile}/{topology_id}",
    )


def build_topology(cfg: dict[str, Any], dataset: str, build_profile: str,
                   topology_id: str, force: bool = False) -> dict[str, Any]:
    ds = cfg["datasets"][dataset]
    build = cfg["build_profiles"][build_profile]
    topology = topology_config(cfg, topology_id)
    out_dir = index_dir(cfg, dataset, build_profile, topology_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / "COMPLETE"
    if complete.exists() and not force:
        raise FileExistsError(f"already complete: {out_dir}")
    base, _, _, _ = dataset_arrays(cfg, dataset)
    permutation, partition_manifest = load_partition(cfg, dataset)
    manifests = []
    shard_size = topology["expected_shard_size"]
    for shard_id in range(topology["shard_count"]):
        start = shard_id * shard_size
        stop = start + shard_size
        labels = permutation[start:stop]
        index_path = out_dir / f"shard-{shard_id:03d}.bin"
        manifest_path = out_dir / f"shard-{shard_id:03d}.build.json"
        started = time.perf_counter()
        started_at = utc_now()
        index = hnswlib.Index(space=ds["space"], dim=ds["dimension"])
        index.init_index(
            max_elements=shard_size,
            M=build["M"],
            ef_construction=build["ef_construction"],
            random_seed=build["random_seed"],
            allow_replace_deleted=False,
        )
        index.set_num_threads(build["add_items_threads"])
        batch_size = build["add_items_batch_size"]
        for batch_start in range(0, shard_size, batch_size):
            batch_labels = np.asarray(
                labels[batch_start : batch_start + batch_size], dtype=np.int64
            )
            vectors = np.asarray(base[batch_labels], dtype=np.float32, order="C")
            index.add_items(
                vectors, batch_labels, num_threads=build["add_items_threads"]
            )
            del vectors, batch_labels
            completed = min(batch_start + batch_size, shard_size)
            if completed % 1_000_000 == 0 or completed == shard_size:
                print(
                    f"  shard {shard_id:03d}: {completed:,}/{shard_size:,} inserted",
                    flush=True,
                )
        if index.get_current_count() != shard_size:
            raise RuntimeError("HNSWLib index count mismatch")
        index.save_index(str(index_path))
        elapsed = time.perf_counter() - started
        manifest = {
            "dataset": dataset,
            "build_profile": build_profile,
            "topology": topology_id,
            "shard_id": shard_id,
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed,
            "count": shard_size,
            "dimension": ds["dimension"],
            "space": ds["space"],
            "M": build["M"],
            "ef_construction": build["ef_construction"],
            "random_seed": build["random_seed"],
            "add_items_threads": build["add_items_threads"],
            "add_items_batch_size": batch_size,
            "ordered_labels_sha256": partition_manifest["topologies"]
                [topology_id]["shards"][shard_id]["ordered_labels_sha256"],
            "index_path": str(index_path.relative_to(Path(cfg["_root"]))),
            "index_bytes": index_path.stat().st_size,
            "index_sha256": sha256_file(index_path),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "environment": environment(),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)
        del index
        gc.collect()
    topo_manifest = {
        "dataset": dataset,
        "build_profile": build_profile,
        "topology": topology_id,
        "shard_count": topology["shard_count"],
        "total_count": sum(item["count"] for item in manifests),
        "total_index_bytes": sum(item["index_bytes"] for item in manifests),
        "total_build_seconds": sum(item["elapsed_seconds"] for item in manifests),
        "shards": manifests,
    }
    write_json(out_dir / "topology_manifest.json", topo_manifest)
    complete.write_text("complete\n")
    del base, permutation
    return topo_manifest
