from __future__ import annotations

import csv
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import yaml

from .common import environment, root_path, sha256_file, utc_now, write_json
from .data import dataset_arrays
from .indexing import index_dir, topology_config


THREAD_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def validate_threads(cfg: dict[str, Any]) -> None:
    required = cfg["search"]["required_environment"]
    wrong = {key: os.environ.get(key) for key in THREAD_KEYS
             if os.environ.get(key) != required[key]}
    if wrong:
        raise RuntimeError(f"single-thread environment not set: {wrong}")


def merge_candidates(labels: list[np.ndarray], distances: list[np.ndarray],
                     k: int) -> tuple[np.ndarray, np.ndarray]:
    all_labels = np.concatenate(labels).astype(np.int64, copy=False)
    all_distances = np.concatenate(distances).astype(np.float32, copy=False)
    order = np.lexsort((all_labels, all_distances))[:k]
    return all_labels[order], all_distances[order]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    return len(set(map(int, found)) & set(map(int, truth))) / len(truth)


def result_dir(cfg: dict[str, Any], dataset: str, build_profile: str,
               topology: str) -> Path:
    return root_path(
        cfg,
        f"{cfg['paths']['raw_results']}/{cfg['experiment_id']}/{dataset}/"
        f"{build_profile}/{topology}",
    )


def run_topology(cfg: dict[str, Any], dataset: str, build_profile: str,
                 topology_id: str, force: bool = False,
                 query_limit: int | None = None,
                 repetitions: int | None = None) -> Path:
    validate_threads(cfg)
    ds = cfg["datasets"][dataset]
    topology = topology_config(cfg, topology_id)
    idx_dir = index_dir(cfg, dataset, build_profile, topology_id)
    if not (idx_dir / "COMPLETE").exists():
        raise FileNotFoundError(f"incomplete index: {idx_dir}")
    out_dir = result_dir(cfg, dataset, build_profile, topology_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / "COMPLETE"
    if complete.exists() and not force:
        raise FileExistsError(f"already complete: {out_dir}")
    raw_path = out_dir / "per_query.csv"
    base, queries, gt_ids, _ = dataset_arrays(cfg, dataset)
    del base
    rng = np.random.Generator(np.random.PCG64(cfg["seeds"]["query_order_seed"]))
    query_order = rng.permutation(len(queries))
    configured_limit = min(cfg["search"]["measured_queries"], len(queries))
    measured_count = min(query_limit or configured_limit, len(queries))
    measured_ids = query_order[:measured_count]
    warmup_count = min(cfg["search"]["warmup_queries"], len(queries))
    warmup_ids = query_order[-warmup_count:]
    reps = repetitions or cfg["search"]["repetitions"]
    indexes = []
    for shard_id in range(topology["shard_count"]):
        index = hnswlib.Index(space=ds["space"], dim=ds["dimension"])
        index.load_index(str(idx_dir / f"shard-{shard_id:03d}.bin"))
        index.set_num_threads(1)
        indexes.append(index)
    fields = [
        "experiment_id", "dataset", "build_profile", "topology",
        "shard_count", "ef_search", "k", "repetition", "query_position",
        "query_id", "end_to_end_ns", "shard_search_ns", "merge_ns",
        "recall_at_k", "returned_count",
    ]
    temp_path = raw_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ef_search in cfg["search"]["ef_search_values"]:
            print(
                f"{dataset} {topology_id}: ef_search={ef_search}", flush=True
            )
            for index in indexes:
                index.set_ef(ef_search)
            for k in cfg["search"]["k_values"]:
                print(f"  K={k}: warming {warmup_count} queries", flush=True)
                for query_id in warmup_ids:
                    for index in indexes:
                        index.knn_query(
                            np.asarray(queries[query_id], dtype=np.float32)[None, :],
                            k=k, num_threads=1,
                        )
                for repetition in range(reps):
                    repetition_start = time.perf_counter()
                    for query_position, query_id in enumerate(measured_ids):
                        query = np.asarray(
                            queries[query_id], dtype=np.float32, order="C"
                        )[None, :]
                        all_labels: list[np.ndarray] = []
                        all_distances: list[np.ndarray] = []
                        start = time.perf_counter_ns()
                        search_start = start
                        for index in indexes:
                            labels, distances = index.knn_query(
                                query, k=k, num_threads=1
                            )
                            all_labels.append(labels[0])
                            all_distances.append(distances[0])
                        search_end = time.perf_counter_ns()
                        found, _ = merge_candidates(all_labels, all_distances, k)
                        end = time.perf_counter_ns()
                        recall = recall_at_k(found, gt_ids[query_id, :k])
                        writer.writerow({
                            "experiment_id": cfg["experiment_id"],
                            "dataset": dataset,
                            "build_profile": build_profile,
                            "topology": topology_id,
                            "shard_count": topology["shard_count"],
                            "ef_search": ef_search,
                            "k": k,
                            "repetition": repetition,
                            "query_position": query_position,
                            "query_id": int(query_id),
                            "end_to_end_ns": end - start,
                            "shard_search_ns": search_end - search_start,
                            "merge_ns": end - search_end,
                            "recall_at_k": f"{recall:.8f}",
                            "returned_count": len(found),
                        })
                    print(
                        f"  K={k} repetition {repetition + 1}/{reps}: "
                        f"{time.perf_counter() - repetition_start:.1f}s",
                        flush=True,
                    )
                handle.flush()
    os.replace(temp_path, raw_path)
    resolved = {key: value for key, value in cfg.items() if not key.startswith("_")}
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )
    manifest = {
        "created_at": utc_now(),
        "dataset": dataset,
        "build_profile": build_profile,
        "topology": topology_id,
        "measured_query_count": measured_count,
        "warmup_query_count": warmup_count,
        "repetitions": reps,
        "ef_search_values": cfg["search"]["ef_search_values"],
        "k_values": cfg["search"]["k_values"],
        "single_thread": True,
        "shard_execution": "sequential",
        "environment": environment(),
        "raw_file": raw_path.name,
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha256_file(raw_path),
        "index_topology_manifest": str(
            (idx_dir / "topology_manifest.json").relative_to(Path(cfg["_root"]))
        ),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    complete.write_text("complete\n")
    del indexes, queries, gt_ids
    gc.collect()
    return raw_path
