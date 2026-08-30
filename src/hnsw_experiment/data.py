from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .common import root_path, sha256_file, utc_now, write_json


SOURCE_URLS = {
    "sift10m": {
        "base": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
        "query": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin",
        "ground_truth": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M",
    },
    "cohere10m": {
        "base": "https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M/wikipedia_base.bin",
        "query": "https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M/wikipedia_query.bin",
        "ground_truth": "https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M/wikipedia-10M",
    },
}


def read_matrix(path: Path, dtype: str, expected_n: int | None = None,
                expected_d: int | None = None) -> np.memmap:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"short matrix header: {path}")
    n, d = struct.unpack("<II", header)
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{path}: rows={n}, expected={expected_n}")
    if expected_d is not None and d != expected_d:
        raise ValueError(f"{path}: dimension={d}, expected={expected_d}")
    itemsize = np.dtype(dtype).itemsize
    expected_size = 8 + n * d * itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path}: bytes={path.stat().st_size}, expected={expected_size}"
        )
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def read_ground_truth(path: Path, expected_n: int | None = None,
                      min_k: int | None = None) -> tuple[np.memmap, np.memmap]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"short ground-truth header: {path}")
    n, k = struct.unpack("<II", header)
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{path}: rows={n}, expected={expected_n}")
    if min_k is not None and k < min_k:
        raise ValueError(f"{path}: k={k}, need at least {min_k}")
    expected_size = 8 + n * k * 8
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path}: bytes={path.stat().st_size}, expected={expected_size}"
        )
    ids = np.memmap(path, dtype="<i4", mode="r", offset=8, shape=(n, k))
    distances = np.memmap(
        path, dtype="<f4", mode="r", offset=8 + n * k * 4, shape=(n, k)
    )
    return ids, distances


def dataset_arrays(cfg: dict[str, Any], name: str):
    ds = cfg["datasets"][name]
    base = read_matrix(
        root_path(cfg, ds["base_path"]), ds["source_dtype"], ds["count"],
        ds["dimension"],
    )
    queries = read_matrix(
        root_path(cfg, ds["query_path"]), ds["source_dtype"],
        ds["query_count"], ds["dimension"],
    )
    gt_ids, gt_distances = read_ground_truth(
        root_path(cfg, ds["ground_truth_path"]), ds["query_count"],
        max(cfg["search"]["k_values"]),
    )
    return base, queries, gt_ids, gt_distances


def finalize_cropped_base(cfg: dict[str, Any], name: str) -> Path:
    ds = cfg["datasets"][name]
    final = root_path(cfg, ds["base_path"])
    part = final.with_suffix(final.suffix + ".part")
    if final.exists():
        return final
    if not part.exists():
        raise FileNotFoundError(f"missing download: {part}")
    expected = 8 + ds["count"] * ds["dimension"] * np.dtype(ds["source_dtype"]).itemsize
    if part.stat().st_size != expected:
        raise ValueError(f"{part}: bytes={part.stat().st_size}, expected={expected}")
    with part.open("r+b") as handle:
        handle.write(struct.pack("<I", ds["count"]))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, final)
    return final


def validate_and_manifest(cfg: dict[str, Any], name: str,
                          with_hashes: bool = True) -> dict[str, Any]:
    ds = cfg["datasets"][name]
    base, queries, gt_ids, gt_distances = dataset_arrays(cfg, name)
    if int(gt_ids.min()) < 0 or int(gt_ids.max()) >= ds["count"]:
        raise ValueError(f"{name}: ground-truth labels outside base range")
    files = {
        "base": root_path(cfg, ds["base_path"]),
        "query": root_path(cfg, ds["query_path"]),
        "ground_truth": root_path(cfg, ds["ground_truth_path"]),
    }
    manifest = {
        "dataset": name,
        "created_at": utc_now(),
        "source_urls": SOURCE_URLS[name],
        "count": ds["count"],
        "dimension": ds["dimension"],
        "query_count": ds["query_count"],
        "source_dtype": ds["source_dtype"],
        "hnsw_space": ds["space"],
        "ground_truth_k": int(gt_ids.shape[1]),
        "files": {
            key: {
                "path": str(path.relative_to(Path(cfg["_root"]))),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path) if with_hashes else None,
            }
            for key, path in files.items()
        },
    }
    write_json(root_path(cfg, ds["manifest_path"]), manifest)
    del base, queries, gt_ids, gt_distances
    return manifest

