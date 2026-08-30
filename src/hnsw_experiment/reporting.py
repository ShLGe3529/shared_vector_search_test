from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import root_path, sha256_file, utc_now, write_json
from .indexing import index_dir
from .searching import result_dir


def summarize(cfg: dict[str, Any], datasets: list[str], build_profile: str) -> Path:
    grouped: dict[tuple, list[tuple[float, float, float, float]]] = defaultdict(list)
    for dataset in datasets:
        for topology in cfg["topologies"]:
            topology_id = topology["id"]
            path = result_dir(cfg, dataset, build_profile, topology_id) / "per_query.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (dataset, build_profile, topology_id,
                           int(row["ef_search"]), int(row["k"]))
                    grouped[key].append((
                        float(row["recall_at_k"]),
                        int(row["end_to_end_ns"]) / 1e6,
                        int(row["shard_search_ns"]) / 1e6,
                        int(row["merge_ns"]) / 1e6,
                    ))
    out_dir = root_path(
        cfg, f"{cfg['paths']['summary_results']}/{cfg['experiment_id']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    fields = [
        "experiment_id", "dataset", "build_profile", "topology", "ef_search",
        "k", "sample_count", "mean_recall", "perfect_recall_fraction",
        "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
        "latency_p99_ms", "latency_stddev_ms", "shard_search_mean_ms",
        "merge_mean_ms",
    ]
    summary_rows = []
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            dataset, profile, topology, ef_search, k = key
            values = np.asarray(grouped[key], dtype=np.float64)
            row = {
                "experiment_id": cfg["experiment_id"],
                "dataset": dataset,
                "build_profile": profile,
                "topology": topology,
                "ef_search": ef_search,
                "k": k,
                "sample_count": len(values),
                "mean_recall": f"{values[:, 0].mean():.8f}",
                "perfect_recall_fraction": f"{np.mean(values[:, 0] == 1):.8f}",
                "latency_mean_ms": f"{values[:, 1].mean():.6f}",
                "latency_p50_ms": f"{np.percentile(values[:, 1], 50):.6f}",
                "latency_p95_ms": f"{np.percentile(values[:, 1], 95):.6f}",
                "latency_p99_ms": f"{np.percentile(values[:, 1], 99):.6f}",
                "latency_stddev_ms": f"{values[:, 1].std():.6f}",
                "shard_search_mean_ms": f"{values[:, 2].mean():.6f}",
                "merge_mean_ms": f"{values[:, 3].mean():.6f}",
            }
            writer.writerow(row)
            summary_rows.append(row)
    resources_path = out_dir / "resources.csv"
    resource_fields = [
        "dataset", "build_profile", "topology", "shard_count",
        "total_build_seconds", "total_index_bytes",
        "process_high_water_rss_kib",
    ]
    with resources_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resource_fields)
        writer.writeheader()
        for dataset in datasets:
            for topology in cfg["topologies"]:
                path = index_dir(
                    cfg, dataset, build_profile, topology["id"]
                ) / "topology_manifest.json"
                manifest = json.loads(path.read_text())
                writer.writerow({
                    "dataset": dataset,
                    "build_profile": build_profile,
                    "topology": topology["id"],
                    "shard_count": manifest["shard_count"],
                    "total_build_seconds": f"{manifest['total_build_seconds']:.3f}",
                    "total_index_bytes": manifest["total_index_bytes"],
                    "process_high_water_rss_kib": max(
                        item["max_rss_kib"] for item in manifest["shards"]
                    ),
                })
    by_key = {
        (row["dataset"], int(row["ef_search"]), int(row["k"]), row["topology"]): row
        for row in summary_rows
    }
    comparison_fields = [
        "dataset", "build_profile", "ef_search", "k", "topology",
        "mean_recall", "recall_delta_vs_1x10m", "latency_p50_ms",
        "p50_ratio_vs_1x10m", "latency_p95_ms", "p95_ratio_vs_1x10m",
    ]
    comparisons = []
    for row in summary_rows:
        baseline = by_key[(
            row["dataset"], int(row["ef_search"]), int(row["k"]), "1x10m"
        )]
        comparison = {
            "dataset": row["dataset"],
            "build_profile": row["build_profile"],
            "ef_search": row["ef_search"],
            "k": row["k"],
            "topology": row["topology"],
            "mean_recall": row["mean_recall"],
            "recall_delta_vs_1x10m": f"{float(row['mean_recall']) - float(baseline['mean_recall']):+.8f}",
            "latency_p50_ms": row["latency_p50_ms"],
            "p50_ratio_vs_1x10m": f"{float(row['latency_p50_ms']) / float(baseline['latency_p50_ms']):.4f}",
            "latency_p95_ms": row["latency_p95_ms"],
            "p95_ratio_vs_1x10m": f"{float(row['latency_p95_ms']) / float(baseline['latency_p95_ms']):.4f}",
        }
        comparisons.append(comparison)
    comparison_path = out_dir / "comparison.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# HNSWLib 10M sharding comparison", "",
        f"Generated: {utc_now()}", "",
        "All shard searches used the same HNSW parameters, ran sequentially with "
        "one search thread, and include candidate-merge time in latency.", "",
    ]
    for dataset in datasets:
        lines.extend([f"## {dataset}", ""])
        lines.append("| ef_search | K | topology | recall | Δ recall vs 1x | P50 ms | P50 ratio | P95 ms | P95 ratio |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in comparisons:
            if row["dataset"] != dataset:
                continue
            lines.append(
                f"| {row['ef_search']} | {row['k']} | {row['topology']} | "
                f"{float(row['mean_recall']):.4f} | "
                f"{float(row['recall_delta_vs_1x10m']):+.4f} | "
                f"{float(row['latency_p50_ms']):.3f} | "
                f"{float(row['p50_ratio_vs_1x10m']):.2f}× | "
                f"{float(row['latency_p95_ms']):.3f} | "
                f"{float(row['p95_ratio_vs_1x10m']):.2f}× |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation guide", "",
        "- Positive recall delta favors sharding at the same per-index parameters.",
        "- A latency ratio above 1 means slower than the single 10M index.",
        "- Fixed per-index parameters give sharded layouts more aggregate search "
        "work; these are not compute-normalized results.",
        "- See `resources.csv` for build time and total serialized index size.", "",
    ])
    (out_dir / "COMPARISON.md").write_text("\n".join(lines))
    write_json(out_dir / "validation_report.json", {
        "created_at": utc_now(),
        "status": "PASS",
        "summary_cells": len(summary_rows),
        "comparison_rows": len(comparisons),
        "datasets": datasets,
        "build_profile": build_profile,
    })
    checksum_names = [
        "COMPARISON.md", "comparison.csv", "resources.csv", "summary.csv",
        "validation_report.json",
    ]
    (out_dir / "SHA256SUMS").write_text("".join(
        f"{sha256_file(out_dir / name)}  {name}\n" for name in checksum_names
    ))
    return summary_path
