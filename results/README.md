# Result artifact contract

Do not commit large raw result files. Preserve them with checksums in the
experiment archive.

```text
results/
├── RESULTS.md
├── raw/<experiment_id>/<dataset>/<build_profile>/<topology>/
│   ├── resolved_config.yaml
│   ├── per_query.csv
│   ├── run_manifest.json
│   └── COMPLETE
└── summary/<experiment_id>/
    ├── COMPARISON.md
    ├── comparison.csv
    ├── summary.csv
    ├── resources.csv
    ├── validation_report.json
    └── SHA256SUMS
```

`COMPLETE` is written atomically only after all expected rows and checksums pass.
Summarization ignores any run without it.

Minimum `per_query.csv` columns:

```text
experiment_id,dataset,build_profile,topology,shard_count,ef_search,k,
repetition,query_position,query_id,end_to_end_ns,shard_search_ns,merge_ns,
recall_at_k,returned_count
```

Minimum `summary.csv` columns:

```text
experiment_id,dataset,build_profile,topology,ef_search,k,query_count,
repetitions,mean_recall,perfect_recall_fraction,latency_mean_ms,
latency_p50_ms,latency_p95_ms,latency_p99_ms,latency_stddev_ms
```

The run manifest contains the raw CSV checksum, resolved search settings, index
manifest reference, environment, and completion state. Dataset, partition, and
index checksums live in their dedicated manifests. `SHA256SUMS` covers the final
summary artifacts.

The RSS value in `resources.csv` is the process high-water mark. Builds for one
dataset were executed in one process, so this counter is cumulative and must not
be interpreted as a topology-specific peak.
