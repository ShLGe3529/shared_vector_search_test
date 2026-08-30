# Experiment plan: HNSWLib index sharding at 10M scale

## 1. Objective

Measure how global nearest-neighbor recall and latency change when the same 10M
vectors are indexed as one, two, or five independent HNSWLib indexes.

The comparison is deliberately **fixed-parameter per index**, not
compute-normalized. For a given dataset/build/search configuration, `M`,
`ef_construction`, `ef_search`, metric, `K`, vector representation, and HNSWLib
version are identical for every index in all three topologies. A sharded query
therefore performs `S` HNSW searches at the configured `ef_search`, where `S` is
1, 2, or 5. This is part of the system being measured and must not be hidden.

### Research questions

1. At the same HNSW parameters, does searching smaller graphs improve or reduce
   global `recall@K` after exact candidate merging?
2. What is the end-to-end latency cost of sequentially searching 2 or 5 indexes?
3. How do those effects change with dataset, `M`, `ef_construction`,
   `ef_search`, and `K`?
4. What storage, build-time, and peak-memory costs accompany each topology?

### Expected behavior, to be tested rather than assumed

- Exact search of top `K` within every shard followed by a global merge is
  sufficient to recover exact global top `K`: a global top-`K` item cannot rank
  below `K` inside its own shard. With approximate HNSW search, the result is not
  guaranteed, so recall must be measured.
- Smaller HNSW graphs may be easier to search at a fixed `ef_search`, but a
  sharded topology performs more searches. Recall may improve while sequential
  end-to-end latency rises. Cache behavior and dataset dimensionality can change
  either effect.

## 2. Experimental units

### Datasets

Run the full matrix independently on:

| Dataset | Base count | Canonical dtype | Metric | Required metadata |
|---|---:|---|---|---|
| SIFT10M | 10,000,000 | float32 | `l2` | dimension, source, checksums, query count |
| Cohere10M | 10,000,000 | float32 | `ip` | model/source, dimension, original dtype, normalization, checksums, query count |

The Big-ANN Wikipedia-Cohere source defines inner product, so the checked-in
configuration uses HNSWLib space `ip`. Do not substitute cosine: that would make
the official 10M ground truth invalid.

Use the dataset's standard held-out queries where available. Queries must not be
sampled from the indexed base unless the benchmark explicitly defines them that
way. Use one fixed query set, query order, and global ground truth for every
topology.

### Topologies

| ID | Shards | Vectors per shard | Searches per query | Candidates merged |
|---|---:|---:|---:|---:|
| `1x10m` | 1 | 10,000,000 | 1 | `K` |
| `2x5m` | 2 | 5,000,000 | 2 sequential | `2K` |
| `5x2m` | 5 | 2,000,000 | 5 sequential | `5K` |

All base labels are global `int64` IDs in `[0, 10_000_000)`. HNSWLib must store
those global labels directly so merging never relies on shard-local IDs.

## 3. Deterministic random partitioning

Create one permutation of all global IDs with NumPy's `PCG64` generator and the
checked-in `partition_seed`.

- `1x10m`: insert the complete permutation into its sole index.
- `2x5m`: split that same permutation into two equal contiguous slices.
- `5x2m`: split that same permutation into five equal contiguous slices.

Within every shard, retain permutation order during insertion. Save each shard's
ordered global IDs and SHA-256 checksum. Validate that each topology has exactly
10M distinct labels, no overlap between its shards, and the same union of labels.

This procedure makes partition membership, insertion order, and reruns
deterministic. It does not make graph structure identical—the graph sizes are the
experimental treatment.

## 4. Controlled variables and fairness rules

For one `(dataset, build_profile, search_profile, K)` comparison, hold all of the
following constant across `1x10m`, `2x5m`, and `5x2m`:

- exact base vectors, query vectors, and global labels;
- HNSWLib and NumPy versions, Python version, compiler/runtime, and CPU host;
- HNSW `space`, `M`, `ef_construction`, `random_seed`, and vector dtype;
- `ef_search` and `K`;
- `add_items` build-thread setting and batch size;
- per-shard `knn_query(..., num_threads=1)`;
- query order, warm-up queries, measured queries, repetitions, and merge code;
- CPU affinity and thread-related environment variables;
- index load/cache policy.

The following values necessarily differ and are not tuning parameters:

- `max_elements` equals that shard's vector count;
- shard count and candidate count (`S * K`);
- number of sequential HNSW searches per query.

Never tune an `ef_search` separately to make a topology hit a desired recall.
Recall-matched latency may be derived later by interpolation from the common
parameter sweep, but it must be labeled as secondary analysis.

## 5. HNSW configurations

### Primary matrix

Run the baseline build profile first:

- `M = 16`
- `ef_construction = 200`
- HNSW random seed `100`
- `ef_search in {100, 200, 400, 800}`
- `K in {10, 100}`

Every `ef_search` is at least the largest `K`. Invoke each `K` separately; do not
assume that truncating a `K=100` call is identical to running `K=10`.

### Extended matrix

After the primary matrix passes validation and storage/time capacity is known,
add:

- `M = 32`
- `ef_construction = 400`
- the same search sweep and `K` values.

The configuration file is authoritative. Each run manifest must contain the
fully resolved configuration, not just the profile name.

## 6. Ground truth and recall

Ground truth is the exact global top `max(K)` over all 10M base vectors—not the
union of approximate shard results. Use source-provided ground truth only when
its base corpus, query corpus, distance metric, and label mapping exactly match
the canonical data. Validate it against an independent blockwise exact scan for
a deterministic sample of at least 100 queries.

If source ground truth is unavailable, generate it with a blockwise NumPy exact
scan. HNSWLib remains the only vector index/search library; do not use FAISS,
Annoy, ScaNN, or another ANN package. For each query, scan every base block,
retain the best `max(K)` candidates, and merge exactly. Use these distance
definitions so ordering matches HNSWLib:

- `l2`: squared Euclidean distance;
- `ip`: `1 - dot(query, base)`;
- `cosine`: `1 - cosine_similarity(query, base)`, with zero-vector handling
  explicitly tested and normalization recorded.

Store ground-truth global IDs and, preferably, distances. Use deterministic
tie-breaking by `(distance, global_label)`. If ties at the K boundary are common,
also report a tie-aware recall, but retain conventional ID-intersection recall as
the primary metric.

For query `q`:

```text
recall@K(q) = |returned_ids(q) intersection exact_ids(q)| / K
mean_recall@K = mean over measured queries
```

Also report recall percentiles and the fraction of queries with perfect recall.

## 7. Search and merge procedure

For each measured query, execute the following inside one process:

1. Start the end-to-end timer.
2. In ascending shard ID, call `knn_query(query[None, :], k=K,
   num_threads=1)` on one shard at a time.
3. Concatenate all `S*K` `(distance, global_label)` pairs.
4. Select the best global `K` by the metric distance returned by HNSWLib and sort
   by `(distance, global_label)` for deterministic output.
5. Stop the timer.
6. Compute recall outside the timed section.

Do not parallelize over indexes, queries, BLAS, or Python workers. Set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`; call the HNSWLib thread setter where available; and pass
`num_threads=1` explicitly to every query call.

The **primary latency** includes Python dispatch, all shard searches, candidate
concatenation, and merge. Additionally instrument:

- sum of shard-search time;
- merge time;
- optional individual shard times.

These components diagnose the result but do not replace end-to-end latency.

## 8. Timing protocol

- Pin the process to one physical CPU core when the host permits it. Record the
  CPU ID and whether simultaneous multithreading shares that core.
- Use `time.perf_counter_ns()` around each individual query.
- Load all indexes needed by one topology before warm-up.
- Run at least 1,000 fixed warm-up queries, excluded from results.
- Measure at least 10,000 fixed queries, or all standard queries if fewer.
- Run five measured repetitions per matrix cell.
- Execute topology order `1x10m`, `2x5m`, then `5x2m` as recorded in the config.
  Keep the host idle and preserve the same CPU affinity throughout. This fixed
  order is a known limitation; a randomized-order replication is recommended if
  very small latency deltas matter.
- Do not perform builds concurrently with search benchmarks.
- Record whether index files were memory-mapped by the runtime, available RAM,
  process RSS, system load, CPU model/governor, and free disk space.

Primary results are steady-state-after-warm-up. A cold-start study may be run as
a separately labeled experiment, but must not be mixed into primary summaries.
Do not flush OS caches unless the procedure is approved and documented.

Report per-query latency distribution as median, P95, P99, arithmetic mean, and
standard deviation in milliseconds. Report throughput only as a secondary
single-thread statistic.

## 9. Build and resource measurements

For every shard, capture:

- vector count and label checksum;
- resolved HNSW parameters and versions;
- build start/end time and elapsed seconds;
- serialized index bytes;
- peak process RSS if measurable;
- host and environment manifest;
- index-file SHA-256 checksum.

For a topology, report total sequential build time, sum of index bytes, and peak
per-process RSS. If shards are ever built concurrently, that is a separate build
experiment and cannot replace the sequential measurements.

Plan disk capacity before execution: all three topologies store approximately
three complete HNSW representations per dataset for each retained build profile,
in addition to canonical vectors and ground truth.

## 10. Run order

1. Freeze dataset manifests and checksums.
2. Create and validate the deterministic partition manifests.
3. Produce or validate exact global ground truth.
4. Run unit tests on merge logic, metrics, partitioning, and distance semantics.
5. Run an end-to-end smoke test on a deterministic small slice.
6. Build all baseline indexes, capturing manifests and resource measurements.
7. Search the full primary matrix in rotated topology order.
8. Validate raw results and generate summaries/plots.
9. Review capacity and anomalies, then optionally repeat for the extended build
   profile.

Never rebuild only one topology after changing code or dependencies. A material
change increments the experiment ID and invalidates the comparable set.

## 11. Analysis and presentation

For each dataset/build profile, create:

1. recall versus P50/P95/P99 latency curves, one line per topology;
2. a table for every `(ef_search, K)` containing recall and latency statistics;
3. paired per-query latency deltas versus `1x10m`;
4. paired per-query recall deltas versus `1x10m`;
5. build time, index size, and peak RSS by topology.

Use paired bootstrap confidence intervals over queries (resampling query IDs, not
individual repetitions) for recall and latency deltas. Aggregate repetitions per
query first so repeated timings are not treated as independent queries. Always
show absolute measurements alongside percentage changes.

Useful derived comparisons:

```text
latency_ratio = topology_latency / latency_1x10m
recall_delta  = topology_recall - recall_1x10m
size_ratio    = topology_total_index_bytes / bytes_1x10m
```

Do not claim one topology is universally better from a single `ef_search` or
dataset. Conclusions should distinguish fixed-parameter behavior from
recall-matched behavior inferred from the common sweep.

## 12. Validity checks and failure conditions

A run is invalid if any of the following occurs:

- dataset, query, or partition checksum differs across compared cells;
- a topology is missing or duplicates any global label;
- an index uses a different HNSW/search parameter other than `max_elements`;
- any shard returns fewer than `K` results;
- shard searches overlap in time or use more than one search thread;
- distances are merged with the wrong ordering/metric;
- recall uses topology-specific or approximate ground truth;
- timed regions include ground-truth/recall computation;
- the process swaps, encounters memory pressure that affects only some cells, or
  runs alongside a material competing workload;
- raw rows, resolved configuration, or environment manifest are missing.

Flag rather than silently discard outliers. Reruns require a recorded reason.

## 13. Pre-10M implementation gates

Before the expensive build, tests must demonstrate that:

- deterministic splitting is balanced, disjoint, exhaustive, and repeatable;
- merged sharded exact results equal unsharded exact results for `K=1,10,100`;
- global labels survive index save/load;
- HNSW distance values/order agree with exact calculations for every metric used;
- the search path passes `num_threads=1` and invokes shards serially;
- the timer includes merge and excludes recall calculation;
- manifests capture config, git revision, package versions, dataset checksums,
  host data, seeds, and command line;
- interrupted runs cannot be mistaken for complete runs.

## 14. Definition of done

The experiment is complete when both datasets have a valid, comparable baseline
matrix; each cell has five repetitions and per-query raw records; summaries have
confidence intervals and resource statistics; every artifact is traceable to
checksummed inputs, resolved configuration, code revision, and environment; and
the final report answers the four research questions without mixing primary and
secondary protocols.
