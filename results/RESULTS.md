# Results: sequential HNSWLib sharding at 10M scale

## Bottom line

With identical HNSW parameters on every index, splitting 10M vectors into more
indexes improved recall on both datasets, but sequential search latency increased
almost in proportion to the shard count.

- `2x5m` used about **1.9–2.0×** the P50 latency of `1x10m`.
- `5x2m` used about **4.4–5.0×** the P50 latency of `1x10m`.
- Recall gains were largest at low `ef_search`, especially for `K=100`, and
  narrowed as the single index approached perfect recall.
- Total serialized index size was effectively unchanged by sharding.
- Smaller graphs reduced aggregate build time by 9.8%/23.7% for SIFT
  (`2x5m`/`5x2m`) and 4.3%/9.8% for Cohere.

If shards must be searched sequentially, `1x10m` is the best latency choice.
`2x5m` is a reasonable recall/latency trade when its recall gain matters.
`5x2m` provides the highest fixed-parameter recall, but its latency penalty is
large. Parallel shard search would answer a different question and was not used.

## Representative result (`ef_search=100`, `K=100`)

| Dataset | Topology | Recall@100 | Recall gain vs 1x | P50 latency | P50 ratio |
|---|---|---:|---:|---:|---:|
| SIFT10M | `1x10m` | 0.8605 | — | 0.281 ms | 1.00× |
| SIFT10M | `2x5m` | 0.9178 | +0.0573 | 0.545 ms | 1.94× |
| SIFT10M | `5x2m` | 0.9638 | +0.1033 | 1.285 ms | 4.57× |
| Cohere10M | `1x10m` | 0.8076 | — | 0.715 ms | 1.00× |
| Cohere10M | `2x5m` | 0.8641 | +0.0565 | 1.431 ms | 2.00× |
| Cohere10M | `5x2m` | 0.9142 | +0.1066 | 3.551 ms | 4.97× |

The effect is consistent across the 128-dimensional L2 and 768-dimensional
inner-product datasets.

## Recall-matched view (`K=100`)

Using only points from the common `ef_search` sweep:

| Dataset | Recall target | `1x10m` best qualifying point | `2x5m` best qualifying point | `5x2m` best qualifying point |
|---|---:|---|---|---|
| SIFT10M | ≥0.96 | ef=400, 0.9782, 0.937 ms | ef=200, 0.9691, 0.976 ms | ef=100, 0.9638, 1.285 ms |
| Cohere10M | ≥0.95 | ef=800, 0.9646, 4.446 ms | ef=400, 0.9575, 4.745 ms | ef=200, 0.9543, 6.368 ms |

Two shards can reach a similar recall target at half the per-index `ef_search`,
but sequentially searching both indexes leaves P50 latency slightly higher in
these examples. Five shards require less per-index effort but remain materially
slower overall.

## Build and storage

| Dataset | Topology | Build time | Change vs 1x | Serialized bytes | Size ratio |
|---|---|---:|---:|---:|---:|
| SIFT10M | `1x10m` | 1,391.5 s | — | 6,605,339,232 | 1.0000× |
| SIFT10M | `2x5m` | 1,255.0 s | -9.8% | 6,605,341,164 | 1.0000× |
| SIFT10M | `5x2m` | 1,062.2 s | -23.7% | 6,605,505,944 | 1.0000× |
| Cohere10M | `1x10m` | 5,349.4 s | — | 32,205,338,552 | 1.0000× |
| Cohere10M | `2x5m` | 5,119.5 s | -4.3% | 32,205,340,892 | 1.0000× |
| Cohere10M | `5x2m` | 4,826.0 s | -9.8% | 32,205,506,012 | 1.0000× |

Build time includes graph construction and index serialization but not the
subsequent SHA-256 calculation. Builds used 16 threads and ran sequentially.

## Experiment facts

- HNSWLib 0.8.0; `M=16`; `ef_construction=200`.
- `ef_search={100,200,400,800}`; `K={10,100}`.
- Five repetitions over all 10,000 SIFT queries and all 5,000 Cohere queries.
- Search pinned to CPU 4 on an Intel Xeon Silver 4316.
- One search thread; shards queried sequentially; merge included in latency.
- SIFT10M: 128D uint8 source vectors, HNSWLib L2.
- Wikipedia-Cohere10M: 768D float32, HNSWLib inner product.
- Exact official global top-100 ground truth shared by all topologies.
- Total validated raw rows: 1,800,000.

The topology execution order was `1x10m`, `2x5m`, then `5x2m`. The host was idle
during search, but this fixed order remains a limitation for detecting very small
latency differences. It does not explain the large shard-count effects observed.

## Detailed artifacts

- `summary/hnswlib-sharding-10m-v1/COMPARISON.md`: every dataset/ef/K/topology.
- `summary/hnswlib-sharding-10m-v1/comparison.csv`: recall deltas and latency ratios.
- `summary/hnswlib-sharding-10m-v1/summary.csv`: full latency distributions.
- `summary/hnswlib-sharding-10m-v1/resources.csv`: build and storage results.
- `summary/hnswlib-sharding-10m-v1/validation_report.json`: validation status.
- `raw/hnswlib-sharding-10m-v1/`: per-query data, configs, manifests, hashes.

Follow `../docs/RUNBOOK.md` exactly to reproduce the results.

