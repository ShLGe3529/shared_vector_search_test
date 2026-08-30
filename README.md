# HNSWLib 10M sharding experiment

This project specifies a controlled experiment comparing three ways to index the
same 10 million vectors with HNSWLib:

- `1x10m`: one index containing all 10M vectors
- `2x5m`: two random, disjoint 5M-vector indexes
- `5x2m`: five random, disjoint 2M-vector indexes

The primary outcomes are global `recall@K` and end-to-end single-query latency.
For a sharded topology, every query is sent to every shard **sequentially**, each
shard returns `K` neighbors, and the candidates are merged to global top `K`.
No shard searches run concurrently.

Start with:

1. [results/RESULTS.md](results/RESULTS.md) — concise findings and representative
   measurements.
2. [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) — hypotheses, fairness
   rules, exact procedures, metrics, analysis, and acceptance criteria.
3. [docs/RUNBOOK.md](docs/RUNBOOK.md) — the reproducible execution sequence and
   intended command-line interface.
4. [configs/experiment.yaml](configs/experiment.yaml) — the versioned experiment
   matrix and all seeds.
5. [data/README.md](data/README.md) — canonical dataset layout and validation.
6. [results/README.md](results/README.md) — raw and summary result contracts.

## Project layout

```text
shared_vector_search_test/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml
├── docs/
│   ├── EXPERIMENT_PLAN.md
│   └── RUNBOOK.md
├── data/                    # local inputs; large files are ignored by git
│   └── README.md
├── indexes/                 # generated HNSWLib indexes; ignored by git
├── logs/                    # build/search logs; ignored by git
├── results/
│   ├── README.md
│   ├── raw/                 # one per-query CSV and manifest per run
│   └── summary/             # aggregated comparison tables
├── scripts/                 # CLI entry points described in the runbook
├── src/                     # shared implementation
└── tests/                   # unit and small end-to-end tests
```

## Current status

The baseline experiment is complete and validated: 1.8 million per-query rows
across both datasets, three topologies, four `ef_search` values, two K values, and
five repetitions. See `results/RESULTS.md` and the runbook.
