# Reproduce the published results

Run every command below from:

```bash
cd /ssd_root/liu3529/shared_vector_search_test
```

The measured experiment uses HNSWLib 0.8.0, Python 3.10, the `baseline` build
profile (`M=16`, `ef_construction=200`), and the checked-in seeds and search
matrix. Do not change `configs/experiment.yaml` when reproducing these results.

## 1. Check capacity

Have at least 64 GiB RAM and 200 GiB free local disk. Close unrelated CPU- and
memory-intensive jobs before building and especially before search timing.

```bash
free -h
df -h .
lscpu | sed -n '1,25p'
```

## 2. Create the exact Python environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > logs/pip-freeze.txt
```

## 3. Download the public datasets

This downloads about 32 GB. The downloader resumes partial files and fetches
only the first 10M base vectors from the larger source objects.

```bash
.venv/bin/python scripts/download_data.py --all
```

The inputs are:

- BIGANN SIFT, first 10M vectors: 128-dimensional uint8, squared L2;
- Wikipedia-Cohere, first 10M vectors: 768-dimensional float32, inner product;
- the official public query sets and exact top-100 ground truth for each 10M
  slice.

## 4. Finalize, checksum, and partition the data

```bash
.venv/bin/python scripts/prepare_data.py --config configs/experiment.yaml --all
.venv/bin/python scripts/make_partitions.py --config configs/experiment.yaml --all
```

This patches each cropped base-file row count to 10M, validates every header and
ground-truth ID, writes SHA-256 manifests, and creates one deterministic PCG64
permutation. All topologies use slices of that same permutation.

## 5. Test the implementation

```bash
.venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/smoke_test.py
```

Do not continue unless all tests and all three smoke-test topologies pass.

## 6. Build the indexes

Builds use 16 threads, but the parameters and thread count are identical for
every shard. Run these commands sequentially:

```bash
.venv/bin/python scripts/build_indexes.py \
  --config configs/experiment.yaml --dataset sift10m \
  --build-profile baseline --all-topologies

.venv/bin/python scripts/build_indexes.py \
  --config configs/experiment.yaml --dataset cohere10m \
  --build-profile baseline --all-topologies
```

Each completed topology contains a `COMPLETE` marker and checksummed build
manifests under `indexes/hnswlib-sharding-10m-v1/`.

## 7. Choose one physical search core

Inspect CPU/core/sibling mappings and choose an otherwise idle physical core.
The reference run used CPU 4; its sibling must also remain idle.

```bash
lscpu -e=CPU,CORE,SOCKET,ONLINE
```

Set all thread controls exactly once in the same shell:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 8. Run searches sequentially

Do not build, download, or run another benchmark concurrently. Execute each
topology separately; every query searches its shards in ascending order and
merges them before the end-to-end timer stops.

```bash
for dataset in sift10m cohere10m; do
  for topology in 1x10m 2x5m 5x2m; do
    taskset -c 4 .venv/bin/python scripts/run_search.py \
      --config configs/experiment.yaml \
      --dataset "$dataset" \
      --build-profile baseline \
      --topology "$topology"
  done
done
```

The runner refuses to proceed unless all four thread environment variables equal
`1`. It also passes `num_threads=1` to every HNSWLib query call. Recall
calculation is outside the timed region.

## 9. Validate and summarize

```bash
.venv/bin/python scripts/validate_results.py \
  --config configs/experiment.yaml --build-profile baseline

.venv/bin/python scripts/summarize.py \
  --config configs/experiment.yaml --build-profile baseline
```

Read the outputs in this order:

1. `results/summary/hnswlib-sharding-10m-v1/COMPARISON.md`
2. `results/summary/hnswlib-sharding-10m-v1/comparison.csv`
3. `results/summary/hnswlib-sharding-10m-v1/summary.csv`
4. `results/summary/hnswlib-sharding-10m-v1/resources.csv`
5. per-query CSV and run manifests under `results/raw/`

## 10. Re-running safely

Completed build and search commands refuse to overwrite artifacts. To reproduce
from scratch without deleting the published run, copy this folder, change
`experiment_id` in the copied config, and run the steps there. Use `--force`
only when intentionally replacing an incomplete or disposable run.

