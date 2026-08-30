# Planned implementation modules

Keep reusable experiment logic here. Suggested modules are `config.py`,
`datasets.py`, `partition.py`, `ground_truth.py`, `index.py`, `search.py`,
`metrics.py`, `manifests.py`, and `report.py`.

The timed search function should have a narrow interface and must return timing
components separately from recall. This makes it straightforward to test that
ground-truth work is outside the timed region and that shard calls are serial.

