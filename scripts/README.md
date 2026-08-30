# Command-line entry points

The reproduction runbook uses:

- `download_data.py` and `prepare_data.py`
- `make_partitions.py`
- `smoke_test.py`
- `build_indexes.py`
- `run_search.py`
- `validate_results.py`
- `summarize.py`

CLI programs contain orchestration and argument parsing only. Shared
partition, build, search, merge, metrics, manifest, and reporting logic belongs
under `src/` so it can be unit-tested.
