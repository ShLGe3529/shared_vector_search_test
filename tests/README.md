# Required tests

Before any 10M build, cover deterministic/exhaustive partitioning, exact merge
equivalence, all configured metric semantics, global-label preservation through
HNSW save/load, single-thread sequential dispatch, timing boundaries, manifest
completeness, and incomplete-run handling. Also run the small end-to-end smoke
test defined in the runbook.

