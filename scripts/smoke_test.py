#!/usr/bin/env python3
"""Small in-memory test of all three HNSW shard/merge paths."""

import tempfile
import sys
from pathlib import Path

import hnswlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.searching import merge_candidates


def main() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(size=(10_000, 32)).astype(np.float32)
    queries = rng.normal(size=(20, 32)).astype(np.float32)
    permutation = rng.permutation(len(base))
    with tempfile.TemporaryDirectory() as temporary:
        for shard_count in (1, 2, 5):
            indexes = []
            for shard_id, labels in enumerate(np.array_split(permutation, shard_count)):
                index = hnswlib.Index(space="l2", dim=base.shape[1])
                index.init_index(
                    max_elements=len(labels), M=16, ef_construction=100,
                    random_seed=100,
                )
                index.add_items(base[labels], labels, num_threads=1)
                path = Path(temporary) / f"{shard_count}-{shard_id}.bin"
                index.save_index(str(path))
                loaded = hnswlib.Index(space="l2", dim=base.shape[1])
                loaded.load_index(str(path))
                loaded.set_ef(100)
                loaded.set_num_threads(1)
                assert loaded.get_current_count() == len(labels)
                indexes.append(loaded)
            for query in queries:
                labels, distances = [], []
                for index in indexes:
                    shard_labels, shard_distances = index.knn_query(
                        query[None, :], k=10, num_threads=1
                    )
                    labels.append(shard_labels[0])
                    distances.append(shard_distances[0])
                merged, _ = merge_candidates(labels, distances, 10)
                assert len(merged) == 10
                assert len(set(map(int, merged))) == 10
                assert np.all((merged >= 0) & (merged < len(base)))
            print(f"smoke PASS: {shard_count} shard(s)")


if __name__ == "__main__":
    main()

