import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hnsw_experiment.searching import merge_candidates, recall_at_k


def test_merge_uses_global_best_distance_then_label():
    labels = [np.array([8, 1]), np.array([3, 2])]
    distances = [np.array([0.1, 0.2]), np.array([0.1, 0.3])]
    found, found_distances = merge_candidates(labels, distances, 3)
    assert found.tolist() == [3, 8, 1]
    np.testing.assert_allclose(found_distances, [0.1, 0.1, 0.2])


def test_recall():
    assert recall_at_k(np.array([1, 2, 9]), np.array([1, 2, 3])) == 2 / 3


def test_exact_shard_merge_property():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(100, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)
    distances = np.sum((base - query) ** 2, axis=1)
    exact = np.lexsort((np.arange(100), distances))[:10]
    labels = []
    shard_distances = []
    for shard in np.array_split(rng.permutation(100), 5):
        order = np.lexsort((shard, distances[shard]))[:10]
        labels.append(shard[order])
        shard_distances.append(distances[shard[order]])
    merged, _ = merge_candidates(labels, shard_distances, 10)
    assert merged.tolist() == exact.tolist()

