"""
Tests for build_class_weights and load_or_build_dataset in
evaluation_pipeline/benchmark_classification/datasets.py.

All tests use synthetic tensors/examples — no real images or GPU required.

Usage:
    pytest tests/test_datasets.py -v
"""

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation_pipeline.benchmark_classification.datasets import (
    BenchmarkDataset,
    build_class_weights,
    load_or_build_dataset,
)


class TestBuildClassWeights(unittest.TestCase):

    def test_all_classes_present_equal_weights(self):
        labels = torch.tensor([0, 0, 1, 1])
        weights = build_class_weights(labels)
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(weights[0].item(), weights[1].item(), places=5)

    def test_absent_class_gets_weight_one(self):
        # Only class 0 present; classes 1 and 2 are absent
        labels = torch.tensor([0, 0, 0])
        weights = build_class_weights(labels, num_classes=3)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights[1].item(), 1.0, places=5)
        self.assertAlmostEqual(weights[2].item(), 1.0, places=5)

    def test_single_class_no_division_by_zero(self):
        labels = torch.tensor([2, 2, 2])
        weights = build_class_weights(labels)
        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights[0].item(), 1.0, places=5)

    def test_imbalanced_minority_gets_higher_weight(self):
        # class 0: 3 samples, class 1: 1 sample → class 1 should weigh more
        labels = torch.tensor([0, 0, 0, 1])
        weights = build_class_weights(labels)
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_returns_float32_tensor(self):
        labels = torch.tensor([0, 1, 2])
        weights = build_class_weights(labels)
        self.assertEqual(weights.dtype, torch.float32)

    def test_num_classes_expands_output_size(self):
        # Only 2 labels present but num_classes=5
        labels = torch.tensor([0, 1])
        weights = build_class_weights(labels, num_classes=5)
        self.assertEqual(len(weights), 5)


class TestBenchmarkDataset(unittest.TestCase):

    def test_len_matches_labels(self):
        data = torch.zeros(10, 512)
        labels = torch.arange(10)
        ds = BenchmarkDataset(data=data, labels=labels)
        self.assertEqual(len(ds), 10)

    def test_getitem_returns_embedding_and_label(self):
        data = torch.ones(4, 128)
        labels = torch.tensor([0, 1, 2, 3])
        ds = BenchmarkDataset(data=data, labels=labels)
        emb, label = ds[2]
        self.assertTrue(torch.equal(emb, data[2]))
        self.assertEqual(label.item(), 2)


class TestLoadOrBuildDataset(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_embed(self, example, label, model, dataset_root):
        return torch.zeros(512)

    def _examples_and_labels(self, n=4):
        return [{"id": i} for i in range(n)], list(range(n))

    def test_cache_miss_calls_embed_and_saves_files(self):
        examples, labels = self._examples_and_labels()
        calls = []

        def embed(example, label, model, dataset_root):
            calls.append(1)
            return torch.zeros(512)

        ds = load_or_build_dataset(
            cache_prefix="miss",
            examples=examples,
            labels=labels,
            model=None,
            dataset_root=self.cache_root,
            cache_root=self.cache_root,
            use_cache=True,
            desc="test",
            embed_example=embed,
        )
        self.assertEqual(len(calls), 4)
        self.assertIsInstance(ds, BenchmarkDataset)
        self.assertEqual(len(ds), 4)
        self.assertTrue((self.cache_root / "miss_embeddings.pt").exists())
        self.assertTrue((self.cache_root / "miss_labels.pt").exists())

    def test_cache_hit_skips_embed(self):
        examples, labels = self._examples_and_labels()
        calls = []

        def embed(example, label, model, dataset_root):
            calls.append(1)
            return torch.zeros(512)

        # First pass: build cache
        load_or_build_dataset(
            cache_prefix="hit",
            examples=examples, labels=labels,
            model=None, dataset_root=self.cache_root,
            cache_root=self.cache_root, use_cache=True,
            desc="test", embed_example=embed,
        )
        first_count = len(calls)

        # Second pass: should load from cache, no new embed calls
        ds = load_or_build_dataset(
            cache_prefix="hit",
            examples=examples, labels=labels,
            model=None, dataset_root=self.cache_root,
            cache_root=self.cache_root, use_cache=True,
            desc="test", embed_example=embed,
        )
        self.assertEqual(len(calls), first_count)
        self.assertIsInstance(ds, BenchmarkDataset)

    def test_use_cache_false_always_reembeds(self):
        examples, labels = self._examples_and_labels()
        calls = []

        def embed(example, label, model, dataset_root):
            calls.append(1)
            return torch.zeros(512)

        load_or_build_dataset(
            cache_prefix="nocache",
            examples=examples, labels=labels,
            model=None, dataset_root=self.cache_root,
            cache_root=self.cache_root, use_cache=True,
            desc="test", embed_example=embed,
        )
        count_after_first = len(calls)

        load_or_build_dataset(
            cache_prefix="nocache",
            examples=examples, labels=labels,
            model=None, dataset_root=self.cache_root,
            cache_root=self.cache_root, use_cache=False,
            desc="test", embed_example=embed,
        )
        self.assertGreater(len(calls), count_after_first)

    def test_returned_dataset_has_correct_shape(self):
        examples, labels = self._examples_and_labels(n=6)

        ds = load_or_build_dataset(
            cache_prefix="shape",
            examples=examples, labels=labels,
            model=None, dataset_root=self.cache_root,
            cache_root=self.cache_root, use_cache=False,
            desc="test", embed_example=self._fake_embed,
        )
        emb, label = ds[0]
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(len(ds), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
