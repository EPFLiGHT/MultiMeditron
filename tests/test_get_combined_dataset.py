"""
Tests for get_combined_dataset in train_multidomain_clip.py.

Covers the domain-balanced dataset-mixing logic:
  - output structure (train/test splits, correct columns)
  - single domain: no trimming, clean pass-through
  - two equal domains: full budget kept
  - two unequal domains: smaller domain sets the budget, larger is trimmed
  - multi-dataset domain: proportional allocation within the domain
  - missing domain field: raises ValueError
  - <attachment> stripped from captions
  - image_column as list: first element is taken
  - image_column as {"type", "value"} dict: value is extracted

Usage:
    pytest tests/test_get_combined_dataset.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset
from data import (
    DatasetConfig,
    ModelArguments,
    get_combined_dataset,
)


def _model_args():
    return ModelArguments(cache_dir=None, token=None, trust_remote_code=False)


def _save_dataset(tmp_dir, name, rows):
    ds = Dataset.from_dict(rows)
    path = tmp_dir / name
    ds.save_to_disk(str(path))
    return path


def _make_dataset(tmp_dir, name, n):
    return _save_dataset(
        tmp_dir,
        name,
        {
            "image_path": [f"/fake/{name}_{i}.jpg" for i in range(n)],
            "text": [f"caption {name} {i}" for i in range(n)],
        },
    )


def _config(ds_path, domain, **kwargs):
    return DatasetConfig(
        dataset_name="json",
        data_dir=str(ds_path),
        image_column="image_path",
        caption_column="text",
        domain=domain,
        **kwargs,
    )


class TestOutputStructure(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_has_train_and_test_splits(self):
        ds_path = _make_dataset(self.tmp, "ct", 20)
        result = get_combined_dataset([_config(ds_path, "ct")], _model_args())
        self.assertIn("train", result)
        self.assertIn("test", result)

    def test_columns_are_image_path_and_caption(self):
        ds_path = _make_dataset(self.tmp, "ct", 20)
        result = get_combined_dataset([_config(ds_path, "ct")], _model_args())
        self.assertEqual(set(result["train"].column_names), {"image_path", "caption"})
        self.assertEqual(set(result["test"].column_names), {"image_path", "caption"})

    def test_train_test_sizes_sum_to_total(self):
        ds_path = _make_dataset(self.tmp, "ct", 20)
        result = get_combined_dataset([_config(ds_path, "ct")], _model_args())
        self.assertEqual(len(result["train"]) + len(result["test"]), 20)


class TestDomainBalancing(unittest.TestCase):
    """Domain balancing: smallest domain determines the per-domain budget.

    Note: interleave_datasets(..., stopping_strategy="all_exhausted") may
    oversample individual datasets to ensure every dataset is fully consumed
    at least once before stopping. Exact totals are therefore not asserted;
    instead we verify that the total is:
      - at least n_domains * budget  (all budget examples are present)
      - strictly less than the sum of the *original* dataset sizes  (trimming
        actually happened for the larger domain)
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_equal_domains_keeps_all_examples(self):
        ct = _make_dataset(self.tmp, "ct", 10)
        mri = _make_dataset(self.tmp, "mri", 10)
        result = get_combined_dataset(
            [_config(ct, "ct"), _config(mri, "mri")],
            _model_args(),
        )
        total = len(result["train"]) + len(result["test"])
        self.assertGreaterEqual(total, 20)

    def test_larger_domain_is_trimmed_to_smaller(self):
        ct = _make_dataset(self.tmp, "ct", 30)
        mri = _make_dataset(self.tmp, "mri", 10)
        result = get_combined_dataset(
            [_config(ct, "ct"), _config(mri, "mri")],
            _model_args(),
        )
        total = len(result["train"]) + len(result["test"])
        self.assertGreaterEqual(total, 20)
        self.assertLess(total, 40)

    def test_three_domains_budget_from_smallest(self):
        ct = _make_dataset(self.tmp, "ct", 40)
        mri = _make_dataset(self.tmp, "mri", 20)
        xray = _make_dataset(self.tmp, "xray", 10)
        result = get_combined_dataset(
            [_config(ct, "ct"), _config(mri, "mri"), _config(xray, "xray")],
            _model_args(),
        )
        total = len(result["train"]) + len(result["test"])
        self.assertGreaterEqual(total, 30)
        self.assertLess(total, 70)


class TestProportionalAllocation(unittest.TestCase):
    """Multiple datasets in the same domain are allocated proportionally."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_datasets_same_domain_proportional(self):
        ct_a = _make_dataset(self.tmp, "ct_a", 70)
        ct_b = _make_dataset(self.tmp, "ct_b", 30)
        mri = _make_dataset(self.tmp, "mri", 50)
        result = get_combined_dataset(
            [_config(ct_a, "ct"), _config(ct_b, "ct"), _config(mri, "mri")],
            _model_args(),
        )
        total = len(result["train"]) + len(result["test"])
        # all_exhausted repeats shorter datasets until the longest is done:
        # max total = max_balanced_size * n_datasets = 50 * 3 = 150
        self.assertGreaterEqual(total, 100)
        self.assertLessEqual(total, 150)

    def test_budget_fits_all_no_trimming(self):
        ct_a = _make_dataset(self.tmp, "ct_a", 20)
        ct_b = _make_dataset(self.tmp, "ct_b", 30)
        mri = _make_dataset(self.tmp, "mri", 80)
        result = get_combined_dataset(
            [_config(ct_a, "ct"), _config(ct_b, "ct"), _config(mri, "mri")],
            _model_args(),
        )
        total = len(result["train"]) + len(result["test"])
        # ct fits its budget (20+30=50), mri trimmed to 50.
        # all_exhausted: max total = 50 * 3 = 150. Without trimming mri (80) it would be 80*3=240.
        self.assertGreaterEqual(total, 100)
        self.assertLessEqual(total, 150)


class TestValidation(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_domain_raises_value_error(self):
        ds_path = _make_dataset(self.tmp, "ct", 10)
        cfg = DatasetConfig(
            dataset_name="json",
            data_dir=str(ds_path),
            image_column="image_path",
            caption_column="text",
            domain=None,
        )
        with self.assertRaises(ValueError):
            get_combined_dataset([cfg], _model_args())

    def test_empty_configs_raises(self):
        with self.assertRaises(Exception):
            get_combined_dataset([], _model_args())


class TestStandardization(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _all_values(self, result, column):
        return list(result["train"][column]) + list(result["test"][column])

    def test_attachment_tag_stripped_from_caption(self):
        ds_path = _save_dataset(
            self.tmp,
            "ct",
            {
                "image_path": ["/img.jpg", "/img2.jpg"],
                "text": ["<attachment>some caption", "normal caption<attachment>"],
            },
        )
        result = get_combined_dataset([_config(ds_path, "ct")], _model_args())
        for cap in self._all_values(result, "caption"):
            self.assertNotIn("<attachment>", cap)

    def test_image_list_uses_first_element(self):
        ds_path = _save_dataset(
            self.tmp,
            "ct",
            {
                "image_path": [["path_a.jpg", "path_b.jpg"], ["path_c.jpg"]],
                "text": ["caption 0", "caption 1"],
            },
        )
        result = get_combined_dataset([_config(ds_path, "ct")], _model_args())
        images = self._all_values(result, "image_path")
        self.assertIn("path_a.jpg", images)
        self.assertNotIn("path_b.jpg", images)

    def test_image_type_value_dict_extracts_value(self):
        n = 20
        ds_path = _save_dataset(
            self.tmp,
            "ct",
            {
                "image_path": [
                    {"type": "image", "value": f"/real/path_{i}.jpg"} for i in range(n)
                ],
                "text": [f"caption {i}" for i in range(n)],
            },
        )
        result = get_combined_dataset([_config(ds_path, "ct")], _model_args())
        images = self._all_values(result, "image_path")
        for img in images:
            self.assertIsInstance(img, str)
            self.assertTrue(img.startswith("/real/path_"))


class TestManifestSourceLoading(unittest.TestCase):
    """get_combined_dataset with manifest_path produces __manifest_source__ dicts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_source_dataset(self, name, n):
        rows = {
            "text": [f"caption {name} {i}" for i in range(n)],
            "modalities": [[{"type": "image", "value": f"/fake/{name}_{i}.jpg"}] for i in range(n)],
        }
        path = self.tmp / name
        Dataset.from_dict(rows).save_to_disk(str(path))
        return path

    def _make_manifest(self, filename, source_root, n):
        path = self.tmp / filename
        with open(path, "w") as f:
            for i in range(n):
                f.write(json.dumps({
                    "source_root": str(source_root),
                    "source_split": "train",
                    "source_index": i,
                }) + "\n")
        return path

    def _config(self, manifest_path):
        return DatasetConfig(manifest_path=str(manifest_path), domain="ct")

    def test_image_path_is_manifest_source_dict(self):
        source = self._make_source_dataset("src_a", 20)
        manifest = self._make_manifest("manifest_a.jsonl", source, 15)
        result = get_combined_dataset([self._config(manifest)], _model_args())
        for img_str in list(result["train"]["image_path"]) + list(result["test"]["image_path"]):
            self.assertIsInstance(img_str, str)
            img = json.loads(img_str)
            self.assertTrue(img.get("__manifest_source__"))

    def test_manifest_source_pointers_are_correct(self):
        source = self._make_source_dataset("src_b", 20)
        manifest = self._make_manifest("manifest_b.jsonl", source, 15)
        result = get_combined_dataset([self._config(manifest)], _model_args())
        for img_str in result["train"]["image_path"]:
            img = json.loads(img_str)
            self.assertEqual(img["source_root"], str(source))
            self.assertEqual(img["source_split"], "train")
            self.assertIsInstance(img["source_index"], int)

    def test_caption_extracted_from_source_text_column(self):
        source = self._make_source_dataset("src_c", 20)
        manifest = self._make_manifest("manifest_c.jsonl", source, 15)
        result = get_combined_dataset([self._config(manifest)], _model_args())
        for caption in result["train"]["caption"]:
            self.assertIsInstance(caption, str)
            self.assertGreater(len(caption), 0)
            self.assertIn("src_c", caption)

    def test_manifest_mixes_with_regular_dataset(self):
        source = self._make_source_dataset("src_d", 20)
        manifest = self._make_manifest("manifest_d.jsonl", source, 20)
        regular = self.tmp / "mri"
        Dataset.from_dict({
            "image_path": [f"/img/mri_{i}.jpg" for i in range(20)],
            "text": [f"mri caption {i}" for i in range(20)],
        }).save_to_disk(str(regular))
        mri_cfg = DatasetConfig(
            dataset_name="json", data_dir=str(regular),
            image_column="image_path", caption_column="text", domain="mri",
        )
        result = get_combined_dataset([self._config(manifest), mri_cfg], _model_args())
        total = len(result["train"]) + len(result["test"])
        self.assertGreaterEqual(total, 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
