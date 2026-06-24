"""
Tests for build_benchmarks_from_names in evaluation_pipeline/build_benchmarks.py.

Covers:
  - ct / mri return the correct benchmark types
  - multiple names returns benchmarks in the requested order
  - case-insensitive matching (CT, Mri, ...)
  - leading/trailing whitespace stripped from each name
  - None or empty list falls back to build_default_benchmarks
  - unknown name raises ValueError with a message listing known names
  - unknown name mixed with valid names raises ValueError
  - optional benchmark (skin/ophthalmology) explicitly requested but not configured
    raises ValueError rather than silently skipping
  - skin builds from hardcoded defaults (no env vars required)
  - ophthalmology builds when OPHTH_* env vars point to EyePACS data

Usage:
    pytest tests/test_build_benchmarks_from_names.py -v
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation_pipeline.build_benchmarks import build_benchmarks_from_names
from evaluation_pipeline.benchmark_classification.ct_benchmark import CTBenchmark
from evaluation_pipeline.benchmark_classification.mri_benchmark import MRIBenchmark
from evaluation_pipeline.benchmark_classification.skin_benchmark import SkinBenchmark
from evaluation_pipeline.benchmark_classification.ophthalmology_benchmark import OphthalmologyBenchmark

# Ophthalmology paths read from env; tests are skipped when unset or missing.
_OPHTH_DATASET_ROOT = os.getenv("OPHTH_DATASET_ROOT", "")
_OPHTH_TRAIN_JSONL = os.getenv("OPHTH_TRAIN_JSONL", "")
_OPHTH_TEST_JSONL = os.getenv("OPHTH_TEST_JSONL", "")


def _types(result):
    return [type(b) for b in result]


class TestSingleNames(unittest.TestCase):

    def test_ct_returns_one_ct_benchmark(self):
        result = build_benchmarks_from_names(["ct"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CTBenchmark)

    def test_mri_returns_one_mri_benchmark(self):
        result = build_benchmarks_from_names(["mri"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MRIBenchmark)


class TestMultipleNames(unittest.TestCase):

    def test_ct_then_mri_returns_two_benchmarks(self):
        result = build_benchmarks_from_names(["ct", "mri"])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], CTBenchmark)
        self.assertIsInstance(result[1], MRIBenchmark)

    def test_order_is_preserved_mri_before_ct(self):
        result = build_benchmarks_from_names(["mri", "ct"])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], MRIBenchmark)
        self.assertIsInstance(result[1], CTBenchmark)

    def test_duplicate_names_build_two_instances(self):
        result = build_benchmarks_from_names(["ct", "ct"])
        self.assertEqual(len(result), 2)
        for b in result:
            self.assertIsInstance(b, CTBenchmark)


class TestNormalization(unittest.TestCase):

    def test_uppercase_ct_works(self):
        result = build_benchmarks_from_names(["CT"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CTBenchmark)

    def test_mixed_case_mri_works(self):
        result = build_benchmarks_from_names(["MrI"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MRIBenchmark)

    def test_whitespace_stripped_from_name(self):
        result = build_benchmarks_from_names(["  ct  "])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CTBenchmark)

    def test_whitespace_stripped_mixed_case(self):
        result = build_benchmarks_from_names(["  CT  ", "  MRI  "])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], CTBenchmark)
        self.assertIsInstance(result[1], MRIBenchmark)


class TestFallbackToDefault(unittest.TestCase):

    def test_none_falls_back_to_default(self):
        result = build_benchmarks_from_names(None)
        types = _types(result)
        self.assertIn(CTBenchmark, types)
        self.assertIn(MRIBenchmark, types)

    def test_empty_list_falls_back_to_default(self):
        result = build_benchmarks_from_names([])
        types = _types(result)
        self.assertIn(CTBenchmark, types)
        self.assertIn(MRIBenchmark, types)

    def test_default_returns_at_least_ct_and_mri(self):
        result = build_benchmarks_from_names(None)
        self.assertGreaterEqual(len(result), 2)


class TestUnknownNames(unittest.TestCase):

    def test_unknown_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_benchmarks_from_names(["nonexistent_modality"])

    def test_unknown_name_mixed_with_valid_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_benchmarks_from_names(["ct", "nonexistent_modality"])

    def test_error_message_names_the_unknown_benchmark(self):
        with self.assertRaises(ValueError) as ctx:
            build_benchmarks_from_names(["bad_benchmark"])
        self.assertIn("bad_benchmark", str(ctx.exception))

    def test_error_message_lists_known_names(self):
        with self.assertRaises(ValueError) as ctx:
            build_benchmarks_from_names(["bad_benchmark"])
        msg = str(ctx.exception).lower()
        self.assertIn("ct", msg)
        self.assertIn("mri", msg)


_OPTIONAL_BENCHMARK_ENVS = {
    "skin": ["SKIN_TRAIN_JSONL", "SKIN_TEST_JSONL", "SKIN_IMAGE_ROOT"],
    "ophthalmology": ["OPHTH_DATASET_ROOT", "OPHTH_TRAIN_JSONL", "OPHTH_TEST_JSONL"],
}


class TestOptionalBenchmarksMissing(unittest.TestCase):

    def _assert_raises_when_unconfigured(self, name, env_keys):
        with patch.dict(os.environ, {}, clear=False):
            for key in env_keys:
                os.environ.pop(key, None)
            with self.assertRaises(ValueError, msg=f"Expected ValueError for unconfigured {name!r}"):
                build_benchmarks_from_names([name])

    def test_skin_not_configured_raises(self):
        self._assert_raises_when_unconfigured("skin", _OPTIONAL_BENCHMARK_ENVS["skin"])

    def test_ophthalmology_not_configured_raises(self):
        self._assert_raises_when_unconfigured("ophthalmology", _OPTIONAL_BENCHMARK_ENVS["ophthalmology"])


class TestSkinIntegrated(unittest.TestCase):

    def test_skin_builds_without_env_vars(self):
        env_keys = [
            "SKIN_INTEGRATED_TRAIN_JSONLS",
            "SKIN_INTEGRATED_TEST_JSONLS",
            "SKIN_INTEGRATED_IMAGE_ROOTS",
        ]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_keys:
                os.environ.pop(key, None)
            result = build_benchmarks_from_names(["skin"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], SkinBenchmark)

    def test_skin_in_default_suite(self):
        result = build_benchmarks_from_names(None)
        self.assertIn(SkinBenchmark, _types(result))


class TestOphthalmologyConfigured(unittest.TestCase):

    _OPHTH_ENV = {
        "OPHTH_DATASET_ROOT": _OPHTH_DATASET_ROOT,
        "OPHTH_TRAIN_JSONL": _OPHTH_TRAIN_JSONL,
        "OPHTH_TEST_JSONL": _OPHTH_TEST_JSONL,
    }

    @classmethod
    def setUpClass(cls):
        missing = [k for k, v in cls._OPHTH_ENV.items() if not v or not Path(v).exists()]
        if missing:
            raise unittest.SkipTest(
                "Set OPHTH_DATASET_ROOT, OPHTH_TRAIN_JSONL, OPHTH_TEST_JSONL to run ophthalmology tests."
            )

    def test_ophthalmology_builds_with_env_vars(self):
        with patch.dict(os.environ, self._OPHTH_ENV):
            result = build_benchmarks_from_names(["ophthalmology"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], OphthalmologyBenchmark)

    def test_ophthalmology_dataset_root_is_set(self):
        with patch.dict(os.environ, self._OPHTH_ENV):
            result = build_benchmarks_from_names(["ophthalmology"])
        self.assertEqual(str(result[0].dataset_root), _OPHTH_DATASET_ROOT)

    def test_ophthalmology_train_test_jsonls_are_set(self):
        with patch.dict(os.environ, self._OPHTH_ENV):
            result = build_benchmarks_from_names(["ophthalmology"])
        self.assertEqual(str(result[0].train_jsonl), _OPHTH_TRAIN_JSONL)
        self.assertEqual(str(result[0].test_jsonl), _OPHTH_TEST_JSONL)

    def test_ophthalmology_in_default_suite_when_configured(self):
        with patch.dict(os.environ, self._OPHTH_ENV):
            result = build_benchmarks_from_names(None)
        self.assertIn(OphthalmologyBenchmark, _types(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
