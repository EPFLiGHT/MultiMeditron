import os
import sys
from pathlib import Path

# Keep this directory on sys.path for older modules that still use script-style
# imports such as `from Benchmark import Benchmark`. benchmark_classification's
# __init__.py also sets this up, but build_benchmarks.py may be imported first.
EVAL_DIR = os.path.dirname(__file__)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from benchmark_classification import (
    MRIBenchmark,
    CTBenchmark,
    HistopathologyBenchmark,
    OphthalmologyBenchmark,
    SkinBenchmark,
    UltrasoundBenchmark,
    XRay_benchmark,
)
from benchmark_classification.xray_benchmark import _resolve_xray_paths


SKIN_INTEGRATED_ENV_VARS = {
    "max_train_examples": "SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "SKIN_INTEGRATED_MAX_TEST_EXAMPLES",
}

OPHTH_ENV_VARS = {
    "max_train_examples": "OPHTH_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "OPHTH_MAX_TEST_EXAMPLES",
}

CT_ENV_VARS = {
    "max_train_examples": "CT_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "CT_MAX_TEST_EXAMPLES",
}

ULTRASOUND_ENV_VARS = {
    "max_train_examples": "ULTRASOUND_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "ULTRASOUND_MAX_TEST_EXAMPLES",
}

XRAY_ENV_VARS = {
    "data_root": "XRAY_DATA_ROOT",
    "max_train_examples": "XRAY_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "XRAY_MAX_TEST_EXAMPLES",
}

MRI_ENV_VARS = {
    "train_jsonl": "MRI_TRAIN_JSONL",
    "test_jsonl": "MRI_TEST_JSONL",
    "max_train_examples": "MRI_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "MRI_MAX_TEST_EXAMPLES",
}

HISTOPATHOLOGY_ENV_VARS = {
    "max_train_examples": "HISTO_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "HISTO_MAX_TEST_EXAMPLES",
}


def _parse_optional_int(raw_value):
    return None if raw_value in (None, "") else int(raw_value)


def _manifest_pair_exists(manifest_root):
    return (manifest_root / "mlp_train.jsonl").exists() and (
        manifest_root / "benchmark_eval.jsonl"
    ).exists()


def _maybe_build_skin_benchmark():
    if not _manifest_pair_exists(SkinBenchmark.default_manifest_root):
        return None
    max_train_examples = _parse_optional_int(
        os.environ.get(SKIN_INTEGRATED_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(SKIN_INTEGRATED_ENV_VARS["max_test_examples"])
    )
    return SkinBenchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _maybe_build_ophthalmology_benchmark():
    if not _manifest_pair_exists(OphthalmologyBenchmark.default_manifest_root):
        return None
    max_train_examples = _parse_optional_int(
        os.environ.get(OPHTH_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(OPHTH_ENV_VARS["max_test_examples"])
    )
    return OphthalmologyBenchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _maybe_build_ct_benchmark():
    if not CTBenchmark.is_available():
        return None
    max_train_examples = _parse_optional_int(
        os.environ.get(CT_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(CT_ENV_VARS["max_test_examples"])
    )
    return CTBenchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _split_env_paths(raw_value):
    if raw_value is None:
        return []
    return [part for part in raw_value.split(os.pathsep) if part]


def _maybe_build_ultrasound_benchmark():
    if not _manifest_pair_exists(UltrasoundBenchmark.default_manifest_root):
        return None
    max_train_examples = _parse_optional_int(
        os.environ.get(ULTRASOUND_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(ULTRASOUND_ENV_VARS["max_test_examples"])
    )
    return UltrasoundBenchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _maybe_build_mri_benchmark():
    train_jsonl = os.environ.get(MRI_ENV_VARS["train_jsonl"])
    test_jsonl = os.environ.get(MRI_ENV_VARS["test_jsonl"])
    max_train_examples = _parse_optional_int(
        os.environ.get(MRI_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(MRI_ENV_VARS["max_test_examples"])
    )

    resolved_train = (
        Path(train_jsonl) if train_jsonl else MRIBenchmark.default_train_jsonl
    )
    resolved_test = (
        Path(test_jsonl) if test_jsonl else MRIBenchmark.default_test_jsonl
    )

    if not resolved_train.exists() or not resolved_test.exists():
        return None

    return MRIBenchmark(
        train_jsonl=resolved_train,
        test_jsonl=resolved_test,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _maybe_build_histopathology_benchmark():
    max_train_examples = _parse_optional_int(
        os.environ.get(HISTOPATHOLOGY_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(HISTOPATHOLOGY_ENV_VARS["max_test_examples"])
    )

    if not HistopathologyBenchmark.is_available():
        return None

    return HistopathologyBenchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _maybe_build_xray_benchmark():
    _, csv_path, _, _ = _resolve_xray_paths()
    if not csv_path.exists():
        return None
    max_train_examples = _parse_optional_int(
        os.environ.get(XRAY_ENV_VARS["max_train_examples"])
    )
    max_test_examples = _parse_optional_int(
        os.environ.get(XRAY_ENV_VARS["max_test_examples"])
    )
    return XRay_benchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def build_benchmarks_from_names(names):
    """Build only the requested benchmarks by stable benchmark name.

    If ``names`` is empty or None, this falls back to the default benchmark suite.
    """

    if not names:
        return build_default_benchmarks()

    requested = [name.strip() for name in names if str(name).strip()]
    normalized = {name.lower(): name for name in requested}

    available_builders = {
        "mri": _maybe_build_mri_benchmark,
        "ct": _maybe_build_ct_benchmark,
        "histopathology": _maybe_build_histopathology_benchmark,
        "skin": _maybe_build_skin_benchmark,
        "ophthalmology": _maybe_build_ophthalmology_benchmark,
        "ultrasound": _maybe_build_ultrasound_benchmark,
        "xray": _maybe_build_xray_benchmark,
    }

    built = []
    unknown = sorted(set(normalized) - set(available_builders))
    if unknown:
        raise ValueError(
            "Unknown benchmark name(s): " + ", ".join(unknown) + ". "
            "Known names: " + ", ".join(sorted(available_builders))
        )

    for name in requested:
        key = name.lower()
        builder = available_builders[key]
        benchmark = builder()
        if benchmark is None:
            raise ValueError(
                f"Benchmark {name!r} was requested in the config but could not be built. "
                "Check its dataset paths / environment variables."
            )
        built.append(benchmark)

    return built


def build_default_benchmarks():
    """Return the default benchmark suite used by train_multidomain_clip.

    Includes MRI, CT, ultrasound, X-ray, skin, ophthalmology,
    and histopathology benchmarks.
    """

    benchmarks = []

    mri_benchmark = _maybe_build_mri_benchmark()
    if mri_benchmark is not None:
        benchmarks.append(mri_benchmark)

    ct_benchmark = _maybe_build_ct_benchmark()
    if ct_benchmark is not None:
        benchmarks.append(ct_benchmark)

    ultrasound_benchmark = _maybe_build_ultrasound_benchmark()
    if ultrasound_benchmark is not None:
        benchmarks.append(ultrasound_benchmark)

    xray_benchmark = _maybe_build_xray_benchmark()
    if xray_benchmark is not None:
        benchmarks.append(xray_benchmark)

    skin_benchmark = _maybe_build_skin_benchmark()
    if skin_benchmark is not None:
        benchmarks.append(skin_benchmark)

    ophthalmology_benchmark = _maybe_build_ophthalmology_benchmark()
    if ophthalmology_benchmark is not None:
        benchmarks.append(ophthalmology_benchmark)

    histopathology_benchmark = _maybe_build_histopathology_benchmark()
    if histopathology_benchmark is not None:
        benchmarks.append(histopathology_benchmark)

    return benchmarks
