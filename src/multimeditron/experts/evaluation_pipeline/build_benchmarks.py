from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List


# Some older benchmark modules still rely on script-style imports such as
# `from Benchmark import Benchmark`. Keep this directory on sys.path so the
# new factory can import both the old and the newer benchmark modules.
EVAL_DIR = os.path.dirname(__file__)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)



SKIN_INTEGRATED_ENV_VARS = {
    "train_jsonls": "SKIN_INTEGRATED_TRAIN_JSONLS",
    "test_jsonls": "SKIN_INTEGRATED_TEST_JSONLS",
    "image_roots": "SKIN_INTEGRATED_IMAGE_ROOTS",
    "max_train_examples": "SKIN_INTEGRATED_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "SKIN_INTEGRATED_MAX_TEST_EXAMPLES",
}

OPHTH_ENV_VARS = {
    "dataset_root": "OPHTH_DATASET_ROOT",
    "train_jsonl": "OPHTH_TRAIN_JSONL",
    "test_jsonl": "OPHTH_TEST_JSONL",
    "max_train_examples": "OPHTH_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "OPHTH_MAX_TEST_EXAMPLES",
}

MRI_ENV_VARS = {
    "dataset_root": "MRI_DATASET_ROOT",
    "dataset_jsonl": "MRI_DATASET_JSONL",
    "subset_fraction": "MRI_SUBSET_FRACTION",
    "max_train_examples": "MRI_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "MRI_MAX_TEST_EXAMPLES",
    "balanced_sampling": "MRI_BALANCED_SAMPLING",
}

CT_ENV_VARS = {
    "dataset_root": "CT_DATASET_ROOT",
    "dataset_jsonl": "CT_DATASET_JSONL",
    "max_train_examples": "CT_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "CT_MAX_TEST_EXAMPLES",
    "balanced_sampling": "CT_BALANCED_SAMPLING",
}

SCIN_ENV_VARS = {
    "eval_jsonls": "SCIN_EVAL_JSONLS",
    "manifest_jsonls": "SCIN_MANIFEST_JSONLS",
    "protocol_cache_path": "SCIN_PROTOCOL_CACHE_PATH",
    "ref_model_name": "SCIN_REF_MODEL_NAME",
}

ULTRASOUND_ENV_VARS = {
    "dataset_root": "ULTRASOUND_DATASET_ROOT",
    "max_train_examples": "ULTRASOUND_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "ULTRASOUND_MAX_TEST_EXAMPLES",
}

XRAY_ENV_VARS = {
    "data_root": "XRAY_DATA_ROOT",
    "max_train_examples": "XRAY_MAX_TRAIN_EXAMPLES",
    "max_test_examples": "XRAY_MAX_TEST_EXAMPLES",
}


def _parse_optional_int(raw_value: str | None) -> int | None:
    return None if raw_value in (None, "") else int(raw_value)


def _validate_optional_env_block(name: str, values: dict[str, str | None], env_names: dict[str, str]):
    provided_count = sum(value is not None for value in values.values())
    if provided_count == 0:
        return False

    missing = [key for key, value in values.items() if not value]
    if missing:
        missing_env_names = ", ".join(env_names[key] for key in missing)
        raise ValueError(
            f"{name} benchmark configuration is incomplete. Missing environment variable(s): "
            f"{missing_env_names}"
        )

    missing_paths = [str(Path(value)) for value in values.values() if value and not Path(value).exists()]
    if missing_paths:
        raise FileNotFoundError(f"{name} benchmark paths do not exist: " + ", ".join(missing_paths))

    return True



def _maybe_build_skin_integrated_benchmark():
    train_jsonls = _split_env_paths(os.environ.get(SKIN_INTEGRATED_ENV_VARS["train_jsonls"]))
    test_jsonls = _split_env_paths(os.environ.get(SKIN_INTEGRATED_ENV_VARS["test_jsonls"]))
    image_roots = _split_env_paths(os.environ.get(SKIN_INTEGRATED_ENV_VARS["image_roots"]))
    max_train_examples = _parse_optional_int(os.environ.get(SKIN_INTEGRATED_ENV_VARS["max_train_examples"]))
    max_test_examples = _parse_optional_int(os.environ.get(SKIN_INTEGRATED_ENV_VARS["max_test_examples"]))

    from evaluation_pipeline.benchmark_classification.skin_integrated_benchmark import SkinIntegratedBenchmark

    provided = {
        "train_jsonls": bool(train_jsonls),
        "test_jsonls": bool(test_jsonls),
        "image_roots": bool(image_roots),
    }
    provided_count = sum(provided.values())
    if provided_count == 0:
        default_train_exists = all(path.exists() for path in SkinIntegratedBenchmark.default_train_jsonls)
        default_test_exists = all(path.exists() for path in SkinIntegratedBenchmark.default_test_jsonls)
        default_roots_exist = all(path.exists() for path in SkinIntegratedBenchmark.default_image_roots)
        if not (default_train_exists and default_test_exists and default_roots_exist):
            return None
        return SkinIntegratedBenchmark(
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )

    missing = [key for key, is_present in provided.items() if not is_present]
    if missing:
        missing_env_names = ", ".join(SKIN_INTEGRATED_ENV_VARS[key] for key in missing)
        raise ValueError(
            "Integrated skin benchmark configuration is incomplete. Missing environment variable(s): "
            f"{missing_env_names}"
        )

    missing_paths = [
        path for path in [*train_jsonls, *test_jsonls, *image_roots]
        if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError("Integrated skin benchmark paths do not exist: " + ", ".join(missing_paths))

    return SkinIntegratedBenchmark(
        train_jsonls=tuple(train_jsonls),
        test_jsonls=tuple(test_jsonls),
        image_roots=tuple(image_roots),
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _ophthalmology_config_from_env():
    return {
        "dataset_root": os.environ.get(OPHTH_ENV_VARS["dataset_root"]),
        "train_jsonl": os.environ.get(OPHTH_ENV_VARS["train_jsonl"]),
        "test_jsonl": os.environ.get(OPHTH_ENV_VARS["test_jsonl"]),
    }


def _maybe_build_ophthalmology_benchmark():
    config = _ophthalmology_config_from_env()
    max_train_examples = _parse_optional_int(os.environ.get(OPHTH_ENV_VARS["max_train_examples"]))
    max_test_examples = _parse_optional_int(os.environ.get(OPHTH_ENV_VARS["max_test_examples"]))

    if not _validate_optional_env_block("Ophthalmology", config, OPHTH_ENV_VARS):
        return None

    from evaluation_pipeline.benchmark_classification.ophthalmology_benchmark import OphthalmologyBenchmark

    return OphthalmologyBenchmark(
        dataset_root=config["dataset_root"],
        train_jsonl=config["train_jsonl"],
        test_jsonl=config["test_jsonl"],
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def _maybe_build_ophthalmology_dr_benchmark():
    config = _ophthalmology_config_from_env()

    if not _validate_optional_env_block("Ophthalmology", config, OPHTH_ENV_VARS):
        return None

    from evaluation_pipeline.benchmark_classification.ophthalmology_dr_benchmark import OphthalmologyDRBenchmark

    return OphthalmologyDRBenchmark(
        train_jsonls=(config["train_jsonl"],),
        test_jsonls=(config["test_jsonl"],),
        image_roots=(config["dataset_root"],),
    )



def _build_ct_benchmark():
    from evaluation_pipeline.benchmark_classification.ct_benchmark import CTBenchmark

    def _parse_optional_bool(raw_value: str | None) -> bool | None:
        if raw_value in (None, ""):
            return None
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(
            f"Invalid boolean value for {CT_ENV_VARS['balanced_sampling']}: {raw_value!r}"
        )

    dataset_root = os.environ.get(CT_ENV_VARS["dataset_root"])
    dataset_jsonl = os.environ.get(CT_ENV_VARS["dataset_jsonl"])
    max_train_examples = _parse_optional_int(os.environ.get(CT_ENV_VARS["max_train_examples"]))
    max_test_examples = _parse_optional_int(os.environ.get(CT_ENV_VARS["max_test_examples"]))
    balanced_sampling = _parse_optional_bool(os.environ.get(CT_ENV_VARS["balanced_sampling"]))
    return CTBenchmark(
        dataset_root=dataset_root,
        dataset_jsonl=dataset_jsonl,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
        balanced_sampling=balanced_sampling,
    )


def _build_mri_benchmark():
    from evaluation_pipeline.benchmark_classification.mri_benchmark import MRIBenchmark

    def _parse_optional_float(raw_value: str | None) -> float | None:
        return None if raw_value in (None, "") else float(raw_value)

    def _parse_optional_bool(raw_value: str | None) -> bool | None:
        if raw_value in (None, ""):
            return None
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(
            f"Invalid boolean value for {MRI_ENV_VARS['balanced_sampling']}: {raw_value!r}"
        )

    dataset_root = os.environ.get(MRI_ENV_VARS["dataset_root"])
    dataset_jsonl = os.environ.get(MRI_ENV_VARS["dataset_jsonl"])
    subset_fraction = _parse_optional_float(os.environ.get(MRI_ENV_VARS["subset_fraction"]))
    max_train_examples = _parse_optional_int(os.environ.get(MRI_ENV_VARS["max_train_examples"]))
    max_test_examples = _parse_optional_int(os.environ.get(MRI_ENV_VARS["max_test_examples"]))
    balanced_sampling = _parse_optional_bool(os.environ.get(MRI_ENV_VARS["balanced_sampling"]))
    return MRIBenchmark(
        dataset_root=dataset_root,
        dataset_jsonl=dataset_jsonl,
        subset_fraction=subset_fraction,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
        balanced_sampling=balanced_sampling,
    )


def _split_env_paths(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [part for part in raw_value.split(os.pathsep) if part]


def _maybe_build_scin_benchmark():
    eval_jsonls = _split_env_paths(os.environ.get(SCIN_ENV_VARS["eval_jsonls"]))
    manifest_jsonls = _split_env_paths(os.environ.get(SCIN_ENV_VARS["manifest_jsonls"]))
    protocol_cache_path = os.environ.get(SCIN_ENV_VARS["protocol_cache_path"])
    ref_model_name = os.environ.get(SCIN_ENV_VARS["ref_model_name"], "openai/clip-vit-base-patch32")

    from evaluation_pipeline.scin_benchmark import SCINBenchmark

    provided = {
        "eval_jsonls": bool(eval_jsonls),
        "manifest_jsonls": bool(manifest_jsonls),
    }
    provided_count = sum(provided.values())
    if provided_count == 0:
        default_eval_exists = all(path.exists() for path in SCINBenchmark.default_eval_jsonls)
        default_manifest_exists = all(path.exists() for path in SCINBenchmark.default_manifest_jsonls)
        if not (default_eval_exists and default_manifest_exists):
            return None
        return SCINBenchmark(
            protocol_cache_path=protocol_cache_path,
            ref_model_name=ref_model_name,
        )

    missing = [key for key, is_present in provided.items() if not is_present]
    if missing:
        missing_env_names = ", ".join(SCIN_ENV_VARS[key] for key in missing)
        raise ValueError(
            "SCIN benchmark configuration is incomplete. Missing environment variable(s): "
            f"{missing_env_names}"
        )

    missing_paths = [
        path for path in [*eval_jsonls, *manifest_jsonls]
        if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError("SCIN benchmark paths do not exist: " + ", ".join(missing_paths))

    return SCINBenchmark(
        eval_jsonls=eval_jsonls,
        manifest_jsonls=manifest_jsonls,
        protocol_cache_path=protocol_cache_path,
        ref_model_name=ref_model_name,
    )


def _maybe_build_ultrasound_benchmark():
    from benchmark_classification.ultrasound_benchmark import UltrasoundBenchmark, DATASET_ROOT

    dataset_root_env = os.environ.get(ULTRASOUND_ENV_VARS["dataset_root"])
    dataset_root = Path(dataset_root_env) if dataset_root_env else DATASET_ROOT
    if not dataset_root.exists():
        return None
    max_train_examples = _parse_optional_int(os.environ.get(ULTRASOUND_ENV_VARS["max_train_examples"]))
    max_test_examples = _parse_optional_int(os.environ.get(ULTRASOUND_ENV_VARS["max_test_examples"]))
    return UltrasoundBenchmark(
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
        dataset_root=dataset_root,
    )


def _maybe_build_xray_benchmark():
    from evaluation_pipeline.xray_eval import XRay_benchmark, _resolve_xray_paths

    _, csv_path, _, _ = _resolve_xray_paths()
    if not csv_path.exists():
        return None
    max_train_examples = _parse_optional_int(os.environ.get(XRAY_ENV_VARS["max_train_examples"]))
    max_test_examples = _parse_optional_int(os.environ.get(XRAY_ENV_VARS["max_test_examples"]))
    return XRay_benchmark(
        is_lion_model=False,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def build_benchmarks_from_names(names: list[str] | tuple[str, ...] | None) -> List[object]:
    """Build only the requested benchmarks by stable benchmark name.

    If ``names`` is empty or None, this falls back to the default benchmark suite.
    """

    if not names:
        return build_default_benchmarks()

    requested = [name.strip() for name in names if str(name).strip()]
    normalized = {name.lower(): name for name in requested}

    available_builders = {
        'ct': _build_ct_benchmark,
        'mri': _build_mri_benchmark,
        'skin_integrated': _maybe_build_skin_integrated_benchmark,
        'ophthalmology': _maybe_build_ophthalmology_benchmark,
        'ophthalmology_dr': _maybe_build_ophthalmology_dr_benchmark,
        'scin': _maybe_build_scin_benchmark,
        'ultrasound': _maybe_build_ultrasound_benchmark,
        'xray': _maybe_build_xray_benchmark,
    }

    built = []
    unknown = sorted(set(normalized) - set(available_builders))
    if unknown:
        raise ValueError(
            'Unknown benchmark name(s): ' + ', '.join(unknown) + '. '
            'Known names: ' + ', '.join(sorted(available_builders))
        )

    for name in requested:
        key = name.lower()
        builder = available_builders[key]
        benchmark = builder()
        if benchmark is None:
            raise ValueError(
                f'Benchmark {name!r} was requested in the config but could not be built. '
                'Check its dataset paths / environment variables.'
            )
        built.append(benchmark)

    return built

def build_default_benchmarks() -> List[object]:
    """Return the default benchmark suite used by train_new_pipeline.

    This integration step keeps the existing ultrasound and XRay benchmarks,
    extends the suite with MRI and CT2D classification, and optionally adds
    the skin and ophthalmology benchmarks when their dataset paths are
    configured via environment variables.
    """

    benchmarks: List[object] = [
        _build_mri_benchmark(),
        _build_ct_benchmark(),
    ]

    ultrasound_benchmark = _maybe_build_ultrasound_benchmark()
    if ultrasound_benchmark is not None:
        benchmarks.append(ultrasound_benchmark)

    xray_benchmark = _maybe_build_xray_benchmark()
    if xray_benchmark is not None:
        benchmarks.append(xray_benchmark)

    skin_benchmark = _maybe_build_skin_benchmark()
    if skin_benchmark is not None:
        benchmarks.append(skin_benchmark)

    skin_integrated_benchmark = _maybe_build_skin_integrated_benchmark()
    if skin_integrated_benchmark is not None:
        benchmarks.append(skin_integrated_benchmark)

    ophthalmology_benchmark = _maybe_build_ophthalmology_benchmark()
    if ophthalmology_benchmark is not None:
        benchmarks.append(ophthalmology_benchmark)

    scin_benchmark = _maybe_build_scin_benchmark()
    if scin_benchmark is not None:
        benchmarks.append(scin_benchmark)

    return benchmarks
