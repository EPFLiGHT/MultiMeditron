from __future__ import annotations

import json
from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset, read_jsonl

CT3D_2D_ROOT = Path("/lightscratch/users/cljordan/datasets/CT3D-2D")
CT3D_REGION_ROOT = Path("/lightscratch/users/nemo/datasets/CT_data/CT3D")

_LABELS = ["chest", "abdomen", "head", "other"]
_LABEL_TO_IDX = {label: idx for idx, label in enumerate(_LABELS)}

_REGION_FILES = {
    "chest": CT3D_REGION_ROOT / "chest.jsonl",
    "abdomen": CT3D_REGION_ROOT / "abdomen.jsonl",
    "head": CT3D_REGION_ROOT / "head.jsonl",
    "other": CT3D_REGION_ROOT / "arms.jsonl",
    "other_leg": CT3D_REGION_ROOT / "leg.jsonl",
    "other_misc": CT3D_REGION_ROOT / "other.jsonl",
}


def _png_to_volume_value(img_relpath: str) -> str:
    """Reverse the encoding done by build_ct3d_2d_dataset.py.

    images/ct_quizze_36__014458.png  →  data/ct_quizze_36/014458
    """
    return "data/" + Path(img_relpath).stem.replace("__", "/")


def _build_label_map() -> dict[str, int]:
    """Map volume value (e.g. data/ct_quizze_36/014458) to label index."""
    label_map: dict[str, int] = {}
    for key, path in _REGION_FILES.items():
        if not path.exists():
            continue
        idx = _LABEL_TO_IDX[key if key in _LABEL_TO_IDX else "other"]
        with path.open() as f:
            for line in f:
                v = json.loads(line)["modalities"][0]["value"]
                label_map[v] = idx
    return label_map


class CT3DBenchmark(ClassificationBenchmark):
    """Body-region classification on CT3D (chest / abdomen / head / other).

    Uses pre-extracted 2D axial slices (PNG) from CT3D-2D — same loading
    pattern as UltrasoundBenchmark (encode_img on JPEG/PNG files).
    Labels come from the CT3D region JSONL files via a path reverse-mapping.
    Train/test splits are CT3D-train.jsonl / CT3D-test.jsonl in CT3D_2D_ROOT.
    """

    name = "ct3d"
    num_classes = 4
    labels = _LABELS
    label_to_idx = _LABEL_TO_IDX

    max_train_examples = 10_000
    max_test_examples = 3_000

    _train_jsonl = CT3D_2D_ROOT / "CT3D-train.jsonl"
    _test_jsonl = CT3D_2D_ROOT / "CT3D-test.jsonl"

    def __init__(self, cache_root=None, max_train_examples=None, max_test_examples=None):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        self._label_map: dict[str, int] | None = None

    def _get_label_map(self) -> dict[str, int]:
        if self._label_map is None:
            self._label_map = _build_label_map()
        return self._label_map

    def _load_labeled_split(self, jsonl_path: Path) -> list[dict]:
        """Return only examples that have a region label."""
        label_map = self._get_label_map()
        return [
            ex for ex in read_jsonl(jsonl_path)
            if _png_to_volume_value(ex["modalities"][0]["value"]) in label_map
        ]

    def _labels_for(self, examples: list[dict]) -> list[int]:
        label_map = self._get_label_map()
        return [label_map[_png_to_volume_value(ex["modalities"][0]["value"])] for ex in examples]

    def build_train_dataset(self, model, model_name, use_cache=True):
        examples = self._load_labeled_split(self._train_jsonl)
        examples = self._sample_examples_random(examples, self.max_train_examples, seed=42)
        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_train",
            examples=examples,
            labels=self._labels_for(examples),
            model=model,
            dataset_root=CT3D_2D_ROOT,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ct3d-train",
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        examples = self._load_labeled_split(self._test_jsonl)
        examples = self._sample_examples_random(examples, self.max_test_examples, seed=43)
        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_test",
            examples=examples,
            labels=self._labels_for(examples),
            model=model,
            dataset_root=CT3D_2D_ROOT,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ct3d-test",
        )
