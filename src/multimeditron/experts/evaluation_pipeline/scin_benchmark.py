from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from transformers import CLIPModel, CLIPProcessor

try:
    from .Benchmark import Benchmark
    from .hard_benchmark_scin_tone_stratified import (
        HARD_TOPK,
        REF_MODEL_NAME,
        SEED,
        build_hard_protocol_from_embeds,
        compute_image_embeds,
        evaluate,
        extract_fst,
        extract_top_differential,
        fst_to_group,
        load_manifest_lookup,
        resolve_many,
    )
except ImportError:
    from Benchmark import Benchmark
    from hard_benchmark_scin_tone_stratified import (
        HARD_TOPK,
        REF_MODEL_NAME,
        SEED,
        build_hard_protocol_from_embeds,
        compute_image_embeds,
        evaluate,
        extract_fst,
        extract_top_differential,
        fst_to_group,
        load_manifest_lookup,
        resolve_many,
    )

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]


def _macro_mean_group_score(scores: dict[str, float]) -> float:
    preferred_groups = [group for group in ("light", "medium", "dark") if group in scores]
    groups = preferred_groups or list(scores.keys())
    if not groups:
        raise ValueError("SCIN benchmark produced no group scores.")
    return float(sum(scores[group] for group in groups) / len(groups))


class SCINBenchmark(Benchmark):
    """SCIN hard-negative retrieval benchmark with skin-tone stratified reporting.

    The scalar value returned to Optuna is the macro-average Recall@1 over the
    available Fitzpatrick groups light/medium/dark. Detailed per-group metrics
    are also written next to the trained model for inspection.
    """

    default_eval_jsonls = (
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/SCIN/scin_api_val.jsonl'),
    )
    default_manifest_jsonls = (
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/SCIN/scin_manifest.jsonl'),
    )
    default_protocol_cache_path = REPO_ROOT / 'benchmark_cache' / 'scin_hard_protocol.json'

    def __init__(
        self,
        eval_jsonls: Sequence[str | Path] | None = None,
        manifest_jsonls: Sequence[str | Path] | None = None,
        protocol_cache_path: str | Path | None = None,
        ref_model_name: str = REF_MODEL_NAME,
        seed: int = SEED,
        hard_topk: int = HARD_TOPK,
    ) -> None:
        chosen_eval = eval_jsonls if eval_jsonls is not None else self.default_eval_jsonls
        chosen_manifest = manifest_jsonls if manifest_jsonls is not None else self.default_manifest_jsonls

        self.eval_jsonls = [str(Path(p)) for p in chosen_eval]
        self.manifest_jsonls = [str(Path(p)) for p in chosen_manifest]
        chosen_cache = protocol_cache_path if protocol_cache_path is not None else self.default_protocol_cache_path
        self.protocol_cache_path = Path(chosen_cache) if chosen_cache else None
        self.ref_model_name = ref_model_name
        self.seed = seed
        self.hard_topk = hard_topk

        self._validate_paths(self.eval_jsonls, 'SCIN eval jsonls')
        self._validate_paths(self.manifest_jsonls, 'SCIN manifest jsonls')

        self.items = self._load_items()
        self.triples = self._load_or_build_protocol()

    def _validate_paths(self, paths: Sequence[str], label: str) -> None:
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f'{label} do not exist: ' + ', '.join(missing))

    def _load_items(self) -> list[dict]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        items = resolve_many(self.eval_jsonls)
        fst_map, diff_map = load_manifest_lookup(self.manifest_jsonls)

        for item in items:
            fname = Path(item["modalities"][0]["value"]).name
            text = item.get("text", "")

            fst = extract_fst(text) or fst_map.get(fname)
            diff = extract_top_differential(text) or diff_map.get(fname)

            item["_fst"] = fst
            item["_skin_group"] = fst_to_group(fst)
            item["_proxy_disease"] = diff or "unknown"

        logger.info("SCIN skin-tone groups: %s", Counter(item["_skin_group"] for item in items))
        logger.info("SCIN Fitzpatrick labels: %s", Counter(item["_fst"] for item in items))
        return items

    def _load_or_build_protocol(self) -> list[tuple[int, int, int, int]]:
        if self.protocol_cache_path and self.protocol_cache_path.exists():
            with self.protocol_cache_path.open("r", encoding="utf-8") as f:
                return [tuple(entry) for entry in json.load(f)]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Building SCIN hard protocol with reference model: %s", self.ref_model_name)

        ref_model = CLIPModel.from_pretrained(self.ref_model_name).to(device).eval()
        ref_processor = CLIPProcessor.from_pretrained(self.ref_model_name)
        ref_img_embeds = compute_image_embeds(ref_model, ref_processor, device, self.items)
        triples = build_hard_protocol_from_embeds(
            self.items,
            ref_img_embeds,
            seed=self.seed,
            top_k=self.hard_topk,
        )

        if self.protocol_cache_path:
            self.protocol_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.protocol_cache_path.open("w", encoding="utf-8") as f:
                json.dump(triples, f)

        return triples

    def evaluate(self, model_path) -> float:
        group_scores = evaluate(model_path, self.items, self.triples)
        scalar_score = _macro_mean_group_score(group_scores)

        metrics_path = Path(model_path) / "scin_benchmark_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": "macro_recall_at_1",
                    "score": scalar_score,
                    "group_scores": group_scores,
                },
                f,
                indent=2,
            )

        logger.info("SCIN benchmark score: %.6f | groups=%s", scalar_score, group_scores)
        return scalar_score
