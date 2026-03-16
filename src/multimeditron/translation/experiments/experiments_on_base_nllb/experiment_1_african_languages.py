#!/usr/bin/env python3
"""
African Languages Translation Experiment

Compares two translation approaches for African languages:
1) Meditron native multilingual translation (African → English)
2) NLLB translation pipeline (fastText detection → NLLB)

Evaluation:
- BLEU
- chrF
- BERTScore (DeBERTa)

Two-pass execution to avoid GPU OOM:
PASS 1: Meditron
PASS 2: NLLB

Output:
src/multimeditron/translation/experiments/results/base_nllb/experiment_1_results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
import fasttext
from huggingface_hub import hf_hub_download

SRC_ROOT = next((parent for parent in Path(__file__).resolve().parents if parent.name == "src"), None)
if SRC_ROOT is not None and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.translation.translator import NLLBTranslator


ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

LANG_MAP = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}


class TranslationEvaluator:
    def __init__(self):
        self.results = {
            "meditron_direct": defaultdict(list),
            "nllb": defaultdict(list),
        }

        self.det = {
            "total": 0,
            "correct": 0,
            "by_language": defaultdict(lambda: {
                "total": 0,
                "correct": 0,
                "misdetections": Counter()
            })
        }

        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")
        self.bertscore = evaluate.load("bertscore")

    def track_detection(self, true_lang, detected_lang):
        self.det["total"] += 1
        self.det["by_language"][true_lang]["total"] += 1
        if detected_lang == true_lang:
            self.det["correct"] += 1
            self.det["by_language"][true_lang]["correct"] += 1
        else:
            self.det["by_language"][true_lang]["misdetections"][detected_lang] += 1

    def add(self, pipeline, lang, src, pred, ref, detected=None):
        self.results[pipeline][lang].append({
            "source": src,
            "prediction": pred,
            "reference": ref,
            "detected_lang": detected
        })

    def compute(self):
        out = {}
        for pipe, langs in self.results.items():
            out[pipe] = {}
            for lang, rows in langs.items():
                if not rows:
                    continue
                preds = [r["prediction"] for r in rows]
                refs = [[r["reference"]] for r in rows]

                bleu = self.bleu.compute(predictions=preds, references=refs)["bleu"] * 100
                chrf = self.chrf.compute(predictions=preds, references=refs)["score"]
                bert = self.bertscore.compute(
                    predictions=preds,
                    references=[r["reference"] for r in rows],
                    model_type="microsoft/deberta-xlarge-mnli"
                )
                f1 = sum(bert["f1"]) / len(bert["f1"]) * 100

                out[pipe][lang] = {
                    "count": len(rows),
                    "bleu": bleu,
                    "chrf": chrf,
                    "bertscore_f1": f1
                }
        return out

    def save(self, path):
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "detection_stats": self.det,
                "metrics": self.compute(),
                "results": self.results
            }, f, indent=2, ensure_ascii=False)


def extract_translation(text, original):
    if original in text:
        text = text.split(original)[-1]
    return text.strip().strip('"').strip("'")


def translate_with_meditron(model, tokenizer, collator, text):
    if len(text) > 400:
        text = text[:400]

    prompt = f"Translate to English:\n\n{text}"

    sample = {
        "conversations": [{"role": "user", "content": prompt}],
        "modalities": []
    }

    batch = collator([sample])
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            batch=batch,
            max_new_tokens=100,
            do_sample=False
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_translation(decoded, text)


def run_experiment(args):
    dataset = load_dataset(
        "ClosedMeditron/MediTranslation",
        split="train",
        token=os.getenv("HF_LAB_TOKEN")
    )

    lid_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    fasttext_model = fasttext.load_model(lid_path)

    evaluator = TranslationEvaluator()

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        padding_side="left",
        token=os.getenv("HF_PERSONAL_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [ATTACHMENT_TOKEN]})

    model = MultiModalModelForCausalLM.from_pretrained(
        "ClosedMeditron/Mulimeditron-Proj-CLIP-generalist",
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda").eval()

    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        modality_processors=model.processors(),
        modality_loaders={},
        attachment_token=ATTACHMENT_TOKEN,
        chat_template=ChatTemplate.from_name("llama"),
        add_generation_prompt=True
    )

    collected = []

    for lang, (nllb_code, _) in LANG_MAP.items():
        processed = 0
        for row in tqdm(dataset, desc=f"Meditron {lang}"):
            if processed >= args.max_samples:
                break

            src = row.get(lang)
            ref = row.get("en")
            if not src or not ref:
                continue
            if len(src) > 400:
                continue

            det = fasttext_model.predict(src.replace("\n", " "), k=1)[0][0].replace("__label__", "")
            evaluator.track_detection(nllb_code, det)

            try:
                pred = translate_with_meditron(model, tokenizer, collator, src)
                evaluator.add("meditron_direct", lang, src, pred, ref)
                collected.append({
                    "lang": lang,
                    "nllb_code": nllb_code,
                    "source": src,
                    "reference": ref,
                    "detected": det
                })
                processed += 1
            except Exception:
                torch.cuda.empty_cache()
                gc.collect()

    del model, tokenizer, collator
    torch.cuda.empty_cache()
    gc.collect()

    translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")

    for item in tqdm(collected, desc="NLLB"):
        try:
            pred = translator.translate(
                item["source"],
                src_lang=item["nllb_code"],
                tgt_lang="eng_Latn"
            )
            evaluator.add(
                "nllb",
                item["lang"],
                item["source"],
                pred,
                item["reference"],
                detected=item["detected"]
            )
        except Exception:
            torch.cuda.empty_cache()

    evaluator.save(args.output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_1_results.json"
    )

    run_experiment(parser.parse_args())
