#!/usr/bin/env python3
"""
High-Resource Language Translation Experiment

Compares Meditron native translation vs NLLB for high-resource languages
on medical questions from MediBench.

Evaluation:
- NLLB used as pseudo-reference
- BLEU, chrF, BERTScore
- Medical term preservation

Runs in two passes to avoid GPU OOM:
1) Meditron translation
2) NLLB translation

Output:
src/multimeditron/translation/experiments/results/base_nllb/experiment_2_results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
import evaluate
import fasttext
from huggingface_hub import hf_hub_download
from bert_score import score as bert_score

SRC_ROOT = next((parent for parent in Path(__file__).resolve().parents if parent.name == "src"), None)
if SRC_ROOT is not None and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.translation.translator import NLLBTranslator


HIGH_RESOURCE_LANGS = {
    'zh': ('zho_Hans', 'Chinese'),
    'es': ('spa_Latn', 'Spanish'),
    'fr': ('fra_Latn', 'French'),
    'ja': ('jpn_Jpan', 'Japanese'),
    'ru': ('rus_Cyrl', 'Russian'),
}

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"


def load_medibench(split="train"):
    path = hf_hub_download(
        repo_id="ClosedMeditron/MediBench",
        filename=f"{split}.jsonl",
        repo_type="dataset",
        token=os.getenv("HF_LAB_TOKEN")
    )

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = json.loads(line)
                if isinstance(s.get("answer"), list):
                    s["answer"] = s["answer"][0] if s["answer"] else None
                if isinstance(s.get("question"), list):
                    s["question"] = " ".join(map(str, s["question"]))
                samples.append(s)
            except Exception:
                continue

    return Dataset.from_list(samples)


class HighResourceEvaluator:
    def __init__(self):
        self.results = defaultdict(list)
        self.det = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": Counter()})
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")

    def track_detection(self, true_lang, detected):
        self.det[true_lang]["total"] += 1
        if detected == true_lang:
            self.det[true_lang]["correct"] += 1
        else:
            self.det[true_lang]["wrong"][detected] += 1

    def add(self, lang, src, med, ref):
        self.results[lang].append({
            "source": src,
            "meditron": med,
            "nllb": ref
        })

    def metrics(self):
        out = {}
        for lang, rows in self.results.items():
            preds = [r["meditron"] for r in rows]
            refs = [[r["nllb"]] for r in rows]

            bleu = self.bleu.compute(predictions=preds, references=refs)["bleu"] * 100
            chrf = self.chrf.compute(predictions=preds, references=refs)["score"]

            _, _, f1 = bert_score(
                preds,
                [r["nllb"] for r in rows],
                lang="en",
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False
            )

            out[lang] = {
                "count": len(rows),
                "bleu": bleu,
                "chrf": chrf,
                "bertscore_f1": f1.mean().item() * 100
            }
        return out

    def save(self, path):
        metrics = self.metrics()
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)


def extract_translation(text):
    m = re.search(r'translation\s*:\s*"(.*?)"', text, re.I | re.S)
    return m.group(1).strip() if m else text.strip()


def translate_with_meditron(model, tokenizer, collator, text):
    prompt = (
        "Translate the following medical text into English.\n"
        "Return only:\ntranslation: \"...\"\n\n"
        f"{text}"
    )

    sample = {
        "conversations": [{"role": "user", "content": prompt}],
        "modalities": []
    }

    batch = collator([sample])
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(batch=batch, max_new_tokens=150, do_sample=False)

    return extract_translation(tokenizer.decode(out[0], skip_special_tokens=True))


def run_experiment(args):
    dataset = load_medibench("train")
    langs = set(dataset["language"])
    langs = {k: v for k, v in HIGH_RESOURCE_LANGS.items() if k in langs}

    lid_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    ft = fasttext.load_model(lid_path)

    evaluator = HighResourceEvaluator()

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

    for lang, (nllb_code, _) in langs.items():
        samples = [s for s in dataset if s["language"] == lang][:args.max_samples]

        for s in tqdm(samples, desc=f"Meditron {lang}"):
            q = s.get("question", "")
            if not q or len(q) > 400:
                continue

            det = ft.predict(q.replace("\n", " "), k=1)[0][0].replace("__label__", "")
            evaluator.track_detection(nllb_code, det)

            try:
                tr = translate_with_meditron(model, tokenizer, collator, q)
                collected.append({
                    "lang": lang,
                    "nllb_code": nllb_code,
                    "source": q,
                    "meditron": tr
                })
            except Exception:
                torch.cuda.empty_cache()
                gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")

    for s in tqdm(collected, desc="NLLB"):
        try:
            ref = translator.translate(
                s["source"],
                src_lang=s["nllb_code"],
                tgt_lang="eng_Latn"
            )
            evaluator.add(s["lang"], s["source"], s["meditron"], ref)
        except Exception:
            torch.cuda.empty_cache()

    evaluator.save(args.output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_2_results.json"
    )

    run_experiment(parser.parse_args())
