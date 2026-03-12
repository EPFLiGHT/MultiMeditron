#!/usr/bin/env python3

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.translation.translator import NLLBTranslator


ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

ISO_TO_NLLB = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
}

HIGH_RESOURCE_LANGS = {"zh", "es", "fr", "ja", "ru"}


def load_medibench():
    try:
        return load_dataset(
            "ClosedMeditron/MediBench",
            split="train",
            token=os.getenv("HF_LAB_TOKEN")
        )
    except Exception:
        path = hf_hub_download(
            repo_id="ClosedMeditron/MediBench",
            filename="train.jsonl",
            repo_type="dataset",
            token=os.getenv("HF_LAB_TOKEN")
        )
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if isinstance(obj.get("answer"), list):
                    obj["answer"] = obj["answer"][0] if obj["answer"] else None
                rows.append(obj)
        return Dataset.from_list(rows)


def validate_sample(sample):
    if not sample.get("question"):
        return "missing_question"
    if not sample.get("answer"):
        return "missing_answer"
    opts = sample.get("options")
    if not opts or not isinstance(opts, list):
        return "missing_options"
    ans = str(sample["answer"]).strip().upper()
    if len(ans) != 1:
        return "invalid_answer"
    if ord(ans) - 65 < 0 or ord(ans) - 65 >= len(opts):
        return "answer_out_of_range"
    return None


def format_mcq(q, opts):
    out = q + "\n\nOptions:\n"
    for i, o in enumerate(opts):
        out += f"{chr(65+i)}. {o}\n"
    out += "\nAnswer with a single letter."
    return out


def generate_answer(model, tokenizer, collator, question, options):
    prompt = format_mcq(question, options)
    sample = {
        "conversations": [{"role": "user", "content": prompt}],
        "modalities": []
    }
    batch = collator([sample])
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(batch=batch, max_new_tokens=8, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_letter(text, n):
    text = text.upper()
    valid = {chr(65+i) for i in range(n)}
    for pat in [r"\b([A-Z])\b"]:
        m = re.search(pat, text)
        if m and m.group(1) in valid:
            return m.group(1)
    return "UNKNOWN"


class Evaluator:
    def __init__(self):
        self.res = {"translation": [], "native": []}
        self.skipped = Counter()
        self.fasttext = Counter()

    def add(self, pipe, lang, correct, pred):
        self.res[pipe].append({
            "lang": lang,
            "correct": correct,
            "predicted": pred,
            "ok": correct == pred
        })

    def stats(self):
        out = {}
        for p, rows in self.res.items():
            tot = len(rows)
            cor = sum(r["ok"] for r in rows)
            by = defaultdict(lambda: {"total": 0, "correct": 0})
            for r in rows:
                by[r["lang"]]["total"] += 1
                if r["ok"]:
                    by[r["lang"]]["correct"] += 1
            out[p] = {
                "accuracy": 100 * cor / tot if tot else 0,
                "total": tot,
                "correct": cor,
                "by_language": {
                    k: {
                        "accuracy": 100 * v["correct"] / v["total"] if v["total"] else 0,
                        **v
                    }
                    for k, v in by.items()
                }
            }
        return out

    def save(self, path):
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stats": self.stats(),
                "results": self.res,
                "skipped": dict(self.skipped)
            }, f, indent=2)


def run(args):
    dataset = load_medibench()

    valid = defaultdict(list)
    for s in dataset:
        err = validate_sample(s)
        if err:
            continue
        lang = s.get("language")
        if lang in HIGH_RESOURCE_LANGS:
            valid[lang].append(s)

    samples = []
    for lang, rows in valid.items():
        take = min(len(rows), args.max_samples)
        samples.extend(rows[:take])

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

    translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")
    evaluator = Evaluator()

    for s in tqdm(samples):
        q = str(s["question"])
        opts = [str(o) for o in s["options"]]
        correct = str(s["answer"]).strip().upper()
        lang = s["language"]
        nllb_lang = ISO_TO_NLLB.get(lang, "eng_Latn")

        try:
            q_en = translator.translate(q, src_lang=nllb_lang, tgt_lang="eng_Latn")
            opts_en = [translator.translate(o, src_lang=nllb_lang, tgt_lang="eng_Latn") for o in opts]
            r_t = generate_answer(model, tokenizer, collator, q_en, opts_en)
            p_t = extract_letter(r_t, len(opts))
            evaluator.add("translation", lang, correct, p_t)

            r_n = generate_answer(model, tokenizer, collator, q, opts)
            p_n = extract_letter(r_n, len(opts))
            evaluator.add("native", lang, correct, p_n)

        except Exception:
            gc.collect()
            torch.cuda.empty_cache()

    evaluator.save(args.output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_0_results.json"
    )
    run(ap.parse_args())
