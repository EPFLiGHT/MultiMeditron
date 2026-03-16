#!/usr/bin/env python3
"""
Experiment 0 - Meditron Output Language Analysis (High Resource Languages)
Native Meditron evaluation ONLY (no translation) with improved language detection.

Key improvement: Only run language detection on outputs >= 10 characters to avoid
misclassification of single-letter answers.
"""

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
import fasttext

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
from multimeditron.model.data_loader import DataCollatorForMultimodal


ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

ISO_TO_NLLB = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
}

LANG_NAMES = {
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian"
}

# Expected fasttext codes for matching input languages
EXPECTED_FASTTEXT_CODES = {
    "zh": ["zho_Hans", "zho_Hant"],
    "es": ["spa_Latn"],
    "fr": ["fra_Latn"],
    "ja": ["jpn_Jpan"],
    "ru": ["rus_Cyrl"]
}

HIGH_RESOURCE_LANGS = {"zh", "es", "fr", "ja", "ru"}

# Minimum characters for reliable language detection
MIN_CHARS_FOR_LANG_DETECTION = 10


def load_fasttext_model():
    """Load fasttext language identification model from HuggingFace."""
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    return fasttext.load_model(model_path)


def detect_language(text: str, ft_model) -> dict:
    """
    Detect language of text using fasttext.
    Returns TOO_SHORT if text is too short for reliable detection.
    """
    clean_text = text.strip().replace("\n", " ")
    
    if len(clean_text) < MIN_CHARS_FOR_LANG_DETECTION:
        return {
            "detected_lang": "TOO_SHORT",
            "confidence": 0.0,
            "top_3": [],
            "text_length": len(clean_text)
        }
    
    predictions = ft_model.predict(clean_text, k=3)
    labels, scores = predictions
    
    top_3 = [
        {"lang": label.replace("__label__", ""), "confidence": float(score)}
        for label, score in zip(labels, scores)
    ]
    
    return {
        "detected_lang": top_3[0]["lang"] if top_3 else "UNKNOWN",
        "confidence": top_3[0]["confidence"] if top_3 else 0.0,
        "top_3": top_3,
        "text_length": len(clean_text)
    }


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
        out = model.generate(batch=batch, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_letter(text, n):
    text = text.upper()
    valid = {chr(65+i) for i in range(n)}
    for pat in [r"\b([A-Z])\b"]:
        m = re.search(pat, text)
        if m and m.group(1) in valid:
            return m.group(1)
    return "UNKNOWN"


def run(args):
    # Load dataset
    print("Loading MediBench dataset...")
    dataset = load_medibench()

    # Filter and validate samples
    valid = defaultdict(list)
    for s in dataset:
        err = validate_sample(s)
        if err:
            continue
        lang = s.get("language")
        if lang in HIGH_RESOURCE_LANGS:
            valid[lang].append(s)

    # Take samples per language
    samples = []
    samples_per_lang = args.max_samples // len(HIGH_RESOURCE_LANGS)
    for lang in HIGH_RESOURCE_LANGS:
        rows = valid[lang]
        take = min(len(rows), samples_per_lang)
        samples.extend(rows[:take])
    
    print(f"Loaded {len(samples)} samples across {len(HIGH_RESOURCE_LANGS)} languages")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
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

    # Load fasttext for language detection
    print("Loading fasttext language detection model...")
    ft_model = load_fasttext_model()

    # Tracking
    results = []
    lang_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    output_lang_counts = defaultdict(int)
    output_lang_by_input = defaultdict(lambda: defaultdict(int))
    short_responses = 0
    long_responses = 0
    skipped = Counter()

    print(f"\nEvaluating {len(samples)} samples (native only, no translation)...")

    for s in tqdm(samples):
        q = str(s["question"])
        opts = [str(o) for o in s["options"]]
        correct = str(s["answer"]).strip().upper()
        lang = s["language"]

        try:
            # Generate answer (native, no translation)
            raw_output = generate_answer(model, tokenizer, collator, q, opts)
            predicted = extract_letter(raw_output, len(opts))
            is_correct = (correct == predicted)

            # Track accuracy
            lang_stats[lang]["total"] += 1
            if is_correct:
                lang_stats[lang]["correct"] += 1

            # Detect output language
            lang_info = detect_language(raw_output, ft_model)

            # Track response lengths
            if lang_info["detected_lang"] == "TOO_SHORT":
                short_responses += 1
            else:
                long_responses += 1
                output_lang_counts[lang_info["detected_lang"]] += 1
                output_lang_by_input[lang][lang_info["detected_lang"]] += 1

            results.append({
                "input_lang": lang,
                "input_lang_name": LANG_NAMES.get(lang, lang),
                "question": q,
                "options": opts,
                "correct_answer": correct,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "output_length": len(raw_output.strip()),
                "output_language": lang_info
            })

        except Exception as e:
            skipped[f"error_{lang}"] += 1
            gc.collect()
            torch.cuda.empty_cache()

    # Compute statistics
    total_correct = sum(s["correct"] for s in lang_stats.values())
    total_samples = sum(s["total"] for s in lang_stats.values())

    # Build per-language output analysis (only for long responses)
    by_input_lang_analysis = {}
    for lang in HIGH_RESOURCE_LANGS:
        lang_output_dist = dict(output_lang_by_input[lang])
        total_long = sum(lang_output_dist.values())

        # Count matches
        matches = sum(
            lang_output_dist.get(code, 0)
            for code in EXPECTED_FASTTEXT_CODES.get(lang, [])
        )
        english_count = lang_output_dist.get("eng_Latn", 0)

        by_input_lang_analysis[lang] = {
            "language_name": LANG_NAMES.get(lang, lang),
            "total_samples": lang_stats[lang]["total"],
            "long_responses": total_long,
            "short_responses": lang_stats[lang]["total"] - total_long,
            "output_language_distribution": lang_output_dist,
            "matches_input_lang_pct": round(100 * matches / total_long, 1) if total_long > 0 else 0,
            "responds_in_english_pct": round(100 * english_count / total_long, 1) if total_long > 0 else 0,
            "examples": [
                {
                    "question_preview": r["question"][:100] + "...",
                    "raw_output": r["raw_output"],
                    "output_length": r["output_length"],
                    "detected_output_lang": r["output_language"]["detected_lang"],
                    "confidence": r["output_language"]["confidence"]
                }
                for r in results
                if r["input_lang"] == lang and r["output_language"]["detected_lang"] != "TOO_SHORT"
            ][:3]
        }

    # Summary output
    output_data = {
        "experiment": "Meditron Output Language Analysis - High Resource Languages (v2)",
        "description": "Native Meditron evaluation with IMPROVED language detection (min 10 chars)",
        "model": "ClosedMeditron/Mulimeditron-Proj-CLIP-generalist",
        "min_chars_for_detection": MIN_CHARS_FOR_LANG_DETECTION,
        "languages": list(HIGH_RESOURCE_LANGS),
        "statistics": {
            "accuracy": {
                "overall_accuracy": round(100 * total_correct / total_samples, 1) if total_samples else 0,
                "total": total_samples,
                "correct": total_correct,
                "by_input_language": {
                    lang: {
                        "accuracy": round(100 * stats["correct"] / stats["total"], 1) if stats["total"] else 0,
                        "language_name": LANG_NAMES.get(lang, lang),
                        "total": stats["total"],
                        "correct": stats["correct"]
                    }
                    for lang, stats in lang_stats.items()
                }
            },
            "response_length_summary": {
                "short_responses": short_responses,
                "long_responses": long_responses,
                "short_pct": round(100 * short_responses / total_samples, 1) if total_samples else 0,
                "long_pct": round(100 * long_responses / total_samples, 1) if total_samples else 0
            },
            "output_language_analysis": {
                "note": "Language detection only performed on responses >= 10 characters",
                "overall_output_language_distribution": dict(output_lang_counts),
                "overall_output_language_pct": {
                    lang: round(100 * count / long_responses, 1)
                    for lang, count in output_lang_counts.items()
                } if long_responses > 0 else {},
                "by_input_language": by_input_lang_analysis
            },
            "skipped": dict(skipped)
        },
        "detailed_results": results
    }

    # Save
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {output_data['statistics']['accuracy']['overall_accuracy']}%")
    print(f"\nResponse Length Distribution:")
    print(f"  Short (<{MIN_CHARS_FOR_LANG_DETECTION} chars): {short_responses} ({output_data['statistics']['response_length_summary']['short_pct']}%)")
    print(f"  Long (>={MIN_CHARS_FOR_LANG_DETECTION} chars): {long_responses} ({output_data['statistics']['response_length_summary']['long_pct']}%)")
    print(f"\nAccuracy by Language:")
    for lang, stats in lang_stats.items():
        acc = round(100 * stats["correct"] / stats["total"], 1) if stats["total"] else 0
        print(f"  {LANG_NAMES.get(lang, lang)}: {acc}%")
    print(f"\nOutput Language Distribution (long responses only):")
    for lang, count in sorted(output_lang_counts.items(), key=lambda x: -x[1])[:10]:
        pct = round(100 * count / long_responses, 1) if long_responses > 0 else 0
        print(f"  {lang}: {count} ({pct}%)")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/analysis_output/output_language_analysis_experiment_0"
    )
    run(ap.parse_args())
