#!/usr/bin/env python3
"""
Experiment 3 - Meditron Output Language Analysis (African Languages)
Native Meditron evaluation ONLY (no translation) with improved language detection.

Uses the same model setup as experiment_0 (high-resource languages).
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
from huggingface_hub import hf_hub_download
import fasttext

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
from multimeditron.model.data_loader import DataCollatorForMultimodal


ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Minimum characters for reliable language detection
MIN_CHARS_FOR_LANG_DETECTION = 10

LANG_NAMES = {
    "am": "Amharic",
    "ha": "Hausa",
    "sw": "Swahili",
    "yo": "Yoruba",
    "zu": "Zulu"
}

EXPECTED_FASTTEXT_CODES = {
    "am": ["amh_Ethi"],
    "ha": ["hau_Latn"],
    "sw": ["swh_Latn"],
    "yo": ["yor_Latn"],
    "zu": ["zul_Latn"]
}

AFRICAN_LANGUAGES = {"am", "ha", "sw", "yo", "zu"}


def load_fasttext_model():
    """Load fasttext language identification model from HuggingFace."""
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    return fasttext.load_model(model_path)


def detect_language(text: str, ft_model) -> dict:
    """Detect language of text using fasttext."""
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


def load_african_data(json_path: str, languages: list, max_per_lang: int = 40):
    """Load pre-translated African language data from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    lang_data = defaultdict(list)
    
    for item in data:
        lang = item.get("language")
        if lang in languages and len(lang_data[lang]) < max_per_lang:
            lang_data[lang].append({
                "question": item["question"],
                "options": item["options"],
                "answer": item.get("answer", "A"),
                "lang": lang,
                "lang_name": LANG_NAMES.get(lang, lang),
                "source_language": item.get("source_language", "")
            })
    
    all_samples = []
    for lang in languages:
        all_samples.extend(lang_data[lang])
    
    return all_samples


def format_mcq(q, opts):
    """Format question and options as MCQ prompt."""
    out = q + "\n\nOptions:\n"
    for i, o in enumerate(opts):
        out += f"{chr(65+i)}. {o}\n"
    out += "\nAnswer with a single letter."
    return out


def generate_answer(model, tokenizer, collator, question, options):
    """Generate answer using the model."""
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
    """Extract answer letter from model output."""
    text = text.upper()
    valid = {chr(65+i) for i in range(n)}
    for pat in [r"\b([A-Z])\b"]:
        m = re.search(pat, text)
        if m and m.group(1) in valid:
            return m.group(1)
    return "UNKNOWN"


def run(args):
    # Load dataset
    print("Loading African languages dataset...")
    languages = list(AFRICAN_LANGUAGES)
    samples_per_lang = args.max_samples // len(languages)
    
    samples = load_african_data(args.input_json, languages, max_per_lang=samples_per_lang)
    print(f"Loaded {len(samples)} samples across {len(languages)} languages")
    
    if len(samples) == 0:
        print("ERROR: No samples loaded! Check input file path.")
        return

    # Load model and tokenizer (same as experiment_0)
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
        use_safetensors=True,
        token=os.getenv("HF_LAB_TOKEN")
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

    for s in tqdm(samples, desc="Processing"):
        q = str(s["question"])
        opts = [str(o) for o in s["options"]]
        correct = str(s["answer"]).strip().upper()
        lang = s["lang"]

        try:
            raw_output = generate_answer(model, tokenizer, collator, q, opts)
            predicted = extract_letter(raw_output, len(opts))
            is_correct = (correct == predicted)

            lang_stats[lang]["total"] += 1
            if is_correct:
                lang_stats[lang]["correct"] += 1

            lang_info = detect_language(raw_output, ft_model)

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

    # Build per-language output analysis
    by_input_lang_analysis = {}
    for lang in languages:
        lang_output_dist = dict(output_lang_by_input[lang])
        total_long = sum(lang_output_dist.values())

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
        "experiment": "Meditron Output Language Analysis - African Languages (v2)",
        "description": "Native Meditron evaluation with IMPROVED language detection (min 10 chars)",
        "model": "ClosedMeditron/Mulimeditron-Proj-CLIP-generalist",
        "min_chars_for_detection": MIN_CHARS_FOR_LANG_DETECTION,
        "languages": languages,
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
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / "african_langs_meditron_only_v2.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
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
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--input_json", type=str,
                    default="src/multimeditron/translation/experiments/results/base_nllb/mediBench_translation/cleaned_high_resource_translation_results.json")
    ap.add_argument("--output_dir", type=str,
                    default="src/multimeditron/translation/experiments/results/base_nllb/analysis_output/output_language_analysis_experiment_3")
    run(ap.parse_args())