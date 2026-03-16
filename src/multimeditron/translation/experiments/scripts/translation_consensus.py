#!/usr/bin/env python3
"""
Multi-model translation pipeline for low-resource African/Asian languages.
Uses TranslateGemma-4B (for supported languages), Qwen3-4B (for unsupported), 
M2M100, and MADLAD-400 with consensus selection.

TranslateGemma supports: Swahili (sw), Tamil (ta), Zulu (zu)
Qwen3 used for: Amharic (amh), Hausa (hau), Yoruba (yor)

v4: Added checkpointing to resume after pod restarts
"""

import json
import gc
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    AutoModelForCausalLM
)
import torch
from sacrebleu.metrics import BLEU, CHRF

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed, using fallback English detection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Flush output immediately for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# TranslateGemma supported languages (from WMT24++ 55 languages)
TRANSLATEGEMMA_SUPPORTED = {"swa", "tam", "zul"}  # sw, ta, zu
QWEN3_LANGUAGES = {"amh", "hau", "yor"}  # Not in TranslateGemma


def log(msg: str):
    """Print with timestamp and flush."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def is_english(text: str, threshold: float = 0.7) -> bool:
    """Check if text is primarily English using langdetect or word frequency heuristic."""
    if not text or len(text.strip()) < 10:
        return True
    
    text = text.strip()
    
    if LANGDETECT_AVAILABLE:
        try:
            return detect(text) == 'en'
        except LangDetectException:
            pass
    
    # Fallback: count common English words
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
        'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over'
    }
    
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return False
    
    english_count = sum(1 for w in words if w in common_words)
    return (english_count / len(words)) > threshold


def detect_repetition_strict(text: str, word_threshold: float = 0.5, ngram_threshold: int = 8) -> bool:
    """
    Check for excessive word or phrase repetition - ONLY for LLMs like Qwen3.
    Uses relaxed thresholds to avoid false positives.
    
    NOT used for dedicated translation models (TranslateGemma, M2M100, MADLAD).
    """
    if not text or len(text.strip()) < 30:
        return False
    
    words = text.strip().split()
    if len(words) < 10:
        return False
    
    # Check single word dominance (very relaxed - 50% threshold)
    word_counts = {}
    for word in words:
        word_lower = word.lower()
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    if max(word_counts.values()) / len(words) > word_threshold:
        return True
    
    # Check n-gram repetition (very relaxed - 8+ occurrences)
    for n in [3, 4]:  # Only check 3-grams and 4-grams
        if len(words) < n * 2:
            continue
        
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        
        if ngrams and max(ngrams.values()) >= ngram_threshold:
            return True
    
    # Check obvious character repetition (e.g., "aaaaaaa")
    if re.search(r'(.)\1{10,}', text):  # 10+ consecutive same chars
        return True
    
    return False


def clean_llm_wrapper_text(text: str) -> str:
    """Remove common LLM preambles and formatting."""
    if not text:
        return text
    
    text = text.strip()
    
    # Remove thinking blocks if present (Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    
    patterns = [
        r'^here\s+is\s+the\s+translation:?\s*',
        r'^here\'s\s+the\s+translation:?\s*',
        r'^translation:?\s*',
        r'^sure,?\s+here\s+is\s+',
        r'^sure,?\s+here\'s\s+',
        r'^certainly,?\s*',
        r'^of\s+course,?\s*',
        r'^yes,?\s+i\s+can\s+translate\s+',
        r'^yes,?\s+',
        r'^the\s+translation\s+is:?\s*',
        r'^okay,?\s*',
        r'^alright,?\s*',
        r'^i\s+will\s+translate\s+',
        r'^i\'ll\s+translate\s+',
        r'^translating\s+to\s+\w+:?\s*',
        r'^in\s+\w+:?\s*',
        r'^translated:?\s*',
        r'^output:?\s*',
        r'^result:?\s*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = text.strip()
    
    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    # Clean markdown
    text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text)
    text = re.sub(r'^`(.+)`$', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================
# Checkpointing Functions
# ============================================================

def get_checkpoint_path(output_dir: Path, lang: str, model_name: str) -> Path:
    """Get path for model-specific checkpoint file."""
    return output_dir / f".checkpoint_{lang}_{model_name}.jsonl"


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[str], List[str], int]:
    """Load translations and statuses from checkpoint file."""
    translations = []
    statuses = []
    
    if not checkpoint_path.exists():
        return translations, statuses, 0
    
    with open(checkpoint_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            translations.append(entry['translation'])
            statuses.append(entry['status'])
    
    return translations, statuses, len(translations)


def save_checkpoint_entry(checkpoint_path: Path, translation: str, status: str):
    """Append a single translation to checkpoint file."""
    with open(checkpoint_path, 'a') as f:
        f.write(json.dumps({'translation': translation, 'status': status}, ensure_ascii=False) + '\n')


def clear_checkpoint(checkpoint_path: Path):
    """Remove checkpoint file after successful completion."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# ============================================================
# TranslateGemma-4B (for Swahili, Tamil, Zulu)
# ============================================================

def translate_single_translategemma(text: str, target_lang: str, model, processor,
                                    tgt_lang_code: str, sample_idx: int = 0,
                                    max_attempts: int = 3) -> Tuple[str, str]:
    """Translate text with TranslateGemma-4B. No repetition check needed."""
    
    if sample_idx == 0:
        log("Starting first TranslateGemma inference...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "en",
                            "target_lang_code": tgt_lang_code,
                            "text": text,
                        }
                    ],
                }
            ]
            
            inputs = processor.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_len = len(inputs['input_ids'][0])
            
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=False
                )
            
            generation = generation[0][input_len:]
            result = processor.decode(generation, skip_special_tokens=True).strip()
            
            if sample_idx == 0 and attempt == 1:
                log("First TranslateGemma inference complete!")
            
            if not result or len(result) < 3:
                if attempt < max_attempts:
                    continue
                return "", "error"
            
            # Only check if output is English (failed translation)
            # No repetition check - TranslateGemma is a dedicated translation model
            if is_english(result):
                if attempt < max_attempts:
                    continue
                return result, "english"
            
            return result, "success"
            
        except Exception as e:
            if sample_idx < 3:
                log(f"TranslateGemma sample {sample_idx} attempt {attempt} error: {str(e)[:100]}")
            if attempt < max_attempts:
                continue
            return "", "error"
    
    return "", "error"


def translate_with_translategemma(texts: List[str], target_lang: str, 
                                  output_dir: Path) -> Tuple[List[str], List[str]]:
    """Batch translate with TranslateGemma-4B with checkpointing."""
    
    # Check for existing checkpoint
    checkpoint_path = get_checkpoint_path(output_dir, target_lang, "translategemma")
    translations, statuses, start_idx = load_checkpoint(checkpoint_path)
    
    if start_idx > 0:
        log(f"[1/3] Resuming TranslateGemma-4B from index {start_idx}/{len(texts)}...")
    else:
        log("[1/3] Loading TranslateGemma-4B...")
    
    # Language code mapping for TranslateGemma (ISO 639-1)
    lang_codes = {
        "swa": "sw",  # Swahili
        "tam": "ta",  # Tamil
        "zul": "zu",  # Zulu
    }
    tgt_lang_code = lang_codes.get(target_lang, target_lang)
    
    lang_names = {
        "swa": "Swahili", "tam": "Tamil", "zul": "Zulu"
    }
    lang_name = lang_names.get(target_lang, target_lang)
    
    # If already complete, return cached results
    if start_idx >= len(texts):
        log(f"TranslateGemma already complete for {lang_name}! Using cached {len(translations)} translations.")
        return translations, statuses
    
    model_id = "google/translategemma-4b-it"
    log(f"Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    log(f"Loading model from {model_id}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    log(f"Model loaded! Translating {len(texts) - start_idx} remaining texts to {lang_name}...")
    
    status_counts = {"success": 0, "english": 0, "error": 0}
    # Count existing statuses
    for s in statuses:
        if s in status_counts:
            status_counts[s] += 1
    
    for i in tqdm(range(start_idx, len(texts)), desc="TranslateGemma", initial=start_idx, total=len(texts)):
        text = texts[i]
        translation, status = translate_single_translategemma(
            text, target_lang, model, processor, tgt_lang_code, sample_idx=i
        )
        translations.append(translation)
        statuses.append(status)
        status_counts[status] += 1
        
        # Save checkpoint after each translation
        save_checkpoint_entry(checkpoint_path, translation, status)
        
        # Log first 5 individually, then every 10, then every 50 after 100
        if i < 5 or (i < 100 and (i + 1) % 10 == 0) or (i + 1) % 50 == 0:
            log(f"TranslateGemma [{i+1}/{len(texts)}] ✓:{status_counts['success']} "
                f"eng:{status_counts['english']} err:{status_counts['error']}")
    
    del model, processor
    clear_gpu_memory()
    
    log(f"TranslateGemma DONE: {status_counts['success']} ok, {status_counts['english']} english, "
        f"{status_counts['error']} errors")
    
    return translations, statuses


# ============================================================
# Qwen3-4B (for Amharic, Hausa, Yoruba)
# ============================================================

def translate_single_qwen3(text: str, target_lang: str, model, tokenizer, 
                          lang_name: str, sample_idx: int = 0, 
                          max_attempts: int = 3) -> Tuple[str, str]:
    """Translate text with Qwen3, retrying on failures. Includes relaxed repetition check."""
    
    if sample_idx == 0:
        log("Starting first Qwen3 inference (CUDA warmup may take 1-2 min)...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            system_msg = (
                f"You are a professional translator. "
                f"Translate the following English medical text to {lang_name}. "
                f"Output ONLY the {lang_name} translation, nothing else. "
                f"Do not add explanations, acknowledgments, or any English text."
            )
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text}
            ]
            
            # Use enable_thinking=False for faster inference (no reasoning)
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Non-thinking mode recommended params from Qwen3 docs
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3
            )
            
            if sample_idx == 0 and attempt == 1:
                log("First Qwen3 inference complete!")
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            result = clean_llm_wrapper_text(result)
            
            if not result or len(result) < 3:
                if attempt < max_attempts:
                    continue
                return "", "error"
            
            if is_english(result):
                if attempt < max_attempts:
                    continue
                return result, "english"
            
            # Relaxed repetition check only for Qwen3 (LLM, not dedicated translator)
            if detect_repetition_strict(result):
                if attempt < max_attempts:
                    continue
                return result, "repetitive"
            
            return result, "success"
            
        except Exception as e:
            if sample_idx < 3:
                log(f"Qwen3 sample {sample_idx} attempt {attempt} error: {str(e)[:100]}")
            if attempt < max_attempts:
                continue
            return "", "error"
    
    return "", "error"


def translate_with_qwen3(texts: List[str], target_lang: str, 
                        output_dir: Path,
                        model_size: str = "4B") -> Tuple[List[str], List[str]]:
    """Batch translate with Qwen3 with checkpointing."""
    
    # Check for existing checkpoint
    checkpoint_path = get_checkpoint_path(output_dir, target_lang, "qwen3")
    translations, statuses, start_idx = load_checkpoint(checkpoint_path)
    
    if start_idx > 0:
        log(f"[1/3] Resuming Qwen3-{model_size} from index {start_idx}/{len(texts)}...")
    else:
        log(f"[1/3] Loading Qwen3-{model_size}...")
    
    lang_names = {
        "amh": "Amharic", "hau": "Hausa", "yor": "Yoruba"
    }
    lang_name = lang_names.get(target_lang, target_lang)
    
    # If already complete, return cached results
    if start_idx >= len(texts):
        log(f"Qwen3 already complete for {lang_name}! Using cached {len(translations)} translations.")
        return translations, statuses
    
    model_name = f"Qwen/Qwen3-{model_size}"
    log(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    log(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    log(f"Model loaded! Translating {len(texts) - start_idx} remaining texts to {lang_name}...")
    
    status_counts = {"success": 0, "english": 0, "repetitive": 0, "error": 0}
    # Count existing statuses
    for s in statuses:
        if s in status_counts:
            status_counts[s] += 1
    
    for i in tqdm(range(start_idx, len(texts)), desc="Qwen3", initial=start_idx, total=len(texts)):
        text = texts[i]
        translation, status = translate_single_qwen3(
            text, target_lang, model, tokenizer, lang_name, sample_idx=i
        )
        translations.append(translation)
        statuses.append(status)
        status_counts[status] += 1
        
        # Save checkpoint after each translation
        save_checkpoint_entry(checkpoint_path, translation, status)
        
        # Log first 5 individually, then every 10, then every 50 after 100
        if i < 5 or (i < 100 and (i + 1) % 10 == 0) or (i + 1) % 50 == 0:
            log(f"Qwen3 [{i+1}/{len(texts)}] ✓:{status_counts['success']} "
                f"eng:{status_counts['english']} rep:{status_counts['repetitive']} "
                f"err:{status_counts['error']}")
    
    del model, tokenizer
    clear_gpu_memory()
    
    log(f"Qwen3 DONE: {status_counts['success']} ok, {status_counts['english']} english, "
        f"{status_counts['repetitive']} repetitive, {status_counts['error']} errors")
    
    return translations, statuses


# ============================================================
# M2M100 (dedicated translation model - no repetition check)
# ============================================================

def translate_single_m2m100(text: str, target_lang: str, model, tokenizer, 
                           tgt_lang: str, sample_idx: int = 0,
                           max_attempts: int = 3) -> Tuple[str, str]:
    """Translate text with M2M100. No repetition check needed."""
    
    if sample_idx == 0:
        log("Starting first M2M100 inference...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            tokenizer.src_lang = "en"
            encoded = tokenizer(text, return_tensors="pt", padding=True, 
                              truncation=True, max_length=512).to(device)
            generated = model.generate(
                **encoded, 
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                max_length=512, 
                num_beams=5, 
                no_repeat_ngram_size=3, 
                repetition_penalty=1.3
            )
            
            if sample_idx == 0 and attempt == 1:
                log("First M2M100 inference complete!")
            
            result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            
            if not result or len(result) < 3:
                if attempt < max_attempts:
                    continue
                return "", "error"
            
            # Only check if output is English (failed translation)
            # No repetition check - M2M100 is a dedicated translation model
            if is_english(result):
                if attempt < max_attempts:
                    continue
                return result, "english"
            
            return result, "success"
            
        except Exception as e:
            if sample_idx < 3:
                log(f"M2M100 sample {sample_idx} attempt {attempt} error: {str(e)[:100]}")
            if attempt < max_attempts:
                continue
            return "", "error"
    
    return "", "error"


def translate_with_m2m100(texts: List[str], target_lang: str,
                          output_dir: Path) -> Tuple[List[str], List[str]]:
    """Batch translate with M2M100 with checkpointing."""
    
    # Check for existing checkpoint
    checkpoint_path = get_checkpoint_path(output_dir, target_lang, "m2m100")
    translations, statuses, start_idx = load_checkpoint(checkpoint_path)
    
    if start_idx > 0:
        log(f"[2/3] Resuming M2M100 from index {start_idx}/{len(texts)}...")
    else:
        log("[2/3] Loading M2M100...")
    
    lang_codes = {"amh": "am", "swa": "sw", "tam": "ta", "hau": "ha", "yor": "yo", "zul": "zu"}
    tgt_lang = lang_codes.get(target_lang, target_lang)
    
    # If already complete, return cached results
    if start_idx >= len(texts):
        log(f"M2M100 already complete! Using cached {len(translations)} translations.")
        return translations, statuses
    
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained(
        "facebook/m2m100_418M",
        torch_dtype=torch.float16
    ).to(device)
    
    log(f"M2M100 loaded! Translating {len(texts) - start_idx} remaining texts...")
    
    status_counts = {"success": 0, "english": 0, "error": 0}
    # Count existing statuses
    for s in statuses:
        if s in status_counts:
            status_counts[s] += 1
    
    for i in tqdm(range(start_idx, len(texts)), desc="M2M100", initial=start_idx, total=len(texts)):
        text = texts[i]
        translation, status = translate_single_m2m100(
            text, target_lang, model, tokenizer, tgt_lang, sample_idx=i
        )
        translations.append(translation)
        statuses.append(status)
        status_counts[status] += 1
        
        # Save checkpoint after each translation
        save_checkpoint_entry(checkpoint_path, translation, status)
        
        # Log first 5 individually, then every 10, then every 50 after 100
        if i < 5 or (i < 100 and (i + 1) % 10 == 0) or (i + 1) % 50 == 0:
            log(f"M2M100 [{i+1}/{len(texts)}] ✓:{status_counts['success']} "
                f"eng:{status_counts['english']} err:{status_counts['error']}")
    
    del model, tokenizer
    clear_gpu_memory()
    
    log(f"M2M100 DONE: {status_counts['success']} ok, {status_counts['english']} english, "
        f"{status_counts['error']} errors")
    
    return translations, statuses


# ============================================================
# MADLAD-400 (dedicated translation model - no repetition check)
# ============================================================

def translate_single_madlad(text: str, target_lang: str, model, tokenizer, 
                           tgt_lang: str, sample_idx: int = 0,
                           max_attempts: int = 3) -> Tuple[str, str]:
    """Translate text with MADLAD. No repetition check needed."""
    
    if sample_idx == 0:
        log("Starting first MADLAD inference...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            prompt = f"<2{tgt_lang}> {text}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs, 
                max_length=512, 
                num_beams=5, 
                no_repeat_ngram_size=3, 
                repetition_penalty=1.3
            )
            
            if sample_idx == 0 and attempt == 1:
                log("First MADLAD inference complete!")
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            if not result or len(result) < 3:
                if attempt < max_attempts:
                    continue
                return "", "error"
            
            # Only check if output is English (failed translation)
            # No repetition check - MADLAD is a dedicated translation model
            if is_english(result):
                if attempt < max_attempts:
                    continue
                return result, "english"
            
            return result, "success"
            
        except Exception as e:
            if sample_idx < 3:
                log(f"MADLAD sample {sample_idx} attempt {attempt} error: {str(e)[:100]}")
            if attempt < max_attempts:
                continue
            return "", "error"
    
    return "", "error"


def translate_with_madlad(texts: List[str], target_lang: str,
                          output_dir: Path) -> Tuple[List[str], List[str]]:
    """Batch translate with MADLAD with checkpointing."""
    
    # Check for existing checkpoint
    checkpoint_path = get_checkpoint_path(output_dir, target_lang, "madlad")
    translations, statuses, start_idx = load_checkpoint(checkpoint_path)
    
    if start_idx > 0:
        log(f"[3/3] Resuming MADLAD-400 from index {start_idx}/{len(texts)}...")
    else:
        log("[3/3] Loading MADLAD-400...")
    
    lang_codes = {"amh": "amh", "swa": "swa", "tam": "tam", "hau": "hau", "yor": "yor", "zul": "zul"}
    tgt_lang = lang_codes.get(target_lang, target_lang)
    
    # If already complete, return cached results
    if start_idx >= len(texts):
        log(f"MADLAD already complete! Using cached {len(translations)} translations.")
        return translations, statuses
    
    tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/madlad400-3b-mt",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    log(f"MADLAD loaded! Translating {len(texts) - start_idx} remaining texts...")
    
    status_counts = {"success": 0, "english": 0, "error": 0}
    # Count existing statuses
    for s in statuses:
        if s in status_counts:
            status_counts[s] += 1
    
    for i in tqdm(range(start_idx, len(texts)), desc="MADLAD-400", initial=start_idx, total=len(texts)):
        text = texts[i]
        translation, status = translate_single_madlad(
            text, target_lang, model, tokenizer, tgt_lang, sample_idx=i
        )
        translations.append(translation)
        statuses.append(status)
        status_counts[status] += 1
        
        # Save checkpoint after each translation
        save_checkpoint_entry(checkpoint_path, translation, status)
        
        # Log first 5 individually, then every 10, then every 50 after 100
        if i < 5 or (i < 100 and (i + 1) % 10 == 0) or (i + 1) % 50 == 0:
            log(f"MADLAD [{i+1}/{len(texts)}] ✓:{status_counts['success']} "
                f"eng:{status_counts['english']} err:{status_counts['error']}")
    
    del model, tokenizer
    clear_gpu_memory()
    
    log(f"MADLAD DONE: {status_counts['success']} ok, {status_counts['english']} english, "
        f"{status_counts['error']} errors")
    
    return translations, statuses


# ============================================================
# Consensus Selection
# ============================================================

def compute_consensus_with_fallback(all_translations: Dict[str, List[str]],
                                    all_statuses: Dict[str, List[str]],
                                    primary_model: str) -> Tuple[List[str], List[str], List[Dict], List[str], List[int]]:
    """Select best translation per entry using BLEU/chrF consensus with fallback logic."""
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    
    num_entries = len(all_translations[primary_model])
    best_translations = []
    best_models = []
    all_scores = []
    final_statuses = []
    skipped_indices = []
    
    model_keys = list(all_translations.keys())
    
    for i in range(num_entries):
        translations = {k: all_translations[k][i] for k in model_keys}
        statuses = {k: all_statuses[k][i] for k in model_keys}
        
        successful = {k: v for k, v in translations.items() if v and statuses[k] == "success"}
        
        if successful:
            if len(successful) == 1:
                model = list(successful.keys())[0]
                best_translations.append(successful[model])
                best_models.append(model)
                all_scores.append({model: 100.0})
                final_statuses.append("success")
                continue
            
            scores = {}
            for model, candidate in successful.items():
                refs = [t for m, t in successful.items() if m != model]
                try:
                    b = bleu.sentence_score(candidate, refs).score
                    c = chrf.sentence_score(candidate, refs).score
                    scores[model] = 0.5 * b + 0.5 * c
                except Exception as score_err:
                    log(f"⚠️  Consensus scoring failed for entry {i}, model {model}: {score_err}")
                    scores[model] = 0.0
            
            best_model = max(scores, key=scores.get)
            best_translations.append(successful[best_model])
            best_models.append(best_model)
            all_scores.append(scores)
            final_statuses.append("success")
            continue
        
        # All models failed - skip this entry entirely
        skipped_indices.append(i)
        best_translations.append(None)
        best_models.append("skipped")
        all_scores.append({})
        final_statuses.append("skipped")
    
    return best_translations, best_models, all_scores, final_statuses, skipped_indices


def main():
    parser = argparse.ArgumentParser(description="Multi-model translation pipeline")
    parser.add_argument("--input", type=str, 
                       default="/mloscratch/users/nemo/datasets/guidelines/chunked_eng.jsonl")
    parser.add_argument("--output", type=str, 
                       default="/mloscratch/users/grieder/MeditronPolyglot/MultiMeditron/src/multimeditron/translation/datasets/generated_datasets/consensus")
    parser.add_argument("--languages", nargs="+", default=["amh", "swa", "tam", "hau", "yor", "zul"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--qwen-size", type=str, default="4B")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("="*60)
    log("Multi-Model Translation Pipeline v4 (with checkpointing)")
    log("TranslateGemma-4B for: swa, tam, zul")
    log(f"Qwen3-{args.qwen_size} for: amh, hau, yor")
    log("M2M100 + MADLAD-400 for all languages")
    log(f"Input: {args.input}")
    log(f"Output: {args.output}")
    log(f"Languages: {args.languages}")
    log(f"Sample: {args.sample}")
    log("Checkpoints will be saved to output directory")
    log("="*60)
    
    log(f"Loading dataset from {args.input}...")
    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if args.sample:
        data = data[:args.sample]
    
    texts = [entry.get('text', entry.get('content', '')) for entry in data]
    texts = [t for t in texts if t]
    log(f"Loaded {len(texts)} texts")
    
    for lang in args.languages:
        log("")
        log("="*60)
        log(f"Processing {lang.upper()}")
        log("="*60)
        
        all_translations = {}
        all_statuses = {}
        
        # Choose primary model based on language support
        if lang in TRANSLATEGEMMA_SUPPORTED:
            log(f"Using TranslateGemma-4B as primary model for {lang}")
            primary_trans, primary_status = translate_with_translategemma(texts, lang, output_dir)
            primary_model = "translategemma"
        else:
            log(f"Using Qwen3-{args.qwen_size} as primary model for {lang} (not in TranslateGemma)")
            primary_trans, primary_status = translate_with_qwen3(texts, lang, output_dir, args.qwen_size)
            primary_model = "qwen3"
        
        all_translations[primary_model] = primary_trans
        all_statuses[primary_model] = primary_status
        
        # M2M100 for all languages
        m2m_trans, m2m_status = translate_with_m2m100(texts, lang, output_dir)
        all_translations["m2m100"] = m2m_trans
        all_statuses["m2m100"] = m2m_status
        
        # MADLAD for all languages
        madlad_trans, madlad_status = translate_with_madlad(texts, lang, output_dir)
        all_translations["madlad-400"] = madlad_trans
        all_statuses["madlad-400"] = madlad_status
        
        log("Computing consensus...")
        best_translations, best_models, scores, final_statuses, skipped = compute_consensus_with_fallback(
            all_translations, all_statuses, primary_model
        )
        
        output_file = output_dir / f"{lang}_train.jsonl"
        
        saved_count = 0
        with open(output_file, 'w') as f:
            for i, entry in enumerate(data[:len(texts)]):
                if i in skipped:
                    continue
                
                output = {
                    **entry,
                    "translated_text": best_translations[i],
                    "target_language": lang,
                    "source_language": "eng",
                    "translation_model": best_models[i],
                    "translation_status": final_statuses[i],
                    "consensus_scores": scores[i],
                    "all_translations": {k: all_translations[k][i] for k in all_translations},
                    "all_statuses": {k: all_statuses[k][i] for k in all_statuses}
                }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
                saved_count += 1
        
        log(f"Saved: {output_file}")
        log(f"Saved {saved_count} translations (skipped {len(skipped)} failed entries)")
        
        # Clean up checkpoint files after successful completion
        for model_name in ["translategemma", "qwen3", "m2m100", "madlad"]:
            checkpoint_path = get_checkpoint_path(output_dir, lang, model_name)
            clear_checkpoint(checkpoint_path)
        log(f"Cleaned up checkpoint files for {lang}")
        
        valid_models = [m for i, m in enumerate(best_models) if i not in skipped]
        valid_statuses = [s for i, s in enumerate(final_statuses) if i not in skipped]
        
        model_counts = {}
        for m in valid_models:
            model_counts[m] = model_counts.get(m, 0) + 1
        
        log("Model distribution:")
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            pct = count / len(valid_models) * 100 if valid_models else 0
            log(f"  {model:15s}: {count:5d} ({pct:5.1f}%)")
        
        status_counts = {}
        for s in valid_statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        
        log("Final status:")
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            pct = count / len(valid_statuses) * 100 if valid_statuses else 0
            log(f"  {status:20s}: {count:5d} ({pct:5.1f}%)")
        
        successful_scores = [s[m] for i, (s, m, fs) in enumerate(zip(scores, best_models, final_statuses)) 
                           if i not in skipped and m in s and fs == "success"]
        if successful_scores:
            log(f"Average consensus score: {np.mean(successful_scores):.2f}")
    
    log("")
    log("="*60)
    log("Complete!")
    log("="*60)


if __name__ == "__main__":
    main()
