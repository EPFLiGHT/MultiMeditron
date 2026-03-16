"""
African Languages Translation Experiment (LLAMA Baseline)

Compares two translation approaches for African languages (Amharic, Hausa, Swahili, Yoruba, Zulu):
1. LLAMA 3.1's native multilingual translation (direct African → English)
2. NLLB translation pipeline (fastText detection → NLLB translation)

Evaluated using BLEU, chrF, and BERTScore against ground truth English.
Runs in two passes to avoid GPU OOM: LLAMA first, then NLLB.

Default: 1000 samples per language

Output: src/multimeditron/translation/experiments/results/base_nllb/experiment_1_llama_results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from collections import defaultdict, Counter
import evaluate
import fasttext
from huggingface_hub import hf_hub_download

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimeditron.translation.translator import NLLBTranslator


class TranslationEvaluator:
    """Evaluates and compares translation quality across pipelines."""
    
    def __init__(self):
        self.results = {
            'llama_direct': defaultdict(list),
            'nllb': defaultdict(list)
        }
        
        self.detection_stats = {
            'total': 0,
            'correct': 0,
            'by_language': defaultdict(lambda: {'total': 0, 'correct': 0, 'misdetections': Counter()})
        }
        
        self.bleu = evaluate.load('bleu')
        self.chrf = evaluate.load('chrf')
        self.bertscore = evaluate.load("bertscore")
    
    def track_detection(self, true_lang: str, detected_lang: str):
        """Track fastText detection against ground truth."""
        self.detection_stats['total'] += 1
        self.detection_stats['by_language'][true_lang]['total'] += 1
        
        if detected_lang == true_lang:
            self.detection_stats['correct'] += 1
            self.detection_stats['by_language'][true_lang]['correct'] += 1
        else:
            self.detection_stats['by_language'][true_lang]['misdetections'][detected_lang] += 1
    
    def add_result(self, pipeline: str, language: str, 
                   source: str, prediction: str, reference: str,
                   detected_lang: str = None):
        """Store translation result."""
        self.results[pipeline][language].append({
            'source': source[:100],
            'prediction': prediction,
            'reference': reference,
            'detected_lang': detected_lang
        })
    
    def compute_metrics(self):
        """Compute BLEU, chrF, and BERTScore for all translations."""
        stats = {}
        
        for pipeline in ['llama_direct', 'nllb']:
            stats[pipeline] = {}
            
            for language, translations in self.results[pipeline].items():
                if not translations:
                    continue
                
                predictions = [t['prediction'] for t in translations]
                references = [[t['reference']] for t in translations]
                
                bleu_result = self.bleu.compute(predictions=predictions, references=references)
                chrf_result = self.chrf.compute(predictions=predictions, references=references)
                bertscore_result = self.bertscore.compute(
                    predictions=predictions,
                    references=[t['reference'] for t in translations],
                    model_type="microsoft/deberta-base-mnli",   # 140MB, needs 2GB GPU
                    batch_size=8                                 # Process in batches
                )
                
                stats[pipeline][language] = {
                    'count': len(translations),
                    'bleu': bleu_result['bleu'] * 100,
                    'chrf': chrf_result['score'],
                    'bertscore_f1': sum(bertscore_result['f1']) / len(bertscore_result['f1']) * 100
                }
        
        return stats
    
    def print_summary(self):
        """Print comparison of LLAMA vs NLLB translation quality."""
        stats = self.compute_metrics()
        
        print("\n" + "="*70)
        print("EXPERIMENT: DO AFRICAN LANGUAGES NEED TRANSLATION OR IS MULTILINGUAL LLAMA SUFFICIENT?")
        print("="*70)
        
        det = self.detection_stats
        if det['total'] > 0:
            det_acc = det['correct'] / det['total'] * 100
            print(f"\n📍 FASTTEXT LANGUAGE DETECTION:")
            print(f"   Overall Accuracy: {det_acc:.1f}% ({det['correct']}/{det['total']})")
            
            print(f"\n   By Language:")
            for lang, data in sorted(det['by_language'].items(), key=lambda x: -x[1]['total']):
                acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
                symbol = "✅" if acc > 90 else "⚠️" if acc > 70 else "❌"
                print(f"      {symbol} {lang}: {acc:>5.1f}% ({data['correct']:3d}/{data['total']:3d})", end='')
                
                if data['misdetections']:
                    top = data['misdetections'].most_common(1)[0]
                    print(f"  → confused with {top[0]} ({top[1]}x)")
                else:
                    print()
        
        print("\n" + "="*70)
        print("📊 TRANSLATION QUALITY COMPARISON")
        print("="*70)
        print(f"\n{'Language':<12} {'Llama 3.1 Direct':<30} {'NLLB + fastText':<30} {'Winner':<10}")
        print(f"{'':12} {'BLEU':>8} {'chrF':>8} {'BERT-F1':>8}   {'BLEU':>8} {'chrF':>8} {'BERT-F1':>8}")
        print("-" * 90)
        
        all_langs = set()
        for pipeline_stats in stats.values():
            all_langs.update(pipeline_stats.keys())
        
        winners = {'LLAMA': 0, 'nllb': 0, 'tie': 0}
        
        for lang in sorted(all_langs):
            llama_stats = stats['llama_direct'].get(lang, {})
            nllb_stats = stats['nllb'].get(lang, {})
            
            llama_bleu = llama_stats.get('bleu', 0)
            llama_chrf = llama_stats.get('chrf', 0)
            nllb_bleu = nllb_stats.get('bleu', 0)
            nllb_chrf = nllb_stats.get('chrf', 0)
            llama_bert = llama_stats.get('bertscore_f1', 0)
            nllb_bert = nllb_stats.get('bertscore_f1', 0)

            if abs(llama_chrf - nllb_chrf) < 2:
                winner = "➡️  Tie"
                winners['tie'] += 1
            elif llama_chrf > nllb_chrf:
                winner = "✅ LLAMA"
                winners['LLAMA'] += 1
            else:
                winner = "✅ NLLB"
                winners['nllb'] += 1

            print(f"{lang:<12} {llama_bleu:>7.1f}% {llama_chrf:>7.1f} {llama_bert:>7.1f}   "
                f"{nllb_bleu:>7.1f}% {nllb_chrf:>7.1f} {nllb_bert:>7.1f}   {winner}")
        
        print("\n" + "-" * 70)
        
        llama_avg_bleu = sum(s['bleu'] for s in stats['llama_direct'].values()) / len(stats['llama_direct']) if stats['llama_direct'] else 0
        llama_avg_chrf = sum(s['chrf'] for s in stats['llama_direct'].values()) / len(stats['llama_direct']) if stats['llama_direct'] else 0
        nllb_avg_bleu = sum(s['bleu'] for s in stats['nllb'].values()) / len(stats['nllb']) if stats['nllb'] else 0
        nllb_avg_chrf = sum(s['chrf'] for s in stats['nllb'].values()) / len(stats['nllb']) if stats['nllb'] else 0        
        llama_avg_bert = sum(s['bertscore_f1'] for s in stats['llama_direct'].values()) / len(stats['llama_direct']) if stats['llama_direct'] else 0
        nllb_avg_bert = sum(s['bertscore_f1'] for s in stats['nllb'].values()) / len(stats['nllb']) if stats['nllb'] else 0

        print(f"{'AVERAGE':<12} {llama_avg_bleu:>7.1f}% {llama_avg_chrf:>7.1f} {llama_avg_bert:>7.1f}   {nllb_avg_bleu:>7.1f}% {nllb_avg_chrf:>7.1f} {nllb_avg_bert:>7.1f}")
        
        print("\n" + "="*70)
        print("📝 SAMPLE TRANSLATIONS")
        print("="*70)
        
        for lang in sorted(all_langs)[:2]:
            if lang not in self.results['llama_direct']:
                continue
                
            samples = self.results['llama_direct'][lang][:2]
            
            for i, sample in enumerate(samples, 1):
                print(f"\n{lang.upper()} Sample {i}:")
                print(f"   Source: {sample['source']}...")
                print(f"   Reference: {sample['reference'][:80]}...")
                
                nllb_sample = self.results['nllb'][lang][i-1] if i <= len(self.results['nllb'][lang]) else None
                
                print(f"   LLAMA:  {sample['prediction'][:80]}...")
                if nllb_sample:
                    print(f"   NLLB (detected as {nllb_sample.get('detected_lang', 'N/A')}): {nllb_sample['prediction'][:80]}...")
        
        print("\n" + "="*70)
        print("💡 CONCLUSION")
        print("="*70)
        
        print(f"\n📊 Translation Winner Count:")
        print(f"   LLAMA (no language hint): {winners['LLAMA']} languages")
        print(f"   NLLB (with fastText detection):    {winners['nllb']} languages")
        print(f"   Tie:                                {winners['tie']} languages")
        
        chrf_diff = llama_avg_chrf - nllb_avg_chrf
        
        print(f"\n📈 Average chrF Score:")
        print(f"   LLAMA: {llama_avg_chrf:.1f}")
        print(f"   NLLB:     {nllb_avg_chrf:.1f}")
        print(f"   Difference: {chrf_diff:+.1f}")
        
        det_acc = det['correct'] / det['total'] * 100 if det['total'] > 0 else 0
        
        if abs(chrf_diff) < 3:
            print(f"\n➡️  RESULT: Translation quality is COMPARABLE")
            print(f"   → Both approaches produce similar quality translations")
            if det_acc < 85:
                print(f"   ⚠️  BUT fastText detection is poor ({det_acc:.1f}%)")
                print(f"   → NLLB pipeline unreliable due to detection errors")
                print(f"   → RECOMMEND: Use LLAMA's native multilingual capability")
            else:
                print(f"   ✅ FastText detection is good ({det_acc:.1f}%)")
                print(f"   → Choice depends on other factors (speed, complexity)")
        elif chrf_diff > 3:
            print(f"\n✅ RESULT: LLAMA's NATIVE multilingual capability is BETTER")
            print(f"   → LLAMA can translate directly without language detection!")
            print(f"   → No need for fastText or NLLB")
            print(f"   → Simpler architecture, fewer failure points")
        else:
            print(f"\n⚠️  RESULT: NLLB translation is BETTER")
            print(f"   → NLLB produces higher quality translations")
            if det_acc > 85:
                print(f"   ✅ FastText detection is reliable ({det_acc:.1f}%)")
                print(f"   → Translation pipeline is JUSTIFIED")
                print(f"   → Use fastText → NLLB → LLAMA pipeline")
            else:
                print(f"   ❌ BUT fastText detection is poor ({det_acc:.1f}%)")
                print(f"   → NLLB pipeline unreliable in practice")
                print(f"   → Consider improving detection or using LLAMA directly")
        
        print("\n" + "="*70)
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        output = {
            'experiment': 'African Languages: Translation Comparison (LLAMA Baseline)',
            'detection_stats': {
                'overall_accuracy': self.detection_stats['correct'] / self.detection_stats['total'] * 100 if self.detection_stats['total'] > 0 else 0,
                'by_language': {
                    k: {
                        'accuracy': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0,
                        'correct': v['correct'],
                        'total': v['total'],
                        'misdetections': dict(v['misdetections'])
                    }
                    for k, v in self.detection_stats['by_language'].items()
                }
            },
            'translation_stats': self.compute_metrics(),
            'detailed_results': {
                k: dict(v) for k, v in self.results.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def save_intermediate_results(results, filename):
    """Save intermediate results between passes."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved intermediate results to {filename}")


def extract_translation(response: str, original_text: str) -> str:
    """Extract clean translation from LLAMA response, removing LLM commentary."""
    if original_text in response:
        response = response.split(original_text)[-1]
    
    prefixes_to_remove = [
        "here is the translation:", "the translation is:", "english translation:",
        "translation:", "here's the translation:", "the english translation is:",
        "translated text:", "in english:", "sure, here is the translation:",
        "certainly! here is the translation:", "okay, here is the translation:",
        "here you go:", "the text translates to:",
    ]
    
    response_lower = response.lower().strip()
    
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            response_lower = response.lower().strip()
    
    response = response.replace('**', '').replace('*', '')
    
    if (response.startswith('"') and response.endswith('"')) or \
       (response.startswith("'") and response.endswith("'")):
        response = response[1:-1].strip()
    
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        first_lower = paragraphs[0].lower()
        if any(word in first_lower for word in ['here', 'translation', 'english', 'okay', 'sure', 'certainly']):
            response = '\n\n'.join(paragraphs[1:])
        else:
            response = '\n\n'.join(paragraphs)
    
    return response.strip()


def translate_with_llama(model, tokenizer, text: str) -> str:
    """Use LLAMA 3.1 to translate text to English."""
    if len(text) > 400:
        text = text[:400]
    
    prompt = f"Translate the following text into English:\n\n{text}\n\nEnglish translation:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = extract_translation(response, text)
    return translation.strip()


def run_experiment(args):
    """Run African languages translation experiment in two passes."""
    
    LANG_MAP = {
        'am': ('amh_Ethi', 'Amharic'),
        'ha': ('hau_Latn', 'Hausa'),
        'sw': ('swh_Latn', 'Swahili'),
        'yo': ('yor_Latn', 'Yoruba'),
        'zu': ('zul_Latn', 'Zulu'),
    }
    
    print("\n" + "="*70)
    print("EXPERIMENT: AFRICAN LANGUAGES TRANSLATION COMPARISON")
    print("(TWO-PASS VERSION - LLAMA first, then NLLB)")
    print("="*70)
    
    print("\n[1/6] Loading MediTranslation dataset...")
    dataset = load_dataset(
        "ClosedMeditron/MediTranslation",
        split="train",
        token=os.getenv("HF_LAB_TOKEN")
    )
    print(f"   ✅ Loaded {len(dataset)} parallel translations")

    max_samples = args.max_samples
    total_per_lang = max_samples * len(LANG_MAP)
    print(f"   Using {max_samples} samples per language ({total_per_lang} total)")
    
    print("\n[2/6] Loading fastText for language detection...")
    lid_model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    fasttext_model = fasttext.load_model(lid_model_path)
    print("   ✅ FastText loaded")
    
    evaluator = TranslationEvaluator()
    
    llama_results_file = "llama_intermediate.json"
    
    print("\n" + "="*70)
    print("[PASS 1/2] LLAMA TRANSLATION")
    print("="*70)
    print("\n[3/6] Loading Llama 3.1-8B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        token=os.getenv("HF_PERSONAL_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("HF_PERSONAL_TOKEN")
    )
    model.eval()
    print("   ✅ Llama 3.1 loaded")

    print(f"   💾 Initial GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    llama_results = []

    print(f"\n[4/6] Running Llama 3.1 translations...")
    for lang_code, (true_nllb_code, lang_name) in LANG_MAP.items():
        print(f"\n   Processing {lang_name} ({lang_code})...")
        processed = 0
        for idx, sample in enumerate(tqdm(dataset, desc=f"  {lang_name}", total=max_samples)):
            if processed >= max_samples:
                break

            source_text = sample.get(lang_code)
            reference_english = sample.get('en')
            if not source_text or not reference_english:
                continue

            pred = fasttext_model.predict(source_text.replace('\n', ' '), k=1)
            detected_lang = pred[0][0].replace('__label__', '')
            evaluator.track_detection(true_nllb_code, detected_lang)

            try:
                llama_translation = translate_with_llama(model, tokenizer, source_text)
                llama_results.append({
                    'lang_code': lang_code,
                    'source': source_text,
                    'prediction': llama_translation,
                    'reference': reference_english,
                    'detected_lang': detected_lang,
                    'true_lang': true_nllb_code
                })
                evaluator.add_result(
                    pipeline='llama_direct',
                    language=lang_code,
                    source=source_text,
                    prediction=llama_translation,
                    reference=reference_english
                )
                processed += 1
            except Exception as e:
                print(f"   ⚠️ Translation failed: {e}")
    
    save_intermediate_results(llama_results, llama_results_file)
    
    print("\n   🗑️  Unloading LLAMA to free GPU memory...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print("   ✅ GPU memory cleared")
    
    print("\n" + "="*70)
    print("[PASS 2/2] NLLB TRANSLATION")
    print("="*70)
    print("\n[5/6] Loading NLLB...")
    
    # ✅ VERIFIED: Using base NLLB-200 3.3B model
    translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")
    print("   ✅ NLLB loaded")
    
    print(f"\n   Running NLLB translations on {len(llama_results)} samples...")
    
    for idx, result in enumerate(tqdm(llama_results, desc="  Translating")):
        try:
            nllb_translation = translator.translate(
                result['source'],
                src_lang=result['true_lang'],
                tgt_lang='eng_Latn'
            )
            
            evaluator.add_result(
                pipeline='nllb',
                language=result['lang_code'],
                source=result['source'],
                prediction=nllb_translation,
                reference=result['reference'],
                detected_lang=result['detected_lang']
            )
            
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n      [ERROR] Sample {idx} failed: {e}")
            torch.cuda.empty_cache()
            continue
    
    print("\n[6/6] Computing results...")
    evaluator.print_summary()
    evaluator.save_results(args.output)
    
    if os.path.exists(llama_results_file):
        os.remove(llama_results_file)
    
    print("\n✅ Experiment complete!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="African languages translation experiment: LLAMA vs NLLB"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Samples per language (default: 1000)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_1_llama_results.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    run_experiment(args)