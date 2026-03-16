"""
Fine-tuned NLLB Translation Evaluation for African Languages

Evaluates translation quality from African languages (Amharic, Hausa, Swahili, 
Yoruba, Zulu) to English using BLEU, chrF, and BERTScore metrics.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import torch
import fasttext
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

SRC_ROOT = next((parent for parent in Path(__file__).resolve().parents if parent.name == "src"), None)
if SRC_ROOT is not None and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimeditron.translation.translator import NLLBTranslator

LANGUAGE_CONFIG = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}

MAX_CONSECUTIVE_ERRORS = 10
MIN_SOURCE_LENGTH = 10
MAX_SOURCE_LENGTH = 400


class TranslationEvaluator:
    
    def __init__(self):
        self.results = defaultdict(list)
        self.detection_stats = {
            'total': 0,
            'correct': 0,
            'by_language': defaultdict(lambda: {
                'total': 0, 
                'correct': 0, 
                'misdetections': Counter()
            })
        }
        self.bleu = evaluate.load('bleu')
        self.chrf = evaluate.load('chrf')
        self.bertscore = evaluate.load('bertscore')
    
    def track_detection(self, true_lang: str, detected_lang: str):
        self.detection_stats['total'] += 1
        self.detection_stats['by_language'][true_lang]['total'] += 1
        
        if detected_lang == true_lang:
            self.detection_stats['correct'] += 1
            self.detection_stats['by_language'][true_lang]['correct'] += 1
        else:
            self.detection_stats['by_language'][true_lang]['misdetections'][detected_lang] += 1
    
    def add_result(self, language: str, source: str, prediction: str, 
                   reference: str, detected_lang: str = None):
        self.results[language].append({
            'source': source[:100],
            'prediction': prediction,
            'reference': reference,
            'detected_lang': detected_lang
        })
    
    def compute_metrics(self):
        stats = {}
        
        for language, translations in self.results.items():
            if not translations:
                continue
            
            predictions = [t['prediction'] for t in translations]
            references_nested = [[t['reference']] for t in translations]
            references_flat = [t['reference'] for t in translations]
            
            bleu_result = self.bleu.compute(
                predictions=predictions, 
                references=references_nested
            )
            chrf_result = self.chrf.compute(
                predictions=predictions, 
                references=references_nested
            )
            bertscore_result = self.bertscore.compute(
                predictions=predictions,
                references=references_flat,
                model_type="microsoft/deberta-xlarge-mnli"
            )
            
            stats[language] = {
                'count': len(translations),
                'bleu': bleu_result['bleu'] * 100,
                'chrf': chrf_result['score'],
                'bertscore_f1': sum(bertscore_result['f1']) / len(bertscore_result['f1']) * 100
            }
        
        return stats
    
    def print_summary(self):
        stats = self.compute_metrics()
        
        print("\n" + "=" * 70)
        print("FINE-TUNED NLLB TRANSLATION EVALUATION RESULTS")
        print("=" * 70)
        
        self._print_detection_stats()
        self._print_translation_stats(stats)
        self._print_sample_translations(stats)
        self._print_final_summary(stats)
    
    def _print_detection_stats(self):
        det = self.detection_stats
        if det['total'] == 0:
            return
        
        accuracy = det['correct'] / det['total'] * 100
        print(f"\nLanguage Detection Accuracy: {accuracy:.1f}% ({det['correct']}/{det['total']})")
        print("\nPer-Language Breakdown:")
        
        for lang, data in sorted(det['by_language'].items(), key=lambda x: -x[1]['total']):
            lang_acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
            status = "OK" if lang_acc > 90 else "WARN" if lang_acc > 70 else "FAIL"
            print(f"  [{status}] {lang}: {lang_acc:>5.1f}% ({data['correct']:3d}/{data['total']:3d})", end='')
            
            if data['misdetections']:
                top_confusion = data['misdetections'].most_common(1)[0]
                print(f"  -> confused with {top_confusion[0]} ({top_confusion[1]}x)")
            else:
                print()
    
    def _print_translation_stats(self, stats):
        print("\n" + "=" * 70)
        print("TRANSLATION QUALITY METRICS")
        print("=" * 70)
        print(f"\n{'Language':<12} {'BLEU':>10} {'chrF':>10} {'BERT-F1':>10} {'Samples':>10}")
        print("-" * 55)
        
        for lang in sorted(stats.keys()):
            s = stats[lang]
            print(f"{lang:<12} {s['bleu']:>9.1f}% {s['chrf']:>9.1f} "
                  f"{s['bertscore_f1']:>9.1f} {s['count']:>10}")
        
        print("-" * 55)
        
        if stats:
            avg_bleu = sum(s['bleu'] for s in stats.values()) / len(stats)
            avg_chrf = sum(s['chrf'] for s in stats.values()) / len(stats)
            avg_bert = sum(s['bertscore_f1'] for s in stats.values()) / len(stats)
            total = sum(s['count'] for s in stats.values())
            print(f"{'AVERAGE':<12} {avg_bleu:>9.1f}% {avg_chrf:>9.1f} {avg_bert:>9.1f} {total:>10}")
    
    def _print_sample_translations(self, stats):
        print("\n" + "=" * 70)
        print("SAMPLE TRANSLATIONS")
        print("=" * 70)
        
        for lang in sorted(stats.keys())[:2]:
            if lang not in self.results:
                continue
            
            for i, sample in enumerate(self.results[lang][:2], 1):
                print(f"\n{lang.upper()} Sample {i}:")
                print(f"  Source:     {sample['source']}...")
                print(f"  Reference:  {sample['reference'][:80]}...")
                print(f"  Predicted:  {sample['prediction'][:80]}...")
    
    def _print_final_summary(self, stats):
        if not stats:
            return
        
        avg_bleu = sum(s['bleu'] for s in stats.values()) / len(stats)
        avg_chrf = sum(s['chrf'] for s in stats.values()) / len(stats)
        avg_bert = sum(s['bertscore_f1'] for s in stats.values()) / len(stats)
        total = sum(s['count'] for s in stats.values())
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n  Average BLEU:     {avg_bleu:.1f}%")
        print(f"  Average chrF:     {avg_chrf:.1f}")
        print(f"  Average BERT-F1:  {avg_bert:.1f}")
        print(f"  Total Samples:    {total}")
        print("\n" + "=" * 70)
    
    def save_results(self, output_path: str):
        det = self.detection_stats
        output = {
            'experiment': 'African Languages: Fine-tuned NLLB Translation',
            'detection_stats': {
                'overall_accuracy': det['correct'] / det['total'] * 100 if det['total'] > 0 else 0,
                'by_language': {
                    k: {
                        'accuracy': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0,
                        'correct': v['correct'],
                        'total': v['total'],
                        'misdetections': dict(v['misdetections'])
                    }
                    for k, v in det['by_language'].items()
                }
            },
            'translation_stats': self.compute_metrics(),
            'detailed_results': dict(self.results)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
    def to_checkpoint(self):
        return {
            'results': dict(self.results),
            'detection_stats': {
                'total': self.detection_stats['total'],
                'correct': self.detection_stats['correct'],
                'by_language': {
                    k: {
                        'total': v['total'],
                        'correct': v['correct'],
                        'misdetections': dict(v['misdetections'])
                    }
                    for k, v in self.detection_stats['by_language'].items()
                }
            }
        }
    
    def load_checkpoint(self, data: dict):
        self.results = defaultdict(list, data.get('results', {}))
        
        det = data.get('detection_stats', {})
        self.detection_stats['total'] = det.get('total', 0)
        self.detection_stats['correct'] = det.get('correct', 0)
        
        for lang, lang_data in det.get('by_language', {}).items():
            self.detection_stats['by_language'][lang]['total'] = lang_data.get('total', 0)
            self.detection_stats['by_language'][lang]['correct'] = lang_data.get('correct', 0)
            self.detection_stats['by_language'][lang]['misdetections'] = Counter(
                lang_data.get('misdetections', {})
            )


class CheckpointManager:
    
    def __init__(self, output_path: str):
        output_path = Path(output_path)
        self.checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
    
    def save(self, evaluator: TranslationEvaluator, completed_languages: list,
             current_language: str = None, current_processed: int = 0, 
             current_dataset_idx: int = 0):
        checkpoint = {
            'completed_languages': completed_languages,
            'current_language': current_language,
            'current_language_processed': current_processed,
            'current_dataset_idx': current_dataset_idx,
            'evaluator_state': evaluator.to_checkpoint()
        }
        
        temp_path = str(self.checkpoint_path) + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        os.replace(temp_path, self.checkpoint_path)
        print(f"  [Checkpoint] {len(completed_languages)} languages complete, "
              f"{current_processed} samples in current")
    
    def load(self) -> dict:
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"  [Checkpoint] Loaded: {len(checkpoint.get('completed_languages', []))} languages complete")
            return checkpoint
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [Checkpoint] Failed to load: {e}")
            return None
    
    def delete(self):
        if self.checkpoint_path.exists():
            os.remove(self.checkpoint_path)
            print("  [Checkpoint] Removed (experiment complete)")


def load_fasttext_model():
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    return fasttext.load_model(model_path)


def is_valid_sample(sample: dict, lang_code: str) -> bool:
    source = sample.get(lang_code)
    reference = sample.get('en')
    
    if not source or not reference:
        return False
    if len(source.strip()) < MIN_SOURCE_LENGTH:
        return False
    if len(source) > MAX_SOURCE_LENGTH:
        return False
    
    return True


def run_experiment(args):
    print("\n" + "=" * 70)
    print("FINE-TUNED NLLB TRANSLATION EVALUATION")
    print("=" * 70)
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_mgr = CheckpointManager(args.output)
    evaluator = TranslationEvaluator()
    
    completed_languages = []
    resume_language = None
    resume_processed = 0
    resume_dataset_idx = 0
    
    print("\n[1/5] Checking for checkpoint...")
    checkpoint = checkpoint_mgr.load()
    if checkpoint:
        evaluator.load_checkpoint(checkpoint.get('evaluator_state', {}))
        completed_languages = checkpoint.get('completed_languages', [])
        resume_language = checkpoint.get('current_language')
        resume_processed = checkpoint.get('current_language_processed', 0)
        resume_dataset_idx = checkpoint.get('current_dataset_idx', 0)
        
        print(f"  Resuming: {len(completed_languages)} languages done")
        if resume_language:
            print(f"  Continuing {resume_language} from sample {resume_processed}")
    else:
        print("  No checkpoint found, starting fresh")
    
    print("\n[2/5] Loading dataset...")
    dataset = load_dataset(
        "ClosedMeditron/MediTranslation",
        split="train",
        token=os.getenv("HF_LAB_TOKEN")
    )
    print(f"  Loaded {len(dataset)} parallel translations")
    print(f"  Using {args.max_samples} samples per language")
    
    print("\n[3/5] Loading fastText language detector...")
    fasttext_model = load_fasttext_model()
    print("  FastText ready")
    
    print("\n[4/5] Loading Fine-tuned NLLB translator...")
    translator = NLLBTranslator()
    print("  NLLB ready")
    
    print(f"\n[5/5] Running translations (checkpoint every {args.checkpoint_interval} samples)...")
    
    for lang_code, (nllb_code, lang_name) in LANGUAGE_CONFIG.items():
        if lang_code in completed_languages:
            print(f"\n  Skipping {lang_name} - already complete")
            continue
        
        print(f"\n  Processing {lang_name} ({lang_code})...")
        
        if lang_code == resume_language:
            processed = resume_processed
            start_idx = resume_dataset_idx
            print(f"    Resuming from sample {processed}")
        else:
            processed = 0
            start_idx = 0
        
        consecutive_errors = 0
        samples_since_checkpoint = 0
        
        progress = tqdm(dataset, desc=f"    {lang_name}", total=args.max_samples, initial=processed)
        
        for idx, sample in enumerate(progress):
            if idx < start_idx:
                continue
            
            if processed >= args.max_samples:
                break
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n    Too many errors, skipping remaining samples")
                break
            
            if idx % 10 == 0 and idx > 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if not is_valid_sample(sample, lang_code):
                continue
            
            source_text = sample[lang_code]
            reference_english = sample['en']
            
            try:
                prediction = fasttext_model.predict(source_text.replace('\n', ' '), k=1)
                detected_lang = prediction[0][0].replace('__label__', '')
                evaluator.track_detection(nllb_code, detected_lang)
                
                translation = translator.translate(
                    source_text,
                    src_lang=nllb_code,
                    tgt_lang='eng_Latn'
                )
                
                evaluator.add_result(
                    language=lang_code,
                    source=source_text,
                    prediction=translation,
                    reference=reference_english,
                    detected_lang=detected_lang
                )
                
                processed += 1
                consecutive_errors = 0
                samples_since_checkpoint += 1
                
                if samples_since_checkpoint >= args.checkpoint_interval:
                    checkpoint_mgr.save(
                        evaluator, completed_languages,
                        current_language=lang_code,
                        current_processed=processed,
                        current_dataset_idx=idx + 1
                    )
                    samples_since_checkpoint = 0
                
            except Exception as e:
                consecutive_errors += 1
                print(f"\n    Error at sample {idx}: {str(e)[:80]}")
                torch.cuda.empty_cache()
                gc.collect()
        
        completed_languages.append(lang_code)
        checkpoint_mgr.save(evaluator, completed_languages)
        print(f"  Completed {lang_name}: {processed} samples")
    
    print("\n" + "=" * 70)
    print("Computing final results...")
    
    evaluator.print_summary()
    evaluator.save_results(args.output)
    checkpoint_mgr.delete()
    
    print("\nExperiment complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned NLLB translation for African languages"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Number of samples per language (default: 1000)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb_consensus/experiment_1.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
