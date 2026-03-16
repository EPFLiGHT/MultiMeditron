"""
Translate MediBench to African Languages (Fine-tuned NLLB)

Translates MediBench medical Q&A from high-resource languages to African languages
using fine-tuned NLLB. Includes preemption-safe checkpointing for cluster environments.

Default: 500 samples × 5 source languages × 5 target languages = 12,500 translations
"""

import os
import sys
import gc
import json
from pathlib import Path
from typing import Set

import torch
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import hf_hub_download

SRC_ROOT = next((parent for parent in Path(__file__).resolve().parents if parent.name == "src"), None)
if SRC_ROOT is not None and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimeditron.translation.translator import NLLBTranslator

AFRICAN_LANGUAGES = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}

LANG_TO_NLLB = {
    'zh': 'zho_Hans',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'en': 'eng_Latn',
    'ar': 'arb_Arab',
    'de': 'deu_Latn',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'it': 'ita_Latn',
    'nl': 'nld_Latn',
    'pl': 'pol_Latn',
    'tr': 'tur_Latn',
    'vi': 'vie_Latn',
    'th': 'tha_Thai',
    'hi': 'hin_Deva',
    'id': 'ind_Latn',
}

HIGH_RESOURCE_LANGS = {'zh', 'es', 'fr', 'ja', 'ru'}


def load_checkpoint_indices(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {int(line.strip()) for line in f if line.strip()}


def append_checkpoint_index(path: Path, idx: int):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"{idx}\n")
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, obj: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class MediBenchTranslator:
    
    def __init__(self):
        print("\n[1/2] Loading fine-tuned NLLB translator...")
        self.translator = NLLBTranslator()
        print("  Fine-tuned NLLB loaded")
        
        self.stats = {
            'total_samples': 0,
            'total_translations': 0,
            'failed': 0,
            'by_target_lang': {lang: 0 for lang in AFRICAN_LANGUAGES},
            'by_source_lang': {},
        }
    
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or not text.strip():
            return text
        try:
            return self.translator.translate(text, src_lang=src_lang, tgt_lang=tgt_lang)
        except Exception as exc:
            print(f"  [WARNING] Translation failed for {src_lang}->{tgt_lang}: {exc}")
            self.stats['failed'] += 1
            return text
    
    def translate_sample(self, sample: dict) -> list:
        source_lang = sample.get('language')
        if source_lang not in LANG_TO_NLLB:
            return []
        question = sample.get('question')
        options = sample.get('options')
        answer = sample.get('answer')
        if not question or not isinstance(options, list) or not options or answer is None:
            self.stats['failed'] += 1
            return []
        
        src_nllb = LANG_TO_NLLB[source_lang]
        self.stats['by_source_lang'][source_lang] = (
            self.stats['by_source_lang'].get(source_lang, 0) + 1
        )
        
        translations = []
        for lang_code, (nllb_code, _) in AFRICAN_LANGUAGES.items():
            translated_question = self.translate_text(question, src_nllb, nllb_code)
            translated_options = [
                self.translate_text(opt, src_nllb, nllb_code)
                for opt in options
            ]
            
            translations.append({
                'language': lang_code,
                'question': translated_question,
                'options': translated_options,
                'answer': answer,
                'source_language': source_lang,
            })
            
            self.stats['by_target_lang'][lang_code] += 1
            self.stats['total_translations'] += 1
        
        return translations
    
    def translate_dataset(self, dataset: Dataset, output_jsonl: Path, checkpoint_path: Path):
        completed = load_checkpoint_indices(checkpoint_path)
        
        print(f"\n[2/2] Translating dataset (checkpoint-enabled)")
        print(f"  Completed samples: {len(completed)}")
        print(f"  Remaining: {len(dataset) - len(completed)}")
        
        self.stats['total_samples'] = len(dataset)
        
        for i in tqdm(range(len(dataset)), desc="  Translating"):
            if i in completed:
                continue
            
            sample = dataset[i]
            translations = self.translate_sample(sample)
            
            for mcq in translations:
                append_jsonl(output_jsonl, mcq)
            
            append_checkpoint_index(checkpoint_path, i)
            
            if (i + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    
    def print_stats(self):
        print("\n" + "=" * 70)
        print("TRANSLATION STATISTICS")
        print("=" * 70)
        
        print(f"\nSource samples: {self.stats['total_samples']}")
        print(f"Total translations: {self.stats['total_translations']}")
        print(f"Failed: {self.stats['failed']}")
        
        print("\nBy target language:")
        for lang_code in sorted(AFRICAN_LANGUAGES.keys()):
            lang_name = AFRICAN_LANGUAGES[lang_code][1]
            count = self.stats['by_target_lang'][lang_code]
            print(f"  {lang_name} ({lang_code}): {count:5d}")
        
        print("\nBy source language:")
        for src_lang, count in sorted(
            self.stats['by_source_lang'].items(), 
            key=lambda x: -x[1]
        ):
            print(f"  {src_lang}: {count}")
        
        print("\n" + "=" * 70)


def load_medibench(split: str = "train", token: str = None) -> Dataset:
    print(f"  Loading MediBench ({split})...")
    
    file_path = hf_hub_download(
        repo_id="ClosedMeditron/MediBench",
        filename=f"{split}.jsonl",
        repo_type="dataset",
        token=token
    )
    
    samples = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line.strip())
                
                if isinstance(sample.get('answer'), list):
                    sample['answer'] = str(sample['answer'][0]) if sample['answer'] else None
                elif sample.get('answer') is not None:
                    sample['answer'] = str(sample['answer'])
                
                if not isinstance(sample.get('options'), list):
                    sample['options'] = []
                
                samples.append(sample)
                
            except Exception:
                skipped += 1
                continue
    
    if skipped > 0:
        print(f"  Skipped {skipped} malformed lines")
    
    print(f"  Loaded {len(samples)} samples")
    return Dataset.from_list(samples)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Translate MediBench to African languages using fine-tuned NLLB"
    )
    parser.add_argument(
        "--samples_per_language",
        type=int,
        default=500,
        help="Samples per high-resource source language (default: 500)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb_consensus/mediBench_translation/african_translations.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb_consensus/mediBench_translation/checkpoints",
        help="Checkpoint directory"
    )
    
    args = parser.parse_args()

    hf_lab_token = os.getenv("HF_LAB_TOKEN")
    if not hf_lab_token:
        raise EnvironmentError(
            "HF_LAB_TOKEN is required to download MediBench. "
            "Set it in the environment before running this script."
        )
    
    print("=" * 70)
    print("MEDIBENCH → AFRICAN LANGUAGES (FINE-TUNED NLLB)")
    print("=" * 70)
    print(f"\nSource languages: {', '.join(sorted(HIGH_RESOURCE_LANGS))}")
    print(f"Target languages: {', '.join(f'{v[1]} ({k})' for k, v in AFRICAN_LANGUAGES.items())}")
    print(f"Samples per source language: {args.samples_per_language}")
    
    total_sources = args.samples_per_language * len(HIGH_RESOURCE_LANGS)
    total_translations = total_sources * len(AFRICAN_LANGUAGES)
    print(f"Total source samples: {total_sources}")
    print(f"Total translations: {total_translations}")
    
    print("\n[0/2] Loading MediBench dataset...")
    dataset = load_medibench(split="train", token=hf_lab_token)
    
    samples_by_lang = {}
    for sample in dataset:
        lang = sample.get('language')
        if lang in HIGH_RESOURCE_LANGS:
            samples_by_lang.setdefault(lang, []).append(sample)
    
    print("\nAvailable samples per language:")
    for lang in sorted(HIGH_RESOURCE_LANGS):
        count = len(samples_by_lang.get(lang, []))
        print(f"  {lang}: {count}")
    
    selected_samples = []
    for lang in HIGH_RESOURCE_LANGS:
        lang_samples = samples_by_lang.get(lang, [])[:args.samples_per_language]
        selected_samples.extend(lang_samples)
        print(f"  Selected {len(lang_samples)} from {lang}")
    
    dataset = Dataset.from_list(selected_samples)
    print(f"\nTotal selected samples: {len(dataset)}")
    
    output_path = Path(args.output)
    output_jsonl = output_path.with_suffix(".jsonl")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (output_path.stem + ".ckpt")
    
    translator = MediBenchTranslator()
    translator.translate_dataset(dataset, output_jsonl, checkpoint_path)
    translator.print_stats()
    
    print("\nConsolidating JSONL to JSON...")
    
    translations = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            translations.append(json.loads(line))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, indent=2, ensure_ascii=False)
    
    stats_path = output_path.parent / "translation_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(translator.stats, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print(f"Statistics saved to: {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
