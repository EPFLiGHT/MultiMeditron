"""
Translate MediBench to African Languages

Translates MediBench medical Q&A from source languages to African languages
using NLLB-200 3.3B. Outputs JSON with questions, options, and answers.

Default: 500 samples × 5 languages into 5 languages = 12500 translations

Output: src/multimeditron/translation/experiments/results/base_nllb/mediBench_translation/african_translations.json
"""

import os
import sys
from pathlib import Path
from datasets import Dataset
import torch
from tqdm import tqdm
import json
import gc
from huggingface_hub import hf_hub_download

src_root = next((parent for parent in Path(__file__).resolve().parents if parent.name == "src"), None)
if src_root is not None and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from multimeditron.translation.translator import NLLBTranslator


AFRICAN_LANGUAGES = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}

LANG_TO_NLLB = {
    'zh': 'zho_Hans', 'es': 'spa_Latn', 'fr': 'fra_Latn', 'en': 'eng_Latn',
    'ar': 'arb_Arab', 'de': 'deu_Latn', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang',
    'pt': 'por_Latn', 'ru': 'rus_Cyrl', 'it': 'ita_Latn', 'nl': 'nld_Latn',
    'pl': 'pol_Latn', 'tr': 'tur_Latn', 'vi': 'vie_Latn', 'th': 'tha_Thai',
    'hi': 'hin_Deva', 'id': 'ind_Latn',
}

HIGH_RESOURCE_LANGS = {'zh', 'es', 'fr', 'ja', 'ru'}


class MediBenchTranslator:
    """Translates MediBench to African languages using NLLB."""
    
    def __init__(self):
        """Initialize with NLLB-200 3.3B model."""
        print("\n[1/2] Loading NLLB-200 3.3B translator...")
        self.translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")
        print("   ✅ NLLB-200 3.3B loaded")
        
        self.stats = {
            'total_samples': 0,
            'total_translations': 0,
            'by_target_lang': {lang: 0 for lang in AFRICAN_LANGUAGES.keys()},
            'by_source_lang': {},
            'failed': 0
        }
    
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text, return original on failure."""
        if not text or not text.strip():
            return text
        
        try:
            return self.translator.translate(text, src_lang=src_lang, tgt_lang=tgt_lang)
        except Exception as e:
            print(f"\n   [WARNING] Translation failed: {str(e)[:100]}")
            self.stats['failed'] += 1
            return text
    
    def translate_sample(self, sample: dict) -> list:
        """Translate one sample to all African languages."""
        question = sample.get('question', '')
        options = sample.get('options', [])
        answer = sample.get('answer', '')
        source_lang = sample.get('language', '')
        
        if not question or not options or not answer or not source_lang:
            return []
        
        src_nllb = LANG_TO_NLLB.get(source_lang)
        if not src_nllb:
            return []
        
        if source_lang not in self.stats['by_source_lang']:
            self.stats['by_source_lang'][source_lang] = 0
        self.stats['by_source_lang'][source_lang] += 1
        
        translations = []
        
        for lang_code, (nllb_code, lang_name) in AFRICAN_LANGUAGES.items():
            try:
                translated_question = self.translate_text(question, src_nllb, nllb_code)
                translated_options = [
                    self.translate_text(opt, src_nllb, nllb_code) for opt in options
                ]
                
                mcq = {
                    'language': lang_code,
                    'question': translated_question,
                    'options': translated_options,
                    'answer': answer,
                    'source_language': source_lang
                }
                
                translations.append(mcq)
                self.stats['by_target_lang'][lang_code] += 1
                self.stats['total_translations'] += 1
                
            except Exception as e:
                print(f"\n   [ERROR] Failed translating to {lang_code}: {str(e)[:100]}")
                self.stats['failed'] += 1
                continue
        
        return translations
    
    def translate_dataset(self, dataset):
        """Translate dataset to all African languages."""
        print("\n[2/2] Translating dataset...")
        
        self.stats['total_samples'] = len(dataset)
        print(f"   Processing {len(dataset)} samples × {len(AFRICAN_LANGUAGES)} languages")
        print(f"   Expected output: {len(dataset) * len(AFRICAN_LANGUAGES)} translations")
        
        all_translations = []
        
        for i in tqdm(range(len(dataset)), desc="   Translating"):
            sample = dataset[i]
            translations = self.translate_sample(sample)
            all_translations.extend(translations)
            
            if (i + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        print(f"\n   ✅ Translation complete! Generated {len(all_translations)} MCQs")
        return all_translations
    
    def save_translations(self, translations: list, output_file: str):
        """Save translations and statistics."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[3/3] Saving translations to {output_file}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved {len(translations)} MCQs")
        
        stats_file = output_path.parent / "translation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"   📊 Statistics saved to {stats_file}")
        self.print_stats()
    
    def print_stats(self):
        """Print translation statistics."""
        print("\n" + "="*70)
        print("TRANSLATION STATISTICS")
        print("="*70)
        
        print(f"\n📊 Source samples: {self.stats['total_samples']}")
        print(f"✅ Total translations: {self.stats['total_translations']}")
        print(f"❌ Failed: {self.stats['failed']}")
        
        print("\n🌍 By target language:")
        for lang_code in sorted(AFRICAN_LANGUAGES.keys()):
            lang_name = AFRICAN_LANGUAGES[lang_code][1]
            count = self.stats['by_target_lang'][lang_code]
            print(f"   {lang_name} ({lang_code}): {count:5d}")
        
        print("\n📚 By source language:")
        for src_lang, count in sorted(self.stats['by_source_lang'].items(), 
                                       key=lambda x: -x[1])[:10]:
            print(f"   {src_lang}: {count}")
        
        print("\n" + "="*70)


def load_medibench_robust(split="train", token=None):
    """Load MediBench with handling for inconsistent format."""
    print(f"   Loading MediBench {split} split...")
    
    file_path = hf_hub_download(
        repo_id="ClosedMeditron/MediBench",
        filename=f"{split}.jsonl",
        repo_type="dataset",
        token=token
    )
    
    samples = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line.strip())
                
                if isinstance(sample.get('answer'), list):
                    sample['answer'] = sample['answer'][0] if sample['answer'] else None
                
                if sample.get('answer') is not None:
                    sample['answer'] = str(sample['answer'])
                
                for field in ['question', 'rationale', 'meta_info', 'subject', 'equation_solution']:
                    if isinstance(sample.get(field), list):
                        sample[field] = ' '.join(str(x) for x in sample[field]) if sample[field] else None
                
                for field in ['options', 'choices', 'metamap_phrases']:
                    if sample.get(field) is None:
                        sample[field] = []
                    elif not isinstance(sample.get(field), list):
                        sample[field] = [str(sample[field])]
                
                samples.append(sample)
                
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"      ⚠️  Skipping malformed line {line_num}: {str(e)[:60]}")
                continue
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"      ⚠️  Error on line {line_num}: {str(e)[:60]}")
                continue
    
    if skipped > 5:
        print(f"      ⚠️  ... and {skipped - 5} more skipped lines")
    
    print(f"   ✅ Successfully loaded {len(samples)} samples")
    return Dataset.from_list(samples)


def main():
    """Main translation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Translate MediBench to African languages using NLLB-200 3.3B"
    )
    parser.add_argument(
        "--samples_per_language",
        type=int,
        default=500,
        help="Samples per high-resource source language (default: 500)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/mediBench_translation/african_translations_subset.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MEDIBENCH → AFRICAN LANGUAGES TRANSLATION")
    print("="*70)
    print(f"\nSource languages: {', '.join(sorted(HIGH_RESOURCE_LANGS))}")
    print(f"Target languages: {', '.join(f'{v[1]} ({k})' for k, v in AFRICAN_LANGUAGES.items())}")
    print(f"Model: NLLB-200 3.3B")
    print(f"Output: {args.output}")
    print(f"Samples per source language: {args.samples_per_language}")
    total_sources = args.samples_per_language * len(HIGH_RESOURCE_LANGS)
    total_translations = total_sources * len(AFRICAN_LANGUAGES)
    print(f"Total source samples: {total_sources}")
    print(f"Total translations: {total_translations}")
    
    print("\n[0/3] Loading MediBench dataset...")
    dataset = load_medibench_robust(
        split="train",
        token=os.getenv("HF_LAB_TOKEN")
    )
    print(f"   ✅ Loaded {len(dataset)} samples")
    
    print(f"\n[1/3] Filtering to high-resource languages...")
    samples_by_lang = {}
    for sample in dataset:
        lang = sample.get('language')
        if lang in HIGH_RESOURCE_LANGS:
            if lang not in samples_by_lang:
                samples_by_lang[lang] = []
            samples_by_lang[lang].append(sample)
    
    print(f"   Found samples per language:")
    for lang in sorted(HIGH_RESOURCE_LANGS):
        count = len(samples_by_lang.get(lang, []))
        print(f"      {lang}: {count}")
    
    selected_samples = []
    for lang in HIGH_RESOURCE_LANGS:
        if lang in samples_by_lang:
            lang_samples = samples_by_lang[lang][:args.samples_per_language]
            selected_samples.extend(lang_samples)
            print(f"   Selected {len(lang_samples)} from {lang}")
    
    filtered_dataset = Dataset.from_list(selected_samples)
    print(f"\n   ✅ Total selected samples: {len(filtered_dataset)}")
    
    translator = MediBenchTranslator()
    
    translations = translator.translate_dataset(filtered_dataset)
    
    translator.save_translations(translations, args.output)
    
    print("\n✅ All done!\n")


if __name__ == "__main__":
    main()
