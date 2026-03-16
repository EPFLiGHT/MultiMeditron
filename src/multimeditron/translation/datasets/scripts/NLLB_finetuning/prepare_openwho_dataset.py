"""
OpenWHO Dataset Preparation for NLLB Fine-tuning

Prepares the OpenWHO multilingual medical dataset for fine-tuning NLLB-200-3.3B.
Processes sentence-level parallel translations from English to 48 target languages,
formatting them according to NLLB's expected input structure with proper language codes.

The script:
- Maps OpenWHO ISO 639-3 codes to NLLB language codes with script variants
- Loads parallel data from the raphaelmerx/openwho HuggingFace dataset
- Formats each language pair with proper NLLB translation structure
- Saves individual language pair datasets for subsequent combination
- Generates sample files and preparation summary

Input: raphaelmerx/openwho dataset from HuggingFace
Output: Individual language pair datasets in src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_formatted/

Usage:
    python prepare_openwho_dataset.py
"""

import os
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
import json
from typing import List


NLLB_LANG_CODES = {
    'eng': 'eng_Latn',
    'amh': 'amh_Ethi',
    'ara': 'arb_Arab',
    'aze': 'azj_Latn',
    'ben': 'ben_Beng',
    'bul': 'bul_Cyrl',
    'cat': 'cat_Latn',
    'ell': 'ell_Grek',
    'fas': 'pes_Arab',
    'fra': 'fra_Latn',
    'hau': 'hau_Latn',
    'hin': 'hin_Deva',
    'hye': 'hye_Armn',
    'ind': 'ind_Latn',
    'ita': 'ita_Latn',
    'jpn': 'jpn_Jpan',
    'kat': 'kat_Geor',
    'kaz': 'kaz_Cyrl',
    'kur': 'kmr_Latn',
    'lao': 'lao_Laoo',
    'lin': 'lin_Latn',
    'mar': 'mar_Deva',
    'mkd': 'mkd_Cyrl',
    'mya': 'mya_Mymr',
    'nld': 'nld_Latn',
    'ori': 'ory_Orya',
    'pan': 'pan_Guru',
    'pol': 'pol_Latn',
    'por': 'por_Latn',
    'pus': 'pbt_Arab',
    'ron': 'ron_Latn',
    'rus': 'rus_Cyrl',
    'sin': 'sin_Sinh',
    'som': 'som_Latn',
    'spa': 'spa_Latn',
    'sqi': 'als_Latn',
    'srp': 'srp_Cyrl',
    'swa': 'swh_Latn',
    'tam': 'tam_Taml',
    'tel': 'tel_Telu',
    'tet': 'tet_Latn',
    'tgk': 'tgk_Cyrl',
    'tha': 'tha_Thai',
    'tur': 'tur_Latn',
    'ukr': 'ukr_Cyrl',
    'urd': 'urd_Arab',
    'vie': 'vie_Latn',
    'yor': 'yor_Latn',
    'zho': 'zho_Hans',
}


class OpenWHODatasetPreparation:
    """Handles preparation of OpenWHO dataset for NLLB fine-tuning."""

    def __init__(self, dataset_name: str = "raphaelmerx/openwho"):
        self.dataset_name = dataset_name
        self.source_lang = 'eng_Latn'

    def get_available_languages(self) -> List[str]:
        """Retrieve list of available language configurations from the dataset."""
        configs = get_dataset_config_names(self.dataset_name)
        return [c.replace("sent__", "") for c in configs if c.startswith("sent__")]

    def load_parallel_data(self, target_lang_iso3: str) -> Dataset:
        """Load parallel data for a specific language pair."""
        config_name = f"sent__{target_lang_iso3}"
        print(f"Loading {config_name}...")
        return load_dataset(self.dataset_name, config_name, split="train")

    def format_for_nllb(self, dataset: Dataset, target_lang_iso3: str) -> Dataset:
        """Format dataset examples to NLLB translation structure."""
        target_nllb = NLLB_LANG_CODES[target_lang_iso3]

        def fmt(example):
            return {
                "translation": {
                    self.source_lang: example["src_text"],
                    target_nllb: example["tgt_text"]
                },
                "course_id": example["course_id"],
                "section_index": example["section_index"],
                "subsection_index": example["subsection_index"]
            }

        return dataset.map(fmt, remove_columns=dataset.column_names)

    def create_train_split(self, dataset: Dataset) -> DatasetDict:
        """Wrap dataset in DatasetDict with train split."""
        return DatasetDict({"train": dataset})

    def prepare_language_pair(self, target_lang_iso3: str, output_dir: str) -> DatasetDict:
        """Prepare and save a single language pair dataset."""
        available = self.get_available_languages()
        if target_lang_iso3 not in available:
            raise ValueError(f"Language '{target_lang_iso3}' not available. Available: {available}")

        ds = self.load_parallel_data(target_lang_iso3)
        print(f"Loaded {len(ds)} examples")

        formatted = self.format_for_nllb(ds, target_lang_iso3)
        split = self.create_train_split(formatted)

        tgt_code = NLLB_LANG_CODES[target_lang_iso3]
        save_dir = os.path.join(output_dir, f"eng_Latn-{tgt_code}")
        os.makedirs(save_dir, exist_ok=True)
        split.save_to_disk(save_dir)

        sample = formatted.select(range(min(5, len(formatted))))
        with open(os.path.join(save_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(sample.to_list(), f, ensure_ascii=False, indent=2)

        print(f"Saved dataset to {save_dir}")
        return split


def prepare_multiple_languages(lang_codes: List[str], output_dir: str):
    """Prepare datasets for multiple language pairs and generate summary."""
    prep = OpenWHODatasetPreparation()
    results = {}

    for lang in lang_codes:
        print("\n" + "="*60)
        print(f"Processing: English → {lang}")
        print("="*60)

        try:
            ds = prep.prepare_language_pair(lang, output_dir)
            results[lang] = {"status": "success", "train_size": len(ds["train"])}
        except Exception as e:
            print(f"❌ Error: {e}")
            results[lang] = {"status": "failed", "error": str(e)}

    with open(os.path.join(output_dir, "preparation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    prep = OpenWHODatasetPreparation()
    available = prep.get_available_languages()
    print(f"Available languages: {len(available)}")
    print(available)

    all_languages = available

    output_dir = "src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_formatted"
    os.makedirs(output_dir, exist_ok=True)

    prepare_multiple_languages(all_languages, output_dir)