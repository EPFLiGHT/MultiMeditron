# MultiMeditron: Multilingual Medical Translation for Low-Resource Languages

Medical domain adaptation of NLLB-200 for clinical translation in low-resource African and Asian languages. This project fine-tunes translation models on consensus-generated parallel medical corpora to enable multilingual medical AI.

## 📖 Project Overview

This repository contains the complete pipeline for:
- Generating high-quality synthetic parallel medical data via translation consensus
- Fine-tuning NLLB-200-3.3B on medical domain translations
- Evaluating translation quality and downstream medical QA performance
- Production-ready translation interface with automatic language detection

**Target Languages**: Swahili, Hausa, Amharic, Yoruba, Zulu, Tamil

## 🎯 Core Component: `translator.py`

**The main translation interface** - production-ready NLLB-200 translator with intelligent language detection.

### Key Features

- **Automatic Language Detection**: Uses fastText with confidence thresholding (80%)
- **Smart Routing Strategy**: 
  - High confidence (≥80%) → Translate via NLLB
  - Low confidence (<80%) → Pass through as-is (prevents mistranslation of ambiguous text)
- **Bidirectional Translation Pipeline**:
  - `translate_to_english()`: Detects language and translates to English
  - `translate_from_english()`: Translates back with explicit target language
- **Flexible Usage**: Direct translation with `translate(text, src_lang, tgt_lang)`
- **Model Agnostic**: Works with both base NLLB-200-3.3B and fine-tuned models

### Usage Example

```python
from multimeditron.translation.translator import NLLBTranslator

# Initialize with base or fine-tuned model
translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")

# Option 1: Automatic detection + bidirectional translation
english_query, user_lang = translator.translate_to_english(
    "Dalili za malaria ni zipi?",
    return_detected_lang=True,
)
# → Automatically detects Swahili, translates to English

# Process with medical LLM...
english_response = "Common symptoms include fever, chills, and headache."

# Translate response back to user's language
user_response = translator.translate_from_english(english_response, tgt_lang=user_lang)
# → Translates back to detected Swahili

# Option 2: Direct translation with explicit language codes
translation = translator.translate(
    text="What are the symptoms of malaria?",
    src_lang="eng_Latn",
    tgt_lang="swh_Latn"
)
```

### Translation Strategy

The confidence threshold prevents incorrect translations:
- **Short/ambiguous text** → Low confidence → Pass through unchanged
- **Clear language signal** → High confidence → Translate accurately

This design choice prioritizes **avoiding mistranslation** over attempting translation of uncertain text.

## 🏗️ Repository Structure

```
src/multimeditron/translation/
├── translator.py               # ⭐ Main translation interface (CORE)
├── datasets/                   # Training and evaluation data
├── experiments/                # Evaluation scripts and results
└── models/                     # Fine-tuned model checkpoints
```

### `datasets/`

Contains medical corpora used for training and evaluation:

- **`raw/`**: Original unprocessed datasets
  - `afriberta_raw/`: African language medical text
  - `xlsum_local/`: Medical news summaries
  - `healthcare_hindi_punjabi/`: Hindi/Punjabi medical datasets

- **`formatted_datasets/`**: Preprocessed datasets ready for training
  - `general_datasets/`: Wikipedia, FineWeb (non-medical)
  - `healthcare_datasets/`: Medical-specific corpora
    - `afriberta_medical_jsonl/`: African medical text
    - `openwho_formatted/`: WHO educational materials (50+ parallel language pairs)
    - `xlsum_medical_jsonl/`: Medical news by language
    - `guidelines_consensus/`: Consensus translation evaluation sets

- **`generated_datasets/consensus/`**: Synthetically generated training data with checkpoints
  - `{lang}_train.jsonl`: Consensus-translated WHO guidelines
  - Each entry contains translations from all 3 models + consensus scores + selected best translation

- **`scripts/`**: Dataset preprocessing and preparation
  - `NLLB_finetuning/`: Prepare datasets for NLLB fine-tuning
  - `reformatting_scripts/`: Convert raw datasets to training format

- **`training/`**: Model fine-tuning scripts
  - `finetune_nllb_consensus.py`: Train on consensus-translated data (main approach)
  - `finetune_nllb_openwho.py`: Train on OpenWHO parallel corpus (exploratory)

### `experiments/`

Systematic evaluation framework:

- **`experiments_on_base_nllb/`**: Baseline evaluation (pretrained NLLB-200-3.3B)
  - `experiment_0_medical_qa.py`: High-resource language QA (native vs translated)
  - `experiment_1_african_languages.py`: African language translation quality (BLEU, chrF, BERTScore)
  - `experiment_2_medical_qa_translate.py`: Meditron vs NLLB translation comparison
  - `experiment_3_medical_qa_lowres.py`: Low-resource end-to-end QA pipeline

- **`experiments_on_finetuned_nllb/`**: Fine-tuned model evaluation
  - Identical experiment structure for direct comparison

- **`results/`**: Experimental outputs (JSON)
  - `base_nllb/`: Baseline NLLB-200 metrics
  - `finetuned_nllb_consensus/`: Consensus fine-tuned model metrics
  - `mediBench_translation/`: Translation quality statistics

- **`scripts/`**: Auxiliary evaluation tools
  - `translation_consensus.py`: Multi-model consensus pipeline (Qwen2.5 + M2M100 + MADLAD-400)
  - `translateMediBench_*.py`: Batch translate evaluation datasets
  - `filter_translated_medibench_*.py`: Quality filtering for translations

### `models/`

Fine-tuned model checkpoints:

- **`nllb-consensus-finetuned/`**: Complete fine-tuned models
  - `checkpoint-*/`: Intermediate training checkpoints
  - `model-*.safetensors`: Sharded model weights
  - `sentencepiece.bpe.model`: Tokenizer vocabulary
  - `training_config.json`: Hyperparameters and dataset configuration
  - `training_metadata.json`: Final training statistics (loss, BLEU, chrF)
  - `logs/`: TensorBoard training logs

## 🚀 Quick Start

### 1. Generate Consensus Translations
```bash
python experiments/scripts/translation_consensus.py \
  --input /path/to/english_medical_texts.jsonl \
  --output datasets/generated_datasets/consensus \
  --languages amh swa tam hau yor zul \
  --batch-size 32
```

### 2. Fine-tune NLLB-200
```bash
python datasets/training/finetune_nllb_consensus.py
```

### 3. Run Baseline Experiments
```bash
cd experiments/experiments_on_base_nllb
python experiment_0_medical_qa.py --max_samples 1000
python experiment_1_african_languages.py
python experiment_2_medical_qa_translate.py
python experiment_3_medical_qa_lowres.py
```

### 4. Use the Translator in Production

```python
from multimeditron.translation.translator import NLLBTranslator

# Initialize translator
translator = NLLBTranslator(
    model_name="path/to/nllb-consensus-finetuned"  # or "facebook/nllb-200-3.3B"
)

# Translate user query to English
user_query = "Ni vipi naweza kuzuia malaria?"
english_query, user_lang = translator.translate_to_english(
    user_query,
    return_detected_lang=True,
)

# Get medical LLM response (in English)
# ... your medical LLM processing here ...

# Translate response back to user's language
english_response = "You can prevent malaria by using mosquito nets and taking antimalarial medication."
user_response = translator.translate_from_english(english_response, tgt_lang=user_lang)
```

## 📊 Key Findings

| Finding | Result |
|---------|--------|
| **Low-resource languages** | Translation essential: 25.9% → 38.8% accuracy (+12.9pts, p<0.001) |
| **High-resource languages** | Translation unnecessary: 60.1% native vs 59.5% with translation |
| **Translation bottleneck** | Input translation acceptable (BERTScore 86-90), output translation poor |
| **Consensus approach** | Successfully generates high-quality synthetic parallel data at scale |

## 📝 Citation

```bibtex
@techreport{grieder2026meditron,
  title={Making Meditron Multilingual: Medical Domain Adaptation of NLLB-200 for Low-Resource Languages},
  author={Grieder, Lea and Nemo, Fabrice and Hartley, Mary-Anne},
  year={2026},
  institution={LiGHT, EPFL}
}
```

## 📧 Contact

- **Author**: Lea Grieder (lea.grieder@epfl.ch)
- **Lab**: [LiGHT @ EPFL](https://light.epfl.ch)

---
