# MultiMeditron Model & Dataset Registry

This document provides a comprehensive reference of all model checkpoints and training datasets used in the MultiMeditron project, including their paths on CSCS, descriptions, and data format details.

---

## Models

All model checkpoints are stored on CSCS under two storage tiers:

- **capstor** (persistent): `/capstor/store/cscs/swissai/a127/homes/meditron/models/multimeditron/`
- **iopsstor** (scratch): `/iopsstor/scratch/cscs/surech/multimeditron/checkpoints/`

> None of the models below have been published to HuggingFace Hub yet.

### Base LLMs

| Model | HuggingFace ID | Parameters |
|---|---|---|
| Meditron3-8B | [OpenMeditron/Meditron3-8B](https://huggingface.co/OpenMeditron/Meditron3-8B) | 8B |
| LLaMA 3.1-8B Instruct | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B |
| Apertus-8B | — | 8B |

### Vision Encoders (CLIP Experts)

| Expert | Local Path | Domain |
|---|---|---|
| Generalist CLIP | `models/CLIP/clip-vit-base-patch32` | General |
| MedExpert-CT | `models/CLIP/MedExpert-CT` | CT imaging |
| MedExpert-MRI | `models/CLIP/MedExpert-MRI` | MRI imaging |
| MedExpert-Ultrasound | `models/CLIP/MedExpert-Ultrasound` | Ultrasound |
| MedExpert-Xray | `models/CLIP/MedExpert-Xray` | X-ray |
| MedExpert-Pathology | `models/CLIP/ClosedMeditron/` | Pathology (histology) |
| MedExpert-Ophthalmology | `models/CLIP/ClosedMeditron/` | Ophthalmology (fundus, OCT) |
| MedExpert-Dermatology | `models/CLIP/ClosedMeditron/` | Dermatology (skin lesions) |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Biomedical (general) |
| Gating Network | `models/CLIP/MultiMeditron-Gating` | Expert routing |

### End-to-end Trained Models (Stage 2)

**Capstor root**: `/capstor/store/cscs/swissai/a127/homes/meditron/models/multimeditron/unfreeze/`

#### MoE Models (5 experts: CT, MRI, Ultrasound, X-ray, generalist CLIP)

| Model | Path | Final Ckpt | Last Trained | Author | Base LLM | Fusion | Projection |
|---|---|---|---|---|---|---|---|
| ATTN-PEP | `attn_pep/MultiMeditron-8B-attn-pep-end2end/` | 3063 | 2026-01-01 | mzhang | LLaMA 3.1-8B | Cross-attention | Per-expert (PEP) |
| ATTN-SHARED | `attn_shared/MultiMeditron-8B-attn-shared-end2end/` | 1532 | 2025-12-22 | theoschiff | LLaMA 3.1-8B | Cross-attention | Shared |
| AVG-PEP | `avg_pep/MultiMeditron-8B-avg-pep-end2end/` | 1532 | 2025-12-22 | theoschiff | LLaMA 3.1-8B | Average | Per-expert (PEP) |
| AVG-SHARED | `avg_shared/MultiMeditron-8B-avg-shared-end2end/` | 1532 | 2025-12-22 | theoschiff | LLaMA 3.1-8B | Average | Shared |
| CAT-PEP | `cat_pep/MultiMeditron-8B-cat-pep-end2end/` | 1532 | 2026-01-07 | theoschiff | LLaMA 3.1-8B | Concatenation | Per-expert (PEP) |
| CAT-SHARED | `cat_shared/MultiMeditron-8B-cat-shared-end2end/` | 1532 | 2026-01-07 | theoschiff | LLaMA 3.1-8B | Concatenation | Shared |

#### Single Vision Encoder Models

| Model | Path | Final Ckpt | Last Trained | Author | Base LLM | Vision Encoder |
|---|---|---|---|---|---|---|
| Qwen3-4B BiomedCLIP | `single_clip/MultiMeditron-Qwen-4B-End2End-BiomedCLIP/` | 3063 | 2026-01-27 | mzhang | Qwen3-4B | BiomedCLIP |
| Apertus-8B BiomedCLIP | `single_clip/MultiMeditron-Apertus-8B-End2End-BiomedCLIP/` | 3063 | 2025-12-30 | mzhang | Apertus-8B | BiomedCLIP |
| LLaMA3.1-8B BiomedCLIP | `single_clip/MultiMeditron-Llama-8B-End2End-BiomedCLIP/` | 3063 | 2026-01-03 | mzhang | LLaMA 3.1-8B | BiomedCLIP |
| LLaMA3.1-8B CLIP | `single_clip/MultiMeditron-Llama-8B-End2End-CLIP/` | 3063 | 2026-01-26 | mzhang | LLaMA 3.1-8B | CLIP ViT-B/32 |

#### 7-expert ATTN-PEP (in-progress)

| Model | Path | Latest Ckpt | Last Trained | Author |
|---|---|---|---|---|
| ATTN-PEP 7-expert | `/iopsstor/.../MultiMeditron-8B-attn-pep-end2end-7exp/` | 800 | 2026-03-24 | surech |

Experts: CT, MRI, Ultrasound, X-ray, Pathology, Ophthalmology, Dermatology.

### Alignment Models (Stage 1)

**Capstor root**: `/capstor/store/cscs/swissai/a127/homes/meditron/models/multimeditron/freeze/`

Alignment (Stage 1) checkpoints exist for the same variants listed above, plus additional MoE-specific variants:
`moe_avg_pep`, `moe_avg_shared`, `moe_cat_pep`, `moe_cat_shared`.

---

## Datasets

All training datasets are stored on capstor in Arrow format at:

```
/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/
```

The HuggingFace dataset is published at [OpenMeditron/MultiMediset](https://huggingface.co/datasets/OpenMeditron/MultiMediset).

### Alignment Datasets (Stage 1)

| Dataset | Path | Domain | Format | Samples | Source |
|---|---|---|---|---|---|
| LLaVA Pretrain | `llava_pretrain_cleaned` | Generalist | Not formatted | 558 000 | [LLaVA](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| pixmo_anything | `pixmo_anything` | Generalist | Not formatted | 101 000 | [Pixmo](https://huggingface.co/datasets/allenai/pixmo-anything) |
| pixmo-cap | `pixmo_cap` | Generalist | Not formatted | 717 000 | [Pixmo](https://huggingface.co/datasets/allenai/pixmo-cap) |
| MedTrinity Alignment | `medtrinity_conversations_1_formatted_alignment` | Medical | Well formatted | 100 000 | MedTrinity |

### End-to-end Datasets (Stage 2)

#### Generalist

| Dataset | Path | Format | Samples | Source |
|---|---|---|---|---|
| LLaVA Instruct | `llava_instruct` | Not formatted | 624 000 | [LLaVA](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| Mammoth | `image_mammoth` | Well formatted | 1 639 212 | [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH) |

#### Medical — General

| Dataset | Path | Format | Samples | Source |
|---|---|---|---|---|
| MedTrinity 1 | `medtrinity_conversations_1_formatted` | Well formatted | 5 032 522 | MedTrinity |
| MedTrinity 2 | `medtrinity_conversations_2_formatted` | Well formatted | 5 032 523 | MedTrinity |
| PMC-VQA Full | `PMC_VQA_FULL` | Well formatted | 176 948 | [PMC-VQA](https://huggingface.co/datasets/RadGenome/PMC-VQA) |

#### Medical — Ultrasound

| Dataset | Path | Format | Samples | Source |
|---|---|---|---|---|
| BUSI | `BUSI` | Well formatted | 773 | [Kaggle](https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset) |
| COVID-US | `COVID_US` | Well formatted | 30 100 | [GitHub](https://github.com/nrc-cnrc/COVID-US) |
| CT2 (Kidney) | `ct2` | Well formatted | 6 460 | [Kaggle](https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg) |
| DDTI (Thyroid) | `DDTI` | Well formatted | 347 | [Kaggle](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) |

#### Medical — Radiology

| Dataset | Path | Format | Samples | Source |
|---|---|---|---|---|
| IU X-Ray | `iu_xray` | Well formatted | 2 960 | [Kaggle](https://www.kaggle.com/datasets/areebbinnadeem/iu-xray) |

#### Medical — Dermatology

| Dataset | Path | Format | Samples | Sources |
|---|---|---|---|---|
| Skin | `skin_dataset_converted` | Well formatted | 70 461 | [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet), [ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic), [SCIN](https://github.com/google-research-datasets/scin), [skin_diseases_10](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset), [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) |

#### Medical — Ophthalmology

| Dataset | Path | Format | Samples | Sources |
|---|---|---|---|---|
| Eye | `eye_dataset_converted` | Well formatted | 36 148 | [EyePACS](https://huggingface.co/datasets/bumbledeep/eyepacs), [SLID-E](https://figshare.com/articles/dataset/_i_SLID-E_Slit_Lamp_Image_Dataset_for_Epiphora_-_A_Benchmark_Resource_for_Automated_Tear_Overflow_Analysis_i_/26172919), [UWF Fundus](https://springernature.figshare.com/articles/dataset/An_ultra-wide-field_fundus_image_dataset_for_intelligent_diagnosis_of_intraocular_tumors/27986258), [Messidor2](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess), [RFMiD2](https://www.mdpi.com/2306-5729/6/2/14), [UWF IQA](https://springernature.figshare.com/articles/dataset/Open_ultrawidefield_fundus_image_dataset_with_disease_diagnosis_and_clinical_image_quality_assessment/26936446) |

---

## Data Format

All datasets are stored as HuggingFace `datasets` Arrow files and can be loaded with:

```python
from datasets import load_from_disk
ds = load_from_disk("/capstor/.../arrow/<dataset_name>")
```

### Schema

Every dataset uses the same Arrow schema with two columns:

| Column | Type | Description |
|---|---|---|
| `modalities` | `List[{type: str, value: {bytes: binary, path: null}}]` | List of image modalities; each entry has a `type` (e.g. `"image"`) and the raw image `bytes` |
| `conversations` | `List[{role: str, content: str}]` | Multi-turn conversation between `user` and `assistant` |

### "Well formatted" vs "Not formatted"

The schema is identical for both. The distinction is about **conversation quality**:

- **Well formatted**: conversations follow the MultiMeditron instruction template with a `<attachment>` placeholder for the image and properly structured question-answer pairs.
- **Not formatted**: raw conversations from the source dataset that may need preprocessing before training. They are still usable but may have inconsistent formatting.

### Example: Well-formatted sample (BUSI)

```json
{
  "modalities": [
    {
      "type": "image",
      "value": {"bytes": "<raw PNG bytes>", "path": null}
    }
  ],
  "conversations": [
    {
      "role": "user",
      "content": "<attachment>\nWhat type of lesion is visible in this breast ultrasound image?"
    },
    {
      "role": "assistant",
      "content": "The ultrasound image shows a benign breast lesion with well-defined margins..."
    }
  ]
}
```

### Example: Not-formatted sample (LLaVA Pretrain)

```json
{
  "modalities": [
    {
      "type": "image",
      "value": {"bytes": "<raw JPEG bytes>", "path": null}
    }
  ],
  "conversations": [
    {
      "role": "user",
      "content": "<attachment>\nDescribe this image in detail."
    },
    {
      "role": "assistant",
      "content": "The image shows a busy street scene with pedestrians..."
    }
  ]
}
```

### Dataset Size Summary

| Stage | Total Samples | Medical | Generalist |
|---|---|---|---|
| Alignment (Stage 1) | ~1 476 000 | 100 000 | 1 376 000 |
| End-to-end (Stage 2) | ~12 652 354 | 10 389 142 | 2 263 212 |
| **Total** | **~14 128 354** | **10 489 142** | **3 639 212** |
