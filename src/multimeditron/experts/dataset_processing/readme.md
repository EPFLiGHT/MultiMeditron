# Multimeditron Dataset Processing

This repository contains dataset processing scripts for two medical imaging experts: **Skin** and **Ophthalmology**. Each expert has dedicated scripts to prepare and process various publicly available datasets into a unified JSONL format for training.

## Directory Structure

```
src/multimeditron/experts/dataset_processing
├── skin/
│   ├── process_skin10.py
│   ├── process_dermnet.py
│   ├── process_fitzpatrick.py
│   └── process_isic.py
├── ophthalmology/
│   ├── process_rfmid2.py
│   ├── process_messidor2.py
│   ├── process_scin.py
│   └── process_uwf_tumor.py
├── train_val_split.py
└── paraphrase_jsonl.py
```

## Overview

### Skin Expert Datasets

The skin expert processes dermatological image datasets for various skin conditions and diseases:

- **`process_skin10.py`**: Processes the Skin-10 dataset
- **`process_dermnet.py`**: Processes the DermNet dataset
- **`process_fitzpatrick.py`**: Processes the Fitzpatrick17k dataset
- **`process_isic.py`**: Processes the ISIC-4 skin disease image dataset (4 classes)

### Ophthalmology Expert Datasets

The ophthalmology expert processes retinal and eye-related medical imaging datasets:

- **`process_rfmid2.py`**: Processes the RFMiD2 (Retinal Fundus Multi-disease Image Dataset v2)
- **`process_messidor2.py`**: Processes the MESSIDOR-2 diabetic retinopathy dataset
- **`process_scin.py`**: Processes the SCIN (Smartphone-based Cataract Image Network) dataset
- **`process_uwf_tumor.py`**: Processes the UWF (Ultra-Wide Field) tumor dataset

## Common Processing Scripts

### `train_val_split.py`
Splits JSONL manifest files into training and validation sets.

**Usage:**
```bash
python train_val_split.py.py \
  --input manifest.jsonl \
  --output_train train.jsonl \
  --output_val val.jsonl \
  --val_ratio 0.2
```

### `paraphrase_jsonl.py`
Paraphrases text descriptions in JSONL files to increase linguistic diversity before feeding them to the llm for data augmentation.

**Usage:**
```bash
python paraphrase_jsonl.py \
  --input manifest.jsonl \
  --output paraphrased.jsonl \
  --seed 42
```

## Dataset Processing Workflow

### General Steps

1. **Download Dataset**: Each processing script downloads the respective dataset (typically from Kaggle or other sources)
2. **Extract and Organize**: Images are organized by class/diagnosis into a structured directory
3. **Generate Manifest**: A JSONL manifest file is created with the following structure:
   ```json
   {
     "text": "The diagnosis for this skin lesion is melanoma.",
     "modalities": [{"type": "image", "value": "path/to/image.jpg"}]
   }
   ```
4. **Optional Paraphrasing**: Text descriptions can be paraphrased for diversity
5. **Train/Val Split**: The manifest is split into training and validation sets

### Example: Processing ISIC Dataset

```bash
python skin/process_isic.py \
  --output_images_root data/isic4/images \
  --output_jsonl data/isic/manifest.jsonl

python paraphrase_jsonl.py \
  --in_jsonl data/isic4/manifest.jsonl \
  --out_jsonl data/isic/manifest_paraphrased.jsonl \
  --seed 42

```

**Arguments:**
- `--output_images_root`: Root directory where images will be stored by class
- `--output_jsonl`: Path to output JSONL manifest
- `--kaggle_dataset`: Kaggle dataset slug (default: `abhii1929/isic-skin-disease-image-dataset-4-classes`)
- `--paraphrase`: Whether to paraphrase diagnosis text
- `--seed`: Random seed for reproducibility

## Output Format

All processing scripts generate:

1. **Organized Image Directory**:
   ```
   output_images_root/
   ├── class-1/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── class-2/
       ├── image3.jpg
       └── image4.jpg
   ```

2. **JSONL Manifest**:
   Each line contains a JSON object with:
   - `text`: Natural language description of the diagnosis/condition
   - `modalities`: List of modality objects (typically one image per record)
     - `type`: Always `"image"`
     - `value`: Relative path to the image file

## Requirements

### Python Dependencies

```bash
pip install kagglehub transformers tqdm
```

### Kaggle API Setup

For datasets hosted on Kaggle, ensure you have:
1. A Kaggle account
2. API credentials configured (`~/.kaggle/kaggle.json`)

## Data Characteristics

### Skin Datasets
- **Image Types**: Dermoscopic images, clinical photographs
- **Conditions**: Melanoma, basal cell carcinoma, actinic keratosis, benign lesions, etc.
- **Format**: JPEG, PNG

### Ophthalmology Datasets
- **Image Types**: Fundus photographs, retinal scans, slit-lamp images
- **Conditions**: Diabetic retinopathy, cataracts, retinal tumors, various retinal diseases
- **Format**: JPEG, PNG