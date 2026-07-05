"""
Script to process COVID-US-2026 dataset and create expert/llm formatted datasets.
Uses MMIRAGE with VLM to generate clinical descriptions from extracted metadata and images.

Outputs:
  - Expert format: {"text": VLM-generated description, "modalities": [{"type": "image", "value": path}]}
  - LLM format: {"conversations": [...], "modalities": [{"type": "image", "value": {"bytes": base64}}]}
"""

import json
import io
import base64
from pathlib import Path
from datasets import load_from_disk, Dataset
from PIL import Image
import argparse
from tqdm import tqdm
import subprocess

# Dataset paths
DS_PATH = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/COVID-US-2026"
OUTPUT_DIR = Path("./covid_us_formatted_datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

# Output paths
EXPERT_OUTPUT = OUTPUT_DIR / "covid_us_expert"
LLM_OUTPUT = OUTPUT_DIR / "covid_us_llm"


def extract_image_bytes(sample: dict) -> bytes | None:
    """Extract first image bytes from modalities_images field."""
    modalities_images = sample.get("modalities_images", [])
    for mod in modalities_images:
        if isinstance(mod, dict) and "bytes" in mod:
            image_bytes = mod.get("bytes")
            if isinstance(image_bytes, bytes) or isinstance(image_bytes, str):
                if isinstance(image_bytes, str):
                    image_bytes = image_bytes.encode('utf-8')
                return image_bytes
    return None


def extract_image_path(sample: dict) -> str | None:
    """Extract reference image path from modalities field."""
    modalities = sample.get("modalities", [])
    for mod in modalities:
        if mod.get("type") == "image":
            return mod.get("value")
    return None


def generate_context_from_fields(sample: dict) -> str:
    """
    Generate clinical context from available metadata fields.
    Uses non-null fields from the sample.
    """
    context_parts = []

    # Clinical metadata fields
    disease = sample.get("disease")
    class_field = sample.get("class")
    probe = sample.get("probe")
    source = sample.get("source")

    # Patient demographics
    age = sample.get("age")
    gender = sample.get("gender")
    patient_id = sample.get("patient")

    # Clinical findings
    findings = []
    for key in ["LUSS", "air_bronchogram", "b_line", "consolidation",
                "irreg_pleural_line", "pleural_effusion", "a_line"]:
        value = sample.get(key)
        if value is not None and value != "":
            findings.append(f"{key}: {value}")

    # Symptoms
    symptoms = []
    for key in ["pain", "fever", "distress", "respiratory_problems"]:
        value = sample.get(key)
        if value and value != "":
            symptoms.append(str(value))

    # Build context string
    if disease:
        context_parts.append(f"Primary diagnosis: {disease}")
    if class_field:
        context_parts.append(f"Classification: {class_field}")
    if source:
        context_parts.append(f"Data source: {source}")
    if probe:
        context_parts.append(f"Probe type: {probe}")

    if age:
        context_parts.append(f"Patient age: {age}")
    if gender:
        context_parts.append(f"Patient gender: {gender}")

    if findings:
        context_parts.append("Ultrasound findings: " + ", ".join(findings))

    if symptoms:
        context_parts.append("Reported symptoms: " + ", ".join(symptoms))

    if context_parts:
        return "\n".join(context_parts)
    else:
        return "Clinical ultrasound examination of respiratory system."


def prepare_mmirage_data(sample: dict, image_path: str) -> dict | None:
    """Prepare data for MMIRAGE processing."""
    if not image_path:
        return None

    context = generate_context_from_fields(sample)

    # Collect raw fields for the model
    findings = []
    for key in ["LUSS", "air_bronchogram", "b_line", "consolidation",
                "irreg_pleural_line", "pleural_effusion", "a_line"]:
        value = sample.get(key)
        if value is not None and value != "":
            findings.append(f"{key}: {value}")

    raw_fields = {
        "disease": sample.get("disease", ""),
        "class": sample.get("class", ""),
        "probe": sample.get("probe", ""),
        "findings": "; ".join(findings) if findings else "No specific findings noted"
    }

    return {
        "image_path": image_path,
        "context": context,
        **raw_fields
    }


def generate_description_from_fields_concise(sample: dict) -> str:
    """Generate concise description (fallback for simpler use cases)."""
    description_parts = []

    class_field = sample.get("class", "Unknown pathology")
    disease = sample.get("disease", "")

    # Start with main finding
    description_parts.append(f"Ultrasound image showing findings consistent with {disease or class_field}.")

    # Add specific findings
    findings = []
    if sample.get("b_line") == "Yes":
        findings.append("B-lines are visible")
    if sample.get("consolidation") == "Yes":
        findings.append("Consolidation is present")
    if sample.get("pleural_effusion") == "Yes":
        findings.append("Pleural effusion is noted")
    if sample.get("air_bronchogram") == "Yes":
        findings.append("Air bronchogram is visible")
    if sample.get("irreg_pleural_line") == "Yes":
        findings.append("Irregular pleural line is observed")
    if sample.get("LUSS") == "Yes" or sample.get("LUSS") == 1:
        findings.append("Lung ultrasound score indicates significant involvement")

    if findings:
        description_parts.append("Key findings: " + ", ".join(findings) + ".")

    # Clinical assessment
    probe = sample.get("probe")
    if probe:
        description_parts.append(f"Examination performed using {probe} probe.")

    return " ".join(description_parts)


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    if isinstance(image_bytes, str):
        return image_bytes
    return base64.b64encode(image_bytes).decode('utf-8')


def process_sample_for_expert_from_description(description: str, image_path: str) -> dict:
    """Convert to expert format using generated description."""
    return {
        "text": description,
        "modalities": [
            {
                "type": "image",
                "value": image_path
            }
        ]
    }


def process_sample_for_llm_from_description(description: str, image_bytes: bytes | None, context: str) -> dict | None:
    """Convert to LLM format using generated description."""
    if not image_bytes:
        return None

    # Encode image to base64
    try:
        if isinstance(image_bytes, bytes):
            img = Image.open(io.BytesIO(image_bytes))
            img_base64 = encode_image_to_base64(image_bytes)
        else:
            img_base64 = encode_image_to_base64(image_bytes)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    return {
        "conversations": [
            {
                "role": "user",
                "content": f"Given the following clinical context:\n{context}\n\nWhat can you say about this ultrasound image?"
            },
            {
                "role": "assistant",
                "content": description
            }
        ],
        "modalities": [
            {
                "type": "image",
                "value": {
                    "bytes": img_base64
                }
            }
        ]
    }


def process_dataset(ds, split_name: str, max_samples: int | None = None):
    """
    Process dataset split and collect data for MMIRAGE processing.
    """
    mmirage_data = []
    expert_infos = []
    llm_infos = []
    contexts = []

    n_samples = len(ds) if max_samples is None else min(max_samples, len(ds))

    print(f"\nProcessing {split_name} split ({n_samples} samples)...")

    for idx in tqdm(range(n_samples)):
        sample = ds[idx]

        # Extract image data
        image_bytes = extract_image_bytes(sample)
        image_path = extract_image_path(sample)

        if image_bytes is None and image_path is None:
            continue

        # Prepare data for MMIRAGE
        if image_path:
            mmirage_item = prepare_mmirage_data(sample, image_path)
            if mmirage_item:
                mmirage_data.append(mmirage_item)
                expert_infos.append(image_path)
                llm_infos.append(image_bytes)
                contexts.append(mmirage_item["context"])

    return mmirage_data, expert_infos, llm_infos, contexts


def main(max_samples_per_split: int | None = None):
    """Main processing workflow with VLM-enhanced descriptions."""
    print(f"Loading dataset from {DS_PATH}...")

    ds = load_from_disk(DS_PATH)

    # Handle DatasetDict or single Dataset
    if hasattr(ds, "keys"):
        splits = ds.keys()
    else:
        splits = ["dataset"]
        ds = {"dataset": ds}

    all_mmirage_data = []
    all_expert_infos = []
    all_llm_infos = []
    all_contexts = []

    # Process each split
    for split_name in splits:
        split_data = ds[split_name]
        mmirage_data, expert_infos, llm_infos, contexts = process_dataset(
            split_data,
            split_name,
            max_samples=max_samples_per_split
        )

        all_mmirage_data.extend(mmirage_data)
        all_expert_infos.extend(expert_infos)
        all_llm_infos.extend(llm_infos)
        all_contexts.extend(contexts)

        print(f"  {split_name}: {len(mmirage_data)} samples prepared for MMIRAGE")

    # Save data for MMIRAGE
    mmirage_jsonl_path = Path("./covid_us_for_mmirage.jsonl")
    print(f"\nSaving data for MMIRAGE to {mmirage_jsonl_path}...")
    with open(mmirage_jsonl_path, "w") as f:
        for item in all_mmirage_data:
            json.dump(item, f)
            f.write("\n")

    # Run MMIRAGE to generate descriptions
    print("\nRunning MMIRAGE to generate descriptions...")
    mmirage_config_path = Path("./mmirage_generate_descriptions.yaml")

    # Run MMIRAGE from its directory
    result = subprocess.run(
        ["python", "-m", "mmirage.main", "--config", "../CovidUS-2026/mmirage_generate_descriptions.yaml"],
        cwd="/users/haaissa/MMIRAGE",
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"MMIRAGE failed: {result.stderr}")
        raise RuntimeError("MMIRAGE processing failed")

    print("MMIRAGE completed successfully.")

    # Load MMIRAGE output
    mmirage_output_path = Path("./covid_us_mmirage_output/data.jsonl")
    print(f"\nLoading MMIRAGE output from {mmirage_output_path}...")

    generated_descriptions = []
    with open(mmirage_output_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            generated_descriptions.append(item["generated_description"])

    if len(generated_descriptions) != len(all_mmirage_data):
        raise ValueError(f"MMIRAGE output length mismatch: {len(generated_descriptions)} vs {len(all_mmirage_data)}")

    # Create final datasets
    print("\nCreating final datasets...")

    expert_samples = []
    llm_samples = []

    for i, desc in enumerate(generated_descriptions):
        # Expert format
        expert_sample = process_sample_for_expert_from_description(desc, all_expert_infos[i])
        expert_samples.append(expert_sample)

        # LLM format
        llm_sample = process_sample_for_llm_from_description(desc, all_llm_infos[i], all_contexts[i])
        if llm_sample:
            llm_samples.append(llm_sample)

    # Create HuggingFace datasets
    expert_dataset = Dataset.from_dict({
        "text": [s["text"] for s in expert_samples],
        "modalities": [s["modalities"] for s in expert_samples]
    })

    llm_dataset = Dataset.from_dict({
        "conversations": [s["conversations"] for s in llm_samples],
        "modalities": [s["modalities"] for s in llm_samples]
    })

    # Save datasets
    print(f"\nSaving expert dataset to {EXPERT_OUTPUT}...")
    expert_dataset.save_to_disk(str(EXPERT_OUTPUT))

    print(f"Saving LLM dataset to {LLM_OUTPUT}...")
    llm_dataset.save_to_disk(str(LLM_OUTPUT))

    # Save summary
    summary = {
        "expert_dataset": {
            "path": str(EXPERT_OUTPUT),
            "num_samples": len(expert_dataset),
            "schema": {
                "text": "VLM-generated clinical description of the image",
                "modalities": [{"type": "image", "value": "path_to_image"}]
            }
        },
        "llm_dataset": {
            "path": str(LLM_OUTPUT),
            "num_samples": len(llm_dataset),
            "schema": {
                "conversations": [
                    {"role": "user", "content": "context + question"},
                    {"role": "assistant", "content": "VLM-generated description"}
                ],
                "modalities": [{"type": "image", "value": {"bytes": "base64_encoded"}}]
            }
        }
    }

    summary_path = OUTPUT_DIR / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Summary saved to {summary_path}")
    print(f"\nDatasets created:")
    print(f"  - Expert: {EXPERT_OUTPUT} ({len(expert_dataset)} samples)")
    print(f"  - LLM: {LLM_OUTPUT} ({len(llm_dataset)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process COVID-US-2026 dataset with VLM-enhanced descriptions")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split to process (for testing)"
    )
    args = parser.parse_args()

    main(max_samples_per_split=args.max_samples)
