"""
Generate structured ultrasound descriptions using Qwen3-VL-8B-Instruct.

Reads from an existing `{DATASET}_expert` arrow dataset (which has `text` +
`modalities_images`), calls the VLM with a 7-section clinical prompt, and writes:

  context_examples/{dataset}.jsonl    — input for pdf_gen.py
  output/{dataset}_expert.jsonl       — expert training format
  output/{dataset}_llm.jsonl          — LLM conversation training format

Usage (inside multimeditron container on a compute node):
    export STORAGE_ROOT=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow
    export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache

    python scripts/generate_us_descriptions.py \\
        --dataset BUSI \\
        --num_samples 50 \\
        --output_dir ./generated_data \\
        --batch_size 4

Supported datasets: BUSI, ct2, DDTI, CovidUS

CovidUS reads from COVID-US-2026 directly (no _expert variant needed).
Clinical context for CovidUS is assembled from structured metadata fields
(disease, class, probe, B-lines, consolidation, etc.) instead of a text caption.
"""

import argparse
import base64
import io
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from datasets import load_from_disk


ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Project constraint: generated descriptions must not exceed this many tokens.
MAX_DESCRIPTION_TOKENS = 500
STORAGE_ROOT = os.environ.get(
    "STORAGE_ROOT",
    "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow",
)

# Maps CLI dataset name → arrow directory name and context extraction strategy.
# Datasets with context_field="text" read an existing text caption from the _expert
# arrow. CovidUS has no text field — context is assembled from structured metadata.
DATASET_SOURCES = {
    "BUSI":    {"arrow_name": "BUSI_expert"},
    "ct2":     {"arrow_name": "ct2_expert"},
    "DDTI":    {"arrow_name": "DDTI_expert"},
    "CovidUS": {"arrow_name": "COVID-US-2026"},
}

# --------------------------------------------------------------------------- #
# 7-section clinical prompt template for ultrasound
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are an expert radiologist specializing in ultrasound imaging. "
    "When given an ultrasound image, you provide precise, structured clinical "
    "descriptions following standard reporting conventions."
)

DESCRIPTION_TEMPLATE = """\
Analyze this ultrasound image and provide a structured clinical description \
with the following seven sections.

LENGTH LIMIT: the entire description must stay under 500 tokens (about 350 words). \
Write tersely — short clauses and phrases, not full paragraphs; omit filler words \
and hedging. Cover all seven sections within the limit; do not exceed it.

1. **Visible organs and structures** — List all anatomical structures visible.
2. **Features of each organ/structure** — Size, shape, echogenicity, texture for each structure.
3. **Additional findings** — Any lesions, cysts, masses, lymph nodes, or incidental findings.
4. **Gray scale and Doppler features** — Echo pattern, posterior acoustic features, \
Doppler signal if applicable.
5. **Dynamic features** — Compressibility, mobility, or pulsatility if inferrable.
6. **Image quality and limitations** — Depth, resolution, artefacts, obscured areas.
7. **Impression/conclusion** — Overall assessment, likely diagnosis or differential diagnosis.

{context_section}"""

CONTEXT_SECTION_TEMPLATE = (
    "Clinical context provided:\n{context}\n\nNow describe the image:"
)


def build_covidus_context(sample: dict) -> str:
    """Assemble a clinical context string from COVID-US-2026 structured metadata."""
    fields = [
        ("disease",             "Disease"),
        ("class",               "Class"),
        ("probe",               "Probe"),
        ("distress",            "Respiratory distress"),
        ("pleural_effusion",    "Pleural effusion"),
        ("fever",               "Fever"),
        ("a_line",              "A-lines"),
        ("b_line",              "B-lines"),
        ("consolidation",       "Consolidation"),
        ("respiratory_problems", "Respiratory problems"),
        ("air_bronchogram",     "Air bronchogram"),
        ("irreg_pleural_line",  "Irregular pleural line"),
    ]
    parts = []
    for key, label in fields:
        val = sample.get(key)
        if val is not None and val != "":
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def build_prompt(existing_text: str) -> str:
    if existing_text and existing_text.strip():
        ctx = CONTEXT_SECTION_TEMPLATE.format(context=existing_text.strip())
    else:
        ctx = "No prior clinical context available. Describe the image:"
    return DESCRIPTION_TEMPLATE.format(context_section=ctx)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model_and_processor(model_name: str):
    """Load Qwen3-VL (or Qwen2.5-VL fallback) via transformers AutoClass."""
    from transformers import AutoProcessor

    print(f"Loading model: {model_name}", flush=True)

    # Use AutoModelForVision2Seq so the correct class is resolved for any VL model
    # (Qwen3-VL uses qwen3_vl architecture, distinct from Qwen2.5-VL — forcing
    # Qwen2_5_VLForConditionalGeneration causes random weight init and garbage output)
    from transformers import AutoModelForVision2Seq as VLModel

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = VLModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}", flush=True)
    return model, processor


# --------------------------------------------------------------------------- #
# Image extraction
# --------------------------------------------------------------------------- #

def get_pil_image(sample: dict) -> Image.Image | None:
    """Extract PIL image from a _expert dataset row."""
    # modalities_images is a list of {"bytes": ..., "path": ...}
    images = sample.get("modalities_images", [])
    for img_entry in images:
        if img_entry is None:
            continue
        raw = img_entry.get("bytes") if isinstance(img_entry, dict) else img_entry
        if raw:
            try:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# Single-sample inference
# --------------------------------------------------------------------------- #

def enforce_token_limit(text: str, processor, max_tokens: int = MAX_DESCRIPTION_TOKENS) -> str:
    """Trim `text` to at most `max_tokens` tokens, cutting at a sentence boundary.

    The prompt already asks the model to stay under the limit; this is a hard
    guarantee for the rare overflow. When trimming is needed we cut back to the
    last sentence terminator (. ! ? or newline) so the output never ends
    mid-word/mid-sentence. Prints a notice when a trim happens.

    Args:
        text: The (already <think>-stripped) description.
        processor: The VL processor; its `.tokenizer` is used to count/slice tokens.
        max_tokens: Hard ceiling on the number of tokens.

    Returns:
        The description, unchanged if within the limit, otherwise trimmed.
    """
    tokenizer = processor.tokenizer
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text

    truncated = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
    # Cut back to the last complete sentence so we never end mid-sentence.
    cut = max(truncated.rfind(". "), truncated.rfind("! "),
              truncated.rfind("? "), truncated.rfind("\n"))
    trimmed = truncated[: cut + 1].strip() if cut > 0 else truncated.strip()
    print(f"  [trim] description {len(token_ids)} -> "
          f"{len(tokenizer(trimmed, add_special_tokens=False)['input_ids'])} tokens "
          f"(limit {max_tokens})", flush=True)
    return trimmed


def generate_description(
    model,
    processor,
    image: Image.Image,
    existing_text: str,
    max_new_tokens: int = 600,
) -> str:
    prompt_text = build_prompt(existing_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Strip the input tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    text = processor.decode(generated, skip_special_tokens=True).strip()

    # Qwen3 models emit <think>...</think> reasoning blocks before the answer.
    # Strip them so only the final structured description is kept.
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Hard-enforce the 500-token project constraint (prompt asks for it; this guarantees it).
    text = enforce_token_limit(text, processor, MAX_DESCRIPTION_TOKENS)
    return text


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def image_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def write_record(
    f_context,
    f_expert,
    f_llm,
    source_subset: str,
    source_index: int,
    existing_text: str,
    description: str,
    image_bytes: bytes,
):
    # ---- context_examples JSONL (for pdf_gen.py) ----
    context_record = {
        "source_subset": source_subset,
        "source_index": source_index,
        "generated_context": existing_text,
        "conversations": [
            {
                "role": "user",
                "content": (
                    f"Given the following clinical context: {existing_text}\n"
                    f"What can you say about this ultrasound image? "
                    f"{ATTACHMENT_TOKEN}"
                ),
            },
            {"role": "assistant", "content": description},
        ],
    }
    f_context.write(json.dumps(context_record) + "\n")

    # ---- Expert format (caption-style) ----
    expert_record = {
        "text": description,
        "modalities": [{"type": "image", "value": "image.jpg"}],
    }
    f_expert.write(json.dumps(expert_record) + "\n")

    # ---- LLM conversation format ----
    user_content = (
        f"Given the following clinical context: {existing_text}\n"
        f"What can you say about this ultrasound image? "
        f"{ATTACHMENT_TOKEN}"
        if existing_text.strip()
        else f"What can you say about this ultrasound image? {ATTACHMENT_TOKEN}"
    )
    llm_record = {
        "conversations": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": description},
        ],
        "modalities": [
            {
                "type": "image",
                "value": {
                    "bytes": base64.b64encode(image_bytes).decode("ascii"),
                    "path": None,
                },
            }
        ],
    }
    f_llm.write(json.dumps(llm_record) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Generate US descriptions with Qwen3-VL.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["BUSI", "ct2", "DDTI", "CovidUS"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size (default 1; increase for throughput on multi-GPU).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate (default 50 for PDF review).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_data",
        help="Output directory. Creates context_examples/ and output/ subdirs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=600,
        help="Generation budget (headroom above the 500-token description limit, "
             "which is enforced separately via enforce_token_limit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all samples (overrides --num_samples).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    context_dir = output_dir / "context_examples"
    training_dir = output_dir / "output"
    context_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    # Source dataset
    source_name = DATASET_SOURCES[args.dataset]["arrow_name"]
    source_path = os.path.join(STORAGE_ROOT, source_name)
    print(f"Loading source dataset: {source_path}", flush=True)

    ds = load_from_disk(source_path)
    # Handle DatasetDict
    if hasattr(ds, "keys"):
        split = "train" if "train" in ds else list(ds.keys())[0]
        ds = ds[split]

    total = len(ds)
    print(f"Source: {source_name}, {total} rows", flush=True)

    if args.all:
        indices = list(range(total))
    else:
        n = min(args.num_samples, total)
        indices = random.sample(range(total), n)

    print(f"Generating descriptions for {len(indices)} samples", flush=True)

    # Load model
    model, processor = load_model_and_processor(args.model)

    # Output file paths
    ctx_path = context_dir / f"{args.dataset}.jsonl"
    exp_path = training_dir / f"{args.dataset}_expert.jsonl"
    llm_path = training_dir / f"{args.dataset}_llm.jsonl"

    skipped = 0
    with open(ctx_path, "w") as f_ctx, \
         open(exp_path, "w") as f_exp, \
         open(llm_path, "w") as f_llm:

        for pos, idx in enumerate(indices):
            sample = ds[idx]

            pil_img = get_pil_image(sample)
            if pil_img is None:
                print(f"  [{pos+1}/{len(indices)}] idx={idx}: no image, skipping")
                skipped += 1
                continue

            if args.dataset == "CovidUS":
                existing_text = build_covidus_context(sample)
            else:
                existing_text = sample.get("text", "") or ""

            try:
                description = generate_description(
                    model, processor, pil_img, existing_text, args.max_new_tokens
                )
            except Exception as e:
                print(f"  [{pos+1}/{len(indices)}] idx={idx}: inference error: {e}")
                skipped += 1
                continue

            img_bytes = image_to_bytes(pil_img)

            write_record(
                f_ctx, f_exp, f_llm,
                source_subset=source_name,
                source_index=idx,
                existing_text=existing_text,
                description=description,
                image_bytes=img_bytes,
            )

            if (pos + 1) % 10 == 0 or (pos + 1) == len(indices):
                print(
                    f"  [{pos+1}/{len(indices)}] idx={idx} done "
                    f"({len(description)} chars)",
                    flush=True,
                )

    print(f"\nDone. Written to:")
    print(f"  PDF review JSONL : {ctx_path}")
    print(f"  Expert training  : {exp_path}")
    print(f"  LLM training     : {llm_path}")
    print(f"  Skipped: {skipped}/{len(indices)}")


if __name__ == "__main__":
    main()
