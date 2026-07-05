import os
import argparse
import io
import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Subsets to process (llava_* and pixmo_* are excluded)
SUBSETS = [
    "image_BUSI",
    "image_COVID_US",
    "image_DDTI",
    "image_PMC_VQA",
    "image_ct2",
    "image_iu_xray",
    "image_mammoth",
    "image_medtrinity_conversations_1",
    "image_medtrinity_conversations_1_formatted",
    "image_medtrinity_conversations_2",
    "image_medtrinity_conversations_2_formatted",
]

SAMPLES_PER_SUBSET = 20

CONTEXT_PROMPT = (
    "You are an expert clinical data synthesizer. You will be provided with a medical image and an expert's visual description of that image.\n\n"
    "Your task is to use these findings to reverse-engineer a highly plausible, realistic patient history that explains exactly why this scan was ordered.\n\n"
    "Here is the expert's description:\n"
    "[INSERT ASSISTANT TEXT HERE]\n\n"
    "Instructions:\n"
    "1. Synthesis: Analyze both the attached image and the text description to understand the primary anatomical region and the confirmed pathology.\n"
    "2. Clinical Plausibility: Generate a realistic patient history based on these findings. What were the patient's symptoms? What specific complaint or event led them to seek medical attention and undergo this imaging? \n"
    "3. Keep it Standard: Focus on the most typical clinical presentation for the described pathology. Do not invent rare or exotic underlying conditions unless the image or text explicitly suggests them.\n\n"
    "Please format your output as plain text (not JSON), using exactly one field per line in this order:\n"
    "age_group: ...\n"
    "sex: ...\n"
    "clinical_setting: ...\n"
    "chief_complaint: ...\n"
    "history_of_present_illness: ...\n"
    "relevant_background: ...\n"
    "Use newline separators (\\n) between lines."
)

# Helpers

def extract_image(sample: dict) -> Image.Image | None:
    """
    Extract the first image from a sample's modalities field.
    The value can be:
      - a dict with a "bytes" key   → raw JPEG/PNG bytes
      - a PIL Image already decoded
      - a file path string (filesystem-based loader)
    """
    modalities = sample.get("modalities", [])
    for mod in modalities:
        if mod.get("type") != "image":
            continue
        value = mod.get("value")
        if value is None:
            continue
        # Already a PIL image
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        # Bytes dict (HuggingFace raw-image format)
        if isinstance(value, dict) and "bytes" in value and value["bytes"]:
            try:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            except Exception:
                continue
        # Flat bytes
        if isinstance(value, (bytes, bytearray)) and len(value) > 0:
            try:
                return Image.open(io.BytesIO(value)).convert("RGB")
            except Exception:
                continue
    return None


def extract_expert_description(sample: dict) -> str:
    """Extract the first assistant message to use as expert image description."""
    conversations = sample.get("conversations", [])
    for turn in conversations:
        if turn.get("role") == "assistant":
            text = str(turn.get("content", "")).strip()
            if text:
                return text
    return "No expert description provided."


def build_messages(image: Image.Image, expert_description: str) -> list[dict]:
    """Build the chat messages list for Qwen-VL."""
    prompt = CONTEXT_PROMPT.replace("[INSERT ASSISTANT TEXT HERE]", expert_description)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_context(
    model,
    processor,
    image: Image.Image,
    expert_description: str,
    device: str,
    max_new_tokens: int = 512,
) -> str:
    """Run a single inference call and return the generated context string."""
    messages = build_messages(image, expert_description)

    # Official Qwen3-VL inference pattern: apply_chat_template handles image
    # preprocessing internally when tokenize=True + return_dict=True.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    n_prompt_tokens = inputs["input_ids"].shape[1]
    print(f"      → prompt tokens: {n_prompt_tokens} | generating up to {max_new_tokens} tokens …", flush=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    n_generated = output_ids.shape[1] - n_prompt_tokens
    print(f"      → generated {n_generated} tokens", flush=True)

    # Trim prompt tokens, decode only the generated part
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    context = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    preview = context[:120].replace("\n", " ")
    print(f"      → preview: {preview!r}", flush=True)
    return context.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient background context with Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=SUBSETS,
        help="Subset names to process (space-separated). Defaults to all non-llava/pixmo subsets.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=SAMPLES_PER_SUBSET,
        help=f"Number of samples per subset (default: {SAMPLES_PER_SUBSET})",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "context_examples"),
        help="Output directory for generated JSONL files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per image (default: 512)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample from (default: train)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of parallel shards/tasks (default: 1)",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index for this process in [0, num_shards) (default: 0)",
    )
    return parser.parse_args()


def load_model(model_id: str, device: str):
    """Load Qwen3-VL model and processor."""
    print(f"Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def sample_indices(dataset_len: int, n: int, seed: int) -> list[int]:
    """Return n unique random indices from [0, dataset_len)."""
    rng = random.Random(seed)
    n = min(n, dataset_len)
    return rng.sample(range(dataset_len), n)


def serialize_sample(sample: dict) -> dict:
    """
    Return a JSON-serialisable copy of a HuggingFace dataset row.
    Bytes fields are dropped (too large) — we keep metadata only.
    """
    out = {}
    for key, val in sample.items():
        if key == "modalities":
            # Strip bytes but keep type / path metadata
            clean_mods = []
            for mod in val:
                m = {k: v for k, v in mod.items() if k != "value"}
                if isinstance(mod.get("value"), dict):
                    m["value"] = {
                        k: v
                        for k, v in mod["value"].items()
                        if k != "bytes"
                    }
                clean_mods.append(m)
            out[key] = clean_mods
        elif isinstance(val, (bytes, bytearray)):
            pass  # skip raw bytes at top level
        else:
            out[key] = val
    return out


def process_subset(
    subset_name: str,
    args,
    model,
    processor,
    device: str,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subset_name}.jsonl"

    # Skip already-completed subsets (allows resuming)
    if out_path.exists():
        with open(out_path) as fh:
            done = sum(1 for _ in fh)
        if done >= args.n:
            print(f"  [{subset_name}] already has {done} examples — skipping.")
            return

    print(f"\n{'='*60}")
    print(f"Processing subset: {subset_name}")
    print(f"{'='*60}")

    # Load the subset from the local arrow files saved by download_data.py
    storage_root = os.environ.get("STORAGE_ROOT")
    if not storage_root:
        print("  [ERROR] STORAGE_ROOT environment variable is not set.")
        return
    subset_path = os.path.join(storage_root, subset_name)
    try:
        dataset_dict = load_from_disk(subset_path)
        ds = dataset_dict[args.split]
    except Exception as e:
        print(f"  [ERROR] Could not load subset '{subset_name}' from '{subset_path}': {e}")
        return

    print(f"  Loaded {len(ds)} samples.")
    indices = sample_indices(len(ds), args.n, args.seed)

    generated = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(indices, desc=subset_name, unit="img"):
            sample = ds[idx]

            image = extract_image(sample)
            if image is None:
                print(f"    [WARN] No image found at index {idx}, skipping.")
                skipped += 1
                continue

            print(f"    [idx={idx}] image extracted ({image.size[0]}×{image.size[1]}) — sending to model …", flush=True)
            expert_description = extract_expert_description(sample)
            try:
                raw_context = generate_context(
                    model,
                    processor,
                    image,
                    expert_description,
                    device,
                    args.max_new_tokens,
                )
            except Exception as e:
                print(f"    [WARN] Generation failed at index {idx}: {e}")
                skipped += 1
                continue

            record = serialize_sample(sample)
            record["generated_context"] = raw_context
            record["source_subset"] = subset_name
            record["source_index"] = idx

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated += 1

    print(
        f"  Done — {generated} contexts saved to {out_path}"
        + (f" ({skipped} skipped)" if skipped else "")
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor = load_model(args.model, device)

    # Filter out llava / pixmo just in case they slip through
    subsets = [
        s for s in args.subsets
        if "llava" not in s.lower() and "pixmo" not in s.lower()
    ]

    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"shard_index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    shard_subsets = [
        subset for i, subset in enumerate(subsets)
        if i % args.num_shards == args.shard_index
    ]
    print(
        f"\nSubsets to process (global={len(subsets)}, "
        f"shard={args.shard_index}/{args.num_shards}, local={len(shard_subsets)}): "
        f"{shard_subsets}"
    )

    for subset_name in shard_subsets:
        process_subset(subset_name, args, model, processor, device)

    print("\nAll subsets processed.")


if __name__ == "__main__":
    main()
