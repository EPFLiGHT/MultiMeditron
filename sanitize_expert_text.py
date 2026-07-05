import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanitize expert dataset text to keep only image-descriptive text."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input expert JSONL file to sanitize.",
    )
    parser.add_argument(
        "--output_file",
        default="~/expert_datasets/sanitized_expert_dataset.jsonl",
        help="Output JSONL file for sanitized records.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-2-7B",  # lighter text model than Qwen3-VL
        help="HuggingFace model to use for text sanitization.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate for sanitized description.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default 1).",
    )
    return parser.parse_args()


def build_prompt(original_text: str) -> str:
    return (
        "You are an expert medical description cleaner.\n"
        "Keep only the part of the text that describes the image findings, anatomy, or visual appearance. "
        "Remove any non-descriptive content (instructions, planning, analysis, comments, unrelated context, or metadata).\n"
        "If the text has multiple sentences, keep only the sentences that describe the image.\n"
        "Only output the cleaned description. Do not output any explanation.\n"
        "If no description remains, output: No image description provided.\n\n"
        "Original text:\n"
        f"{original_text.strip()}\n\n"
        "Cleaned description:\n"
    )


def sanitize_text(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Keep text after prompt
    if prompt in decoded:
        decoded = decoded.split(prompt, 1)[1]
    # Remove possible trailing segments from prompt text
    cleaned = decoded.strip()
    # If model returns empty, fallback phrase
    if not cleaned:
        return "No image description provided."
    return cleaned


def sanitize_dataset(
    input_file: Path,
    output_file: Path,
    model_name: str,
    max_new_tokens: int,
    device: str,
    batch_size: int,
):
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            original_text = str(sample.get("text", "")).strip()
            prompt = build_prompt(original_text)
            sanitized = sanitize_text(model, tokenizer, prompt, max_new_tokens, device)
            sample["original_text"] = sample.get("text", "")
            sample["text"] = sanitized
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Sanitized {total} lines and wrote to {output_file}.")


def main():
    args = parse_args()
    input_file = Path(args.input_file).expanduser()
    output_file = Path(args.output_file).expanduser()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    sanitize_dataset(
        input_file=input_file,
        output_file=output_file,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
