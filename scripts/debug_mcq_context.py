"""
debug_mcq_context.py  — inspect exactly what context MultiMeditron sees on an MCQ.

Usage (on the login node or in an interactive job):

    # 1. Reconstruct the raw token string (no GPU needed):
    python3 scripts/debug_mcq_context.py --mode text

    # 2. Run a single forward pass and decode the model output (needs GPU):
    python3 scripts/debug_mcq_context.py --mode generate \
        --checkpoint /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800

    # 3. Compare two checkpoints side-by-side on the same sample:
    python3 scripts/debug_mcq_context.py --mode compare \
        --checkpoint <ckpt_7exp> \
        --checkpoint2 <ckpt_5exp>

What this shows:
  - The exact decoded text that the tokenizer produces (system / user / assistant turns)
  - The positions of image-placeholder tokens inside the sequence
  - The number of image tokens vs text tokens
  - The raw model output (free generation) for the MCQ
  - Side-by-side comparison between two checkpoints (useful for nanoVLM vs MultiMeditron)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third-party", "lmms-eval"))

import torch
from PIL import Image
from transformers import AutoTokenizer

from multimeditron.model.model import ChatTemplate, MultiModalModelForCausalLM
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.dataset.loader import RawImageLoader
from multimeditron.model.constants import IGNORE_TOKEN_INDEX

# ---------------------------------------------------------------------------
# A synthetic MCQ sample that mimics GMAI format (no real image needed for
# "text" mode — a small solid-colour PIL image is used as a placeholder).
# ---------------------------------------------------------------------------
FAKE_MCQ = {
    "question": "What is the most likely diagnosis based on the findings?",
    "options": [
        ("A", "Pneumonia"),
        ("B", "Pleural effusion"),
        ("C", "Pneumothorax"),
        ("D", "Pulmonary oedema"),
    ],
    "answer": "C",
    "post_prompt": (
        "\nLet's think step by step. Detail your reasoning before answering carefully."
        " Your final answer should have the following format: Answer: <letter>."
        " The <letter> is chosen amongst the possible options (A, B, C, D or E)"
    ),
}

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"


def build_fake_image(size: int = 224) -> Image.Image:
    """Return a tiny solid-grey PIL image (avoids needing a real DICOM)."""
    return Image.new("RGB", (size, size), color=(128, 128, 128))


def build_messages(mcq: dict, attachment_token: str) -> dict:
    """Build the conversation dict in MultiMeditron format."""
    question_text = mcq["question"] + "\nOptions:"
    for letter, option in mcq["options"]:
        question_text += f"\n{letter}: {option}"
    question_text += mcq["post_prompt"]

    return {
        "conversations": [
            {
                "role": "user",
                "content": attachment_token + "\n" + question_text,
            }
        ],
        "modalities": [
            {
                "type": "image",
                "value": build_fake_image(),
            }
        ],
    }


def make_collator(tokenizer, model_or_none, tokenizer_type: str = "llama"):
    """Build a DataCollatorForMultimodal using the model's own processors (or a
    dummy processor list if model is None)."""
    chat_template = ChatTemplate.from_name(tokenizer_type)

    if model_or_none is not None:
        processors = model_or_none.processors()
    else:
        # Lazy import — only needed for text-only mode
        from multimeditron.model.modalities.image_modality import ImageModality, ImageConfig
        dummy_cfg = ImageConfig(
            hidden_size=4096,
            clip_name=os.path.join(
                os.path.dirname(__file__), "..", "models", "CLIP", "clip-vit-base-patch32"
            ),
        )
        modality = ImageModality(dummy_cfg)
        processors = {"image": modality.processor()}  # type: ignore

    return DataCollatorForMultimodal(
        tokenizer=tokenizer,
        attachment_token=ATTACHMENT_TOKEN,
        chat_template=chat_template,
        modality_processors=processors,
        modality_loaders={"image": RawImageLoader()},
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_token_sequence(tokenizer, input_ids: torch.Tensor, attachment_token_id: int):
    """
    Decode a token sequence to text, marking image-placeholder regions visually.
    Returns the decoded string and a summary of the image span.
    """
    ids = input_ids.tolist()
    total = len(ids)
    image_positions = [i for i, t in enumerate(ids) if t == attachment_token_id]

    # Replace image tokens with a visible marker for decode
    marked = [
        token if token != attachment_token_id else tokenizer.eos_token_id
        for token in ids
    ]
    decoded = tokenizer.decode(marked, skip_special_tokens=False)

    # Replace eos_token back with readable marker
    eos_str = tokenizer.eos_token or "<eos>"
    for _ in image_positions:
        decoded = decoded.replace(eos_str, "«IMG»", 1)

    return decoded, image_positions, total


def print_context_summary(tokenizer, batch: dict, label: str = ""):
    att_id = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    decoded, img_positions, total_tokens = decode_token_sequence(tokenizer, input_ids, att_id)

    print(f"\n{'='*70}")
    print(f"  CONTEXT SUMMARY  {label}")
    print(f"{'='*70}")
    print(f"  Total tokens in sequence : {total_tokens}")
    print(f"  Image placeholder tokens : {len(img_positions)}")
    if img_positions:
        print(f"  Image token span         : [{img_positions[0]}, {img_positions[-1]+1})")
    text_tokens = total_tokens - len(img_positions)
    print(f"  Text tokens              : {text_tokens}")
    labelled = (labels != IGNORE_TOKEN_INDEX).sum().item()
    print(f"  Tokens with label (loss) : {labelled}")
    print(f"\n--- DECODED CONTEXT (image tokens shown as «IMG») ---\n")
    print(decoded)
    print(f"\n--- END OF CONTEXT ---\n")


# ---------------------------------------------------------------------------
# Mode: text
# ---------------------------------------------------------------------------

def run_text_mode(tokenizer_path: str, tokenizer_type: str):
    """Reconstruct and print the context without loading the full model."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Use a tiny local CLIP just to get a processor (no GPU needed)
    local_clip = os.path.join(os.path.dirname(__file__), "..", "models", "CLIP", "clip-vit-base-patch32")
    if not os.path.exists(local_clip):
        print(f"ERROR: local CLIP not found at {local_clip}. Run in 'generate' mode instead.")
        sys.exit(1)

    from multimeditron.model.modalities.image_modality import ImageModality, ImageConfig
    cfg = ImageConfig(hidden_size=4096, clip_name=local_clip)
    img_processor = ImageModality.preprocessor_class(cfg)

    from multimeditron.dataset.loader import RawImageLoader
    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        attachment_token=ATTACHMENT_TOKEN,
        chat_template=ChatTemplate.from_name(tokenizer_type),
        modality_processors={"image": img_processor},
        modality_loaders={"image": RawImageLoader()},
        add_generation_prompt=True,
    )

    sample = build_messages(FAKE_MCQ, ATTACHMENT_TOKEN)
    batch = collator([sample])
    print_context_summary(tokenizer, batch, label="(text mode — clip-base-patch32 processor)")

    # Explain what to look for
    print("WHAT TO CHECK:")
    print("  1. Does the user turn start with «IMG» tokens (image before text)?")
    print("  2. Is there a system prompt? There should NOT be one unless training data had it.")
    print("  3. Are the image tokens contiguous (one block)?")
    print("  4. Does the question text exactly match the GMAI eval format?")
    print("  5. Does 'Tokens with label (loss)' == 0? (it should — generation prompt mode)")


# ---------------------------------------------------------------------------
# Mode: generate
# ---------------------------------------------------------------------------

def run_generate_mode(checkpoint: str, tokenizer_type: str):
    print(f"Loading model from: {checkpoint}")
    dtype = torch.bfloat16
    model = MultiModalModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, padding_side="left")
    except Exception:
        llm_path = "/capstor/store/cscs/swissai/a127/meditron/hf_cache/hub/models--OpenMeditron--Meditron3-8B/snapshots/15914bcb040cd1a4f263afcd85b84f09ad2efd95"
        print(f"Tokenizer not in checkpoint, falling back to: {llm_path}")
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        attachment_token=ATTACHMENT_TOKEN,
        chat_template=ChatTemplate.from_name(tokenizer_type),
        modality_processors=model.processors(),
        modality_loaders={"image": RawImageLoader()},
        add_generation_prompt=True,
    )

    sample = build_messages(FAKE_MCQ, ATTACHMENT_TOKEN)
    batch = collator([sample])
    print_context_summary(tokenizer, batch, label=f"  checkpoint: {os.path.basename(checkpoint)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Running generation on device: {device}")
    with torch.autocast(device, dtype=dtype):
        out_ids = model.generate(batch=batch, max_new_tokens=512, temperature=0.0, do_sample=False)

    response = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"\n--- MODEL RESPONSE ---\n{response}\n--- END RESPONSE ---\n")
    print(f"Expected answer: {FAKE_MCQ['answer']}")


# ---------------------------------------------------------------------------
# Mode: compare
# ---------------------------------------------------------------------------

def run_compare_mode(checkpoint1: str, checkpoint2: str, tokenizer_type: str):
    """
    Forward pass both checkpoints on the same synthetic MCQ and compare outputs.
    Useful for nanoVLM (SmolLM2) vs MultiMeditron (Meditron3-8B).
    """
    for label, ckpt in [("Checkpoint 1", checkpoint1), ("Checkpoint 2", checkpoint2)]:
        print(f"\n{'#'*70}\n  {label}: {ckpt}\n{'#'*70}")
        run_generate_mode(ckpt, tokenizer_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["text", "generate", "compare"], default="text",
                        help="'text': decode context only (no GPU). 'generate': run model. 'compare': two checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (required for generate/compare modes).")
    parser.add_argument("--checkpoint2", default=None, help="Second checkpoint path for compare mode.")
    parser.add_argument("--tokenizer_type", default="llama", help="Chat template name: llama | qwen3 | apertus")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Override tokenizer path for text mode (default: uses local Meditron3 path).")
    args = parser.parse_args()

    default_tokenizer = (
        args.tokenizer_path
        or "/capstor/store/cscs/swissai/a127/meditron/hf_cache/hub/models--OpenMeditron--Meditron3-8B/snapshots/15914bcb040cd1a4f263afcd85b84f09ad2efd95"
    )

    if args.mode == "text":
        run_text_mode(default_tokenizer, args.tokenizer_type)
    elif args.mode == "generate":
        if not args.checkpoint:
            parser.error("--checkpoint is required for generate mode")
        run_generate_mode(args.checkpoint, args.tokenizer_type)
    elif args.mode == "compare":
        if not args.checkpoint or not args.checkpoint2:
            parser.error("--checkpoint and --checkpoint2 are required for compare mode")
        run_compare_mode(args.checkpoint, args.checkpoint2, args.tokenizer_type)


if __name__ == "__main__":
    main()
