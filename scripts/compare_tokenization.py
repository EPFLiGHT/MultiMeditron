"""
compare_tokenization.py
=======================
Compares the tokenization pipelines of nanoVLM and MultiMeditron on the
same example from the clean dataset.

Run on the cluster:
    python compare_tokenization.py

Expected output:
    - Token sequences from both pipelines side by side
    - Number of image tokens in each
    - Loss mask coverage
    - Diff summary
"""

import sys
import os
import io
import torch

# ── Paths (adjust if needed) ───────────────────────────────────────────────────
NANO_ROOT        = "/users/haaissa/nanoVLM"
MULTI_ROOT       = "/users/haaissa/MultiMeditron/src"
DATASET_PATH     = "/iopsstor/scratch/cscs/haaissa/MultiMeditron_Clean_Arrow"
LLM_NAME         = "HuggingFaceTB/SmolLM2-360M-Instruct"
EXAMPLE_IDX      = 0   # which example from the dataset to test
# ───────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, NANO_ROOT)
sys.path.insert(0, MULTI_ROOT)

from datasets import load_from_disk
from transformers import AutoTokenizer
from PIL import Image

print("=" * 70)
print("Loading dataset...")
ds = load_from_disk(DATASET_PATH)
example = ds[EXAMPLE_IDX]

print(f"Example keys      : {list(example.keys())}")
print(f"Number of texts   : {len(example['texts'])}")
print(f"Number of images  : {len(example['images'])}")
print(f"First user turn   : {example['texts'][0]['user'][:120]!r}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — nanoVLM pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("NANO VLM PIPELINE")
print("=" * 70)

from data.datasets import BaseDataset
from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig

vlm_cfg = VLMConfig()

# Load tokenizer the nanoVLM way
nano_tokenizer = get_tokenizer(
    LLM_NAME,
    extra_special_tokens=vlm_cfg.vlm_extra_tokens,
    chat_template=vlm_cfg.lm_chat_template,
)
nano_tokenizer.pad_token = nano_tokenizer.eos_token

image_processor = get_image_processor(
    vlm_cfg.max_img_size,
    vlm_cfg.max_img_size,
    resize_to_max_side_len=False,
)

# Build a minimal BaseDataset-like processor
class MinimalNanoProcessor(BaseDataset):
    def __init__(self, tokenizer, image_processor, mp_image_token_length):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.prefix_len = self._get_prefix_len()
        self.relevance_min_rating = 1
        self.image_correspondence_min_rating = 1
        self.visual_dependency_min_rating = 1
        self.formatting_min_rating = 1

nano_proc = MinimalNanoProcessor(nano_tokenizer, image_processor, vlm_cfg.mp_image_token_length)

# Process images
images = example['images']  # list of PIL Images
processed_images, splitted_image_counts = nano_proc._process_images(images)

# Build messages
messages = nano_proc._get_messages(example, splitted_image_counts)

# Tokenize
nano_input_ids, nano_loss_mask, _ = nano_proc._prepare_inputs_and_loss_mask(messages)

print(f"mp_image_token_length : {vlm_cfg.mp_image_token_length}")
print(f"splitted_image_counts : {splitted_image_counts}")
print(f"Messages[0] content   : {messages[0]['content'][:200]!r}")
print(f"Total tokens          : {len(nano_input_ids)}")
print(f"Image token ID        : {nano_tokenizer.convert_tokens_to_ids(nano_tokenizer.image_token)}")

# Count image tokens
img_tok_id = nano_tokenizer.convert_tokens_to_ids(nano_tokenizer.image_token)
nano_n_image_toks = (nano_input_ids == img_tok_id).sum().item()
print(f"# image tokens in seq : {nano_n_image_toks}")
print(f"Loss mask coverage    : {nano_loss_mask.sum().item()} / {len(nano_loss_mask)} tokens "
      f"({100*nano_loss_mask.float().mean():.1f}%)")

# Show first 20 and last 20 token IDs
print(f"Token IDs [0:20]      : {nano_input_ids[:20].tolist()}")
print(f"Token IDs [-20:]      : {nano_input_ids[-20:].tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MultiMeditron pipeline (via hf_m4_transform + PromptTokenizer)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("MULTIMEDITRON PIPELINE")
print("=" * 70)

from multimeditron.model.model import ChatTemplate

multi_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, padding_side='right', use_fast=True)
multi_tokenizer.pad_token = multi_tokenizer.eos_token

chat_template = ChatTemplate.from_name("qwen3")
attachment_token = "<|image|>"

special_tokens_list = [attachment_token]
for v in chat_template.special_tokens.values():
    if v is not None:
        special_tokens_list.append(v)
multi_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})

attachment_token_id = multi_tokenizer.convert_tokens_to_ids(attachment_token)
print(f"<|image|> token ID    : {attachment_token_id}")

# Apply hf_m4_transform manually on the single example
batch = {"texts": [example["texts"]], "images": [example["images"]]}

def hf_m4_transform(batch):
    new_examples = {"conversations": [], "modalities": []}
    for texts, images in zip(batch.get("texts", []), batch.get("images", [])):
        convs = []
        for turn in texts:
            convs.append({"role": "user", "content": turn["user"]})
            convs.append({"role": "assistant", "content": turn["assistant"]})

        mods = []
        if images is not None:
            for img in images:
                if img.mode not in ("RGB", "RGBA", "L", "1"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                mods.append({"type": "image", "value": {"bytes": buf.getvalue()}})

        if mods and convs:
            content = convs[0]["content"]
            content = content.replace("<|image|>", "").replace("<image>", "").strip()
            image_tags = "<|image|>\n" * len(mods)
            convs[0]["content"] = image_tags + content

        new_examples["conversations"].append(convs)
        new_examples["modalities"].append(mods)
    return new_examples

transformed = hf_m4_transform(batch)
conversations = transformed["conversations"][0]
modalities    = transformed["modalities"][0]

print(f"Conversations[0] content : {conversations[0]['content'][:200]!r}")
print(f"# modalities             : {len(modalities)}")

# Tokenize via PromptTokenizer
from multimeditron.model.prompt_tokenizers import PromptTokenizer
prompt_tokenizer = PromptTokenizer(
    tokenizer=multi_tokenizer,
    chat_template=chat_template,
    attachment_token=attachment_token,
    modalities_num_embeddings={"image": 64},
)

result = prompt_tokenizer.tokenize_samples(
    [{"conversations": conversations, "modalities": modalities}],
    add_eos_token=True,
)

multi_input_ids  = result[0]["input_ids"]
multi_loss_mask  = result[0]["labels"] != -100  # -100 = IGNORE_TOKEN_INDEX

multi_n_image_toks = (multi_input_ids == attachment_token_id).sum().item()
print(f"Total tokens             : {len(multi_input_ids)}")
print(f"# image tokens in seq    : {multi_n_image_toks}")
print(f"Loss mask coverage       : {multi_loss_mask.sum().item()} / {len(multi_loss_mask)} tokens "
      f"({100*multi_loss_mask.float().mean():.1f}%)")
print(f"Token IDs [0:20]         : {multi_input_ids[:20].tolist()}")
print(f"Token IDs [-20:]         : {multi_input_ids[-20:].tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Diff summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("DIFF SUMMARY")
print("=" * 70)

print(f"Total tokens   : nano={len(nano_input_ids)}  multi={len(multi_input_ids)}  "
      f"{'✅ SAME' if len(nano_input_ids)==len(multi_input_ids) else '❌ DIFFERENT'}")

print(f"# image tokens : nano={nano_n_image_toks}  multi={multi_n_image_toks}  "
      f"{'✅ SAME' if nano_n_image_toks==multi_n_image_toks else '❌ DIFFERENT'}")

print(f"Loss coverage  : nano={nano_loss_mask.sum().item()}  multi={multi_loss_mask.sum().item()}  "
      f"{'✅ SAME' if nano_loss_mask.sum()==multi_loss_mask.sum() else '⚠️  DIFFERENT'}")

# Check if the image token ID is the same
print(f"Image token ID : nano={img_tok_id}  multi={attachment_token_id}  "
      f"{'✅ SAME' if img_tok_id==attachment_token_id else '❌ DIFFERENT ← CRITICAL BUG'}")

# Compare token IDs where possible
min_len = min(len(nano_input_ids), len(multi_input_ids))
n_diff  = (nano_input_ids[:min_len] != multi_input_ids[:min_len]).sum().item()
print(f"Token ID diffs : {n_diff} / {min_len} positions differ in the shared prefix")

if n_diff == 0 and len(nano_input_ids) == len(multi_input_ids):
    print("\n🎉 PIPELINES IDENTIQUES — le problème n'est PAS dans la tokenisation")
    print("   → Investiguer : packing, loss masking, pixel shuffle output")
else:
    print(f"\n⚠️  LES PIPELINES DIVERGENT")
    # Find first diff position
    for i in range(min_len):
        if nano_input_ids[i] != multi_input_ids[i]:
            print(f"   Premier token différent à position {i}")
            print(f"   nano  token {i}: id={nano_input_ids[i].item()} "
                  f"→ '{nano_tokenizer.decode([nano_input_ids[i].item()])}'")
            print(f"   multi token {i}: id={multi_input_ids[i].item()} "
                  f"→ '{multi_tokenizer.decode([multi_input_ids[i].item()])}'")
            break
    print("   → Investiguer : chat template, image token format, hf_m4_transform")

print("=" * 70)

