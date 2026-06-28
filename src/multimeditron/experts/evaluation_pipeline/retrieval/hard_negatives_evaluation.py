"""
Hard-Negative Image–Text Retrieval Evaluation for CLIP-Style Models.

This script evaluates vision–language dual-encoder models using a controlled
4-way (1 positive + 3 negative) image-to-text retrieval protocol. For each image
query, three hard negative text descriptions are selected based on visual
similarity computed by a fixed reference CLIP model, while enforcing different
disease-class labels to avoid trivial duplicates. Evaluation measures Recall@1,
i.e., whether the correct caption is ranked highest among the four candidates.

The hard-negative protocol is constructed once and reused across all evaluated
models to ensure fair comparison. Both image and text embeddings are L2-normalized,
cosine similarity is used for scoring, and ties are handled explicitly according
to a configurable policy. This script is intended for benchmark-style evaluation
of representation quality, not for training or hyperparameter selection.
"""

import os
import json
import random
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    VisionTextDualEncoderProcessor,
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoProcessor,
)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from load_from_clip import load_model

# =====================================
# Config — fill in before running
# =====================================
EVAL_DATASETS = [
    # "/path/to/skin10_val.jsonl",
]
LINE_NUMBER = None
SEED = 14
IMG_BS = 32
TXT_BS = 64
TXT_MAX_LEN = 128
TIE_EPS = 1e-7
TIE_POLICY = "count_incorrect"  # or "skip" / "argmax"

REF_MODEL_NAME = "openai/clip-vit-base-patch32"
HARD_TOPK = 3

RESULTS_TXT = os.environ.get("HARD_NEG_RESULTS_TXT", "hard_negatives_results.txt")

CLIP_CONFIGS = [
    # ("my_model", "/path/to/model_checkpoint"),
]

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("clip-eval")


# =====================================
# BiomedCLIP loader (reference model)
# =====================================
def load_clip_like(model_name_or_path, device):
    """
    Load a CLIP-like model + processor.

    - For vanilla CLIP: CLIPModel + CLIPProcessor
    - For BiomedCLIP (and other custom models): AutoModel + AutoProcessor with trust_remote_code=True
    """
    lower_name = model_name_or_path.lower()

    if "biomedclip" in lower_name:
        # BiomedCLIP or similar custom model
        model = (
            AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    else:
        # vanilla CLIP (openai/clip-vit-base-patch32, etc.)
        model = CLIPModel.from_pretrained(model_name_or_path).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name_or_path)

    return model, processor


# =====================================
# Data utilities
# =====================================
def resolve_items(jsonl_path, line_number=None):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if line_number:
        lines = lines[:line_number]

    items = [json.loads(l) for l in lines]
    base_dir = Path(jsonl_path).parent
    for it in items:
        v = it["modalities"][0]["value"]
        p = Path(v)
        if not p.is_absolute():
            it["modalities"][0]["value"] = str((base_dir / v).resolve())
    return items


def resolve_many(paths, line_number=None):
    all_items = []
    for p in paths:
        cur = resolve_items(p, line_number)
        all_items.extend(cur)
    if len(all_items) < 4:
        raise ValueError(
            f"Need at least 4 total items for 1+3 setup, got {len(all_items)}"
        )
    return all_items


def get_disease_label(item):
    """
    Return a disease-class label for an item.
    Here we derive it from the parent folder, e.g.
    'rebuilt/melanocytic-nevi/xxx.jpg' -> 'melanocytic-nevi'.
    """
    path_str = item["modalities"][0]["value"]
    path = Path(path_str)
    return path.parent.name


def build_hard_protocol_from_embeds(items, img_embeds, seed=SEED, top_k=50):
    """
    Build a 1+3 multiple-choice protocol with *hard negatives*.
    For each query image i, choose negatives that are:
      - visually closest in CLIP embedding space
      - but have a *different disease label* (class)
    img_embeds must be L2-normalized, same order as items.
    """
    rng = random.Random(seed)
    n = len(items)
    disease_labels = [get_disease_label(it) for it in items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_embeds_dev = img_embeds.to(device)

    triples = []
    for i in range(n):
        img_vec = img_embeds_dev[i]  # (D,)
        sims = torch.matmul(img_embeds_dev, img_vec)  # (N,)
        sims[i] = -1.0  # exclude self

        # indices sorted by descending similarity
        sorted_idx = torch.argsort(sims, descending=True).tolist()

        # keep only visually-close *different-disease* neighbours
        hard_candidates = [
            j for j in sorted_idx if disease_labels[j] != disease_labels[i]
        ]
        # restrict to top_k closest
        hard_candidates = hard_candidates[: max(top_k, 3)]

        # fallbacks if dataset is weird / small
        if len(hard_candidates) < 3:
            hard_candidates = [
                j for j in range(n) if j != i and disease_labels[j] != disease_labels[i]
            ]
            if len(hard_candidates) < 3:
                hard_candidates = [j for j in range(n) if j != i]

        a, b, c = rng.sample(hard_candidates, 3)
        triples.append((i, a, b, c))

    return items, triples


# =====================================
# Dataset / Embeddings
# =====================================
class ImageDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]["modalities"][0]["value"]
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im


def _check_embeddings(name, embeds):
    assert embeds.ndim == 2, f"{name} embeds must be 2D, got {embeds.shape}"
    finite = torch.isfinite(embeds).all().item()
    stdv = float(embeds.std().item())
    logger.info(
        f"{name.capitalize()} embeds: shape={tuple(embeds.shape)}, finite={finite}, std={stdv:.6f}"
    )
    if not finite:
        raise ValueError(f"{name} embeds contain non-finite values.")
    if stdv < 1e-6:
        logger.warning(f"{name.capitalize()} embeds nearly constant (std≈0).")


@torch.no_grad()
def compute_image_embeds(model, processor, device, items, batch_size=IMG_BS):
    dataset = ImageDataset(items)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b
    )

    chunks = []
    for batch_images in tqdm(dataloader, desc="Computing image embeddings"):
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if hasattr(model, "get_image_features"):
            emb = model.get_image_features(**inputs)
        else:
            # BiomedCLIP's forward (trust_remote_code) returns image_embeds
            out = model(**inputs)
            if hasattr(out, "image_embeds"):
                emb = out.image_embeds
            else:
                raise ValueError(
                    "Model has neither get_image_features nor image_embeds."
                )
        emb = emb.image_embeds if hasattr(emb, "image_embeds") else emb.pooler_output
        emb = torch.nn.functional.normalize(emb, dim=1)
        chunks.append(emb.cpu())

    img_embeds = torch.cat(chunks, dim=0)
    _check_embeddings("image", img_embeds)
    return img_embeds


def _filter_text_inputs(toks):
    # Keep only what text towers universally accept
    return {k: v for k, v in toks.items() if k in ("input_ids", "attention_mask")}


@torch.no_grad()
def compute_label_embeds(
    model, processor, device, labels, txt_bs=TXT_BS, max_len=TXT_MAX_LEN
):
    """
    Here, `labels` are the *text descriptions* (enhanced or raw),
    NOT the disease classes. We want one embedding per distinct text.
    """
    uniq = sorted(set(labels))
    chunks = []
    for i in range(0, len(uniq), txt_bs):
        batch_labels = uniq[i : i + txt_bs]
        toks = processor(
            text=batch_labels,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        toks = _filter_text_inputs(toks)
        assert (
            toks["attention_mask"].sum().item() > 0
        ), "All-pad attention_mask for label batch!"
        toks = {k: v.to(device) for k, v in toks.items()}
        if hasattr(model, "get_text_features"):
            emb = model.get_text_features(**toks)
        else:
            out = model(**toks)
            if hasattr(out, "text_embeds"):
                emb = out.text_embeds
            else:
                raise ValueError("Model has neither get_text_features nor text_embeds.")
        emb = emb.image_embeds if hasattr(emb, "image_embeds") else emb.pooler_output
        emb = torch.nn.functional.normalize(emb, dim=1).cpu()
        chunks.append(emb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    txt_embeds = torch.cat(chunks, dim=0)
    _check_embeddings("text", txt_embeds)

    uniq_vecs = txt_embeds.unique(dim=0).shape[0]
    logger.info(f"Text embeddings unique vectors: {uniq_vecs} / {txt_embeds.shape[0]}")
    if uniq_vecs <= 1:
        logger.warning(
            "All text embeddings are identical → degenerate retrieval expected."
        )

    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    return txt_embeds, label_to_idx


# =====================================
# Evaluation
# =====================================
@torch.no_grad()
def evaluate_model(model_dir, items, triples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {model_dir} on device={device}")

    model = load_model(model_dir).to(device).eval()
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_dir)

    # --- Embeddings & retrieval ---
    img_embeds = compute_image_embeds(model, processor, device, items)
    all_labels = [it["text"] for it in items]  # enhanced descriptions
    txt_embeds, label_to_idx = compute_label_embeds(
        model, processor, device, all_labels
    )

    good = 0
    tie_count = 0
    nonfinite_sims = 0

    for i, a, b, c in tqdm(triples, desc="Evaluating"):
        img = img_embeds[i]  # (D,)

        # candidates: text for correct + 3 negatives
        labs = [
            items[i]["text"],
            items[a]["text"],
            items[b]["text"],
            items[c]["text"],
        ]
        idxs = [label_to_idx[lab] for lab in labs]
        cand_txt = txt_embeds[idxs]  # (4, D)
        sims = torch.matmul(cand_txt, img)  # (4,)

        if not torch.isfinite(sims).all():
            nonfinite_sims += 1
            logger.error(f"Non-finite similarities at item {i}: {sims}")
            pred = -1
        else:
            if float(sims.max().item() - sims.min().item()) < TIE_EPS:
                tie_count += 1
                if TIE_POLICY == "count_incorrect":
                    pred = -1
                elif TIE_POLICY == "skip":
                    pred = -1
                elif TIE_POLICY == "argmax":
                    pred = int(sims.argmax().item())
                else:
                    raise ValueError(f"Unknown TIE_POLICY: {TIE_POLICY}")
            else:
                pred = int(sims.argmax().item())

        if pred == 0:
            good += 1

    acc = good / len(items)
    logger.info(
        f"Evaluation done. "
        f"good={good}, total={len(items)}, ties={tie_count}, nonfinite_sims={nonfinite_sims}"
    )
    return acc, {"ties": tie_count, "nonfinite_sims": nonfinite_sims}


@torch.no_grad()
def evaluate_hf_clip(model_name, items, triples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading HF model {model_name} on device={device}")

    model, processor = load_clip_like(
        model_name, device
    )  # returns CLIPModel+CLIPProcessor

    img_embeds = compute_image_embeds(model, processor, device, items)
    all_labels = [it["text"] for it in items]
    txt_embeds, label_to_idx = compute_label_embeds(
        model, processor, device, all_labels
    )

    good = 0
    tie_count = 0
    nonfinite_sims = 0

    for i, a, b, c in tqdm(triples, desc=f"Evaluating {model_name}"):
        img = img_embeds[i]
        labs = [items[i]["text"], items[a]["text"], items[b]["text"], items[c]["text"]]
        idxs = [label_to_idx[lab] for lab in labs]
        cand_txt = txt_embeds[idxs]
        sims = torch.matmul(cand_txt, img)

        if not torch.isfinite(sims).all():
            nonfinite_sims += 1
            pred = -1
        else:
            if float(sims.max().item() - sims.min().item()) < TIE_EPS:
                tie_count += 1
                pred = (
                    -1
                    if TIE_POLICY in ("count_incorrect", "skip")
                    else int(sims.argmax().item())
                )
            else:
                pred = int(sims.argmax().item())

        if pred == 0:
            good += 1

    acc = good / len(items)
    return acc, {"ties": tie_count, "nonfinite_sims": nonfinite_sims}


# =====================================
# Main
# =====================================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load all items
    items = resolve_many(EVAL_DATASETS, LINE_NUMBER)
    logger.info(f"Loaded {len(items)} items from {len(EVAL_DATASETS)} datasets.")

    # 2) Build hard protocol using BiomedCLIP as reference
    logger.info(f"Building hard protocol using reference model: {REF_MODEL_NAME}")
    ref_model, ref_processor = load_clip_like(REF_MODEL_NAME, device)
    ref_img_embeds = compute_image_embeds(ref_model, ref_processor, device, items)
    items, triples = build_hard_protocol_from_embeds(
        items,
        ref_img_embeds,
        seed=SEED,
        top_k=HARD_TOPK,
    )
    logger.info(f"Hard protocol built: {len(items)} items, {len(triples)} triples.")

    # 3) Evaluate all configs on this fixed hard protocol and write to txt
    os.makedirs(os.path.dirname(RESULTS_TXT), exist_ok=True)
    with open(RESULTS_TXT, "w") as f:
        f.write("# name\taccuracy\tties\tnonfinite_sims\n")
        for name, path in CLIP_CONFIGS:
            logger.info(f"Evaluating config: {name} at {path}")
            acc, stats = evaluate_model(path, items, triples)
            line = f"{name}\t{acc:.6f}\t{stats['ties']}\t{stats['nonfinite_sims']}\n"
            f.write(line)
            f.flush()
            print(line, end="")

    logger.info(f"Saved results to {RESULTS_TXT}")


if __name__ == "__main__":
    main()
