# ============================================================
# SCIN Hard Benchmark + Skin-Tone Stratified Evaluation
# (Augmented val split + manifest metadata join)
# CONSISTENT protocol: build ONCE using vanilla CLIP reference
# Hard-negative selection logic matches black-skin script
# ============================================================

import os
import json
import random
import logging
import re
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    CLIPModel,
    CLIPProcessor,
    VisionTextDualEncoderProcessor,
)

from load_from_clip import load_model


# =========================
# CONFIG
# =========================

EVAL_DATASETS = [
    "/mloscratch/users/turan/datasets/skin_expert_datasets/SCIN/scin_api_val.jsonl",
]

MANIFEST_DATASETS = [
    "/mloscratch/users/turan/datasets/skin_expert_datasets/SCIN/scin_manifest.jsonl",
]

CLIP_CONFIGS = [
    ("skin_clip_config_10_before", "/mloscratch/users/turan/training/models_skin/combined_dataset_skin_regularization_focused_config_1"),
    ("skin_clip_config_10_after",  "/mloscratch/users/turan/training/models/combined_dataset_skin_regularization_focused_config_1"),
]

# Reference model used ONLY to build fixed protocol
REF_MODEL_NAME = "openai/clip-vit-base-patch32"

RESULTS_TXT = "/mloscratch/users/turan/evaluation_clip/scin_skin_tone_hard_results.txt"

SEED        = 14
IMG_BS      = 32
TXT_BS      = 64
TXT_MAX_LEN = 128

HARD_TOPK   = 3   # <-- match black-skin script logic (pool size)
TIE_EPS     = 1e-7

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("scin-hard-eval")


# =========================
# FITZPATRICK EXTRACTION
# =========================

FST_RE = re.compile(
    r"Fitzpatrick(?: skin type)?[:\s]*(?:type\s*)?(FST\s*\d|[1-6]|VI|IV|V|III|II|I)",
    re.IGNORECASE,
)

ROMAN_TO_FST = {
    "I": "FST1", "II": "FST2", "III": "FST3",
    "IV": "FST4", "V": "FST5", "VI": "FST6",
}

def extract_fst(text):
    if not text:
        return None
    m = FST_RE.search(text)
    if not m:
        return None
    raw = m.group(1).upper().replace(" ", "")
    if raw.startswith("FST"):
        return raw
    if raw.isdigit():
        return f"FST{raw}"
    if raw in ROMAN_TO_FST:
        return ROMAN_TO_FST[raw]
    return None

def fst_to_group(fst):
    if fst in ("FST1", "FST2"):
        return "light"
    if fst == "FST3":
        return "medium"
    if fst in ("FST4", "FST5", "FST6"):
        return "dark"
    return "unknown"


# =========================
# PROXY DISEASE EXTRACTION
# =========================

DIFF_RE = re.compile(
    r"Dermatologist differential \(weighted\):(.+?)(?:<attachment>|$)",
    re.IGNORECASE | re.DOTALL,
)

DIAG_RE = re.compile(
    r"([A-Za-z0-9\s\-/]+)\s*\((0?\.\d+|1\.00)\)"
)

def extract_top_differential(text):
    if not text:
        return None
    m = DIFF_RE.search(text)
    if not m:
        return None
    block = m.group(1)
    matches = DIAG_RE.findall(block)
    if not matches:
        return None
    diag, _ = max(matches, key=lambda x: float(x[1]))
    return diag.strip().lower()


# =========================
# DATA LOADING
# =========================

def resolve_items(jsonl_path):
    items = []
    base_dir = Path(jsonl_path).parent
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            img = Path(it["modalities"][0]["value"])
            if not img.is_absolute():
                img = (base_dir / img).resolve()
            it["modalities"][0]["value"] = str(img)
            items.append(it)
    return items

def resolve_many(paths):
    out = []
    for p in paths:
        out.extend(resolve_items(p))
    return out


# =========================
# LOAD MANIFEST LOOKUP
# =========================

def load_manifest_lookup(manifest_paths):
    fst_by_fname = {}
    diff_by_fname = {}

    for p in manifest_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                fname = Path(it["modalities"][0]["value"]).name
                text = it["text"]

                fst = extract_fst(text)
                diff = extract_top_differential(text)

                if fst:
                    fst_by_fname[fname] = fst
                if diff:
                    diff_by_fname[fname] = diff

    return fst_by_fname, diff_by_fname


# =========================
# DATASET
# =========================

class ImageDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        img = Image.open(self.items[idx]["modalities"][0]["value"])
        return img.convert("RGB")


# =========================
# EMBEDDINGS
# =========================

@torch.no_grad()
def compute_image_embeds(model, processor, device, items):
    dl = DataLoader(
        ImageDataset(items),
        batch_size=IMG_BS,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    outs = []
    for batch in tqdm(dl, desc="Image embeds"):
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if hasattr(model, "get_image_features"):
            emb = model.get_image_features(**inputs)
        else:
            emb = model(**inputs).image_embeds

        outs.append(torch.nn.functional.normalize(emb, dim=1).cpu())

    return torch.cat(outs, dim=0)

@torch.no_grad()
def compute_text_embeds(model, processor, device, texts):
    uniq = list(dict.fromkeys(texts))
    outs = []
    for i in range(0, len(uniq), TXT_BS):
        toks = processor(
            text=uniq[i:i+TXT_BS],
            padding=True,
            truncation=True,
            max_length=TXT_MAX_LEN,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()
                if k in ("input_ids", "attention_mask")}
        emb = (
            model.get_text_features(**toks)
            if hasattr(model, "get_text_features")
            else model(**toks).text_embeds
        )
        outs.append(torch.nn.functional.normalize(emb, dim=1).cpu())
    embeds = torch.cat(outs)
    return embeds, {t: i for i, t in enumerate(uniq)}


# =========================
# HARD PROTOCOL (FIXED, ONCE) — matches black-skin logic
# =========================

def build_hard_protocol_from_embeds(items, img_embeds, seed=SEED, top_k=HARD_TOPK):
    """
    Match black-skin script logic exactly:
      - candidates = top_k nearest neighbors with different proxy label
      - if <3, fallback to anywhere with different proxy label
      - if still <3, fallback to any non-self
    """
    rng = random.Random(seed)
    n = len(items)
    labels = [it["_proxy_disease"] for it in items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_embeds_dev = img_embeds.to(device)

    triples = []
    for i in range(n):
        img_vec = img_embeds_dev[i]
        sims = torch.matmul(img_embeds_dev, img_vec)
        sims[i] = -1.0

        sorted_idx = torch.argsort(sims, descending=True).tolist()

        hard_candidates = [j for j in sorted_idx if labels[j] != labels[i]]
        hard_candidates = hard_candidates[:max(top_k, 3)]

        if len(hard_candidates) < 3:
            hard_candidates = [j for j in range(n) if j != i and labels[j] != labels[i]]
            if len(hard_candidates) < 3:
                hard_candidates = [j for j in range(n) if j != i]

        a, b, c = rng.sample(hard_candidates, 3)
        triples.append((i, a, b, c))

    return triples


# =========================
# EVALUATION
# =========================

@torch.no_grad()
def evaluate(model_dir, items, triples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_dir).to(device).eval()
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_dir)

    img_embeds = compute_image_embeds(model, processor, device, items)
    texts = [it["text"] for it in items]
    txt_embeds, txt_map = compute_text_embeds(model, processor, device, texts)

    tot = defaultdict(int)
    ok  = defaultdict(int)

    for (i, a, b, c) in tqdm(triples, desc="Evaluating"):
        img = img_embeds[i]
        labs = [items[i]["text"], items[a]["text"], items[b]["text"], items[c]["text"]]
        sims = torch.matmul(txt_embeds[[txt_map[l] for l in labs]], img)

        correct = (
            torch.isfinite(sims).all()
            and (sims.max() - sims.min()) > TIE_EPS
            and sims.argmax().item() == 0
        )

        g = items[i]["_skin_group"]
        tot[g] += 1
        if correct:
            ok[g] += 1

    return {g: (ok[g] / tot[g]) for g in tot}


# =========================
# MAIN
# =========================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    items = resolve_many(EVAL_DATASETS)
    fst_map, diff_map = load_manifest_lookup(MANIFEST_DATASETS)

    # Attach fst / group / proxy label
    for it in items:
        fname = Path(it["modalities"][0]["value"]).name
        text = it.get("text", "")

        fst = extract_fst(text) or fst_map.get(fname)
        diff = extract_top_differential(text) or diff_map.get(fname)

        it["_fst"] = fst
        it["_skin_group"] = fst_to_group(fst)
        it["_proxy_disease"] = diff or "unknown"

    logger.info("Skin-tone group counts: %s", Counter(it["_skin_group"] for it in items))
    logger.info("Fitzpatrick counts: %s", Counter(it["_fst"] for it in items))
    logger.info("Proxy disease counts (top 20): %s", Counter(it["_proxy_disease"] for it in items).most_common(20))

    # ------------------------------------------------------------
    # Build ONE fixed hard protocol using vanilla CLIP reference
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Building FIXED hard protocol using reference model: {REF_MODEL_NAME}")

    ref_model = CLIPModel.from_pretrained(REF_MODEL_NAME).to(device).eval()
    ref_proc  = CLIPProcessor.from_pretrained(REF_MODEL_NAME)

    ref_img_embeds = compute_image_embeds(ref_model, ref_proc, device, items)
    triples = build_hard_protocol_from_embeds(items, ref_img_embeds, seed=SEED, top_k=HARD_TOPK)

    # Save protocol for reproducibility
    protocol_path = RESULTS_TXT.replace(".txt", "_protocol.json")
    try:
        with open(protocol_path, "w") as pf:
            json.dump(triples, pf)
        logger.info(f"Saved fixed hard protocol to {protocol_path}")
    except Exception as e:
        logger.warning(f"Could not save protocol to {protocol_path}: {e}")

    # ------------------------------------------------------------
    # Evaluate both configs on the SAME fixed triples + write results
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_TXT), exist_ok=True)
    with open(RESULTS_TXT, "w") as f:
        f.write("model\tlight\tmedium\tdark\tunknown\n")
        for name, path in CLIP_CONFIGS:
            logger.info(f"Evaluating {name} on fixed protocol")
            acc = evaluate(path, items, triples)

            line = (
                f"{name}\t"
                f"{acc.get('light', 0):.4f}\t"
                f"{acc.get('medium', 0):.4f}\t"
                f"{acc.get('dark', 0):.4f}\t"
                f"{acc.get('unknown', 0):.4f}"
            )
            print(line)
            f.write(line + "\n")

    logger.info(f"Saved results to {RESULTS_TXT}")


if __name__ == "__main__":
    main()
