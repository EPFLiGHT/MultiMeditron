"""
Comprehensive gating routing analysis for both the 5-expert and 7-expert gating
networks. Tests each network on 5 modality-pure datasets and reports top-1
routing percentages and average softmax weights per expert.

Datasets tested (5/7 expert modalities) — ALL held-out splits never seen
during gating training:
  ct2_expert/test       → CT scans  (821 images)
  XR-glob_expert/test   → chest X-rays  (745 images)
  BUSI_expert/test      → breast ultrasound  (156 images)
  eye_dataset/val       → ophthalmology / fundus  (903 images)
  skin_dataset/val      → dermatology  (2348 images)

NOTE on previous contaminated results:
  The original run used ct2 / iu_xray / BUSI (same source as training data
  image_ct2 / image_iu_xray / image_BUSI) and eye_dataset/train + 
  skin_dataset/train (actual training splits).  100% accuracy was partly
  an artefact of evaluating on training data.

NOT TESTED (no standalone dataset available on cluster):
  MRI       → No MRI-only arrow dataset exists in multimediset/arrow/*.
              To test MRI routing, download an MRI-only dataset (e.g.,
              BraTS 2021 or IXI) and add it to DATASETS below.
  Generalist → Not a specific imaging modality; no pure "generalist" dataset.

Gating models compared:
  5-expert  → ClosedMeditron/MultiMeditron-Gating (HF cache, original)
  7-expert  → models/CLIP/MultiMeditron-Gating (retrained, adds Ophtho + Skin)

Usage (on the login node — no GPU required for ResNet50):
    python3 scripts/gating_routing_analysis.py
"""

import sys
import os
import io
import random
from collections import Counter, defaultdict

import torch
import pyarrow.ipc as ipc
from PIL import Image
from torchvision import transforms

# ── config ────────────────────────────────────────────────────────────────────

GATING_5EXP = (
    "/capstor/store/cscs/swissai/a127/meditron/hf_cache/hub/"
    "models--ClosedMeditron--MultiMeditron-Gating/snapshots/"
    "e1d1310b6e1962857b61b0009b9a4d7e196e84fa"
)
GATING_7EXP = (
    "/users/surech/meditron/MultiMeditron/models/CLIP/MultiMeditron-Gating"
)

ARROW_ROOT = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow"

# All paths point to held-out splits (test or val) that were never used
# during 7-expert gating training (training used: image_ct2, image_iu_xray,
# image_BUSI, image_COVID_US, image_DDTI, eye_dataset/train, skin_dataset/train).
DATASETS = {
    "CT       (ct2_expert/test)":       os.path.join(ARROW_ROOT, "ct2_expert",      "test"),
    "X-ray    (XR-glob_expert/test)":   os.path.join(ARROW_ROOT, "XR-glob_expert",  "test"),
    "Ultrasnd (BUSI_expert/test)":      os.path.join(ARROW_ROOT, "BUSI_expert",     "test"),
    "Eye      (eye_dataset/val)":        os.path.join(ARROW_ROOT, "eye_dataset",     "val"),
    "Skin     (skin_dataset/val)":       os.path.join(ARROW_ROOT, "skin_dataset",    "val"),
}

N_SAMPLES = 200   # images per dataset
SEED = 42

# ── short label for each expert path ─────────────────────────────────────────
EXPERT_LABELS = {
    "MedExpert-CT": "CT",
    "clip-vit-base-patch32": "Generalist",
    "MedExpert-MRI": "MRI",
    "MedExpert-Ultrasound": "Ultrasound",
    "checkpoint-4350": "Ultrasound",   # UltraSoundCLIP checkpoint alias
    "MedExpert-Xray": "X-ray",
    "OphthalmologyExpert": "Ophthalmology",
    "SkinExpert": "Skin",
}


def short_name(full_path: str) -> str:
    """Extract a human-readable label from a full expert path."""
    basename = os.path.basename(full_path.rstrip("/"))
    return EXPERT_LABELS.get(basename, basename)


# ── paths ─────────────────────────────────────────────────────────────────────

sys.path.insert(0, "/users/surech/meditron/MultiMeditron/src")
from multimeditron.model.modalities.moe.gating import GatingNetwork

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ResNet50 / ImageNet normalisation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── image loading ─────────────────────────────────────────────────────────────

def load_images_from_arrow(dataset_path: str, n: int) -> list:
    """Read up to n images (as PIL.Image) from HF-format Arrow shards."""
    arrow_files = sorted([
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.endswith(".arrow")
    ])
    # Shuffle so we sample different shards on each run
    rng = random.Random(SEED)
    rng.shuffle(arrow_files)

    images = []
    for arrow_file in arrow_files:
        if len(images) >= n:
            break
        try:
            with open(arrow_file, "rb") as fh:
                reader = ipc.open_stream(fh)
                table = reader.read_all()
        except Exception:
            try:
                with open(arrow_file, "rb") as fh:
                    reader = ipc.open_file(fh)
                    table = reader.read_all()
            except Exception as e:
                print(f"  [WARN] cannot read {os.path.basename(arrow_file)}: {e}")
                continue

        col_names = table.schema.names
        # ── schema 1: direct image column (eye_dataset/val, skin_dataset/val) ─
        # Columns: image (struct{bytes, path})
        # ── schema 2: *_expert datasets ─────────────────────────────────────
        # Columns: modalities_images (list<struct{bytes, path}>)
        # ── schema 3: old multimeditron conversation format ───────────────────
        # Columns: modalities (list<struct{type, value{bytes, path}}>)

        if "image" in col_names:
            img_col = table.column("image")
            for i in range(len(img_col)):
                if len(images) >= n:
                    break
                try:
                    row = img_col[i].as_py()
                    if isinstance(row, dict):
                        img_bytes = row.get("bytes")
                    elif isinstance(row, bytes):
                        img_bytes = row
                    else:
                        continue
                    if not img_bytes:
                        continue
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(pil)
                except Exception:
                    continue

        elif "modalities_images" in col_names:
            # *_expert datasets: list of {bytes, path} structs
            img_list_col = table.column("modalities_images")
            for i in range(len(img_list_col)):
                if len(images) >= n:
                    break
                try:
                    img_list = img_list_col[i].as_py()
                    if not img_list:
                        continue
                    img_bytes = img_list[0].get("bytes")
                    if not img_bytes:
                        continue
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(pil)
                except Exception:
                    continue

        elif "modalities" in col_names:
            mod_col = table.column("modalities")
            for i in range(len(mod_col)):
                if len(images) >= n:
                    break
                try:
                    modalities = mod_col[i].as_py()
                    if not modalities:
                        continue
                    for mod in modalities:
                        if mod.get("type") != "image":
                            continue
                        img_bytes = (mod.get("value") or {}).get("bytes")
                        if not img_bytes:
                            continue
                        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        images.append(pil)
                        break  # one image per row
                except Exception:
                    continue

    return images


# ── gating inference ──────────────────────────────────────────────────────────

def run_gating(gating_model, class_names: list, images: list) -> dict:
    """Return routing_pct and avg_weights for a list of PIL images."""
    vote_counter = Counter()
    score_accum = defaultdict(float)
    batch_size = 32

    for start in range(0, len(images), batch_size):
        batch = images[start: start + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)

        with torch.no_grad():
            logits, topk_indices, weights = gating_model(tensors)

        top1 = topk_indices[:, 0].cpu().tolist()
        for idx in top1:
            vote_counter[class_names[idx]] += 1

        w = weights.cpu()
        for c_idx, name in enumerate(class_names):
            score_accum[name] += w[:, c_idx].sum().item()

    n = len(images)
    return {
        "n": n,
        "routing_pct": {name: 100.0 * vote_counter[name] / n for name in class_names},
        "avg_weights": {name: score_accum[name] / n for name in class_names},
    }


# ── print helpers ─────────────────────────────────────────────────────────────

def print_routing(stats: dict, class_names: list):
    """Pretty-print routing stats for one (gating, dataset) pair."""
    pcts = stats["routing_pct"]
    wgts = stats["avg_weights"]
    print(f"  {'Expert':<30} {'Top-1 %':>8}  {'Avg weight':>10}  Bar")
    print(f"  {'-'*70}")
    for name in sorted(class_names, key=lambda n: -pcts[n]):
        label = short_name(name)
        pct = pcts[name]
        wgt = wgts[name]
        bar = "█" * int(pct / 2)
        print(f"  {label:<30} {pct:>7.1f}%  {wgt:>10.4f}  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def load_gating(path: str, label: str):
    print(f"\nLoading {label} gating from:\n  {path}")
    m = GatingNetwork.from_pretrained(path)
    m.eval().to(device)
    names = m.config.class_names
    n = m.config.num_classes
    short = [short_name(x) for x in names]
    print(f"  {n} experts: {short}")
    return m, names


gating_models = {
    "5-expert (original, CT/MRI/US/Xray/Generalist)": load_gating(GATING_5EXP, "5-expert"),
    "7-expert (retrained, +Ophthalmology +Skin)":      load_gating(GATING_7EXP, "7-expert"),
}

# Collect all per-dataset results so we can also print a summary matrix
all_results = {}   # {dataset_label: {model_label: stats}}

for ds_label, ds_path in DATASETS.items():
    print(f"\n{'='*72}")
    print(f"  Dataset: {ds_label}")
    print(f"  Path   : {ds_path}")
    print(f"{'='*72}")

    images = load_images_from_arrow(ds_path, N_SAMPLES)
    if not images:
        print("  [ERROR] No images loaded — skipping.")
        continue
    print(f"  Loaded {len(images)} images.\n")

    all_results[ds_label] = {}
    for model_label, (gating, class_names) in gating_models.items():
        print(f"  -- {model_label} --")
        stats = run_gating(gating, class_names, images)
        all_results[ds_label][model_label] = (stats, class_names)
        print_routing(stats, class_names)
        print()


# ── summary: top-1 expert per (dataset × model) ──────────────────────────────
print("\n" + "="*72)
print("SUMMARY — Top-1 routed expert (% of images)")
print("="*72)
header = f"{'Dataset':<28}"
for ml in gating_models:
    header += f"  {ml[:30]:<30}"
print(header)
print("-"*72)

for ds_label, model_dict in all_results.items():
    row = f"{ds_label:<28}"
    for model_label in gating_models:
        if model_label not in model_dict:
            row += f"  {'N/A':<30}"
            continue
        stats, class_names = model_dict[model_label]
        top1_name = max(class_names, key=lambda n: stats["routing_pct"][n])
        top1_pct  = stats["routing_pct"][top1_name]
        cell = f"{short_name(top1_name)} ({top1_pct:.0f}%)"
        row += f"  {cell:<30}"
    print(row)

print()
print("Done.")
