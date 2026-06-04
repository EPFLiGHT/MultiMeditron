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

import os
import random
import sys

import torch

# Make the multimeditron package importable for gating_utils.load_gating.
sys.path.insert(0, "/users/surech/meditron/MultiMeditron/src")

from gating_utils import load_gating, load_images_from_arrow, print_routing, run_gating, short_name

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

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── main ──────────────────────────────────────────────────────────────────────

gating_models = {
    "5-expert (original, CT/MRI/US/Xray/Generalist)": load_gating(GATING_5EXP, "5-expert", device),
    "7-expert (retrained, +Ophthalmology +Skin)":      load_gating(GATING_7EXP, "7-expert", device),
}

# Collect all per-dataset results so we can also print a summary matrix
all_results = {}   # {dataset_label: {model_label: (stats, class_names)}}

for ds_label, ds_path in DATASETS.items():
    print(f"\n{'='*72}")
    print(f"  Dataset: {ds_label}")
    print(f"  Path   : {ds_path}")
    print(f"{'='*72}")

    images = load_images_from_arrow(ds_path, N_SAMPLES, seed=SEED, shuffle_shards=True)
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
