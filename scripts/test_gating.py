"""
Test a trained gating network on eye and skin images to see how it routes them.

The 7-expert gating routes over 7 classes:
  CT, Generalist, MRI, Ultrasound, X-ray, Ophthalmology, Skin

Usage (inside container):
    python3 scripts/test_gating.py
"""

import random
import sys

import torch

# Make the multimeditron package importable for gating_utils.load_gating.
sys.path.insert(0, "/users/surech/meditron/MultiMeditron/src")
sys.path.insert(0, "/users/surech/meditron/MultiMeditron/third-party/lmms-eval")

from gating_utils import load_gating, load_images_from_arrow, print_routing, run_gating

# ── paths ──────────────────────────────────────────────────────────────────────
GATING_MODEL_PATH = "/users/surech/meditron/MultiMeditron/models/CLIP/MultiMeditron-Gating"
EYE_DATASET_PATH  = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/eye_dataset/train"
SKIN_DATASET_PATH = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/skin_dataset/train"

N_SAMPLES = 100   # images to sample per dataset
SEED      = 42
# ───────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── load gating network ────────────────────────────────────────────────────────
gating, class_names = load_gating(GATING_MODEL_PATH, "gating", device)

# ── run on both datasets ───────────────────────────────────────────────────────
datasets = {
    "EyeDataset  (Ophthalmology)": EYE_DATASET_PATH,
    "SkinDataset (Dermatology)  ": SKIN_DATASET_PATH,
}

for label, path in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    images = load_images_from_arrow(path, N_SAMPLES, seed=SEED)
    if not images:
        print("  [ERROR] No images loaded, skipping.")
        continue
    print(f"  Loaded {len(images)} images.\n")

    stats = run_gating(gating, class_names, images, batch_size=16)
    print_routing(stats, class_names)

print("\nDone.")
