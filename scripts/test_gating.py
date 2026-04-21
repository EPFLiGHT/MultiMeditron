"""
Test the existing gating network on eye and skin images to see how it routes them.

The gating was trained on 5 classes:
  ClosedMeditron/MedExpert-CT
  ClosedMeditron/clip-vit-base-patch32  (general)
  ClosedMeditron/MedExpert-MRI
  ClosedMeditron/MedExpert-Ultrasound
  ClosedMeditron/MedExpert-Xray

Usage (inside container):
    python3 scripts/test_gating.py
"""

import sys
import os
import io
import random
from collections import Counter, defaultdict

import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────────────
GATING_MODEL_PATH = "/users/surech/meditron/MultiMeditron/models/CLIP/MultiMeditron-Gating"
EYE_DATASET_PATH  = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/eye_dataset/train"
SKIN_DATASET_PATH = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/skin_dataset/train"

N_SAMPLES = 100   # images to sample per dataset
SEED      = 42
# ───────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, "/users/surech/meditron/MultiMeditron/src")
sys.path.insert(0, "/users/surech/meditron/MultiMeditron/third-party/lmms-eval")

from multimeditron.model.modalities.moe.gating import GatingNetwork

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── load gating network ────────────────────────────────────────────────────────
print(f"\nLoading gating network from {GATING_MODEL_PATH} ...")
gating = GatingNetwork.from_pretrained(GATING_MODEL_PATH)
gating.eval().to(device)

class_names = gating.config.class_names
num_classes = gating.config.num_classes
print(f"Classes ({num_classes}): {class_names}")

# ── ResNet50 expects ImageNet normalization ────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_images_from_arrow(dataset_path: str, n: int) -> list:
    """Read up to n images (as PIL) from HF Arrow shards."""
    arrow_files = sorted([
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.startswith("data-") and f.endswith(".arrow")
    ])

    images = []
    for arrow_file in arrow_files:
        if len(images) >= n:
            break
        try:
            with open(arrow_file, "rb") as f:
                reader = ipc.open_stream(f)
                table = reader.read_all()
        except Exception:
            try:
                with open(arrow_file, "rb") as f:
                    reader = ipc.open_file(f)
                    table = reader.read_all()
            except Exception as e:
                print(f"  [WARN] could not read {arrow_file}: {e}")
                continue

        # Image column: struct with 'bytes' and 'path'
        if "image" not in table.schema.names:
            print(f"  [WARN] no 'image' column in {arrow_file}")
            continue

        img_col = table.column("image")
        for i in range(len(img_col)):
            if len(images) >= n:
                break
            try:
                row = img_col[i].as_py()
                img_bytes = row.get("bytes") if isinstance(row, dict) else None
                if img_bytes is None:
                    continue
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(pil)
            except Exception:
                continue

    print(f"  Loaded {len(images)} images from {dataset_path}")
    return images


def run_gating(images: list) -> dict:
    """Run gating on a list of PIL images, return routing statistics."""
    vote_counter  = Counter()
    score_accum   = defaultdict(float)
    top1_per_image = []

    batch_size = 16
    for start in range(0, len(images), batch_size):
        batch_pils = images[start : start + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch_pils]).to(device)

        with torch.no_grad():
            logits, topk_indices, weights = gating(tensors)

        # top-1 routing decision per image
        top1 = topk_indices[:, 0].cpu().tolist()
        for idx in top1:
            vote_counter[class_names[idx]] += 1
            top1_per_image.append(class_names[idx])

        # accumulate softmax weights
        w = weights.cpu()
        for c_idx, name in enumerate(class_names):
            score_accum[name] += w[:, c_idx].sum().item()

    n = len(images)
    avg_weights = {name: score_accum[name] / n for name in class_names}
    routing_pct = {name: 100.0 * vote_counter[name] / n for name in class_names}

    return {"routing_pct": routing_pct, "avg_weights": avg_weights}


# ── run on both datasets ───────────────────────────────────────────────────────
datasets = {
    "EyeDataset  (Ophthalmology)": EYE_DATASET_PATH,
    "SkinDataset (Dermatology)  ": SKIN_DATASET_PATH,
}

for label, path in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    images = load_images_from_arrow(path, N_SAMPLES)
    if not images:
        print("  [ERROR] No images loaded, skipping.")
        continue

    stats = run_gating(images)

    print(f"\n  Top-1 routing (% of {len(images)} images routed to each expert):")
    for name, pct in sorted(stats["routing_pct"].items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        short = name.split("/")[-1]
        print(f"    {short:<30} {pct:5.1f}%  {bar}")

    print(f"\n  Average softmax weight per expert:")
    for name, w in sorted(stats["avg_weights"].items(), key=lambda x: -x[1]):
        short = name.split("/")[-1]
        print(f"    {short:<30} {w:.4f}")

print("\nDone.")
