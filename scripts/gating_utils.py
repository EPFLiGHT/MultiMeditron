"""Shared helpers for the gating-network analysis / test scripts.

Used by ``gating_routing_analysis.py``, ``pathvqa_routing_analysis.py`` and
``test_gating.py`` to avoid duplicating the Arrow image loader, the ResNet/
ImageNet preprocessing transform, the expert-label mapping, and the gating
inference loop.

The gating backbone is a ResNet50 (see
``multimeditron.model.modalities.moe.gating.GatingNetwork``), so images are
preprocessed with the standard ImageNet resize/crop/normalize pipeline.
"""

import io
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.ipc as ipc
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Resolve the repository root relative to this file (scripts/ -> repo root) so the
# scripts work from any checkout / account, not just the original one. Make the
# multimeditron package and the lmms-eval submodule importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "src", REPO_ROOT / "third-party" / "lmms-eval"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Cluster paths default to the current CSCS locations but can be overridden via
# environment variables for use on another account or machine.
ARROW_ROOT = os.environ.get(
    "MM_ARROW_ROOT",
    "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow",
)
# 7-expert gating checkpoint (defaults to the in-repo copy).
GATING_7EXP = os.environ.get("MM_GATING_MODEL", str(REPO_ROOT / "models/CLIP/MultiMeditron-Gating"))
# Original 5-expert gating checkpoint (HF cache snapshot).
GATING_5EXP = os.environ.get(
    "MM_GATING_5EXP",
    "/capstor/store/cscs/swissai/a127/meditron/hf_cache/hub/"
    "models--ClosedMeditron--MultiMeditron-Gating/snapshots/"
    "e1d1310b6e1962857b61b0009b9a4d7e196e84fa",
)

# ResNet50 / ImageNet preprocessing.
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Map an expert checkpoint basename -> short, human-readable label.
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
    """Extract a human-readable expert label from a full expert path."""
    basename = os.path.basename(full_path.rstrip("/"))
    return EXPERT_LABELS.get(basename, basename)


def _read_arrow_table(arrow_file: str):
    """Read an Arrow file, trying IPC stream format first, then file format."""
    try:
        with open(arrow_file, "rb") as fh:
            return ipc.open_stream(fh).read_all()
    except Exception:
        with open(arrow_file, "rb") as fh:
            return ipc.open_file(fh).read_all()


def _decode_image(img_bytes) -> Optional[Image.Image]:
    """Decode raw image bytes into an RGB PIL image (None if empty)."""
    if not img_bytes:
        return None
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def load_images_from_arrow(
    dataset_path: str,
    n: int,
    seed: int = 42,
    shuffle_shards: bool = False,
) -> List[Image.Image]:
    """Read up to ``n`` PIL images from the HF-format Arrow shards in a directory.

    Handles the three schemas found across MultiMediset datasets:

    * ``image``             — ``struct{bytes, path}``               (eye / skin datasets)
    * ``modalities_images`` — ``list<struct{bytes, path}>``         (``*_expert`` datasets)
    * ``modalities``        — ``list<struct{type, value{bytes}}>``  (conversation format)

    Args:
        dataset_path: Directory containing ``*.arrow`` shard files.
        n: Maximum number of images to return.
        seed: RNG seed used when ``shuffle_shards`` is True.
        shuffle_shards: If True, shuffle the shard read order so repeated runs
            sample different shards.

    Returns:
        A list of up to ``n`` RGB PIL images.
    """
    arrow_files = sorted(
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.endswith(".arrow")
    )
    if shuffle_shards:
        random.Random(seed).shuffle(arrow_files)

    images: List[Image.Image] = []
    for arrow_file in arrow_files:
        if len(images) >= n:
            break
        try:
            table = _read_arrow_table(arrow_file)
        except Exception as e:
            print(f"  [WARN] cannot read {os.path.basename(arrow_file)}: {e}")
            continue

        cols = table.schema.names

        if "image" in cols:
            col = table.column("image")
            for i in range(len(col)):
                if len(images) >= n:
                    break
                try:
                    row = col[i].as_py()
                    if isinstance(row, dict):
                        b = row.get("bytes")
                    elif isinstance(row, bytes):
                        b = row
                    else:
                        continue
                    pil = _decode_image(b)
                    if pil is not None:
                        images.append(pil)
                except Exception as e:
                    logger.debug("skipping unreadable row %d in %s: %s",
                                 i, os.path.basename(arrow_file), e)
                    continue

        elif "modalities_images" in cols:
            col = table.column("modalities_images")
            for i in range(len(col)):
                if len(images) >= n:
                    break
                try:
                    img_list = col[i].as_py()
                    if not img_list:
                        continue
                    pil = _decode_image(img_list[0].get("bytes"))
                    if pil is not None:
                        images.append(pil)
                except Exception as e:
                    logger.debug("skipping unreadable row %d in %s: %s",
                                 i, os.path.basename(arrow_file), e)
                    continue

        elif "modalities" in cols:
            col = table.column("modalities")
            for i in range(len(col)):
                if len(images) >= n:
                    break
                try:
                    mods = col[i].as_py()
                    if not mods:
                        continue
                    for mod in mods:
                        if mod.get("type") != "image":
                            continue
                        pil = _decode_image((mod.get("value") or {}).get("bytes"))
                        if pil is not None:
                            images.append(pil)
                        break  # one image per row
                except Exception as e:
                    logger.debug("skipping unreadable row %d in %s: %s",
                                 i, os.path.basename(arrow_file), e)
                    continue

        else:
            print(f"  [WARN] no recognized image column in "
                  f"{os.path.basename(arrow_file)} (cols={cols})")

    return images


def run_gating(
    gating_model,
    class_names: List[str],
    images: List[Image.Image],
    batch_size: int = 32,
) -> Dict:
    """Run a gating model over PIL images and return routing statistics.

    Args:
        gating_model: A loaded ``GatingNetwork`` in eval mode.
        class_names: Ordered expert class names (``model.config.class_names``).
        images: PIL images to route.
        batch_size: Inference batch size.

    Returns:
        Dict with ``n`` (image count), ``routing_pct`` (top-1 % per expert) and
        ``avg_weights`` (mean softmax weight per expert).
    """
    device = next(gating_model.parameters()).device
    vote_counter: Counter = Counter()
    score_accum: Dict[str, float] = defaultdict(float)

    for start in range(0, len(images), batch_size):
        batch = images[start: start + batch_size]
        tensors = torch.stack([PREPROCESS(img) for img in batch]).to(device)

        with torch.no_grad():
            _logits, topk_indices, weights = gating_model(tensors)

        for idx in topk_indices[:, 0].cpu().tolist():
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


def print_routing(stats: Dict, class_names: List[str], label_width: int = 30) -> None:
    """Pretty-print routing stats (top-1 % + avg weight + bar) for one model."""
    pcts = stats["routing_pct"]
    wgts = stats["avg_weights"]
    print(f"  {'Expert':<{label_width}} {'Top-1 %':>8}  {'Avg weight':>10}  Bar")
    print(f"  {'-' * 70}")
    for name in sorted(class_names, key=lambda nm: -pcts[nm]):
        bar = "█" * int(pcts[name] / 2)
        print(f"  {short_name(name):<{label_width}} {pcts[name]:>7.1f}%  "
              f"{wgts[name]:>10.4f}  {bar}")


def load_gating(path: str, label: str = "", device: Optional[str] = None) -> Tuple:
    """Load a ``GatingNetwork`` checkpoint and print its expert list.

    The import of ``GatingNetwork`` is deferred to call time so callers can set
    up ``sys.path`` (to find the ``multimeditron`` package) before invoking this.

    Returns:
        Tuple of ``(model, class_names)``.
    """
    from multimeditron.model.modalities.moe.gating import GatingNetwork

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {label or 'gating'} from:\n  {path}")
    model = GatingNetwork.from_pretrained(path)
    model.eval().to(device)
    names = model.config.class_names
    print(f"  {model.config.num_classes} experts: {[short_name(x) for x in names]}")
    return model, names
