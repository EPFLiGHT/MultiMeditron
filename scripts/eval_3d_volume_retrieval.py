"""
Zero-shot image-text retrieval sanity check for the M3D-CLIP backend.

Loads M3D-CLIP, embeds (volume, caption) pairs from a user-provided
CSV, and reports R@1 / R@5 / R@10. The published baseline on the
M3D-Cap 2k-sample test split is R@1 = 19.10 (M3D-LaMed paper,
arXiv 2404.00578, Table 1).

Usage:
    python scripts/eval_3d_volume_retrieval.py \\
        --pairs path/to/pairs.csv \\
        --base-path path/to/volumes/

The CSV must have header ``volume,caption``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from multimeditron.model.modalities.volume.volume_modality import (
    _patch_monai_for_m3dclip,
)


def load_volume(path: Path, target=(32, 256, 256)) -> torch.Tensor:
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    t = torch.from_numpy(arr).unsqueeze(0)
    t = F.interpolate(t, size=target, mode="trilinear", align_corners=False)
    vmin, vmax = t.min(), t.max()
    return (t - vmin) / (vmax - vmin).clamp_min(1e-6)


def recall_at_k(sim: torch.Tensor, k: int) -> float:
    n = sim.size(0)
    topk = sim.topk(k, dim=1).indices
    targets = torch.arange(n).unsqueeze(1)
    return (topk == targets).any(dim=1).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--base-path", type=Path, default=Path("."))
    ap.add_argument("--clip-name", default="GoodBaiBai88/M3D-CLIP")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _patch_monai_for_m3dclip()
    tok = AutoTokenizer.from_pretrained(
        args.clip_name, model_max_length=512, padding_side="right", use_fast=False
    )
    model = AutoModel.from_pretrained(
        args.clip_name, trust_remote_code=True
    ).to(args.device).eval()

    img_feats, txt_feats = [], []
    with open(args.pairs) as f:
        rows = list(csv.DictReader(f))

    with torch.inference_mode():
        for row in rows:
            vol = load_volume(args.base_path / row["volume"]).to(args.device)
            img = model.encode_image(vol)[:, 0]
            txt_in = tok(
                row["caption"], max_length=512, truncation=True,
                padding="max_length", return_tensors="pt",
            ).to(args.device)
            txt = model.encode_text(txt_in["input_ids"], txt_in["attention_mask"])[:, 0]
            img_feats.append(F.normalize(img, dim=-1))
            txt_feats.append(F.normalize(txt, dim=-1))

    img = torch.cat(img_feats, dim=0)
    txt = torch.cat(txt_feats, dim=0)
    sim = img @ txt.T

    print(f"pairs:  {len(rows)}")
    for k in (1, 5, 10):
        print(f"R@{k:<2}:   {recall_at_k(sim, k) * 100:.2f}")


if __name__ == "__main__":
    main()
