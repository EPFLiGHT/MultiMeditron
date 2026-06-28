import argparse, json, re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

SPLITS = {
    "train": ("Training", "RFMiD_2_Training_labels.csv"),
    "val":   ("Validation", "RFMiD_2_Validation_labels.csv"),
    "test":  ("Test", "RFMiD_2_Testing_labels.csv"),
}

# Fix occasional header glitches
HEADER_FIXES = {
    "ON�": "ON",     # garbled ON
    "OPDM": "ODPM",  # typo in some files
}

# Map dataset abbreviations to their full label names (wrt the convention provided in their paper)
LABEL_RENAMES = {
    "WNL":  "Within normal limits",
    "AH":   "Asteroid hyalosis",
    "AION": "Anterior ischemic optic neuropathy",
    "ARMD": "Age-related macular degeneration",
    "BRVO": "Branch retinal vein occlusion",
    "CB":   "Coloboma",
    "CF":   "Choroidal folds",
    "CL":   "Collateral vessels",
    "CME":  "Cystoid macular edema",
    "CNV":  "Choroidal neovascularization",
    "CRAO": "Central retinal artery occlusion",
    "CRS":  "Chorioretinitis",
    "CRVO": "Central retinal vein occlusion",
    "CSR":  "Central serous retinopathy",
    "CWS":  "Cotton wool spots",
    "CSC":  "Cysticercosis",
    "DN":   "Drusen",
    "DR":   "Diabetic retinopathy",
    "EDN":  "Edema (unspecified)",     # if your CSV’s EDN means something specific, swap here
    "ERM":  "Epiretinal membrane",
    "GRT":  "Giant retinal tear",
    "HPED": "Hemorrhagic pigment epithelial detachment",
    "HR":   "Hemorrhagic retinopathy",
    "LS":   "Laser scar",
    "MCA":  "Microaneurysm",
    "ME":   "Macular edema",
    "MH":   "Media haze",
    "MHL":  "Macular hole",
    "MS":   "Macular scar",
    "MYA":  "Myopia",
    "ODC":  "Optic disc cupping",
    "ODE":  "Optic disc edema",
    "ODP":  "Optic disc pallor",
    "ON":   "Optic neuritis",
    "ODPM": "Optic disc pit maculopathy",
    "PRH":  "Preretinal hemorrhage",
    "RD":   "Retinal detachment",
    "RHL":  "Retinal holes",
    "RTR":  "Retinal tear",
    "RP":   "Retinitis pigmentosa",
    "RPEC": "Retinal pigment epithelium changes",
    "RS":   "Retinitis",
    "RT":   "Retinal traction",
    "SOFE": "Silicone oil–filled eye",
    "ST":   "Optociliary shunt",
    "TD":   "Tilted disc",
    "TSLN": "Tessellation",
    "TV":   "Tortuous vessels",
    "VS":   "Vasculitis",
    "HTN":  "Hypertensive retinopathy",
    "IIH":  "Idiopathic intracranial hypertension",
}

def tidy_fallback(name: str) -> str:
    # Fallback if a column isn’t in LABEL_RENAMES
    s = name.strip().replace("_", " ")
    # common uppercase abbreviations to lower-case words (e.g., 'EDN' -> 'Edn')
    s = re.sub(r"\s+", " ", s)
    return s.title()

def full_label_name(col: str) -> str:
    key = col.strip()
    return LABEL_RENAMES.get(key, tidy_fallback(key))

def load_labels_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # Drop unnamed/empty trailing columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    # Fix header glitches
    df.rename(columns={k: v for k, v in HEADER_FIXES.items() if k in df.columns}, inplace=True)

    if "ID" not in df.columns:
        raise ValueError(f"'ID' column not found in {path}")
    df["ID"] = df["ID"].astype(str).str.strip()
    return df

def find_image_by_id(split_dir: Path, id_str: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = split_dir / f"{id_str}{ext}"
        if p.exists():
            return p
    for p in split_dir.rglob("*"):
        if p.is_file() and p.stem == id_str and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return p
    return None

def positive_labels(row: pd.Series, label_cols: List[str]) -> List[str]:
    pos = []
    for c in label_cols:
        v = row[c]
        if pd.isna(v):
            continue
        is_pos = False
        if isinstance(v, (int, float)):
            is_pos = float(v) > 0.5
        else:
            is_pos = str(v).strip().lower() in {"1", "true", "yes", "y"}
        if is_pos:
            pos.append(full_label_name(c))
    return pos

def build_text(img_name: str, labels: List[str]) -> str:
    parts = [f"RFMiD2 image {img_name}."]
    parts.append("Positive labels: " + (", ".join(labels) if labels else "none") + ".")
    parts.append("<attachment>")
    return " ".join(parts)

def process_split(base_dir: Path, split_key: str, out_prefix: Path) -> int:
    split_dir_name, csv_name = SPLITS[split_key]
    split_dir = base_dir / split_dir_name
    csv_path  = split_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels CSV: {csv_path}")

    df = load_labels_csv(csv_path)
    # treat every column except ID as a label flag
    label_cols = [c for c in df.columns if c != "ID"]

    out_path = out_prefix.with_name(out_prefix.name + f"_{split_key}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_key}"):
            id_str = row["ID"]
            img_path = find_image_by_id(split_dir, id_str)
            if img_path is None:
                continue
            labels = positive_labels(row, label_cols)
            record = {
                "text": build_text(img_path.name, labels),
                "modalities": [{"type": "image", "value": str(img_path)}],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} → {out_path}")
    return written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/mloscratch/users/turan/datasets/rfmid2",
                    help="Folder with Training/, Validation/, Test/ and their CSVs.")
    ap.add_argument("--out_prefix", default="/mloscratch/users/turan/datasets/rfmid2/rfmid2_manifest",
                    help="Prefix for output JSONL files (suffix _train/_val/_test added).")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_prefix = Path(args.out_prefix)

    total = 0
    for split_key in ["train", "val", "test"]:
        total += process_split(base_dir, split_key, out_prefix)
    print(f"Done. Total JSONL lines: {total}")

if __name__ == "__main__":
    main()
