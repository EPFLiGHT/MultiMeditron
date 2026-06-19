"""
Paraphrase the "text" field of a JSONL dataset into standardized diagnostic-style sentences.

This script reads an input JSON Lines file where each non-empty line is a JSON object that
may contain a "text" field (typically describing an image and/or label). For each record, it:

1) Extracts a concise label from "text" using extract_label():
   - Prefers preserving structured "key: value" phrases (e.g., "epiphora stage: normal")
   - Falls back to other heuristics (keyword-based patterns, generic key:value, last clause)

2) Rewrites `text` by inserting the extracted label into a randomly chosen natural-language
   template, producing paraphrased, report-like phrasing.

The purpose is to introduce controlled textual variety (paraphrasing) while retaining the
same underlying label/meaning, which can help reduce brittleness to fixed phrasing.

Example:
    python paraphrase_fixed_text.py \
        --in_jsonl data/input.jsonl \
        --out_jsonl data/output_paraphrased.jsonl \
        --seed 123
"""
import argparse
import json
import random
import re
from pathlib import Path
from typing import List

TEMPLATES: List[str] = [
    "{v}.",  # use as-is when v already contains "key: value"
    "Imaging assessment: {v}.",
    "This study is consistent with {v}.",
    "Findings indicate {v}.",
    "Clinical impression: {v}.",
    "Reported diagnosis: {v}.",
    "Label assigned: {v}.",
    "Overall interpretation: {v}.",
    "The image corresponds to {v}.",
    "Target condition: {v}.",
    "Final read: {v}.",
]
TEMPLATES_KEYVAL = [
    "{v}.",
    "Assessment: {v}.",
    "Clinical impression: {v}.",
    "Overall interpretation: {v}.",
    "Findings are consistent with {v}.",
    "This study supports {v}.",
    "The image corresponds to {v}.",
    "Reported diagnosis — {v}.",
    "Label assigned → {v}.",
    "Final read: {v}.",
]


def extract_label(text: str) -> str:
    """Extracts a concise label, preserving 'key: value' (e.g., 'epiphora stage: normal')."""
    if not text:
        return "Unknown"

    # Remove placeholder tokens but keep spacing sane
    t = re.sub(r"\s*<attachment>\s*", "", text, flags=re.I).strip()

    # 1) Prefer phrases like: "with epiphora stage: normal"
    m = re.search(r"\bwith\s+([^.;:]+?\s*:\s*[^.;]+)", t, flags=re.I)
    if m:
        return m.group(1).strip()

    # 2) Generic 'key: value' after typical medical/label keywords
    m = re.search(
        r"(labels?|label|stage|diagnosis|dx|impression|finding|condition)\s*:\s*([^.;]+)",
        t, flags=re.I
    )
    if m:
        key = m.group(1).strip()
        val = m.group(2).strip()
        # Use the key as written in text (preserve case as much as possible)
        return f"{key}: {val}"

    # 3) Any other 'key: value' anywhere
    m = re.search(r"([A-Za-z][\w \-\/]{1,60}?)\s*:\s*([^.;]+)", t)
    if m:
        return f"{m.group(1).strip()}: {m.group(2).strip()}"

    # 4) Fallback: take the last meaningful clause
    parts = [p.strip() for p in re.split(r"[.;]", t) if p.strip()]
    for part in reversed(parts):
        if len(part.split()) > 1:
            return part
    return t or "Unknown"


def main():
    ap = argparse.ArgumentParser(description="Paraphrase text fields in existing JSONL dataset.")
    ap.add_argument("--in_jsonl", type=Path, required=True, help="Input JSONL file")
    ap.add_argument("--out_jsonl", type=Path, required=True, help="Output paraphrased JSONL file")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for paraphrasing variety")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total, written = 0, 0
    with args.in_jsonl.open("r", encoding="utf-8") as fin, \
         args.out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            orig_text = obj.get("text", "")
            label = extract_label(orig_text).strip()

            # Normalize trailing period on label (we'll add one via templates)
            label = label[:-1] if label.endswith(".") else label

            # If label looks like key:value (esp. epiphora stage: ...), use key:value-safe templates
            if ":" in label:
                template = rng.choice(TEMPLATES_KEYVAL)
                new_text = template.format(v=label)
            else:
                # Fallback: use your original free-form templates for plain labels like "normal"
                template = rng.choice(TEMPLATES[1:])  # skip the bare "{v}." if you like
                new_text = template.format(v=label)

            # Safety: collapse accidental double spaces and ensure terminal period
            new_text = re.sub(r"\s{2,}", " ", new_text).strip()
            if not new_text.endswith("."):
                new_text += "."

            new_obj = dict(obj)
            new_obj["text"] = new_text
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"[DONE] Processed {total} lines, wrote {written} → {args.out_jsonl}")

if __name__ == "__main__":
    main()
