"""
Structural validator for the training config matrix (cookbook/sft/**.yaml).

Catches the drift that's easy to introduce across ~30 near-identical configs:
missing required keys, an empty/!moe expert list, a missing gating path, or a
deepspeed file that doesn't exist. It is a *structural* check (keys and obvious
consistency), not a full JSON-schema — fast enough to run in CI or pre-push.

Exit code is non-zero if any config fails, so it can gate a commit.

Usage:
    python scripts/validate_configs.py                       # all cookbook/sft configs
    python scripts/validate_configs.py cookbook/sft/moe/attn/pep/*.yaml
"""

import glob
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level keys every training config must define.
REQUIRED_TOP = ["base_llm", "tokenizer_type", "attachment_token", "modalities", "training_args"]
# Keys required inside a MoE modality block.
REQUIRED_MOE = ["expert_clip_names", "gating_path", "fusion_method"]


def _resolve(path_str: str) -> str:
    """Expand ${REPO_DIR}/${USER} the way the train CLI does, for existence checks."""
    os.environ.setdefault("REPO_DIR", str(REPO_ROOT))
    return os.path.expandvars(path_str)


def validate_file(path: str) -> list:
    """Return a list of human-readable problems for one config (empty == valid)."""
    problems = []
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return [f"YAML parse error: {e}"]

    if not isinstance(cfg, dict):
        return ["top level is not a mapping"]

    for key in REQUIRED_TOP:
        if key not in cfg:
            problems.append(f"missing top-level key: {key}")

    # pack_sequences requires max_sequence_length (mirrors the collator's own check)
    if cfg.get("pack_sequences") and not cfg.get("max_sequence_length"):
        problems.append("pack_sequences=true requires max_sequence_length")

    for mod in cfg.get("modalities", []) or []:
        mtype = mod.get("model_type", "")
        if "moe" in mtype:
            for key in REQUIRED_MOE:
                if not mod.get(key):
                    problems.append(f"MoE modality missing/empty: {key}")
            experts = mod.get("expert_clip_names") or []
            gidx = mod.get("generalist_idx")
            if isinstance(gidx, int) and experts and not (-len(experts) <= gidx < len(experts)):
                problems.append(f"generalist_idx {gidx} out of range for {len(experts)} experts")

    # deepspeed config (if referenced) should resolve to a real file
    ds = cfg.get("training_args", {}).get("deepspeed") if isinstance(cfg.get("training_args"), dict) else None
    if ds:
        resolved = _resolve(ds)
        if resolved.startswith("/") and "$" not in resolved and not os.path.isfile(resolved):
            problems.append(f"deepspeed file not found: {resolved}")

    return problems


def main():
    args = sys.argv[1:]
    files = args or sorted(glob.glob(str(REPO_ROOT / "cookbook/sft/**/*.yaml"), recursive=True))
    if not files:
        print("no config files found")
        return 1

    n_bad = 0
    for path in files:
        problems = validate_file(path)
        rel = os.path.relpath(path, REPO_ROOT)
        if problems:
            n_bad += 1
            print(f"✗ {rel}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"✓ {rel}")

    print(f"\n{len(files) - n_bad}/{len(files)} configs valid"
          + (f"; {n_bad} FAILED" if n_bad else ""))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
