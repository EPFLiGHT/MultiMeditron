"""
Statistical significance helpers for the MultiMeditron evaluation analysis.

Two tests:

1. Wilson 95% confidence interval for a single accuracy (needs only n and the
   accuracy/proportion). Non-overlapping CIs between two models are sufficient (but
   conservative) evidence of a real difference.

2. McNemar's exact test for a *paired* comparison of two models on the same items
   (the correct test for "did model B change accuracy vs model A on these examples").
   This needs the per-item correctness of both models, which lmms-eval writes to its
   per-sample result JSONL — pass the two files with --preds-a/--preds-b.

Examples
--------
    # CIs straight from reported numbers:
    python scripts/eval_significance.py ci --acc 0.471 --n 3362

    # Paired McNemar from two lmms-eval per-sample result files:
    python scripts/eval_significance.py mcnemar \
        --preds-a results/5exp/pathvqa/samples.jsonl \
        --preds-b results/7exp/pathvqa/samples.jsonl
"""

import argparse
import json
import math
from typing import Dict, List, Tuple

Z95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval


def wilson_ci(p: float, n: int, z: float = Z95) -> Tuple[float, float]:
    """Return the (low, high) Wilson score interval for proportion p over n trials."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_correct(path: str) -> Dict[str, bool]:
    """Load {item_id: is_correct} from an lmms-eval per-sample JSONL.

    Tolerant to common field spellings; raises if it can't find an id or a
    correctness/score field so failures are loud, not silent.
    """
    out: Dict[str, bool] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = row.get("doc_id", row.get("id", row.get("question_id")))
            score = row.get("exact_match", row.get("correct", row.get("acc", row.get("score"))))
            if item_id is None or score is None:
                raise ValueError(f"{path}: row missing an id or score field: {list(row)[:8]}")
            out[str(item_id)] = bool(round(float(score)))
    return out


def mcnemar(correct_a: Dict[str, bool], correct_b: Dict[str, bool]) -> Dict[str, float]:
    """McNemar's exact (binomial) test on the paired discordant counts.

    b = A correct & B wrong; c = A wrong & B correct. Returns b, c, the two-sided
    exact p-value, and each model's accuracy on the shared items.
    """
    shared = sorted(set(correct_a) & set(correct_b))
    if not shared:
        raise ValueError("no shared item ids between the two prediction files")
    b = sum(1 for k in shared if correct_a[k] and not correct_b[k])
    c = sum(1 for k in shared if not correct_a[k] and correct_b[k])
    n = b + c
    # Exact two-sided binomial p-value under H0: P(swap)=0.5.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) if n else 1.0
    p_value = min(1.0, 2 * tail)
    return {
        "n_shared": len(shared),
        "b_A_only_correct": b,
        "c_B_only_correct": c,
        "acc_a": sum(correct_a[k] for k in shared) / len(shared),
        "acc_b": sum(correct_b[k] for k in shared) / len(shared),
        "p_value": p_value,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    ci = sub.add_parser("ci", help="Wilson 95%% CI for one accuracy")
    ci.add_argument("--acc", type=float, required=True, help="accuracy/proportion in [0,1]")
    ci.add_argument("--n", type=int, required=True, help="number of items")

    mc = sub.add_parser("mcnemar", help="paired McNemar test from two prediction files")
    mc.add_argument("--preds-a", required=True)
    mc.add_argument("--preds-b", required=True)

    args = ap.parse_args()
    if args.cmd is None:
        ap.error("choose a subcommand: 'ci' or 'mcnemar'")
    if args.cmd == "ci":
        lo, hi = wilson_ci(args.acc, args.n)
        print(f"{args.acc*100:.1f}%  95% CI [{lo*100:.1f}, {hi*100:.1f}]  (n={args.n})")
    else:
        res = mcnemar(_load_correct(args.preds_a), _load_correct(args.preds_b))
        print(json.dumps(res, indent=2))
        verdict = "significant" if res["p_value"] < 0.05 else "not significant"
        print(f"\nMcNemar two-sided p = {res['p_value']:.4g} -> {verdict} at alpha=0.05")


if __name__ == "__main__":
    main()
