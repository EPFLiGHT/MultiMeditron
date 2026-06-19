"""
Diagnostic analysis of lexical overlap between ground-truth captions and randomly
sampled negative captions in a medical image-text validation dataset.

Uses token-level Jaccard similarity to quantify how similar positive captions are
to negatives drawn for forced-choice retrieval evaluation (4-way Recall@1).
Reported statistics (mean, median, p95, max) characterize the difficulty of the
negative pool. Intended for evaluation sanity checks, not as a standalone benchmark.
"""

import argparse
import json
import random
import re
import numpy as np


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def jaccard(a, b):
    ta = set(norm(a).split())
    tb = set(norm(b).split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    parser = argparse.ArgumentParser(
        description="Measure lexical overlap between positive captions and random negatives."
    )
    parser.add_argument("--dataset", required=True, help="Path to a JSONL eval dataset.")
    parser.add_argument("--line-number", type=int, default=1000, help="Max lines to read.")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = f.readlines()

    N = min(args.line_number, len(lines))
    lines = lines[:N]

    sims = []
    for i in range(N):
        pos = json.loads(lines[i])["text"]
        candidates = list(range(N))
        candidates.remove(i)
        a, b, c = random.sample(candidates, 3)
        for j in [a, b, c]:
            neg = json.loads(lines[j])["text"]
            sims.append(jaccard(pos, neg))

    arr = np.array(sims)
    print("pairs:", len(arr))
    print("mean:", arr.mean())
    print("median:", np.median(arr))
    print("p95:", np.quantile(arr, 0.95))
    print("max:", arr.max())


if __name__ == "__main__":
    main()
