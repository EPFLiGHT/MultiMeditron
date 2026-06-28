import re
from datasets import load_from_disk

ds = load_from_disk(
    "/lightscratch/datasets/MultiMediset/general_purpose/CT2D-glob-mini",
    keep_in_memory=False,
)
train = ds["train"]

ct_pattern = re.compile(
    r"\bct\b|ct scan|computed tomography|ct image|ct of|axial ct|coronal ct|ct slice|ct chest|ct abdomen",
    re.IGNORECASE,
)

ct_count = 0
total = len(train)
for i, ex in enumerate(train):
    if ct_pattern.search(ex["text"]):
        ct_count += 1
    if i % 100_000 == 0:
        print(f"{i}/{total} — CT so far: {ct_count}", flush=True)

print(f"Total: {total}")
print(f"CT scanner: {ct_count} ({100*ct_count/total:.1f}%)")
