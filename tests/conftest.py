import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "multimeditron" / "experts"

sys.path.insert(0, str(SRC_ROOT))
