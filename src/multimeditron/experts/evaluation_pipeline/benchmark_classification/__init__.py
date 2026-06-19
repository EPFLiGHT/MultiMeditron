"""Shared structure for embedding-based classification benchmarks.

This package is meant for benchmarks following the same high-level pattern:
image -> embedding -> MLP -> metric.
"""

import os
import sys

# Benchmark modules use script-style absolute imports (e.g. `from load_from_clip import encode_img`)
# that require the evaluation_pipeline directory to be on sys.path.
_EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from .base import ClassificationBenchmark
from .mri_benchmark import MRIBenchmark
from .ct_benchmark import CTBenchmark
from .histopathology_benchmark import HistopathologyBenchmark
from .ophthalmology_benchmark import OphthalmologyBenchmark
from .skin_benchmark import SkinBenchmark
from .ultrasound_benchmark import UltrasoundBenchmark
from .xray_benchmark import XRay_benchmark

__all__ = [
    "ClassificationBenchmark",
    "MRIBenchmark",
    "CTBenchmark",
    "HistopathologyBenchmark",
    "OphthalmologyBenchmark",
    "SkinBenchmark",
    "UltrasoundBenchmark",
    "XRay_benchmark",
]
