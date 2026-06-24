"""Shared structure for embedding-based classification benchmarks.

This package is meant for benchmarks following the same high-level pattern:
image -> embedding -> MLP -> metric.
"""

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
