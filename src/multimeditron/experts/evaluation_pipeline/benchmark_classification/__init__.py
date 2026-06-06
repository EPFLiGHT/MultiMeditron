"""Shared structure for embedding-based classification benchmarks.

This package is meant for benchmarks following the same high-level pattern:
image -> embedding -> MLP -> metric.
"""

from .base import ClassificationBenchmark
from .ct3d_benchmark import CT3DBenchmark

__all__ = ["ClassificationBenchmark", "CT3DBenchmark"]
