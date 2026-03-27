"""Shared structure for embedding-based classification benchmarks.

This package is meant for benchmarks following the same high-level pattern:
image -> embedding -> MLP -> metric.
"""

from .base import ClassificationBenchmark

__all__ = ["ClassificationBenchmark"]
