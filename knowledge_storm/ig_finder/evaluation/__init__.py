"""
IG-Finder 2.0 Evaluation Framework

Provides comprehensive evaluation metrics and tools for assessing
the quality of multi-view taxonomy construction and evolution proposals.
"""

from .time_slice import TimeSliceDataset, TimeSliceEvaluator
from .metrics import BranchHitAtK, TaxonomyEditDistance, compute_all_metrics
from .human_eval import HumanEvaluationInterface, EvaluationDimension

__all__ = [
    'TimeSliceDataset',
    'TimeSliceEvaluator',
    'BranchHitAtK',
    'TaxonomyEditDistance',
    'compute_all_metrics',
    'HumanEvaluationInterface',
    'EvaluationDimension',
]
