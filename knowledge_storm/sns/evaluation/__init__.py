"""
SNS Evaluation Framework

Provides comprehensive evaluation metrics and tools for assessing
self-nonself modeling quality, including:
- Time-slice taxonomy shift analysis
- Branch Hit@K metrics
- Human evaluation interface
"""

from .time_slice import TimeSliceDataset, TimeSliceEvaluator
from .metrics import (
    BranchHitAtK, 
    TaxonomyEditDistance, 
    compute_all_metrics,
    print_metrics_report,
)
from .human_eval import HumanEvaluationInterface, EvaluationDimension

__all__ = [
    'TimeSliceDataset',
    'TimeSliceEvaluator',
    'BranchHitAtK',
    'TaxonomyEditDistance',
    'compute_all_metrics',
    'print_metrics_report',
    'HumanEvaluationInterface',
    'EvaluationDimension',
]
