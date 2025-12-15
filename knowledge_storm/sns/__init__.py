"""
SNS: Self-Nonself Modeling for Automatic Survey Systems

A framework for mitigating cognitive lag in automatic survey generation
through self-nonself modeling, multi-view stress testing, and minimal
necessary structural evolution.

Core Innovation:
- Self: Existing consensus (multi-view taxonomy atlas)
- Nonself: New research that doesn't fit (stress test results)
- Adaptation: Minimal evolution + delta-aware guidance
"""

# Core IG-Finder 2.0 imports
from .engine_v2 import (
    SNSRunner,
    SNSArguments,
)

from .dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    TaxonomyTree,
    TaxonomyNode,
    NodeDefinition,
    FacetLabel,
    FitVector,
    FitLabel,
    FitReport,
    FitScores,
    StressCluster,
    ClusterType,
    EvolutionProposal,
    EvolutionOperation,
    AddNodeOperation,
    SplitNodeOperation,
    RenameNodeOperation,
    NewNodeProposal,
    DeltaAwareGuidance,
    Section,
    Subsection,
    EvidenceCard,
    EvolutionSummaryItem,
    SNSResults,
)

# Legacy data classes (shared between v1 and v2)
from .dataclass import (
    ReviewPaper,
    ResearchPaper,
    EvolutionState,
    TimeRange,
    Evidence,
)

# Evaluation framework
from .evaluation import (
    TimeSliceDataset,
    TimeSliceEvaluator,
    BranchHitAtK,
    TaxonomyEditDistance,
    compute_all_metrics,
    print_metrics_report,
    HumanEvaluationInterface,
    EvaluationDimension,
)

# Infrastructure
from .embeddings import (
    create_embedding_model,
    SPECTER2Embedding,
    SciNCLEmbedding,
    SentenceBERTEmbedding,
    compute_hybrid_similarity,
    compute_top_k_matches,
)

from .nli import (
    create_nli_model,
    NLIModel,
    NLILabel,
    compute_max_conflict_score,
)

__version__ = "2.0.0"

__all__ = [
    # Engine
    "SNSRunner",
    "SNSArguments",
    
    # Core data structures
    "MultiViewBaseline",
    "TaxonomyView",
    "TaxonomyTree",
    "TaxonomyNode",
    "NodeDefinition",
    "FacetLabel",
    "FitVector",
    "FitLabel",
    "FitReport",
    "FitScores",
    "StressCluster",
    "ClusterType",
    "EvolutionProposal",
    "EvolutionOperation",
    "AddNodeOperation",
    "SplitNodeOperation",
    "RenameNodeOperation",
    "NewNodeProposal",
    "DeltaAwareGuidance",
    "Section",
    "Subsection",
    "EvidenceCard",
    "EvolutionSummaryItem",
    "SNSResults",
    
    # Shared data classes
    "ReviewPaper",
    "ResearchPaper",
    "EvolutionState",
    "TimeRange",
    "Evidence",
    
    # Evaluation
    "TimeSliceDataset",
    "TimeSliceEvaluator",
    "BranchHitAtK",
    "TaxonomyEditDistance",
    "compute_all_metrics",
    "print_metrics_report",
    "HumanEvaluationInterface",
    "EvaluationDimension",
    
    # Infrastructure
    "create_embedding_model",
    "SPECTER2Embedding",
    "SciNCLEmbedding",
    "SentenceBERTEmbedding",
    "compute_hybrid_similarity",
    "compute_top_k_matches",
    "create_nli_model",
    "NLIModel",
    "NLILabel",
    "compute_max_conflict_score",
]
