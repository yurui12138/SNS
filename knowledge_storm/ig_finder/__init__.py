"""
IG-Finder 2.0: Multi-view Atlas Stress Test Framework

A framework for identifying structural taxonomy gaps through multi-view
stress testing and minimal necessary evolution planning.
"""

# Core IG-Finder 2.0 imports
from .engine_v2 import (
    IGFinder2Runner,
    IGFinder2Arguments,
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
    IGFinder2Results,
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
    "IGFinder2Runner",
    "IGFinder2Arguments",
    
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
    "IGFinder2Results",
    
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
