"""
IG-Finder: Innovation Gap Finder Framework

A framework for identifying verifiable innovation gaps in scientific knowledge
by modeling the immune system's self-nonself recognition mechanism.
"""

from .dataclass import (
    CognitiveBaseline,
    EvolutionState,
    InnovationCluster,
    InnovationGapReport,
    ReviewPaper,
    ResearchPaper,
    ResearchParadigm,
    DeviationAnalysis,
    GapAnalysis,
    TimeRange,
    Boundary,
    Method,
    Evidence,
)

from .engine import (
    IGFinderRunner,
    IGFinderLMConfigs,
    IGFinderArguments,
)

# IG-Finder 2.0 imports
from .engine_v2 import (
    IGFinder2Runner,
    IGFinder2Arguments,
)

from .dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    FitVector,
    FitLabel,
    FacetLabel,
    IGFinder2Results,
)

__all__ = [
    # Data classes (v1)
    "CognitiveBaseline",
    "EvolutionState",
    "InnovationCluster",
    "InnovationGapReport",
    "ReviewPaper",
    "ResearchPaper",
    "ResearchParadigm",
    "DeviationAnalysis",
    "GapAnalysis",
    "TimeRange",
    "Boundary",
    "Method",
    "Evidence",
    # Engine classes (v1)
    "IGFinderRunner",
    "IGFinderLMConfigs",
    "IGFinderArguments",
    # IG-Finder 2.0
    "IGFinder2Runner",
    "IGFinder2Arguments",
    "MultiViewBaseline",
    "TaxonomyView",
    "FitVector",
    "FitLabel",
    "FacetLabel",
    "IGFinder2Results",
]
