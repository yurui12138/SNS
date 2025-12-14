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

__all__ = [
    # Data classes
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
    # Engine classes
    "IGFinderRunner",
    "IGFinderLMConfigs",
    "IGFinderArguments",
]
