"""
Modules for IG-Finder framework.
"""

from .cognitive_self_construction import (
    ReviewRetriever,
    ConsensusExtractor,
    CognitiveBaselineBuilder,
    CognitiveSelfConstructionModule,
)

from .innovative_nonself_identification import (
    FrontierPaperRetriever,
    ExpertPerspectiveGenerator,
    DifferenceAwareAnalyzer,
    InnovationClusterIdentifier,
    InnovativeNonSelfIdentificationModule,
)

from .mind_map_manager import (
    DynamicMindMapManager,
    EvolutionStateAnnotator,
)

from .report_generation import (
    InnovationGapReportGenerator,
)

__all__ = [
    # Cognitive Self Construction
    "ReviewRetriever",
    "ConsensusExtractor",
    "CognitiveBaselineBuilder",
    "CognitiveSelfConstructionModule",
    # Innovative Non-self Identification
    "FrontierPaperRetriever",
    "ExpertPerspectiveGenerator",
    "DifferenceAwareAnalyzer",
    "InnovationClusterIdentifier",
    "InnovativeNonSelfIdentificationModule",
    # Mind Map Management
    "DynamicMindMapManager",
    "EvolutionStateAnnotator",
    # Report Generation
    "InnovationGapReportGenerator",
]
