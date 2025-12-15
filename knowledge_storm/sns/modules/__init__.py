"""
SNS Framework Modules - Phase implementations.

Four-phase pipeline:
- Phase 1: Multi-view Baseline (Self Construction)
- Phase 2: Multi-view Stress Test (Nonself Identification)
- Phase 3: Stress Clustering & Evolution (Adaptation)
- Phase 4: Delta-aware Guidance Generation
"""

from .phase1_multiview_baseline import (
    Phase1Pipeline,
    ReviewRetriever,
    TaxonomyViewExtractor,
    NodeDefinitionBuilder,
    MultiViewBaselineBuilder,
)

from .phase2_stress_test import (
    Phase2Pipeline,
    PaperClaimExtractor,
    EmbeddingBasedRetriever,
    FitTester,
)

from .phase3_evolution import (
    Phase3Pipeline,
    StressClusterer,
    EvolutionPlanner,
)

from .phase4_guidance import (
    Phase4Pipeline,
    AxisSelector,
    GuidanceGenerator,
)

__all__ = [
    # Phase 1: Self Construction
    "Phase1Pipeline",
    "ReviewRetriever",
    "TaxonomyViewExtractor",
    "NodeDefinitionBuilder",
    "MultiViewBaselineBuilder",
    
    # Phase 2: Nonself Identification
    "Phase2Pipeline",
    "PaperClaimExtractor",
    "EmbeddingBasedRetriever",
    "FitTester",
    
    # Phase 3: Adaptation
    "Phase3Pipeline",
    "StressClusterer",
    "EvolutionPlanner",
    
    # Phase 4: Writing Guidance
    "Phase4Pipeline",
    "AxisSelector",
    "GuidanceGenerator",
]
