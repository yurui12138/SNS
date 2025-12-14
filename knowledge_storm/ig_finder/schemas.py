from typing import Any, Dict, List, Literal
from pydantic import BaseModel


class ConsensusExtractionResult(BaseModel):
    field_development_history: str
    research_paradigms: List[Dict[str, Any]]
    mainstream_methods: List[Dict[str, Any]]
    knowledge_boundaries: List[Dict[str, Any]]
    key_concepts_hierarchy: Dict[str, Any]


class DeviationAnalysisResult(BaseModel):
    matched_baseline_concepts: List[str]
    deviation_description: str
    deviation_dimensions: List[str]
    deviation_score: float
    innovation_potential: Literal['high', 'medium', 'low']
    reasoning: str


class ClusterValidationResult(BaseModel):
    is_coherent_cluster: Literal['yes', 'no']
    cluster_name: str
    coherence_reasoning: str
    innovation_dimensions: List[str]
    cluster_summary: str
    potential_impact: str
