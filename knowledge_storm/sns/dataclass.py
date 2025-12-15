"""
Data classes for IG-Finder framework.

This module defines the core data structures used throughout the IG-Finder system,
including cognitive baseline, innovation clusters, and evolution states.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
from datetime import datetime

from ..interface import Information
from ..dataclass import KnowledgeNode


class EvolutionState(Enum):
    """
    Represents the evolution state of a knowledge node in the mind map.
    
    Based on the immune system's self-nonself recognition:
    - CONSENSUS: Established knowledge from review papers (the "self")
    - CONTINUATION: New research continuing the consensus direction
    - DEVIATION: Research deviating from consensus but isolated
    - INNOVATION: Clustered deviations with internal logical coherence (the "nonself")
    - POTENTIAL_GAP: Preliminary identified gap requiring further validation
    """
    CONSENSUS = "consensus"
    CONTINUATION = "continuation"
    DEVIATION = "deviation"
    INNOVATION = "innovation"
    POTENTIAL_GAP = "potential_gap"


@dataclass
class TimeRange:
    """Represents a time range for temporal coverage."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    
    def to_dict(self):
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            start=datetime.fromisoformat(data["start"]) if data.get("start") else None,
            end=datetime.fromisoformat(data["end"]) if data.get("end") else None,
        )


@dataclass
class ReviewPaper:
    """Represents a review/survey paper."""
    title: str
    authors: List[str]
    year: int
    url: str
    abstract: str
    citations: int = 0
    venue: str = ""
    key_contributions: List[str] = field(default_factory=list)
    extracted_consensus: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "url": self.url,
            "abstract": self.abstract,
            "citations": self.citations,
            "venue": self.venue,
            "key_contributions": self.key_contributions,
            "extracted_consensus": self.extracted_consensus,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class ResearchPaper:
    """Represents a research paper (non-review)."""
    title: str
    authors: List[str]
    year: int
    url: str
    abstract: str
    citations: int = 0
    venue: str = ""
    core_claims: List[str] = field(default_factory=list)
    methodology: str = ""
    key_findings: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "url": self.url,
            "abstract": self.abstract,
            "citations": self.citations,
            "venue": self.venue,
            "core_claims": self.core_claims,
            "methodology": self.methodology,
            "key_findings": self.key_findings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class ResearchParadigm:
    """Represents a research paradigm extracted from reviews."""
    name: str
    description: str
    representative_works: List[str] = field(default_factory=list)
    time_period: Optional[TimeRange] = None
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "representative_works": self.representative_works,
            "time_period": self.time_period.to_dict() if self.time_period else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            name=data["name"],
            description=data["description"],
            representative_works=data.get("representative_works", []),
            time_period=TimeRange.from_dict(data["time_period"]) if data.get("time_period") else None,
        )


@dataclass
class Method:
    """Represents a mainstream method."""
    name: str
    description: str
    category: str = ""
    advantages: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "advantages": self.advantages,
            "limitations": self.limitations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class Boundary:
    """Represents a knowledge boundary."""
    dimension: str  # e.g., "methodology", "data", "application domain"
    description: str
    known_limits: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "dimension": self.dimension,
            "description": self.description,
            "known_limits": self.known_limits,
            "open_questions": self.open_questions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class Evidence:
    """Represents supporting evidence for a claim."""
    paper: ResearchPaper
    claim: str
    supporting_text: str
    confidence_score: float = 1.0
    
    def to_dict(self):
        return {
            "paper": self.paper.to_dict(),
            "claim": self.claim,
            "supporting_text": self.supporting_text,
            "confidence_score": self.confidence_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            paper=ResearchPaper.from_dict(data["paper"]),
            claim=data["claim"],
            supporting_text=data["supporting_text"],
            confidence_score=data.get("confidence_score", 1.0),
        )


@dataclass
class DeviationAnalysis:
    """Analysis of how a paper/cluster deviates from consensus."""
    baseline_node_path: List[str]  # Path to the matched consensus node
    deviation_dimensions: List[str]  # e.g., ["methodology", "data paradigm"]
    deviation_description: str
    deviation_score: float  # 0-1, higher means more deviation
    expert_perspectives: Dict[str, str] = field(default_factory=dict)  # expert_name -> analysis
    
    def to_dict(self):
        return {
            "baseline_node_path": self.baseline_node_path,
            "deviation_dimensions": self.deviation_dimensions,
            "deviation_description": self.deviation_description,
            "deviation_score": self.deviation_score,
            "expert_perspectives": self.expert_perspectives,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class InnovationCluster:
    """
    Represents an innovation cluster - a group of papers that deviate from
    consensus but share internal logical coherence.
    """
    cluster_id: str
    name: str
    core_papers: List[ResearchPaper]
    deviation_from_consensus: DeviationAnalysis
    internal_coherence_score: float  # 0-1, measures logical consistency within cluster
    innovation_dimensions: List[str]  # e.g., ["novel methodology", "new data paradigm"]
    supporting_evidence: List[Evidence]
    knowledge_path: List[str]  # Path in the mind map
    cluster_summary: str = ""
    potential_impact: str = ""
    
    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "core_papers": [p.to_dict() for p in self.core_papers],
            "deviation_from_consensus": self.deviation_from_consensus.to_dict(),
            "internal_coherence_score": self.internal_coherence_score,
            "innovation_dimensions": self.innovation_dimensions,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "knowledge_path": self.knowledge_path,
            "cluster_summary": self.cluster_summary,
            "potential_impact": self.potential_impact,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            cluster_id=data["cluster_id"],
            name=data["name"],
            core_papers=[ResearchPaper.from_dict(p) for p in data["core_papers"]],
            deviation_from_consensus=DeviationAnalysis.from_dict(data["deviation_from_consensus"]),
            internal_coherence_score=data["internal_coherence_score"],
            innovation_dimensions=data["innovation_dimensions"],
            supporting_evidence=[Evidence.from_dict(e) for e in data["supporting_evidence"]],
            knowledge_path=data["knowledge_path"],
            cluster_summary=data.get("cluster_summary", ""),
            potential_impact=data.get("potential_impact", ""),
        )


@dataclass
class GapAnalysis:
    """Analysis of innovation gaps in a specific dimension."""
    dimension: str  # e.g., "Methodology", "Data Paradigm", "Application Domain"
    gap_description: str
    related_clusters: List[str]  # cluster_ids
    evidence_strength: float  # 0-1
    research_opportunities: List[str]
    
    def to_dict(self):
        return {
            "dimension": self.dimension,
            "gap_description": self.gap_description,
            "related_clusters": self.related_clusters,
            "evidence_strength": self.evidence_strength,
            "research_opportunities": self.research_opportunities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class CognitiveBaseline:
    """
    Represents the cognitive baseline extracted from review papers.
    This is the "self" in the immune system metaphor.
    """
    topic: str
    review_papers: List[ReviewPaper]
    consensus_map: KnowledgeBase  # The dynamic mind map with CONSENSUS nodes
    research_paradigms: List[ResearchParadigm]
    mainstream_methods: List[Method]
    knowledge_boundaries: Dict[str, Boundary]  # dimension -> boundary
    temporal_coverage: TimeRange
    field_evolution_timeline: List[Dict] = field(default_factory=list)  # [{year, milestone, description}]
    
    def to_dict(self):
        return {
            "topic": self.topic,
            "review_papers": [p.to_dict() for p in self.review_papers],
            "consensus_map": self.consensus_map.to_dict(),
            "research_paradigms": [p.to_dict() for p in self.research_paradigms],
            "mainstream_methods": [m.to_dict() for m in self.mainstream_methods],
            "knowledge_boundaries": {k: v.to_dict() for k, v in self.knowledge_boundaries.items()},
            "temporal_coverage": self.temporal_coverage.to_dict(),
            "field_evolution_timeline": self.field_evolution_timeline,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            topic=data["topic"],
            review_papers=[ReviewPaper.from_dict(p) for p in data["review_papers"]],
            consensus_map=KnowledgeBase.from_dict(data["consensus_map"]),
            research_paradigms=[ResearchParadigm.from_dict(p) for p in data["research_paradigms"]],
            mainstream_methods=[Method.from_dict(m) for m in data["mainstream_methods"]],
            knowledge_boundaries={k: Boundary.from_dict(v) for k, v in data["knowledge_boundaries"].items()},
            temporal_coverage=TimeRange.from_dict(data["temporal_coverage"]),
            field_evolution_timeline=data.get("field_evolution_timeline", []),
        )


@dataclass
class InnovationGapReport:
    """
    The final output report identifying innovation gaps.
    This replaces simple topic descriptions as input to downstream review systems.
    """
    topic: str
    generation_date: datetime
    cognitive_baseline_summary: str
    identified_clusters: List[InnovationCluster]
    gap_analysis_by_dimension: Dict[str, GapAnalysis]  # dimension -> analysis
    evolution_narrative: str  # Narrative describing knowledge evolution from consensus to innovation
    mind_map_visualization_data: Dict  # Data for interactive visualization
    recommendations_for_review: str  # Suggestions for downstream review generation
    statistics: Dict = field(default_factory=dict)  # Various statistics
    
    def to_dict(self):
        return {
            "topic": self.topic,
            "generation_date": self.generation_date.isoformat(),
            "cognitive_baseline_summary": self.cognitive_baseline_summary,
            "identified_clusters": [c.to_dict() for c in self.identified_clusters],
            "gap_analysis_by_dimension": {k: v.to_dict() for k, v in self.gap_analysis_by_dimension.items()},
            "evolution_narrative": self.evolution_narrative,
            "mind_map_visualization_data": self.mind_map_visualization_data,
            "recommendations_for_review": self.recommendations_for_review,
            "statistics": self.statistics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            topic=data["topic"],
            generation_date=datetime.fromisoformat(data["generation_date"]),
            cognitive_baseline_summary=data["cognitive_baseline_summary"],
            identified_clusters=[InnovationCluster.from_dict(c) for c in data["identified_clusters"]],
            gap_analysis_by_dimension={k: GapAnalysis.from_dict(v) for k, v in data["gap_analysis_by_dimension"].items()},
            evolution_narrative=data["evolution_narrative"],
            mind_map_visualization_data=data["mind_map_visualization_data"],
            recommendations_for_review=data["recommendations_for_review"],
            statistics=data.get("statistics", {}),
        )


class ExtendedKnowledgeNode(KnowledgeNode):
    """
    Extended KnowledgeNode with evolution state tracking for IG-Finder.
    """
    def __init__(
        self,
        name: str,
        content: Optional[str] = None,
        parent: Optional["ExtendedKnowledgeNode"] = None,
        children: Optional[List["ExtendedKnowledgeNode"]] = None,
        synthesize_output: Optional[str] = None,
        need_regenerate_synthesize_output: bool = True,
        evolution_state: EvolutionState = EvolutionState.CONSENSUS,
        deviation_metrics: Optional[Dict] = None,
        source_papers: Optional[List[str]] = None,  # URLs of source papers
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(
            name=name,
            content=content,
            parent=parent,
            children=children,
            synthesize_output=synthesize_output,
            need_regenerate_synthesize_output=need_regenerate_synthesize_output,
        )
        self.evolution_state = evolution_state
        self.deviation_metrics = deviation_metrics if deviation_metrics else {}
        self.source_papers = source_papers if source_papers else []
        self.timestamp = timestamp if timestamp else datetime.now()
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "evolution_state": self.evolution_state.value,
            "deviation_metrics": self.deviation_metrics,
            "source_papers": self.source_papers,
            "timestamp": self.timestamp.isoformat(),
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Construct ExtendedKnowledgeNode from dictionary."""
        def helper(cls, data, parent_node=None):
            if parent_node is not None:
                assert data["parent"] is not None and data["parent"] == parent_node.name
            
            node = cls(
                name=data["name"],
                content=data["content"],
                parent=parent_node,
                children=None,
                synthesize_output=data.get("synthesize_output", None),
                need_regenerate_synthesize_output=data.get("need_regenerate_synthesize_output", True),
                evolution_state=EvolutionState(data.get("evolution_state", "consensus")),
                deviation_metrics=data.get("deviation_metrics", {}),
                source_papers=data.get("source_papers", []),
                timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            )
            
            for child_data in data.get("children", []):
                child_node = helper(cls, child_data, parent_node=node)
                node.children.append(child_node)
            
            return node
        
        return helper(cls, data)


class KnowledgeBase:
    def __init__(self, topic: str):
        self.topic = topic
        self.root = ExtendedKnowledgeNode(name="root")

    def to_dict(self):
        return {
            "topic": self.topic,
            "tree": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict):
        kb = cls(topic=data.get("topic", ""))
        kb.root = ExtendedKnowledgeNode.from_dict(data["tree"]) if data.get("tree") else ExtendedKnowledgeNode(name="root")
        return kb
