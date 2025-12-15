"""
Data classes for IG-Finder 2.0 framework.

This module defines the new data structures for the multi-view taxonomy atlas,
stress test, and minimal evolution components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .dataclass import ResearchPaper, Evidence


# ============================================================================
# Phase 1: Multi-view Baseline Data Structures
# ============================================================================

class FacetLabel(Enum):
    """
    Facet labels for taxonomy views.
    Represents different organizational dimensions used in review papers.
    """
    MODEL_ARCHITECTURE = "MODEL_ARCHITECTURE"
    TRAINING_PARADIGM = "TRAINING_PARADIGM"
    TASK_SETTING = "TASK_SETTING"
    THEORY = "THEORY"
    EVALUATION_PROTOCOL = "EVALUATION_PROTOCOL"
    APPLICATION_DOMAIN = "APPLICATION_DOMAIN"
    DATA_PARADIGM = "DATA_PARADIGM"
    OTHER = "OTHER"


@dataclass
class EvidenceSpan:
    """
    Represents a span of evidence from source text.
    Used to anchor all claims to original text.
    """
    claim: str  # The claim being made
    page: int  # Page number (if available)
    section: str  # Section name (e.g., "Introduction", "Related Work")
    char_start: int  # Character offset start
    char_end: int  # Character offset end
    quote: str  # The actual quoted text
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "page": self.page,
            "section": self.section,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "quote": self.quote,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> EvidenceSpan:
        return cls(**data)


@dataclass
class NodeDefinition:
    """
    Testable definition for a taxonomy node.
    Contains all criteria needed for fit testing.
    """
    node_path: str  # e.g., "ROOT/CNN-based/ResNet"
    definition: str  # Clear definition of what belongs in this node
    inclusion_criteria: List[str]  # What papers SHOULD be classified here
    exclusion_criteria: List[str]  # What papers should NOT be classified here
    canonical_keywords: List[str]  # Key terms associated with this node
    boundary_statements: List[str]  # Edge cases and limitations
    evidence_spans: List[EvidenceSpan]  # Evidence from original review
    
    def to_dict(self) -> Dict:
        return {
            "node_path": self.node_path,
            "definition": self.definition,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "canonical_keywords": self.canonical_keywords,
            "boundary_statements": self.boundary_statements,
            "evidence_spans": [e.to_dict() for e in self.evidence_spans],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> NodeDefinition:
        return cls(
            node_path=data["node_path"],
            definition=data["definition"],
            inclusion_criteria=data["inclusion_criteria"],
            exclusion_criteria=data["exclusion_criteria"],
            canonical_keywords=data["canonical_keywords"],
            boundary_statements=data["boundary_statements"],
            evidence_spans=[EvidenceSpan.from_dict(e) for e in data.get("evidence_spans", [])],
        )


@dataclass
class TaxonomyTreeNode:
    """
    A node in the taxonomy tree.
    """
    name: str
    path: str  # Full path from root (e.g., "ROOT/CNN-based/ResNet")
    parent: Optional[str] = None  # Parent node path
    children: List[str] = field(default_factory=list)  # Child node paths
    is_leaf: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "path": self.path,
            "parent": self.parent,
            "children": self.children,
            "is_leaf": self.is_leaf,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> TaxonomyTreeNode:
        return cls(**data)


@dataclass
class TaxonomyTree:
    """
    Hierarchical tree structure extracted from a review.
    """
    root: TaxonomyTreeNode
    nodes: Dict[str, TaxonomyTreeNode] = field(default_factory=dict)  # path -> node
    
    def __post_init__(self):
        if not self.nodes:
            self.nodes = {self.root.path: self.root}
    
    def add_node(self, node: TaxonomyTreeNode):
        """Add a node to the tree."""
        self.nodes[node.path] = node
        if node.parent and node.parent in self.nodes:
            parent_node = self.nodes[node.parent]
            if node.path not in parent_node.children:
                parent_node.children.append(node.path)
    
    def get_leaf_nodes(self) -> List[TaxonomyTreeNode]:
        """Return all leaf nodes."""
        return [node for node in self.nodes.values() if node.is_leaf]
    
    def get_internal_nodes(self) -> List[TaxonomyTreeNode]:
        """Return all internal (non-leaf) nodes."""
        return [node for node in self.nodes.values() if not node.is_leaf]
    
    def to_dict(self) -> Dict:
        return {
            "root": self.root.to_dict(),
            "nodes": {path: node.to_dict() for path, node in self.nodes.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> TaxonomyTree:
        root = TaxonomyTreeNode.from_dict(data["root"])
        tree = cls(root=root)
        tree.nodes = {path: TaxonomyTreeNode.from_dict(node_data) 
                      for path, node_data in data["nodes"].items()}
        return tree


@dataclass
class TaxonomyView:
    """
    A single view (perspective) extracted from one review paper.
    """
    view_id: str  # Unique identifier (e.g., "T1", "T2")
    review_id: str  # Source review paper ID
    review_title: str  # Source review paper title
    facet_label: FacetLabel  # Organizational dimension
    facet_rationale: str  # Explanation of why this facet was chosen
    tree: TaxonomyTree  # Hierarchical structure
    node_definitions: Dict[str, NodeDefinition] = field(default_factory=dict)  # node_path -> definition
    weight: float = 1.0  # View weight (calculated based on quality metrics)
    evidence: List[EvidenceSpan] = field(default_factory=list)  # Overall evidence for this view
    
    def to_dict(self) -> Dict:
        return {
            "view_id": self.view_id,
            "review_id": self.review_id,
            "review_title": self.review_title,
            "facet_label": self.facet_label.value,
            "facet_rationale": self.facet_rationale,
            "tree": self.tree.to_dict(),
            "node_definitions": {path: node_def.to_dict() 
                               for path, node_def in self.node_definitions.items()},
            "weight": self.weight,
            "evidence": [e.to_dict() for e in self.evidence],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> TaxonomyView:
        return cls(
            view_id=data["view_id"],
            review_id=data["review_id"],
            review_title=data["review_title"],
            facet_label=FacetLabel(data["facet_label"]),
            facet_rationale=data["facet_rationale"],
            tree=TaxonomyTree.from_dict(data["tree"]),
            node_definitions={path: NodeDefinition.from_dict(node_def) 
                            for path, node_def in data.get("node_definitions", {}).items()},
            weight=data.get("weight", 1.0),
            evidence=[EvidenceSpan.from_dict(e) for e in data.get("evidence", [])],
        )


@dataclass
class MultiViewBaseline:
    """
    Multi-view cognitive baseline: collection of taxonomy views from multiple reviews.
    """
    topic: str
    views: List[TaxonomyView]
    creation_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Normalize weights
        total_weight = sum(v.weight for v in self.views)
        if total_weight > 0:
            for v in self.views:
                v.weight = v.weight / total_weight
    
    def get_view_by_id(self, view_id: str) -> Optional[TaxonomyView]:
        """Get view by ID."""
        for view in self.views:
            if view.view_id == view_id:
                return view
        return None
    
    def get_views_by_facet(self, facet: FacetLabel) -> List[TaxonomyView]:
        """Get all views with a specific facet."""
        return [v for v in self.views if v.facet_label == facet]
    
    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "views": [v.to_dict() for v in self.views],
            "creation_date": self.creation_date.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> MultiViewBaseline:
        return cls(
            topic=data["topic"],
            views=[TaxonomyView.from_dict(v) for v in data["views"]],
            creation_date=datetime.fromisoformat(data.get("creation_date", datetime.now().isoformat())),
        )


# ============================================================================
# Phase 2: Stress Test Data Structures
# ============================================================================

@dataclass
class PaperClaim:
    """
    A claim extracted from a research paper with evidence.
    """
    text: str
    evidence: List[EvidenceSpan]
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "evidence": [e.to_dict() for e in self.evidence],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> PaperClaim:
        return cls(
            text=data["text"],
            evidence=[EvidenceSpan.from_dict(e) for e in data.get("evidence", [])],
        )


@dataclass
class PaperClaims:
    """
    Structured claims extracted from a research paper.
    """
    paper_id: str
    problem: Optional[PaperClaim] = None
    core_idea: List[PaperClaim] = field(default_factory=list)
    mechanism: List[PaperClaim] = field(default_factory=list)
    training: List[PaperClaim] = field(default_factory=list)
    evaluation: List[PaperClaim] = field(default_factory=list)
    novelty_bullets: List[PaperClaim] = field(default_factory=list)  # Must have exactly 3
    keywords: List[str] = field(default_factory=list)
    tasks_datasets: List[str] = field(default_factory=list)
    methods_components: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "problem": self.problem.to_dict() if self.problem else None,
            "core_idea": [c.to_dict() for c in self.core_idea],
            "mechanism": [c.to_dict() for c in self.mechanism],
            "training": [c.to_dict() for c in self.training],
            "evaluation": [c.to_dict() for c in self.evaluation],
            "novelty_bullets": [c.to_dict() for c in self.novelty_bullets],
            "keywords": self.keywords,
            "tasks_datasets": self.tasks_datasets,
            "methods_components": self.methods_components,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> PaperClaims:
        return cls(
            paper_id=data["paper_id"],
            problem=PaperClaim.from_dict(data["problem"]) if data.get("problem") else None,
            core_idea=[PaperClaim.from_dict(c) for c in data.get("core_idea", [])],
            mechanism=[PaperClaim.from_dict(c) for c in data.get("mechanism", [])],
            training=[PaperClaim.from_dict(c) for c in data.get("training", [])],
            evaluation=[PaperClaim.from_dict(c) for c in data.get("evaluation", [])],
            novelty_bullets=[PaperClaim.from_dict(c) for c in data.get("novelty_bullets", [])],
            keywords=data.get("keywords", []),
            tasks_datasets=data.get("tasks_datasets", []),
            methods_components=data.get("methods_components", []),
        )


class FitLabel(Enum):
    """
    Labels for fit test results.
    """
    FIT = "FIT"  # Paper fits well into the node
    FORCE_FIT = "FORCE_FIT"  # Paper can be placed but loses key contributions
    UNFITTABLE = "UNFITTABLE"  # Paper cannot be reasonably classified


@dataclass
class LostNovelty:
    """
    Represents a novelty contribution that cannot be captured by the node.
    """
    bullet: str  # The novelty bullet text
    evidence: List[EvidenceSpan]
    similarity_to_leaf: float  # Cosine similarity to best leaf (low = lost)
    
    def to_dict(self) -> Dict:
        return {
            "bullet": self.bullet,
            "evidence": [e.to_dict() for e in self.evidence],
            "similarity_to_leaf": self.similarity_to_leaf,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> LostNovelty:
        return cls(
            bullet=data["bullet"],
            evidence=[EvidenceSpan.from_dict(e) for e in data.get("evidence", [])],
            similarity_to_leaf=data["similarity_to_leaf"],
        )


@dataclass
class ConflictEvidence:
    """
    Evidence of conflict between paper and node boundaries.
    """
    boundary: str  # The boundary/exclusion statement from node definition
    nli_contradiction: float  # NLI contradiction probability
    paper_claim: str  # The conflicting claim from paper
    
    def to_dict(self) -> Dict:
        return {
            "boundary": self.boundary,
            "nli_contradiction": self.nli_contradiction,
            "paper_claim": self.paper_claim,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> ConflictEvidence:
        return cls(**data)


@dataclass
class FitScores:
    """
    Detailed scores from fit test.
    """
    coverage: float  # 0-1, semantic + lexical coverage
    conflict: float  # 0-1, boundary violation score
    residual: float  # 0-1, contribution loss score
    fit_score: float  # Combined score: coverage - 0.8*conflict - 0.4*residual
    
    def to_dict(self) -> Dict:
        return {
            "coverage": self.coverage,
            "conflict": self.conflict,
            "residual": self.residual,
            "fit_score": self.fit_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> FitScores:
        return cls(**data)


@dataclass
class FitReport:
    """
    Complete fit test report for one paper against one view.
    """
    paper_id: str
    view_id: str
    facet_label: FacetLabel
    best_leaf_path: Optional[str]  # Path to best matching leaf node
    label: FitLabel  # FIT / FORCE_FIT / UNFITTABLE
    scores: FitScores
    lost_novelty: List[LostNovelty] = field(default_factory=list)
    conflict_evidence: List[ConflictEvidence] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "view_id": self.view_id,
            "facet_label": self.facet_label.value,
            "best_leaf_path": self.best_leaf_path,
            "label": self.label.value,
            "scores": self.scores.to_dict(),
            "lost_novelty": [ln.to_dict() for ln in self.lost_novelty],
            "conflict_evidence": [ce.to_dict() for ce in self.conflict_evidence],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> FitReport:
        return cls(
            paper_id=data["paper_id"],
            view_id=data["view_id"],
            facet_label=FacetLabel(data["facet_label"]),
            best_leaf_path=data.get("best_leaf_path"),
            label=FitLabel(data["label"]),
            scores=FitScores.from_dict(data["scores"]),
            lost_novelty=[LostNovelty.from_dict(ln) for ln in data.get("lost_novelty", [])],
            conflict_evidence=[ConflictEvidence.from_dict(ce) for ce in data.get("conflict_evidence", [])],
        )


@dataclass
class FitVector:
    """
    Multi-view fit vector for a paper.
    """
    paper_id: str
    fit_reports: List[FitReport]
    stress_score: float  # Weighted sum of non-FIT labels
    unfittable_score: float  # Weighted sum of UNFITTABLE labels
    
    def get_vector(self) -> List[str]:
        """Get the fit vector as list of labels."""
        return [report.label.value for report in self.fit_reports]
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "fit_reports": [fr.to_dict() for fr in self.fit_reports],
            "stress_score": self.stress_score,
            "unfittable_score": self.unfittable_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> FitVector:
        return cls(
            paper_id=data["paper_id"],
            fit_reports=[FitReport.from_dict(fr) for fr in data["fit_reports"]],
            stress_score=data["stress_score"],
            unfittable_score=data["unfittable_score"],
        )


# ============================================================================
# Phase 3: Stress Clustering & Evolution Data Structures
# ============================================================================

class ClusterType(Enum):
    """
    Types of stress clusters based on cross-view consistency.
    """
    STRONG_SHIFT = "STRONG_SHIFT"  # High unfittable across multiple high-weight views
    FACET_DEPENDENT = "FACET_DEPENDENT"  # Some views fail, others fit
    STABLE = "STABLE"  # Most high-weight views fit


@dataclass
class StressCluster:
    """
    A cluster of papers with similar failure patterns.
    """
    cluster_id: str
    papers: List[ResearchPaper]
    cluster_type: ClusterType
    stress_score: float  # Average S(p) for papers in cluster
    unfittable_score: float  # Average U(p) for papers in cluster
    view_failure_rates: Dict[str, float] = field(default_factory=dict)  # view_id -> failure rate
    failure_signature: str = ""  # Text description of common failure pattern
    
    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "papers": [p.to_dict() for p in self.papers],
            "cluster_type": self.cluster_type.value,
            "stress_score": self.stress_score,
            "unfittable_score": self.unfittable_score,
            "view_failure_rates": self.view_failure_rates,
            "failure_signature": self.failure_signature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> StressCluster:
        return cls(
            cluster_id=data["cluster_id"],
            papers=[ResearchPaper.from_dict(p) for p in data["papers"]],
            cluster_type=ClusterType(data["cluster_type"]),
            stress_score=data["stress_score"],
            unfittable_score=data["unfittable_score"],
            view_failure_rates=data.get("view_failure_rates", {}),
            failure_signature=data.get("failure_signature", ""),
        )


class OperationType(Enum):
    """
    Types of taxonomy evolution operations.
    """
    ADD_NODE = "ADD_NODE"
    SPLIT_NODE = "SPLIT_NODE"
    RENAME_NODE = "RENAME_NODE"


@dataclass
class NewNodeProposal:
    """
    Proposal for a new taxonomy node.
    """
    name: str
    definition: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    keywords: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "definition": self.definition,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "keywords": self.keywords,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> NewNodeProposal:
        return cls(**data)


@dataclass
class EvolutionOperation:
    """
    Base class for taxonomy evolution operations.
    """
    operation_type: OperationType
    view_id: str
    evidence: List[EvidenceSpan]
    fit_gain: float  # Improvement in fit after applying operation
    edit_cost: float  # Cost of this operation
    
    def to_dict(self) -> Dict:
        return {
            "operation_type": self.operation_type.value,
            "view_id": self.view_id,
            "evidence": [e.to_dict() for e in self.evidence],
            "fit_gain": self.fit_gain,
            "edit_cost": self.edit_cost,
        }


@dataclass
class AddNodeOperation(EvolutionOperation):
    """
    Operation to add a new node to taxonomy.
    """
    parent_path: str
    new_node: NewNodeProposal
    improvement_rate: float  # Percentage of cluster papers that improve
    
    def __init__(self, view_id: str, parent_path: str, new_node: NewNodeProposal,
                 evidence: List[EvidenceSpan], fit_gain: float, improvement_rate: float):
        super().__init__(
            operation_type=OperationType.ADD_NODE,
            view_id=view_id,
            evidence=evidence,
            fit_gain=fit_gain,
            edit_cost=1.0  # ADD_NODE cost
        )
        self.parent_path = parent_path
        self.new_node = new_node
        self.improvement_rate = improvement_rate
    
    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "parent_path": self.parent_path,
            "new_node": self.new_node.to_dict(),
            "improvement_rate": self.improvement_rate,
        })
        return d


@dataclass
class SplitNodeOperation(EvolutionOperation):
    """
    Operation to split an overcrowded node.
    """
    node_path: str
    sub_nodes: List[NewNodeProposal]
    
    def __init__(self, view_id: str, node_path: str, sub_nodes: List[NewNodeProposal],
                 evidence: List[EvidenceSpan], fit_gain: float):
        super().__init__(
            operation_type=OperationType.SPLIT_NODE,
            view_id=view_id,
            evidence=evidence,
            fit_gain=fit_gain,
            edit_cost=2.0  # SPLIT_NODE cost
        )
        self.node_path = node_path
        self.sub_nodes = sub_nodes
    
    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "node_path": self.node_path,
            "sub_nodes": [sn.to_dict() for sn in self.sub_nodes],
        })
        return d


@dataclass
class RenameNodeOperation(EvolutionOperation):
    """
    Operation to rename a node due to semantic drift.
    """
    node_path: str
    old_name: str
    new_name: str
    new_definition: str
    drift_score: float
    
    def __init__(self, view_id: str, node_path: str, old_name: str, new_name: str,
                 new_definition: str, drift_score: float, evidence: List[EvidenceSpan],
                 fit_gain: float):
        super().__init__(
            operation_type=OperationType.RENAME_NODE,
            view_id=view_id,
            evidence=evidence,
            fit_gain=fit_gain,
            edit_cost=0.5  # RENAME_NODE cost
        )
        self.node_path = node_path
        self.old_name = old_name
        self.new_name = new_name
        self.new_definition = new_definition
        self.drift_score = drift_score
    
    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "node_path": self.node_path,
            "old_name": self.old_name,
            "new_name": self.new_name,
            "new_definition": self.new_definition,
            "drift_score": self.drift_score,
        })
        return d


@dataclass
class EvolutionProposal:
    """
    Collection of proposed evolution operations.
    """
    operations: List[EvolutionOperation]
    total_fit_gain: float
    total_edit_cost: float
    objective_value: float  # fit_gain - lambda * edit_cost
    
    def to_dict(self) -> Dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "total_fit_gain": self.total_fit_gain,
            "total_edit_cost": self.total_edit_cost,
            "objective_value": self.objective_value,
        }


# ============================================================================
# Phase 4: Delta-aware Guidance Data Structures
# ============================================================================

class WritingMode(Enum):
    """
    Writing mode for organizing the survey based on reconstruction analysis.
    
    DELTA_FIRST: Organize primarily by stress/evolution (when main axis collapses badly)
    ANCHOR_PLUS_DELTA: Use main axis structure + highlight evolution (when main axis is stable)
    """
    DELTA_FIRST = "DELTA_FIRST"
    ANCHOR_PLUS_DELTA = "ANCHOR_PLUS_DELTA"


@dataclass
class ViewReconstructionScore:
    """
    Scores for a view after reconstruction with stress clusters.
    
    Used to determine optimal main axis and writing mode.
    """
    view_id: str
    facet_label: FacetLabel
    fit_gain: float  # How much stress papers would improve
    stress_reduction: float  # Reduction in average stress score
    coverage: float  # Coverage of taxonomy (normalized by max leaves)
    edit_cost: float  # Total edit cost of operations needed
    combined_score: float  # α·FitGain + β·Stress + γ·Coverage - λ·EditCost
    
    # Weights for combined score (from design spec)
    ALPHA = 0.4  # Weight for FitGain
    BETA = 0.3   # Weight for Stress reduction
    GAMMA = 0.2  # Weight for Coverage
    LAMBDA = 0.1 # Weight for EditCost penalty
    
    def __init__(self, view_id: str, facet_label: FacetLabel, 
                 fit_gain: float, stress_reduction: float, 
                 coverage: float, edit_cost: float):
        self.view_id = view_id
        self.facet_label = facet_label
        self.fit_gain = fit_gain
        self.stress_reduction = stress_reduction
        self.coverage = coverage
        self.edit_cost = edit_cost
        # Calculate combined score
        self.combined_score = (
            self.ALPHA * fit_gain + 
            self.BETA * stress_reduction + 
            self.GAMMA * coverage - 
            self.LAMBDA * edit_cost
        )
    
    def to_dict(self) -> Dict:
        return {
            "view_id": self.view_id,
            "facet_label": self.facet_label.value,
            "fit_gain": self.fit_gain,
            "stress_reduction": self.stress_reduction,
            "coverage": self.coverage,
            "edit_cost": self.edit_cost,
            "combined_score": self.combined_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> ViewReconstructionScore:
        return cls(
            view_id=data["view_id"],
            facet_label=FacetLabel(data["facet_label"]),
            fit_gain=data["fit_gain"],
            stress_reduction=data["stress_reduction"],
            coverage=data["coverage"],
            edit_cost=data["edit_cost"],
        )


@dataclass
class WritingRules:
    """
    Executable writing constraints based on writing mode.
    """
    do: List[str]  # Things the writer MUST do
    dont: List[str]  # Things the writer MUST NOT do
    
    def to_dict(self) -> Dict:
        return {
            "do": self.do,
            "dont": self.dont,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> WritingRules:
        return cls(
            do=data.get("do", []),
            dont=data.get("dont", []),
        )


@dataclass
class EvidenceCard:
    """
    Evidence card for a paper citation.
    """
    paper_id: str
    title: str
    claim: str
    quote: str
    page: int
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "claim": self.claim,
            "quote": self.quote,
            "page": self.page,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> EvidenceCard:
        return cls(**data)


@dataclass
class Subsection:
    """
    A subsection in the outline.
    """
    subsection: str  # Subsection name
    required_nodes: List[str]  # Node paths that must be covered
    required_citations: List[str]  # Paper IDs that must be cited
    must_answer: List[str]  # Questions that must be answered
    evidence_cards: List[EvidenceCard]  # Evidence for citations
    
    def to_dict(self) -> Dict:
        return {
            "subsection": self.subsection,
            "required_nodes": self.required_nodes,
            "required_citations": self.required_citations,
            "must_answer": self.must_answer,
            "evidence_cards": [ec.to_dict() for ec in self.evidence_cards],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> Subsection:
        return cls(
            subsection=data["subsection"],
            required_nodes=data["required_nodes"],
            required_citations=data["required_citations"],
            must_answer=data["must_answer"],
            evidence_cards=[EvidenceCard.from_dict(ec) for ec in data["evidence_cards"]],
        )


@dataclass
class Section:
    """
    A section in the outline.
    """
    section: str  # Section name
    subsections: List[Subsection]
    
    def to_dict(self) -> Dict:
        return {
            "section": self.section,
            "subsections": [ss.to_dict() for ss in self.subsections],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> Section:
        return cls(
            section=data["section"],
            subsections=[Subsection.from_dict(ss) for ss in data["subsections"]],
        )


@dataclass
class EvolutionSummaryItem:
    """
    Summary of one evolution operation.
    """
    operation: str  # Operation type
    view: str  # View ID
    parent: Optional[str]  # Parent node path (for ADD_NODE)
    new_node: Optional[str]  # New node name (for ADD_NODE)
    trigger_cluster: Optional[str]  # Cluster ID that triggered this
    justification_evidence: List[EvidenceSpan]
    
    def to_dict(self) -> Dict:
        return {
            "operation": self.operation,
            "view": self.view,
            "parent": self.parent,
            "new_node": self.new_node,
            "trigger_cluster": self.trigger_cluster,
            "justification_evidence": [e.to_dict() for e in self.justification_evidence],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> EvolutionSummaryItem:
        return cls(
            operation=data["operation"],
            view=data["view"],
            parent=data.get("parent"),
            new_node=data.get("new_node"),
            trigger_cluster=data.get("trigger_cluster"),
            justification_evidence=[EvidenceSpan.from_dict(e) 
                                   for e in data.get("justification_evidence", [])],
        )


@dataclass
class DeltaAwareGuidance:
    """
    Complete delta-aware writing guidance for downstream survey generation.
    
    This class now includes:
    - main_axis_mode: Writing mode (DELTA_FIRST vs ANCHOR_PLUS_DELTA)
    - writing_rules: Executable do/dont constraints
    - reconstruction_scores: All view scores for transparency
    """
    topic: str
    main_axis: TaxonomyView  # Selected main axis
    aux_axis: Optional[TaxonomyView]  # Selected auxiliary axis (if any)
    main_axis_mode: WritingMode  # NEW: Writing mode determination
    outline: List[Section]  # Structured outline with evidence
    evolution_summary: List[EvolutionSummaryItem]  # Summary of structure updates
    must_answer_questions: List[str]  # Overall questions the survey must address
    writing_rules: WritingRules  # NEW: Executable writing constraints
    reconstruction_scores: List[ViewReconstructionScore] = field(default_factory=list)  # NEW: All view scores
    generation_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "main_axis": self.main_axis.to_dict(),
            "aux_axis": self.aux_axis.to_dict() if self.aux_axis else None,
            "main_axis_mode": self.main_axis_mode.value,
            "outline": [s.to_dict() for s in self.outline],
            "evolution_summary": [es.to_dict() for es in self.evolution_summary],
            "must_answer_questions": self.must_answer_questions,
            "writing_rules": self.writing_rules.to_dict(),
            "reconstruction_scores": [rs.to_dict() for rs in self.reconstruction_scores],
            "generation_date": self.generation_date.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> DeltaAwareGuidance:
        return cls(
            topic=data["topic"],
            main_axis=TaxonomyView.from_dict(data["main_axis"]),
            aux_axis=TaxonomyView.from_dict(data["aux_axis"]) if data.get("aux_axis") else None,
            main_axis_mode=WritingMode(data.get("main_axis_mode", "ANCHOR_PLUS_DELTA")),
            outline=[Section.from_dict(s) for s in data["outline"]],
            evolution_summary=[EvolutionSummaryItem.from_dict(es) 
                             for es in data["evolution_summary"]],
            must_answer_questions=data["must_answer_questions"],
            writing_rules=WritingRules.from_dict(data.get("writing_rules", {"do": [], "dont": []})),
            reconstruction_scores=[ViewReconstructionScore.from_dict(rs) 
                                  for rs in data.get("reconstruction_scores", [])],
            generation_date=datetime.fromisoformat(data.get("generation_date", 
                                                           datetime.now().isoformat())),
        )


# ============================================================================
# Complete Result Data Structures
# ============================================================================

@dataclass
class IGFinder2Results:
    """
    Complete results from IG-Finder 2.0 pipeline.
    """
    topic: str
    multiview_baseline: MultiViewBaseline
    fit_vectors: List[FitVector]
    stress_clusters: List[StressCluster]
    evolution_proposal: EvolutionProposal
    delta_aware_guidance: DeltaAwareGuidance
    statistics: Dict[str, Any] = field(default_factory=dict)
    generation_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "multiview_baseline": self.multiview_baseline.to_dict(),
            "fit_vectors": [fv.to_dict() for fv in self.fit_vectors],
            "stress_clusters": [sc.to_dict() for sc in self.stress_clusters],
            "evolution_proposal": self.evolution_proposal.to_dict(),
            "delta_aware_guidance": self.delta_aware_guidance.to_dict(),
            "statistics": self.statistics,
            "generation_date": self.generation_date.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> IGFinder2Results:
        return cls(
            topic=data["topic"],
            multiview_baseline=MultiViewBaseline.from_dict(data["multiview_baseline"]),
            fit_vectors=[FitVector.from_dict(fv) for fv in data["fit_vectors"]],
            stress_clusters=[StressCluster.from_dict(sc) for sc in data["stress_clusters"]],
            evolution_proposal=EvolutionProposal(
                operations=[],  # Simplified, would need full deserialization
                total_fit_gain=data["evolution_proposal"]["total_fit_gain"],
                total_edit_cost=data["evolution_proposal"]["total_edit_cost"],
                objective_value=data["evolution_proposal"]["objective_value"],
            ),
            delta_aware_guidance=DeltaAwareGuidance.from_dict(data["delta_aware_guidance"]),
            statistics=data.get("statistics", {}),
            generation_date=datetime.fromisoformat(data.get("generation_date", 
                                                           datetime.now().isoformat())),
        )
