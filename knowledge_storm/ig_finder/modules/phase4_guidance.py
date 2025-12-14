"""
Phase 4: Delta-aware Guidance Generation

This module implements main/aux axis selection and generates structured
guidance for downstream survey generation.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from ..dataclass import ResearchPaper
from ..dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    FitVector,
    FitLabel,
    StressCluster,
    EvolutionProposal,
    DeltaAwareGuidance,
    Section,
    Subsection,
    EvidenceCard,
    EvolutionSummaryItem,
)

logger = logging.getLogger(__name__)


class AxisSelector:
    """
    Selects main and auxiliary axes for organizing the survey.
    
    Main axis: Most stable and broadly applicable
    Aux axis: Best discriminator for stress clusters
    """
    
    def __init__(self):
        pass
    
    def select_main_axis(
        self,
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector]
    ) -> TaxonomyView:
        """
        Select main axis based on FIT rate, stability, and coverage.
        
        Formula: score = 0.6 * fit_rate + 0.3 * stability + 0.1 * coverage
        
        Args:
            baseline: Multi-view baseline
            fit_vectors: All fit vectors
            
        Returns:
            Selected TaxonomyView as main axis
        """
        logger.info("Selecting main axis...")
        
        axis_scores = []
        
        for view in baseline.views:
            # Calculate FIT rate for this view
            fit_count = 0
            total = 0
            
            for fv in fit_vectors:
                for report in fv.fit_reports:
                    if report.view_id == view.view_id:
                        total += 1
                        if report.label == FitLabel.FIT:
                            fit_count += 1
            
            fit_rate = fit_count / total if total > 0 else 0.0
            
            # Calculate stability (cross-review consistency)
            # Approximation: count views with same facet
            same_facet_count = sum(
                1 for v in baseline.views
                if v.facet_label == view.facet_label
            )
            stability = same_facet_count / len(baseline.views)
            
            # Calculate coverage (richness of taxonomy)
            num_leaves = len(view.tree.get_leaf_nodes())
            coverage = min(1.0, num_leaves / 50.0)
            
            # Combined score
            score = 0.6 * fit_rate + 0.3 * stability + 0.1 * coverage
            
            axis_scores.append((view, score))
            
            logger.info(f"  {view.view_id} ({view.facet_label.value}): "
                       f"fit_rate={fit_rate:.3f}, stability={stability:.3f}, "
                       f"coverage={coverage:.3f}, score={score:.3f}")
        
        # Select main axis
        main_axis = max(axis_scores, key=lambda x: x[1])[0]
        
        logger.info(f"Selected main axis: {main_axis.view_id} ({main_axis.facet_label.value})")
        
        return main_axis
    
    def select_aux_axis(
        self,
        baseline: MultiViewBaseline,
        clusters: List[StressCluster],
        main_axis: TaxonomyView
    ) -> Optional[TaxonomyView]:
        """
        Select auxiliary axis based on discriminativeness for stress clusters.
        
        Formula: discriminativeness = Var(failure_rates_across_clusters)
        
        Args:
            baseline: Multi-view baseline
            clusters: Stress clusters
            main_axis: Already selected main axis
            
        Returns:
            Selected TaxonomyView as aux axis, or None if no good candidate
        """
        logger.info("Selecting auxiliary axis...")
        
        if not clusters:
            logger.info("No clusters available, skipping aux axis selection")
            return None
        
        discriminativeness_scores = []
        
        for view in baseline.views:
            if view.view_id == main_axis.view_id:
                continue  # Skip main axis
            
            # Calculate failure rate for each cluster
            cluster_failure_rates = []
            
            for cluster in clusters:
                failure_rate = cluster.view_failure_rates.get(view.view_id, 0.0)
                cluster_failure_rates.append(failure_rate)
            
            # Calculate variance (high variance = good discriminator)
            if len(cluster_failure_rates) > 1:
                variance = np.var(cluster_failure_rates)
            else:
                variance = 0.0
            
            discriminativeness_scores.append((view, variance))
            
            logger.info(f"  {view.view_id} ({view.facet_label.value}): "
                       f"variance={variance:.3f}")
        
        if not discriminativeness_scores:
            return None
        
        # Select aux axis with highest variance
        aux_axis = max(discriminativeness_scores, key=lambda x: x[1])[0]
        
        logger.info(f"Selected aux axis: {aux_axis.view_id} ({aux_axis.facet_label.value})")
        
        return aux_axis


class GuidanceGenerator:
    """
    Generates delta-aware writing guidance for downstream survey generation.
    """
    
    def __init__(self):
        pass
    
    def generate_guidance(
        self,
        topic: str,
        main_axis: TaxonomyView,
        aux_axis: Optional[TaxonomyView],
        clusters: List[StressCluster],
        evolution_proposal: EvolutionProposal,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper]
    ) -> DeltaAwareGuidance:
        """
        Generate complete delta-aware guidance.
        
        Args:
            topic: Research topic
            main_axis: Main organizational axis
            aux_axis: Auxiliary axis (optional)
            clusters: Stress clusters
            evolution_proposal: Proposed evolution operations
            fit_vectors: All fit vectors
            papers: Research papers
            
        Returns:
            DeltaAwareGuidance object
        """
        logger.info("Generating delta-aware guidance...")
        
        # Generate outline
        outline = self._generate_outline(
            main_axis,
            aux_axis,
            clusters,
            fit_vectors,
            papers
        )
        
        # Generate evolution summary
        evolution_summary = self._generate_evolution_summary(evolution_proposal)
        
        # Generate must-answer questions
        must_answer = self._generate_must_answer_questions(
            main_axis,
            aux_axis,
            clusters,
            evolution_proposal
        )
        
        guidance = DeltaAwareGuidance(
            topic=topic,
            main_axis=main_axis,
            aux_axis=aux_axis,
            outline=outline,
            evolution_summary=evolution_summary,
            must_answer_questions=must_answer
        )
        
        logger.info(f"Generated guidance with {len(outline)} sections")
        
        return guidance
    
    def _generate_outline(
        self,
        main_axis: TaxonomyView,
        aux_axis: Optional[TaxonomyView],
        clusters: List[StressCluster],
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper]
    ) -> List[Section]:
        """Generate structured outline."""
        
        outline = []
        
        # Use main axis children as top-level sections
        main_children = main_axis.tree.root.children
        
        for child_path in main_children:
            child_node = main_axis.tree.nodes.get(child_path)
            if not child_node:
                continue
            
            section_name = f"{child_node.name} (Main Axis: {main_axis.facet_label.value})"
            
            subsections = []
            
            if aux_axis:
                # Cross-organize with aux axis
                aux_children = aux_axis.tree.root.children
                
                for aux_path in aux_children[:3]:  # Limit to 3 for brevity
                    aux_node = aux_axis.tree.nodes.get(aux_path)
                    if not aux_node:
                        continue
                    
                    subsection = self._create_subsection(
                        child_node,
                        aux_node,
                        clusters,
                        fit_vectors,
                        papers,
                        main_axis,
                        aux_axis
                    )
                    
                    subsections.append(subsection)
            else:
                # Just use main axis children
                subsection = self._create_subsection(
                    child_node,
                    None,
                    clusters,
                    fit_vectors,
                    papers,
                    main_axis,
                    None
                )
                subsections.append(subsection)
            
            section = Section(
                section=section_name,
                subsections=subsections
            )
            
            outline.append(section)
        
        return outline
    
    def _create_subsection(
        self,
        main_node,
        aux_node,
        clusters: List[StressCluster],
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper],
        main_axis: TaxonomyView,
        aux_axis: Optional[TaxonomyView]
    ) -> Subsection:
        """Create a subsection for the outline."""
        
        # Subsection name
        if aux_node:
            subsection_name = f"{main_node.name} × {aux_node.name}"
        else:
            subsection_name = main_node.name
        
        # Find relevant papers
        relevant_papers = self._find_relevant_papers(
            main_node,
            aux_node,
            fit_vectors,
            papers,
            main_axis,
            aux_axis
        )
        
        # Create evidence cards
        evidence_cards = []
        for paper in relevant_papers[:5]:  # Top 5
            evidence_cards.append(EvidenceCard(
                paper_id=paper.url,
                title=paper.title,
                claim=f"Key contribution from {paper.title[:50]}...",
                quote=paper.abstract[:200] if paper.abstract else "",
                page=0
            ))
        
        # Required citations
        required_citations = [p.url for p in relevant_papers[:5]]
        
        # Must-answer questions for this subsection
        must_answer = [
            f"What are the key approaches in {main_node.name}?"
        ]
        
        if aux_node:
            must_answer.append(
                f"How does {aux_node.name} intersect with {main_node.name}?"
            )
        
        subsection = Subsection(
            subsection=subsection_name,
            required_nodes=[main_node.path] + ([aux_node.path] if aux_node else []),
            required_citations=required_citations,
            must_answer=must_answer,
            evidence_cards=evidence_cards
        )
        
        return subsection
    
    def _find_relevant_papers(
        self,
        main_node,
        aux_node,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper],
        main_axis: TaxonomyView,
        aux_axis: Optional[TaxonomyView]
    ) -> List[ResearchPaper]:
        """Find papers relevant to this subsection."""
        
        paper_map = {p.url: p for p in papers}
        relevant = []
        
        for fv in fit_vectors:
            # Check if paper fits into main_node
            main_fits = False
            aux_fits = True  # Default true if no aux axis
            
            for report in fv.fit_reports:
                if report.view_id == main_axis.view_id:
                    if report.best_leaf_path and main_node.path in report.best_leaf_path:
                        main_fits = True
                
                if aux_axis and report.view_id == aux_axis.view_id:
                    if aux_node and report.best_leaf_path:
                        aux_fits = aux_node.path in report.best_leaf_path
            
            if main_fits and aux_fits:
                if fv.paper_id in paper_map:
                    relevant.append(paper_map[fv.paper_id])
        
        return relevant
    
    def _generate_evolution_summary(
        self,
        proposal: EvolutionProposal
    ) -> List[EvolutionSummaryItem]:
        """Generate summary of evolution operations."""
        
        summary = []
        
        for op in proposal.operations:
            item = EvolutionSummaryItem(
                operation=op.operation_type.value,
                view=op.view_id,
                parent=getattr(op, 'parent_path', None),
                new_node=getattr(op, 'new_node', None).name if hasattr(op, 'new_node') else None,
                trigger_cluster=None,  # Would need to track this
                justification_evidence=op.evidence
            )
            summary.append(item)
        
        return summary
    
    def _generate_must_answer_questions(
        self,
        main_axis: TaxonomyView,
        aux_axis: Optional[TaxonomyView],
        clusters: List[StressCluster],
        proposal: EvolutionProposal
    ) -> List[str]:
        """Generate overall must-answer questions."""
        
        questions = [
            f"What are the main organizational dimensions in {main_axis.facet_label.value}?",
            "How has the field evolved beyond existing reviews?",
        ]
        
        if aux_axis:
            questions.append(
                f"How does {aux_axis.facet_label.value} provide additional perspective?"
            )
        
        if clusters:
            questions.append(
                f"What are the {len(clusters)} identified stress points in existing taxonomies?"
            )
        
        if proposal.operations:
            questions.append(
                f"Why are the {len(proposal.operations)} proposed structure updates necessary?"
            )
        
        return questions


# ============================================================================
# Main Phase 4 Pipeline
# ============================================================================

class Phase4Pipeline:
    """
    Complete Phase 4 pipeline: Delta-aware Guidance Generation.
    """
    
    def __init__(self):
        self.axis_selector = AxisSelector()
        self.guidance_generator = GuidanceGenerator()
    
    def run(
        self,
        topic: str,
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper],
        clusters: List[StressCluster],
        evolution_proposal: EvolutionProposal
    ) -> DeltaAwareGuidance:
        """
        Run complete Phase 4 pipeline.
        
        Args:
            topic: Research topic
            baseline: Multi-view baseline from Phase 1
            fit_vectors: Fit vectors from Phase 2
            papers: Research papers
            clusters: Stress clusters from Phase 3
            evolution_proposal: Evolution proposal from Phase 3
            
        Returns:
            DeltaAwareGuidance object
        """
        logger.info("="*80)
        logger.info("PHASE 4: Delta-aware Guidance Generation")
        logger.info("="*80)
        
        # Step 1: Select main axis
        logger.info("Step 1: Selecting main axis...")
        main_axis = self.axis_selector.select_main_axis(baseline, fit_vectors)
        
        # Step 2: Select aux axis
        logger.info("Step 2: Selecting auxiliary axis...")
        aux_axis = self.axis_selector.select_aux_axis(baseline, clusters, main_axis)
        
        # Step 3: Generate guidance
        logger.info("Step 3: Generating delta-aware guidance...")
        guidance = self.guidance_generator.generate_guidance(
            topic=topic,
            main_axis=main_axis,
            aux_axis=aux_axis,
            clusters=clusters,
            evolution_proposal=evolution_proposal,
            fit_vectors=fit_vectors,
            papers=papers
        )
        
        logger.info("Phase 4 completed successfully!")
        logger.info("="*80)
        
        return guidance
