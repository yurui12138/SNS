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
    ViewReconstructionScore,
    WritingMode,
    WritingRules,
)
from .taxonomy_evolution_applier import TaxonomyEvolutionApplier

logger = logging.getLogger(__name__)


class AxisSelector:
    """
    Selects main and auxiliary axes for organizing the survey.
    
    NEW DESIGN (reconstruct-then-select):
    - Main axis: Selected based on reconstruction scores (FitGain, Stress, Coverage, EditCost)
    - Writing mode: Determined by reconstruction analysis (DELTA_FIRST vs ANCHOR_PLUS_DELTA)
    - Aux axis: Best discriminator for stress clusters
    """
    
    def __init__(self):
        pass
    
    def select_main_axis_with_mode(
        self,
        reconstruction_scores: List[ViewReconstructionScore],
        baseline: MultiViewBaseline
    ) -> Tuple[TaxonomyView, WritingMode]:
        """
        Select main axis and determine writing mode based on reconstruction scores.
        
        This implements the 'reconstruct-then-select' design principle:
        1. Use reconstruction_scores (already sorted by combined score)
        2. Pick the best view as main axis
        3. Determine writing mode based on edit cost and fit gain
        
        Writing Mode Rules:
        - DELTA_FIRST: If best view needs heavy reconstruction (EditCost > 3.0 or FitGain > 10.0)
        - ANCHOR_PLUS_DELTA: If best view is relatively stable (low EditCost, moderate FitGain)
        
        Args:
            reconstruction_scores: Sorted reconstruction scores from Phase 3
            baseline: Multi-view baseline
            
        Returns:
            Tuple of (selected_main_axis_view, writing_mode)
        """
        from ..dataclass_v2 import WritingMode
        
        logger.info("Selecting main axis with writing mode (reconstruct-then-select)...")
        
        if not reconstruction_scores:
            # Fallback: use first view
            logger.warning("No reconstruction scores provided, falling back to first view")
            if not baseline.views:
                raise ValueError("No views in baseline to select main axis from.")
            main_axis = baseline.views[0]
            return main_axis, WritingMode.ANCHOR_PLUS_DELTA
        
        # Best view is already at index 0 (sorted by combined_score in Phase 3)
        best_score = reconstruction_scores[0]
        
        # Find corresponding view
        main_axis = baseline.get_view_by_id(best_score.view_id)
        if not main_axis:
            logger.error(f"View {best_score.view_id} not found in baseline")
            main_axis = baseline.views[0]
        
        # Determine writing mode based on reconstruction needs
        # Delta-first: Heavy reconstruction needed (structure collapsed)
        # Anchor+Delta: Light reconstruction (structure is stable)
        
        if best_score.edit_cost > 3.0 or best_score.fit_gain > 10.0:
            # Significant reconstruction needed -> Delta-first mode
            mode = WritingMode.DELTA_FIRST
            mode_rationale = f"Heavy reconstruction (EditCost={best_score.edit_cost:.1f}, FitGain={best_score.fit_gain:.1f})"
        else:
            # Stable structure -> Anchor+Delta mode
            mode = WritingMode.ANCHOR_PLUS_DELTA
            mode_rationale = f"Stable structure (EditCost={best_score.edit_cost:.1f}, FitGain={best_score.fit_gain:.1f})"
        
        logger.info(f"Selected main axis: {main_axis.view_id} ({main_axis.facet_label.value})")
        logger.info(f"  Reconstruction metrics:")
        logger.info(f"    FitGain: {best_score.fit_gain:.3f}")
        logger.info(f"    StressReduction: {best_score.stress_reduction:.3f}")
        logger.info(f"    Coverage: {best_score.coverage:.3f}")
        logger.info(f"    EditCost: {best_score.edit_cost:.3f}")
        logger.info(f"    CombinedScore: {best_score.combined_score:.3f}")
        logger.info(f"Writing mode: {mode.value} - {mode_rationale}")
        
        return main_axis, mode
    
    def select_main_axis(
        self,
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector]
    ) -> TaxonomyView:
        """
        DEPRECATED: Use select_main_axis_with_mode() instead.
        
        This method is kept for backward compatibility but should not be used
        in the new design. It uses the old logic (FIT rate, stability, coverage)
        instead of reconstruction scores.
        """
        logger.warning("Using deprecated select_main_axis(). Use select_main_axis_with_mode() instead.")
        
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
            same_facet_count = sum(
                1 for v in baseline.views
                if v.facet_label == view.facet_label
            )
            stability = same_facet_count / len(baseline.views)
            
            # Calculate coverage (richness of taxonomy)
            num_leaves = len(view.tree.get_leaf_nodes())
            coverage = min(1.0, num_leaves / 50.0)
            
            # Combined score (old formula)
            score = 0.6 * fit_rate + 0.3 * stability + 0.1 * coverage
            
            axis_scores.append((view, score))
        
        # Select main axis
        main_axis = max(axis_scores, key=lambda x: x[1])[0]
        
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
        main_axis_mode: WritingMode,
        reconstruction_scores: List[ViewReconstructionScore],
        clusters: List[StressCluster],
        evolution_proposal: EvolutionProposal,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper]
    ) -> DeltaAwareGuidance:
        """
        Generate complete delta-aware guidance with writing mode and rules.
        
        NEW DESIGN: Now includes:
        - main_axis_mode: Writing mode (DELTA_FIRST vs ANCHOR_PLUS_DELTA)
        - writing_rules: Executable do/dont constraints
        - reconstruction_scores: All view scores for transparency
        
        Args:
            topic: Research topic
            main_axis: Main organizational axis
            aux_axis: Auxiliary axis (optional)
            main_axis_mode: Writing mode determination
            reconstruction_scores: All view reconstruction scores
            clusters: Stress clusters
            evolution_proposal: Proposed evolution operations
            fit_vectors: All fit vectors
            papers: Research papers
            
        Returns:
            DeltaAwareGuidance object with all fields
        """
        logger.info(f"Generating delta-aware guidance (mode: {main_axis_mode.value})...")
        
        # Generate outline (mode-aware)
        outline = self._generate_outline(
            main_axis,
            aux_axis,
            main_axis_mode,
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
        
        # Generate writing rules based on mode
        writing_rules = self._generate_writing_rules(
            main_axis_mode,
            evolution_proposal,
            clusters
        )
        
        guidance = DeltaAwareGuidance(
            topic=topic,
            main_axis=main_axis,
            aux_axis=aux_axis,
            main_axis_mode=main_axis_mode,
            outline=outline,
            evolution_summary=evolution_summary,
            must_answer_questions=must_answer,
            writing_rules=writing_rules,
            reconstruction_scores=reconstruction_scores
        )
        
        logger.info(f"Generated guidance with {len(outline)} sections, "
                   f"{len(writing_rules.do)} do-rules, {len(writing_rules.dont)} dont-rules")
        
        return guidance
    
    def _generate_outline(
        self,
        main_axis: TaxonomyView,
        aux_axis: Optional[TaxonomyView],
        main_axis_mode: WritingMode,
        clusters: List[StressCluster],
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper]
    ) -> List[Section]:
        """
        Generate structured outline based on writing mode.
        
        DELTA_FIRST: Emphasize evolution and stress clusters
        ANCHOR_PLUS_DELTA: Use existing structure + highlight changes
        """
        
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
                quote=paper.description[:200] if paper.description else "",
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
    
    def _generate_writing_rules(
        self,
        main_axis_mode: WritingMode,
        evolution_proposal: EvolutionProposal,
        clusters: List[StressCluster]
    ) -> WritingRules:
        """
        Generate executable writing constraints based on writing mode.
        
        DELTA_FIRST mode:
        - DO: Lead with evolution and stress points
        - DO: Organize by emerging patterns
        - DONT: Over-rely on existing taxonomy structure
        
        ANCHOR_PLUS_DELTA mode:
        - DO: Use existing structure as foundation
        - DO: Highlight specific evolution points
        - DONT: Ignore baseline taxonomy
        """
        
        do_rules = []
        dont_rules = []
        
        if main_axis_mode == WritingMode.DELTA_FIRST:
            # Delta-first: Emphasize change and evolution
            do_rules.extend([
                "Lead with emerging trends and structural shifts",
                "Organize content primarily by innovation clusters and stress points",
                "Emphasize papers that don't fit existing taxonomies",
                f"Address all {len(clusters)} identified stress clusters explicitly",
                f"Justify all {len(evolution_proposal.operations)} proposed structural changes"
            ])
            
            dont_rules.extend([
                "Don't force-fit new work into inadequate old categories",
                "Don't assume existing review structure is still valid",
                "Don't bury evolution insights in traditional sections"
            ])
            
        else:  # ANCHOR_PLUS_DELTA
            # Anchor+Delta: Use structure but highlight changes
            do_rules.extend([
                "Use main axis structure as the organizational foundation",
                "Integrate new papers into existing taxonomy where they fit",
                "Clearly mark and explain structural updates",
                "Maintain continuity with established review structure",
                f"Call out {len(clusters)} specific areas needing attention" if clusters else "Maintain comprehensive coverage"
            ])
            
            dont_rules.extend([
                "Don't ignore evolution and stress points",
                "Don't present the taxonomy as static or complete",
                "Don't omit discussion of structural limitations"
            ])
        
        # Common rules for both modes
        do_rules.extend([
            "Provide evidence for all major claims",
            "Cite specific papers with page numbers where possible",
            "Answer all must-answer questions explicitly"
        ])
        
        dont_rules.extend([
            "Don't make unsupported generalizations",
            "Don't cite papers without reading them"
        ])
        
        return WritingRules(do=do_rules, dont=dont_rules)
    
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
        """
        Generate specific, actionable must-answer questions.
        
        Enhanced to provide:
        1. Category-specific questions (from taxonomy nodes)
        2. Evolution-specific questions (from operations)
        3. Stress-cluster-specific questions
        4. Cross-cutting synthesis questions
        """
        
        questions = []
        
        # Core organizational questions
        leaf_nodes = main_axis.tree.get_leaf_nodes()
        leaf_names = [n.name for n in leaf_nodes[:5]]  # Top 5
        
        questions.append(
            f"What are the defining characteristics that distinguish {', '.join(leaf_names[:3])}?"
        )
        
        questions.append(
            f"How does the {main_axis.facet_label.value} perspective organize the {len(leaf_nodes)} "
            f"distinct categories identified in this survey?"
        )
        
        # Auxiliary axis questions
        if aux_axis:
            aux_leaf_nodes = aux_axis.tree.get_leaf_nodes()
            questions.append(
                f"How do the {len(aux_leaf_nodes)} {aux_axis.facet_label.value} dimensions "
                f"cross-cut the main {main_axis.facet_label.value} categories?"
            )
        
        # Operation-specific questions (detailed)
        if proposal.operations:
            # Group operations by type
            add_ops = [op for op in proposal.operations if op.operation_type.value == "ADD_NODE"]
            split_ops = [op for op in proposal.operations if op.operation_type.value == "SPLIT_NODE"]
            rename_ops = [op for op in proposal.operations if op.operation_type.value == "RENAME_NODE"]
            
            if add_ops:
                new_categories = [op.new_node.name for op in add_ops]
                questions.append(
                    f"Why are the new categories {', '.join(new_categories[:3])} necessary, "
                    f"and what emerging research do they capture that existing taxonomies miss?"
                )
            
            if split_ops:
                split_nodes = [op.node_path.split('/')[-1] for op in split_ops]
                questions.append(
                    f"What evidence justifies splitting the {', '.join(split_nodes)} "
                    f"category/categories into finer-grained sub-categories?"
                )
            
            if rename_ops:
                renamings = [(op.old_name, op.new_name) for op in rename_ops]
                for old, new in renamings[:2]:  # Top 2
                    questions.append(
                        f"What semantic drift motivated renaming '{old}' to '{new}', "
                        f"and how does this better reflect current research trends?"
                    )
        
        # Cluster-specific questions
        if clusters:
            strong_shift_clusters = [c for c in clusters if c.cluster_type.value == "STRONG_SHIFT"]
            if strong_shift_clusters:
                questions.append(
                    f"What are the {len(strong_shift_clusters)} major paradigm shifts or emerging trends "
                    f"that challenge existing taxonomies, and which papers exemplify these shifts?"
                )
            
            # Ask about specific clusters
            for i, cluster in enumerate(clusters[:2], 1):  # Top 2
                cluster_papers = [p.title for p in cluster.papers[:3]]
                questions.append(
                    f"Stress Cluster {i}: What common innovations do papers like "
                    f"'{cluster_papers[0]}' represent, and why do they resist classification "
                    f"in existing categories (stress score: {cluster.stress_score:.2f})?"
                )
        
        # Synthesis and future directions
        questions.extend([
            "How has the field evolved since the most recent comprehensive reviews, "
            "and what gaps remain in the current taxonomy?",
            "What are the most promising directions for future research based on the "
            "identified stress points and structural updates?"
        ])
        
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
        self.evolution_applier = TaxonomyEvolutionApplier()
    
    def run(
        self,
        topic: str,
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper],
        clusters: List[StressCluster],
        evolution_proposal: EvolutionProposal,
        reconstruction_scores: List[ViewReconstructionScore]
    ) -> DeltaAwareGuidance:
        """
        Run complete Phase 4 pipeline with reconstruction-based axis selection.
        
        NEW DESIGN (reconstruct-then-select):
        1. Use reconstruction_scores from Phase 3 to select main axis
        2. Determine writing mode (DELTA_FIRST vs ANCHOR_PLUS_DELTA)
        3. Select auxiliary axis for cross-organization
        4. Generate guidance with mode-specific writing rules
        
        Args:
            topic: Research topic
            baseline: Multi-view baseline from Phase 1
            fit_vectors: Fit vectors from Phase 2
            papers: Research papers
            clusters: Stress clusters from Phase 3
            evolution_proposal: Evolution proposal from Phase 3
            reconstruction_scores: View reconstruction scores from Phase 3
            
        Returns:
            DeltaAwareGuidance object with all fields
        """
        logger.info("="*80)
        logger.info("PHASE 4: Delta-aware Guidance Generation (reconstruct-then-select)")
        logger.info("="*80)
        
        # Step 1: Select main axis AND determine writing mode (NEW DESIGN)
        logger.info("Step 1: Selecting main axis with writing mode...")
        main_axis, main_axis_mode = self.axis_selector.select_main_axis_with_mode(
            reconstruction_scores,
            baseline
        )
        
        # Step 2: Apply evolution operations to main axis (NEW: taxonomy_v2)
        logger.info("Step 2: Applying evolution operations to main axis...")
        main_axis_evolved = self.evolution_applier.apply_evolution(
            base_view=main_axis,
            operations=evolution_proposal.operations
        )
        logger.info(f"  Evolved main axis now has {len(main_axis_evolved.tree.nodes)} nodes")
        
        # Step 3: Select aux axis
        logger.info("Step 3: Selecting auxiliary axis...")
        aux_axis = self.axis_selector.select_aux_axis(baseline, clusters, main_axis)
        
        # Apply evolution to aux axis if present
        if aux_axis:
            aux_axis_evolved = self.evolution_applier.apply_evolution(
                base_view=aux_axis,
                operations=evolution_proposal.operations
            )
            logger.info(f"  Evolved aux axis now has {len(aux_axis_evolved.tree.nodes)} nodes")
        else:
            aux_axis_evolved = None
        
        # Step 4: Generate guidance (NEW: uses evolved taxonomies)
        logger.info("Step 4: Generating delta-aware guidance with evolved taxonomies...")
        guidance = self.guidance_generator.generate_guidance(
            topic=topic,
            main_axis=main_axis_evolved,  # Use evolved taxonomy
            aux_axis=aux_axis_evolved,     # Use evolved taxonomy
            main_axis_mode=main_axis_mode,
            reconstruction_scores=reconstruction_scores,
            clusters=clusters,
            evolution_proposal=evolution_proposal,
            fit_vectors=fit_vectors,
            papers=papers
        )
        
        logger.info("Phase 4 completed successfully!")
        logger.info(f"  Writing mode: {main_axis_mode.value}")
        logger.info(f"  Main axis: {main_axis.view_id} ({main_axis.facet_label.value})")
        logger.info(f"  Aux axis: {aux_axis.view_id if aux_axis else 'None'}")
        logger.info("="*80)
        
        return guidance
