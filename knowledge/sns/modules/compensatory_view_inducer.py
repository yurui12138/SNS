"""
Compensatory View Inducer

This module implements the compensatory view induction strategy for Phase 1.
When the baseline quality check fails (e.g., dominant facet, insufficient diversity),
this inducer generates synthetic views to balance the cognitive structure.

Key Features:
1. Detects quality issues (dominant facet, low diversity)
2. Analyzes missing perspectives using LLM
3. Generates compensatory taxonomy views
4. Ensures minimal disruption to existing baseline

Reference: SNS Method Specification - Phase 1 Compensatory Strategy
"""

import logging
from typing import List, Dict, Optional, Set
from collections import Counter
from dataclasses import dataclass
import dspy

from ..dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    TaxonomyTree,
    TaxonomyTreeNode,
    NodeDefinition,
    EvidenceSpan,
    FacetLabel,
)
from ..parsing import safe_json_loads

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a quality issue in the baseline."""
    issue_type: str  # "dominant_facet", "low_diversity", "missing_perspective"
    severity: float  # 0.0 - 1.0
    description: str
    missing_facets: List[FacetLabel]
    recommended_action: str


class BaselineQualityAnalyzer:
    """
    Analyzes baseline quality and identifies issues requiring compensation.
    
    Quality Metrics:
    1. Facet diversity: Number of unique facets
    2. Facet balance: Distribution uniformity (Gini coefficient)
    3. Perspective coverage: Essential perspectives present
    """
    
    # Quality thresholds
    MIN_UNIQUE_FACETS = 2
    MAX_DOMINANT_FACET_RATIO = 0.6
    MIN_COVERAGE_RATIO = 0.4  # At least 40% of essential facets
    
    # Essential facets for comprehensive survey (heuristic)
    ESSENTIAL_FACETS = {
        FacetLabel.APPROACH,
        FacetLabel.APPLICATION,
        FacetLabel.EVALUATION,
    }
    
    def analyze(self, baseline: MultiViewBaseline) -> List[QualityIssue]:
        """
        Analyze baseline and identify quality issues.
        
        Args:
            baseline: The multi-view baseline to analyze
            
        Returns:
            List of QualityIssue objects
        """
        issues = []
        
        if not baseline.views:
            issues.append(QualityIssue(
                issue_type="empty_baseline",
                severity=1.0,
                description="No views in baseline",
                missing_facets=list(self.ESSENTIAL_FACETS),
                recommended_action="induce_all_essential"
            ))
            return issues
        
        # Count facets
        facet_counts = Counter([v.facet_label for v in baseline.views])
        present_facets = set(facet_counts.keys())
        num_unique_facets = len(facet_counts)
        
        # Issue 1: Low diversity (< 2 unique facets)
        if num_unique_facets < self.MIN_UNIQUE_FACETS:
            severity = 1.0 - (num_unique_facets / self.MIN_UNIQUE_FACETS)
            missing = list(self.ESSENTIAL_FACETS - present_facets)
            
            issues.append(QualityIssue(
                issue_type="low_diversity",
                severity=severity,
                description=f"Only {num_unique_facets} unique facet(s) found (minimum: {self.MIN_UNIQUE_FACETS})",
                missing_facets=missing if missing else [FacetLabel.OTHER],
                recommended_action="induce_diverse_views"
            ))
        
        # Issue 2: Dominant facet (> 60% of views)
        for facet, count in facet_counts.items():
            ratio = count / len(baseline.views)
            if ratio > self.MAX_DOMINANT_FACET_RATIO:
                severity = (ratio - self.MAX_DOMINANT_FACET_RATIO) / (1.0 - self.MAX_DOMINANT_FACET_RATIO)
                missing = list(self.ESSENTIAL_FACETS - present_facets)
                
                issues.append(QualityIssue(
                    issue_type="dominant_facet",
                    severity=severity,
                    description=f"Facet {facet.value} dominates with {ratio:.1%} of views (max: {self.MAX_DOMINANT_FACET_RATIO:.0%})",
                    missing_facets=missing if missing else [FacetLabel.OTHER],
                    recommended_action="balance_with_underrepresented"
                ))
        
        # Issue 3: Missing essential perspectives
        missing_essential = self.ESSENTIAL_FACETS - present_facets
        if missing_essential:
            coverage_ratio = len(present_facets & self.ESSENTIAL_FACETS) / len(self.ESSENTIAL_FACETS)
            if coverage_ratio < self.MIN_COVERAGE_RATIO:
                severity = 1.0 - coverage_ratio
                
                issues.append(QualityIssue(
                    issue_type="missing_perspective",
                    severity=severity,
                    description=f"Missing essential perspectives: {', '.join(f.value for f in missing_essential)}",
                    missing_facets=list(missing_essential),
                    recommended_action="induce_missing_essentials"
                ))
        
        # Sort by severity
        issues.sort(key=lambda x: x.severity, reverse=True)
        
        return issues


class CompensatoryViewGenerator:
    """
    Generates compensatory taxonomy views to address quality issues.
    
    Uses LLM to generate synthetic views based on:
    1. Topic
    2. Existing views (to avoid redundancy)
    3. Target facet (to address specific gap)
    """
    
    def __init__(self, lm):
        self.lm = lm
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup DSPy signature for compensatory view generation."""
        
        class CompensatoryViewSignature(dspy.Signature):
            """Generate a compensatory taxonomy view to fill gaps in baseline."""
            
            topic: str = dspy.InputField(desc="Research topic")
            target_facet: str = dspy.InputField(desc="Target facet to address (e.g., APPROACH, APPLICATION)")
            existing_views_summary: str = dspy.InputField(desc="Summary of existing views to avoid redundancy")
            rationale: str = dspy.InputField(desc="Why this perspective is needed")
            
            taxonomy_json: str = dspy.OutputField(desc="JSON taxonomy with format: {facet_label: str, facet_rationale: str, taxonomy_tree: {name: str, children: [...]}, evidence_spans: []}")
        
        self.generator = dspy.ChainOfThought(CompensatoryViewSignature)
    
    def generate_view(
        self,
        topic: str,
        baseline: MultiViewBaseline,
        target_facet: FacetLabel,
        view_id: str,
        rationale: str
    ) -> Optional[TaxonomyView]:
        """
        Generate a compensatory view for the target facet.
        
        Args:
            topic: Research topic
            baseline: Existing baseline
            target_facet: Target facet to generate
            view_id: Unique ID for new view
            rationale: Explanation of why this view is needed
            
        Returns:
            TaxonomyView or None if generation fails
        """
        logger.info(f"Generating compensatory view {view_id} for facet {target_facet.value}")
        
        # Summarize existing views
        existing_summary = self._summarize_existing_views(baseline)
        
        try:
            # Call LLM (wrap in dspy context with temperature=0.3 for controlled creativity)
            with dspy.context(lm=self.lm):
                result = self.generator(
                    topic=topic,
                    target_facet=target_facet.value,
                    existing_views_summary=existing_summary,
                    rationale=rationale
                )
            
            # Parse JSON output
            taxonomy_data = safe_json_loads(result.taxonomy_json)
            
            if not taxonomy_data:
                logger.warning(f"Failed to parse JSON for compensatory view {view_id}")
                return None
            
            # Build TaxonomyView
            view = self._build_taxonomy_view(taxonomy_data, topic, view_id, target_facet)
            
            # Mark as synthetic
            view.review_id = f"synthetic_{view_id}"
            view.review_title = f"Compensatory View: {target_facet.value} for {topic}"
            
            logger.info(f"Successfully generated compensatory view {view_id}")
            return view
            
        except Exception as e:
            logger.error(f"Error generating compensatory view {view_id}: {e}", exc_info=True)
            return None
    
    def _summarize_existing_views(self, baseline: MultiViewBaseline) -> str:
        """Create concise summary of existing views."""
        if not baseline.views:
            return "No existing views."
        
        summaries = []
        for view in baseline.views:
            leaf_nodes = view.tree.get_leaf_nodes()
            summaries.append(
                f"- {view.facet_label.value}: {len(leaf_nodes)} categories "
                f"(e.g., {', '.join(n.name for n in leaf_nodes[:3])})"
            )
        
        return "\n".join(summaries)
    
    def _build_taxonomy_view(
        self,
        data: Dict,
        topic: str,
        view_id: str,
        target_facet: FacetLabel
    ) -> TaxonomyView:
        """Build TaxonomyView from parsed JSON data."""
        
        # Parse facet label (fallback to target if not in data)
        try:
            facet_label = FacetLabel[data.get("facet_label", target_facet.name)]
        except KeyError:
            facet_label = target_facet
        
        # Build taxonomy tree
        tree = self._build_tree(data.get("taxonomy_tree", {"name": "ROOT", "children": []}))
        
        # Parse evidence spans (synthetic views may have empty evidence)
        evidence = [
            EvidenceSpan(
                claim=e.get("claim", ""),
                page=0,
                section="synthetic",
                char_start=0,
                char_end=0,
                quote=e.get("quote", "")
            )
            for e in data.get("evidence_spans", [])
        ]
        
        # Generate basic node definitions (synthetic, testable)
        node_definitions = self._generate_basic_definitions(tree)
        
        return TaxonomyView(
            view_id=view_id,
            review_id=f"synthetic_{view_id}",
            review_title=f"Compensatory View: {facet_label.value} for {topic}",
            facet_label=facet_label,
            facet_rationale=data.get("facet_rationale", f"Induced to balance baseline for {topic}"),
            tree=tree,
            node_definitions=node_definitions,
            weight=0.5,  # Lower weight for synthetic views
            evidence=evidence
        )
    
    def _build_tree(self, tree_data: Dict, parent_path: str = "") -> TaxonomyTree:
        """Recursively build TaxonomyTree from JSON data."""
        
        # Create root node
        root_name = tree_data.get("name", "ROOT")
        root_path = f"{parent_path}/{root_name}" if parent_path else root_name
        
        root = TaxonomyTreeNode(
            name=root_name,
            path=root_path,
            parent=parent_path if parent_path else None,
            children=[],
            is_leaf=len(tree_data.get("children", [])) == 0
        )
        
        tree = TaxonomyTree(root=root)
        
        # Recursively add children
        for child_data in tree_data.get("children", []):
            self._add_node_recursive(tree, child_data, root_path)
        
        return tree
    
    def _add_node_recursive(self, tree: TaxonomyTree, node_data: Dict, parent_path: str):
        """Recursively add nodes to tree."""
        node_name = node_data.get("name", "Unknown")
        node_path = f"{parent_path}/{node_name}"
        
        children_data = node_data.get("children", [])
        is_leaf = len(children_data) == 0
        
        node = TaxonomyTreeNode(
            name=node_name,
            path=node_path,
            parent=parent_path,
            children=[],
            is_leaf=is_leaf
        )
        
        tree.add_node(node)
        
        # Recursively add children
        for child_data in children_data:
            self._add_node_recursive(tree, child_data, node_path)
    
    def _generate_basic_definitions(self, tree: TaxonomyTree) -> Dict[str, NodeDefinition]:
        """Generate basic testable definitions for synthetic view nodes."""
        definitions = {}
        
        for node_path, node in tree.nodes.items():
            if node.name == "ROOT":
                continue
            
            # Create basic definition (heuristic-based)
            definitions[node_path] = NodeDefinition(
                node_path=node_path,
                definition=f"Methods or concepts related to {node.name}",
                inclusion_criteria=[f"Must be directly related to {node.name}"],
                exclusion_criteria=[f"Not related to {node.name}"],
                canonical_keywords=[node.name.lower()],
                boundary_statements=[f"Boundary: {node.name} vs other categories"],
                evidence_spans=[]
            )
        
        return definitions


class CompensatoryViewInducer:
    """
    Main interface for compensatory view induction.
    
    Workflow:
    1. Analyze baseline quality
    2. Identify critical issues
    3. Generate compensatory views
    4. Integrate into baseline
    """
    
    def __init__(self, lm, max_compensatory_views: int = 3):
        """
        Initialize inducer.
        
        Args:
            lm: Language model for view generation
            max_compensatory_views: Maximum number of compensatory views to generate
        """
        self.analyzer = BaselineQualityAnalyzer()
        self.generator = CompensatoryViewGenerator(lm)
        self.max_compensatory_views = max_compensatory_views
    
    def induce_if_needed(
        self,
        topic: str,
        baseline: MultiViewBaseline
    ) -> MultiViewBaseline:
        """
        Analyze baseline and induce compensatory views if needed.
        
        Args:
            topic: Research topic
            baseline: Current baseline
            
        Returns:
            Enhanced baseline (may be same as input if no issues found)
        """
        logger.info("="*80)
        logger.info("COMPENSATORY VIEW INDUCTION")
        logger.info("="*80)
        
        # Step 1: Analyze baseline quality
        logger.info("Step 1: Analyzing baseline quality...")
        issues = self.analyzer.analyze(baseline)
        
        if not issues:
            logger.info("✓ No quality issues detected. Baseline is adequate.")
            logger.info("="*80)
            return baseline
        
        # Log issues
        logger.info(f"Found {len(issues)} quality issue(s):")
        for i, issue in enumerate(issues, 1):
            logger.info(f"  {i}. [{issue.issue_type.upper()}] {issue.description}")
            logger.info(f"     Severity: {issue.severity:.2f} | Action: {issue.recommended_action}")
        
        # Step 2: Determine which views to induce
        logger.info("\nStep 2: Determining compensatory views to induce...")
        views_to_induce = self._plan_compensatory_views(issues)
        
        if not views_to_induce:
            logger.info("No compensatory views needed (issues are minor).")
            logger.info("="*80)
            return baseline
        
        logger.info(f"Planning to induce {len(views_to_induce)} compensatory view(s):")
        for facet, rationale in views_to_induce:
            logger.info(f"  - {facet.value}: {rationale}")
        
        # Step 3: Generate compensatory views
        logger.info("\nStep 3: Generating compensatory views...")
        new_views = []
        base_view_count = len(baseline.views)
        
        for i, (facet, rationale) in enumerate(views_to_induce, 1):
            view_id = f"C{i}"  # C for Compensatory
            view = self.generator.generate_view(
                topic=topic,
                baseline=baseline,
                target_facet=facet,
                view_id=view_id,
                rationale=rationale
            )
            
            if view:
                new_views.append(view)
                logger.info(f"  ✓ Generated compensatory view {view_id} ({facet.value})")
            else:
                logger.warning(f"  ✗ Failed to generate compensatory view for {facet.value}")
        
        if not new_views:
            logger.warning("Failed to generate any compensatory views.")
            logger.info("="*80)
            return baseline
        
        # Step 4: Integrate into baseline
        logger.info("\nStep 4: Integrating compensatory views into baseline...")
        enhanced_baseline = self._integrate_views(baseline, new_views)
        
        logger.info(f"✓ Enhanced baseline now has {len(enhanced_baseline.views)} views "
                   f"(original: {base_view_count}, added: {len(new_views)})")
        
        # Step 5: Re-analyze quality
        logger.info("\nStep 5: Re-analyzing enhanced baseline quality...")
        final_issues = self.analyzer.analyze(enhanced_baseline)
        
        if not final_issues:
            logger.info("✓ All quality issues resolved!")
        else:
            logger.info(f"Remaining issues: {len(final_issues)}")
            for issue in final_issues:
                logger.info(f"  - {issue.description} (severity: {issue.severity:.2f})")
        
        logger.info("="*80)
        return enhanced_baseline
    
    def _plan_compensatory_views(self, issues: List[QualityIssue]) -> List[tuple]:
        """
        Determine which compensatory views to induce based on issues.
        
        Args:
            issues: List of quality issues
            
        Returns:
            List of (facet, rationale) tuples
        """
        views_to_induce = []
        seen_facets = set()
        
        for issue in issues:
            if len(views_to_induce) >= self.max_compensatory_views:
                break
            
            # Only induce for high-severity issues (>= 0.5)
            if issue.severity < 0.5:
                continue
            
            # Select facets to induce
            for facet in issue.missing_facets:
                if facet not in seen_facets:
                    views_to_induce.append((
                        facet,
                        f"Address {issue.issue_type}: {issue.description}"
                    ))
                    seen_facets.add(facet)
                    
                    if len(views_to_induce) >= self.max_compensatory_views:
                        break
        
        return views_to_induce
    
    def _integrate_views(
        self,
        baseline: MultiViewBaseline,
        new_views: List[TaxonomyView]
    ) -> MultiViewBaseline:
        """
        Integrate compensatory views into baseline.
        
        Args:
            baseline: Original baseline
            new_views: Compensatory views to add
            
        Returns:
            New MultiViewBaseline with integrated views
        """
        # Combine all views
        all_views = baseline.views + new_views
        
        # Create new baseline (weight normalization happens automatically in __post_init__)
        enhanced_baseline = MultiViewBaseline(
            topic=baseline.topic,
            views=all_views
        )
        
        return enhanced_baseline
