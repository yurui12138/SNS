"""
Innovation Gap Report Generation Module

Generates comprehensive reports identifying innovation gaps for downstream
review generation systems.
"""

import dspy
import logging
from typing import List, Dict, Tuple
from datetime import datetime

from ...dataclass import KnowledgeBase
from ..dataclass import (
    CognitiveBaseline,
    InnovationCluster,
    InnovationGapReport,
    GapAnalysis,
    ResearchPaper,
    DeviationAnalysis,
)

logger = logging.getLogger(__name__)


class SummarizeCognitiveBaseline(dspy.Signature):
    """Generate a concise summary of the cognitive baseline."""
    
    topic = dspy.InputField(desc="The research topic")
    num_reviews = dspy.InputField(desc="Number of review papers analyzed")
    paradigms = dspy.InputField(desc="List of research paradigms")
    methods = dspy.InputField(desc="List of mainstream methods")
    boundaries = dspy.InputField(desc="List of knowledge boundaries")
    temporal_coverage = dspy.InputField(desc="Temporal coverage of reviews")
    
    baseline_summary = dspy.OutputField(desc="Comprehensive summary of the cognitive baseline (3-5 paragraphs)")


class GenerateEvolutionNarrative(dspy.Signature):
    """Generate a narrative describing knowledge evolution from consensus to innovation."""
    
    topic = dspy.InputField(desc="The research topic")
    baseline_summary = dspy.InputField(desc="Summary of cognitive baseline")
    innovation_clusters = dspy.InputField(desc="Description of identified innovation clusters")
    innovation_paths = dspy.InputField(desc="Paths from consensus to innovation in the mind map")
    
    evolution_narrative = dspy.OutputField(desc="Narrative describing the knowledge evolution (5-7 paragraphs)")


class GenerateReviewRecommendations(dspy.Signature):
    """Generate recommendations for downstream review generation systems."""
    
    topic = dspy.InputField(desc="The research topic")
    innovation_clusters = dspy.InputField(desc="Description of innovation clusters")
    gap_analysis = dspy.InputField(desc="Gap analysis by dimensions")
    
    recommendations = dspy.OutputField(desc="Detailed recommendations for review generation (organizational structure, innovation emphasis, citation priorities)")


class InnovationGapReportGenerator:
    """
    Generates comprehensive innovation gap reports.
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
    
    def generate_report(
        self,
        topic: str,
        cognitive_baseline: CognitiveBaseline,
        innovation_clusters: List[InnovationCluster],
        papers_with_deviations: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]],
        mind_map_visualization_data: Dict,
    ) -> InnovationGapReport:
        """
        Generate comprehensive innovation gap report.
        
        Args:
            topic: Research topic
            cognitive_baseline: The cognitive baseline
            innovation_clusters: Identified innovation clusters
            papers_with_deviations: Papers with deviation analyses
            mind_map_visualization_data: Visualization data from mind map manager
            
        Returns:
            InnovationGapReport object
        """
        logger.info("Generating Innovation Gap Report...")
        
        # Step 1: Summarize cognitive baseline
        logger.info("Step 1: Summarizing cognitive baseline...")
        baseline_summary = self._generate_baseline_summary(topic, cognitive_baseline)
        
        # Step 2: Perform gap analysis by dimension
        logger.info("Step 2: Performing gap analysis by dimension...")
        gap_analysis_by_dim = self._perform_gap_analysis(
            topic,
            innovation_clusters,
            papers_with_deviations,
        )
        
        # Step 3: Generate evolution narrative
        logger.info("Step 3: Generating evolution narrative...")
        evolution_narrative = self._generate_evolution_narrative(
            topic,
            baseline_summary,
            innovation_clusters,
            mind_map_visualization_data.get("innovation_paths", []),
        )
        
        # Step 4: Generate recommendations for review generation
        logger.info("Step 4: Generating recommendations...")
        recommendations = self._generate_recommendations(
            topic,
            innovation_clusters,
            gap_analysis_by_dim,
        )
        
        # Step 5: Compile statistics
        statistics = self._compile_statistics(
            cognitive_baseline,
            innovation_clusters,
            papers_with_deviations,
            mind_map_visualization_data,
        )
        
        # Create report
        report = InnovationGapReport(
            topic=topic,
            generation_date=datetime.now(),
            cognitive_baseline_summary=baseline_summary,
            identified_clusters=innovation_clusters,
            gap_analysis_by_dimension=gap_analysis_by_dim,
            evolution_narrative=evolution_narrative,
            mind_map_visualization_data=mind_map_visualization_data,
            recommendations_for_review=recommendations,
            statistics=statistics,
        )
        
        logger.info("Innovation Gap Report generation complete")
        return report
    
    def _generate_baseline_summary(self, topic: str, baseline: CognitiveBaseline) -> str:
        paradigms_str = "; ".join([f"{p.name}: {p.description}" for p in baseline.research_paradigms[:5]])
        methods_str = "; ".join([f"{m.name}: {m.description}" for m in baseline.mainstream_methods[:5]])
        boundaries_str = "; ".join([f"{b.dimension}: {b.description}" for b in baseline.knowledge_boundaries.values()])
        temporal_str = f"{baseline.temporal_coverage.start.year if baseline.temporal_coverage.start else 'Unknown'} to {baseline.temporal_coverage.end.year if baseline.temporal_coverage.end else 'Unknown'}"
        prompt = (
            f"Summarize the cognitive baseline for the topic.\n\n"
            f"Topic: {topic}\n"
            f"Reviews analyzed: {len(baseline.review_papers)}\n"
            f"Research paradigms: {paradigms_str if paradigms_str else 'None identified'}\n"
            f"Mainstream methods: {methods_str if methods_str else 'None identified'}\n"
            f"Knowledge boundaries: {boundaries_str if boundaries_str else 'None identified'}\n"
            f"Temporal coverage: {temporal_str}\n\n"
            f"Write a comprehensive summary (3-5 paragraphs)."
        )
        outputs = self.lm(prompt=prompt)
        return outputs[0] if outputs else ""
    
    def _perform_gap_analysis(
        self,
        topic: str,
        innovation_clusters: List[InnovationCluster],
        papers_with_deviations: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]],
    ) -> Dict[str, GapAnalysis]:
        """Perform gap analysis by dimension."""
        # Collect all innovation dimensions
        dimension_to_clusters = {}
        
        for cluster in innovation_clusters:
            for dim in cluster.innovation_dimensions:
                if dim not in dimension_to_clusters:
                    dimension_to_clusters[dim] = []
                dimension_to_clusters[dim].append(cluster)
        
        # Create gap analysis for each dimension
        gap_analyses = {}
        
        for dimension, clusters in dimension_to_clusters.items():
            # Aggregate information
            cluster_ids = [c.cluster_id for c in clusters]
            
            # Create description
            cluster_descriptions = []
            for cluster in clusters[:3]:  # Top 3 clusters per dimension
                cluster_descriptions.append(
                    f"{cluster.name}: {cluster.cluster_summary}"
                )
            
            gap_description = f"Innovation identified in {dimension}. " + " ".join(cluster_descriptions)
            
            # Calculate evidence strength
            evidence_strength = min(1.0, len(clusters) * 0.2 + 0.3)  # Simple heuristic
            
            # Extract research opportunities
            research_opportunities = []
            for cluster in clusters:
                if cluster.potential_impact:
                    research_opportunities.append(cluster.potential_impact)
            
            gap_analysis = GapAnalysis(
                dimension=dimension,
                gap_description=gap_description,
                related_clusters=cluster_ids,
                evidence_strength=evidence_strength,
                research_opportunities=research_opportunities[:5],
            )
            
            gap_analyses[dimension] = gap_analysis
        
        return gap_analyses
    
    def _generate_evolution_narrative(
        self,
        topic: str,
        baseline_summary: str,
        innovation_clusters: List[InnovationCluster],
        innovation_paths: List[List[str]],
    ) -> str:
        clusters_desc = []
        for i, cluster in enumerate(innovation_clusters[:5], 1):
            clusters_desc.append(
                f"{i}. {cluster.name} ({len(cluster.core_papers)} papers): {cluster.cluster_summary}"
            )
        clusters_str = "\n".join(clusters_desc)
        paths_desc = []
        for i, path in enumerate(innovation_paths[:5], 1):
            paths_desc.append(f"{i}. {' → '.join(path)}")
        paths_str = "\n".join(paths_desc) if paths_desc else "No clear paths identified"
        prompt = (
            f"Generate a knowledge evolution narrative (5-7 paragraphs).\n\n"
            f"Topic: {topic}\n\n"
            f"Baseline summary:\n{baseline_summary}\n\n"
            f"Innovation clusters:\n{clusters_str}\n\n"
            f"Innovation paths:\n{paths_str}"
        )
        outputs = self.lm(prompt=prompt)
        return outputs[0] if outputs else ""
    
    def _generate_recommendations(
        self,
        topic: str,
        innovation_clusters: List[InnovationCluster],
        gap_analysis: Dict[str, GapAnalysis],
    ) -> str:
        clusters_desc = []
        for cluster in innovation_clusters:
            clusters_desc.append(
                f"- {cluster.name}: {cluster.cluster_summary} (Impact: {cluster.potential_impact})"
            )
        clusters_str = "\n".join(clusters_desc) if clusters_desc else "No clusters identified"
        gaps_desc = []
        for dim, gap in gap_analysis.items():
            gaps_desc.append(
                f"- {dim}: {gap.gap_description} (Evidence strength: {gap.evidence_strength:.2f})"
            )
        gaps_str = "\n".join(gaps_desc) if gaps_desc else "No gaps identified"
        prompt = (
            f"Generate detailed recommendations for downstream review generation.\n\n"
            f"Topic: {topic}\n\n"
            f"Innovation clusters:\n{clusters_str}\n\n"
            f"Gap analysis by dimension:\n{gaps_str}\n\n"
            f"Include organizational structure, innovation emphasis, and citation priorities."
        )
        outputs = self.lm(prompt=prompt)
        return outputs[0] if outputs else ""
    
    def _compile_statistics(
        self,
        cognitive_baseline: CognitiveBaseline,
        innovation_clusters: List[InnovationCluster],
        papers_with_deviations: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]],
        mind_map_visualization_data: Dict,
    ) -> Dict:
        """Compile statistics for the report."""
        statistics = {
            "num_review_papers": len(cognitive_baseline.review_papers),
            "num_research_papers_analyzed": len(papers_with_deviations),
            "num_innovation_clusters": len(innovation_clusters),
            "total_papers_in_clusters": sum(len(c.core_papers) for c in innovation_clusters),
            "temporal_coverage_start": cognitive_baseline.temporal_coverage.start.year if cognitive_baseline.temporal_coverage.start else None,
            "temporal_coverage_end": cognitive_baseline.temporal_coverage.end.year if cognitive_baseline.temporal_coverage.end else None,
            "num_paradigms_identified": len(cognitive_baseline.research_paradigms),
            "num_mainstream_methods": len(cognitive_baseline.mainstream_methods),
            "num_knowledge_boundaries": len(cognitive_baseline.knowledge_boundaries),
            "evolution_state_distribution": mind_map_visualization_data.get("statistics", {}),
            "num_innovation_paths": len(mind_map_visualization_data.get("innovation_paths", [])),
        }
        
        return statistics
    
    def format_report_as_markdown(self, report: InnovationGapReport) -> str:
        """Format report as markdown for easy reading."""
        md_parts = []
        
        # Header
        md_parts.append(f"# Innovation Gap Report: {report.topic}")
        md_parts.append(f"\n**Generated:** {report.generation_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_parts.append("---\n")
        
        # Executive Summary
        md_parts.append("## Executive Summary\n")
        md_parts.append(f"This report identifies {report.statistics['num_innovation_clusters']} innovation clusters ")
        md_parts.append(f"based on analysis of {report.statistics['num_research_papers_analyzed']} research papers, ")
        md_parts.append(f"contextualized against a cognitive baseline derived from {report.statistics['num_review_papers']} review papers.\n")
        
        # Part I: Cognitive Baseline
        md_parts.append("\n## Part I: Cognitive Baseline\n")
        md_parts.append(report.cognitive_baseline_summary)
        md_parts.append("\n")
        
        # Part II: Innovation Clusters
        md_parts.append("\n## Part II: Identified Innovation Clusters\n")
        for i, cluster in enumerate(report.identified_clusters, 1):
            md_parts.append(f"\n### {i}. {cluster.name}\n")
            md_parts.append(f"**Cluster ID:** {cluster.cluster_id}\n\n")
            md_parts.append(f"**Summary:** {cluster.cluster_summary}\n\n")
            md_parts.append(f"**Core Papers:** {len(cluster.core_papers)}\n")
            for j, paper in enumerate(cluster.core_papers[:5], 1):
                md_parts.append(f"  {j}. {paper.title} ({paper.year})\n")
            if len(cluster.core_papers) > 5:
                md_parts.append(f"  ... and {len(cluster.core_papers) - 5} more\n")
            md_parts.append(f"\n**Innovation Dimensions:** {', '.join(cluster.innovation_dimensions)}\n\n")
            md_parts.append(f"**Internal Coherence Score:** {cluster.internal_coherence_score:.2f}\n\n")
            md_parts.append(f"**Deviation from Consensus:** {cluster.deviation_from_consensus.deviation_description}\n\n")
            md_parts.append(f"**Potential Impact:** {cluster.potential_impact}\n\n")
        
        # Part III: Gap Analysis
        md_parts.append("\n## Part III: Gap Analysis by Dimension\n")
        for dimension, gap in report.gap_analysis_by_dimension.items():
            md_parts.append(f"\n### {dimension}\n")
            md_parts.append(f"**Evidence Strength:** {gap.evidence_strength:.2f}\n\n")
            md_parts.append(f"**Gap Description:** {gap.gap_description}\n\n")
            md_parts.append(f"**Related Clusters:** {', '.join(gap.related_clusters)}\n\n")
            if gap.research_opportunities:
                md_parts.append("**Research Opportunities:**\n")
                for opp in gap.research_opportunities:
                    md_parts.append(f"- {opp}\n")
            md_parts.append("\n")
        
        # Part IV: Evolution Narrative
        md_parts.append("\n## Part IV: Knowledge Evolution Narrative\n")
        md_parts.append(report.evolution_narrative)
        md_parts.append("\n")
        
        # Part V: Mind Map Visualization
        md_parts.append("\n## Part V: Mind Map Visualization\n")
        md_parts.append("**Evolution State Distribution:**\n")
        for state, count in report.mind_map_visualization_data.get("statistics", {}).items():
            md_parts.append(f"- {state}: {count}\n")
        md_parts.append("\n")
        
        # Part VI: Recommendations
        md_parts.append("\n## Part VI: Recommendations for Review Generation\n")
        md_parts.append(report.recommendations_for_review)
        md_parts.append("\n")
        
        # Statistics Appendix
        md_parts.append("\n## Appendix: Statistics\n")
        for key, value in report.statistics.items():
            if key != "evolution_state_distribution":
                md_parts.append(f"- **{key}:** {value}\n")
        
        return "".join(md_parts)
