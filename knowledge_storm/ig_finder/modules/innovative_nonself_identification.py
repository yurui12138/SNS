"""
Innovative Non-self Identification Module

This module implements Phase 2 of IG-Finder: identifying innovation clusters
that deviate from the cognitive baseline but maintain internal coherence.
"""

import dspy
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict

from ...interface import Retriever, Information, Agent
from ..dataclass import KnowledgeBase
from ...dataclass import ConversationTurn
from ...logging_wrapper import LoggingWrapper
from ..dataclass import (
    CognitiveBaseline,
    ResearchPaper,
    InnovationCluster,
    DeviationAnalysis,
    Evidence,
    EvolutionState,
    ExtendedKnowledgeNode,
)
from ..schemas import DeviationAnalysisResult, ClusterValidationResult
from ..parsing import parse_json_array_or_csv

logger = logging.getLogger(__name__)


class FrontierPaperRetriever:
    """
    Retrieves frontier research papers (non-review) published after
    the temporal coverage of the cognitive baseline.
    """
    
    def __init__(self, retriever: Retriever, top_k: int = 30):
        self.retriever = retriever
        self.top_k = top_k
    
    def retrieve_frontier_papers(
        self,
        topic: str,
        baseline_temporal_coverage: Optional[datetime] = None,
    ) -> List[Information]:
        """
        Retrieve research papers (non-review) for the topic.
        
        Args:
            topic: The research topic
            baseline_temporal_coverage: End date of baseline temporal coverage
            
        Returns:
            List of Information objects containing research papers
        """
        logger.info(f"Retrieving frontier research papers for topic: {topic}")
        
        # Construct queries for research papers (avoid review-specific terms)
        research_queries = [
            f"{topic}",
            f"{topic} method",
            f"{topic} approach",
            f"{topic} model",
            f"{topic} framework",
            f"recent advances in {topic}",
        ]
        
        all_results = []
        for query in research_queries:
            results = self.retriever.retrieve(query=query, exclude_urls=[])
            all_results.extend(results)
        
        # Remove duplicates
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Filter out review papers
        research_papers = self._filter_research_papers(unique_results)
        
        # Sort by recency (heuristic: prefer papers with "2023", "2024", etc. in snippets)
        sorted_papers = self._sort_by_recency(research_papers)
        
        return sorted_papers[:self.top_k]
    
    def _filter_research_papers(self, results: List[Information]) -> List[Information]:
        """Filter to keep research papers and exclude reviews."""
        review_keywords = ['survey', 'review', 'overview', 'systematic review']
        
        filtered = []
        for result in results:
            title_lower = result.title.lower()
            
            # Exclude if title contains review keywords
            has_review_keyword = any(keyword in title_lower for keyword in review_keywords)
            
            if not has_review_keyword:
                filtered.append(result)
        
        return filtered
    
    def _sort_by_recency(self, results: List[Information]) -> List[Information]:
        """Sort by recency (heuristic based on year mentions in text)."""
        def recency_score(info: Information) -> float:
            score = 0.0
            text = (info.title + " " + info.description).lower()
            
            # Check for recent years
            current_year = datetime.now().year
            for year in range(current_year, current_year - 5, -1):
                if str(year) in text:
                    score += (year - 2020) * 2.0  # Higher score for more recent years
            
            # Prefer papers with "new", "novel", "recent" in title
            novelty_keywords = ['new', 'novel', 'recent', 'emerging', 'latest']
            for keyword in novelty_keywords:
                if keyword in text:
                    score += 1.0
            
            return score
        
        return sorted(results, key=recency_score, reverse=True)


class ExtractPaperMetadata(dspy.Signature):
    """Extract structured metadata from a research paper."""
    
    title = dspy.InputField(desc="Title of the research paper")
    abstract = dspy.InputField(desc="Abstract or description of the research paper")
    url = dspy.InputField(desc="URL of the research paper")
    
    year = dspy.OutputField(desc="Publication year (estimate if not explicitly stated)")
    authors = dspy.OutputField(desc="List of authors (comma-separated, or 'Unknown' if not available)")
    venue = dspy.OutputField(desc="Publication venue (conference/journal name, or 'Unknown' if not available)")
    core_claims = dspy.OutputField(desc="List of core claims made in the paper (comma-separated)")
    methodology = dspy.OutputField(desc="Brief description of the methodology used")
    key_findings = dspy.OutputField(desc="List of key findings (comma-separated)")


class ExpertPerspectiveGenerator:
    """
    Generates expert perspectives for multi-angle analysis.
    Similar to STORM's perspective-guided question asking, but focused on
    identifying deviations from consensus.
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
    
    def generate_expert_perspectives(
        self,
        topic: str,
        cognitive_baseline: CognitiveBaseline,
    ) -> List[Dict[str, str]]:
        """
        Generate expert perspectives based on topic and baseline.
        
        Returns:
            List of expert dictionaries with 'name' and 'description'
        """
        # Default expert perspectives for scientific research
        base_experts = [
            {
                "name": "Methodology Expert",
                "description": f"An expert in research methodologies for {topic}. Focuses on identifying novel methodological approaches, experimental designs, and analytical techniques that differ from established methods."
            },
            {
                "name": "Data Paradigm Expert",
                "description": f"An expert in data and datasets for {topic}. Focuses on identifying new data sources, data collection strategies, and data paradigm shifts."
            },
            {
                "name": "Theoretical Framework Expert",
                "description": f"An expert in theoretical foundations of {topic}. Focuses on identifying new conceptual frameworks, theoretical innovations, and paradigm shifts."
            },
            {
                "name": "Application Domain Expert",
                "description": f"An expert in applications of {topic}. Focuses on identifying novel application scenarios, use cases, and domain extensions."
            },
        ]
        
        # Could enhance with LLM-generated domain-specific experts
        # For now, return base experts
        return base_experts


class AnalyzePaperDeviation(dspy.Signature):
    """Analyze how a paper deviates from established consensus."""
    
    topic = dspy.InputField(desc="The research topic")
    expert_perspective = dspy.InputField(desc="The expert perspective for analysis")
    paper_title = dspy.InputField(desc="Title of the research paper")
    paper_content = dspy.InputField(desc="Content of the research paper (abstract and key findings)")
    consensus_summary = dspy.InputField(desc="Summary of established consensus in this area")
    baseline_concepts = dspy.InputField(desc="List of key concepts in the cognitive baseline")
    
    matched_baseline_concepts = dspy.OutputField(desc="Strict JSON array of concept names")
    deviation_description = dspy.OutputField(desc="Detailed description of deviation or extension")
    deviation_dimensions = dspy.OutputField(desc="Strict JSON array of dimensions: e.g., methodology, data, theory, application")
    deviation_score = dspy.OutputField(desc="Deviation score from 0 to 10, where 0=fully aligned with consensus, 10=completely novel direction")
    innovation_potential = dspy.OutputField(desc="Assessment of innovation potential: 'high', 'medium', or 'low'")
    reasoning = dspy.OutputField(desc="Reasoning for the deviation assessment")


class DifferenceAwareAnalyzer:
    """
    Performs difference-aware analysis by comparing frontier papers
    with the cognitive baseline from multiple expert perspectives.
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.paper_metadata_extractor = dspy.ChainOfThought(ExtractPaperMetadata)
        self.deviation_analyzer = dspy.ChainOfThought(AnalyzePaperDeviation)
    
    def analyze_paper(
        self,
        topic: str,
        paper_info: Information,
        cognitive_baseline: CognitiveBaseline,
        expert_perspectives: List[Dict[str, str]],
    ) -> Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]:
        """
        Analyze a single paper from multiple expert perspectives.
        
        Args:
            topic: Research topic
            paper_info: Information about the paper
            cognitive_baseline: The cognitive baseline
            expert_perspectives: List of expert perspective dictionaries
            
        Returns:
            Tuple of (ResearchPaper, dict of expert_name -> DeviationAnalysis)
        """
        logger.info(f"Analyzing paper: {paper_info.title}")
        
        # Extract paper metadata
        with dspy.context(lm=self.lm):
            metadata_result = self.paper_metadata_extractor(
                title=paper_info.title,
                abstract=paper_info.description,
                url=paper_info.url,
            )
        
        # Parse metadata
        try:
            year = int(metadata_result.year.strip())
        except:
            year = datetime.now().year
        
        authors = [a.strip() for a in str(metadata_result.authors).split(',')] if metadata_result.authors != 'Unknown' else []
        venue = metadata_result.venue if metadata_result.venue != 'Unknown' else ""
        core_claims = parse_json_array_or_csv(getattr(metadata_result, 'core_claims', ''))
        methodology = str(getattr(metadata_result, 'methodology', ''))
        key_findings = parse_json_array_or_csv(getattr(metadata_result, 'key_findings', ''))
        
        research_paper = ResearchPaper(
            title=paper_info.title,
            authors=authors,
            year=year,
            url=paper_info.url,
            abstract=paper_info.description,
            citations=0,
            venue=venue,
            core_claims=core_claims,
            methodology=methodology,
            key_findings=key_findings,
        )
        
        # Prepare consensus summary and baseline concepts
        consensus_summary = self._summarize_consensus(cognitive_baseline)
        baseline_concepts = self._extract_baseline_concepts(cognitive_baseline)
        
        # Analyze from each expert perspective
        deviation_analyses = {}
        paper_content = f"Abstract: {paper_info.description}\n\nMethodology: {methodology}\n\nKey Findings: {', '.join(key_findings)}"
        
        for expert in expert_perspectives:
            expert_name = expert["name"]
            expert_description = expert["description"]
            
            try:
                with dspy.context(lm=self.lm):
                    deviation_result = self.deviation_analyzer(
                        topic=topic,
                        expert_perspective=f"{expert_name}: {expert_description}",
                        paper_title=paper_info.title,
                        paper_content=paper_content,
                        consensus_summary=consensus_summary,
                        baseline_concepts=baseline_concepts,
                    )

                matched_concepts = parse_json_array_or_csv(deviation_result.matched_baseline_concepts)
                deviation_dims = parse_json_array_or_csv(deviation_result.deviation_dimensions)
                try:
                    deviation_score = float(deviation_result.deviation_score) / 10.0
                except:
                    deviation_score = 0.5

                potential_raw = str(getattr(deviation_result, "innovation_potential", "medium")).lower()
                if "high" in potential_raw:
                    potential_norm = "high"
                elif "med" in potential_raw or "moderate" in potential_raw or "avg" in potential_raw:
                    potential_norm = "medium"
                elif "low" in potential_raw or "small" in potential_raw or "minor" in potential_raw:
                    potential_norm = "low"
                else:
                    potential_norm = "medium"

                validated_dev = DeviationAnalysisResult(
                    matched_baseline_concepts=matched_concepts,
                    deviation_description=deviation_result.deviation_description,
                    deviation_dimensions=deviation_dims,
                    deviation_score=deviation_score,
                    innovation_potential=potential_norm,
                    reasoning=deviation_result.reasoning,
                )

                deviation_analysis = DeviationAnalysis(
                    baseline_node_path=validated_dev.matched_baseline_concepts,
                    deviation_dimensions=validated_dev.deviation_dimensions,
                    deviation_description=validated_dev.deviation_description,
                    deviation_score=validated_dev.deviation_score,
                    expert_perspectives={expert_name: validated_dev.reasoning},
                )

                deviation_analyses[expert_name] = deviation_analysis
            except Exception as e:
                logger.error(f"Deviation analysis failed for {paper_info.title} [{expert_name}]: {e}")
                continue
        
        return research_paper, deviation_analyses
    
    def _summarize_consensus(self, baseline: CognitiveBaseline) -> str:
        """Create a summary of the cognitive baseline."""
        summary_parts = []
        
        # Add paradigms
        if baseline.research_paradigms:
            paradigm_names = [p.name for p in baseline.research_paradigms[:3]]
            summary_parts.append(f"Established paradigms: {', '.join(paradigm_names)}")
        
        # Add methods
        if baseline.mainstream_methods:
            method_names = [m.name for m in baseline.mainstream_methods[:3]]
            summary_parts.append(f"Mainstream methods: {', '.join(method_names)}")
        
        # Add boundaries
        if baseline.knowledge_boundaries:
            boundary_dims = list(baseline.knowledge_boundaries.keys())[:3]
            summary_parts.append(f"Known boundaries: {', '.join(boundary_dims)}")
        
        return ". ".join(summary_parts) if summary_parts else "No consensus established yet."
    
    def _extract_baseline_concepts(self, baseline: CognitiveBaseline) -> str:
        """Extract key concepts from baseline mind map."""
        concepts = []
        
        def collect_concepts(node: ExtendedKnowledgeNode, depth: int = 0):
            if depth > 2:  # Limit depth to avoid too long list
                return
            concepts.append(node.name)
            for child in node.children:
                collect_concepts(child, depth + 1)
        
        if hasattr(baseline.consensus_map, 'root'):
            for child in baseline.consensus_map.root.children:
                collect_concepts(child, 0)
        
        return ", ".join(concepts[:20])  # Limit to top 20 concepts


class IdentifyInnovationClusters(dspy.Signature):
    """Identify and validate innovation clusters from grouped papers."""
    
    topic = dspy.InputField(desc="The research topic")
    paper_group = dspy.InputField(desc="Group of papers with titles and key findings")
    common_deviation_pattern = dspy.InputField(desc="Common deviation pattern across these papers")
    
    is_coherent_cluster = dspy.OutputField(desc="'yes' if papers form a logically coherent innovation cluster, 'no' otherwise")
    cluster_name = dspy.OutputField(desc="A descriptive name for this innovation cluster (if coherent)")
    coherence_reasoning = dspy.OutputField(desc="Reasoning for coherence assessment")
    innovation_dimensions = dspy.OutputField(desc="Strict JSON array of key innovation dimensions")
    cluster_summary = dspy.OutputField(desc="Brief summary of what makes this cluster innovative")
    potential_impact = dspy.OutputField(desc="Potential impact of this innovation cluster")


class InnovationClusterIdentifier:
    """
    Identifies innovation clusters by:
    1. Grouping papers with similar deviation patterns
    2. Validating internal logical coherence
    3. Assigning appropriate evolution states
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.cluster_identifier = dspy.ChainOfThought(IdentifyInnovationClusters)
    
    def identify_clusters(
        self,
        topic: str,
        papers_with_deviations: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]],
        min_cluster_size: int = 2,
        deviation_threshold: float = 0.5,
    ) -> List[InnovationCluster]:
        """
        Identify innovation clusters from papers and their deviation analyses.
        
        Args:
            topic: Research topic
            papers_with_deviations: List of (paper, deviation_analyses) tuples
            min_cluster_size: Minimum papers to form a cluster
            deviation_threshold: Minimum deviation score to consider
            
        Returns:
            List of InnovationCluster objects
        """
        logger.info("Identifying innovation clusters...")
        
        # Filter papers with significant deviation
        significant_deviations = []
        for paper, deviations in papers_with_deviations:
            avg_deviation = np.mean([d.deviation_score for d in deviations.values()])
            if avg_deviation >= deviation_threshold:
                significant_deviations.append((paper, deviations, avg_deviation))
        
        logger.info(f"Found {len(significant_deviations)} papers with significant deviation")
        
        if len(significant_deviations) < min_cluster_size:
            logger.info("Not enough papers to form clusters")
            return []
        
        # Group papers by deviation dimensions (simple clustering)
        dimension_groups = defaultdict(list)
        for paper, deviations, avg_dev in significant_deviations:
            # Get all deviation dimensions across experts
            all_dims = set()
            for deviation in deviations.values():
                all_dims.update(deviation.deviation_dimensions)
            
            # Create a key from sorted dimensions
            dim_key = tuple(sorted(all_dims))
            dimension_groups[dim_key].append((paper, deviations, avg_dev))
        
        # Validate and create clusters
        clusters = []
        for dim_key, papers_in_group in dimension_groups.items():
            if len(papers_in_group) >= min_cluster_size:
                cluster = self._validate_and_create_cluster(
                    topic,
                    papers_in_group,
                    list(dim_key),
                )
                if cluster:
                    clusters.append(cluster)
        
        logger.info(f"Identified {len(clusters)} innovation clusters")
        return clusters
    
    def _validate_and_create_cluster(
        self,
        topic: str,
        papers_in_group: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis], float]],
        common_dimensions: List[str],
    ) -> Optional[InnovationCluster]:
        """Validate cluster coherence and create InnovationCluster object."""
        
        # Prepare paper group description
        paper_descriptions = []
        for i, (paper, _, _) in enumerate(papers_in_group[:5], 1):  # Limit to 5 for prompt
            paper_descriptions.append(
                f"{i}. {paper.title}\n   Key findings: {', '.join(paper.key_findings[:3])}"
            )
        paper_group_text = "\n".join(paper_descriptions)
        
        common_pattern = f"Papers deviate in: {', '.join(common_dimensions)}"
        
        # Use LLM to validate coherence
        with dspy.context(lm=self.lm):
            cluster_result = self.cluster_identifier(
                topic=topic,
                paper_group=paper_group_text,
                common_deviation_pattern=common_pattern,
            )
        
        validated_cluster = ClusterValidationResult(
            is_coherent_cluster=str(cluster_result.is_coherent_cluster).lower().strip(),
            cluster_name=cluster_result.cluster_name,
            coherence_reasoning=cluster_result.coherence_reasoning,
            innovation_dimensions=parse_json_array_or_csv(cluster_result.innovation_dimensions),
            cluster_summary=cluster_result.cluster_summary,
            potential_impact=cluster_result.potential_impact,
        )

        if validated_cluster.is_coherent_cluster != 'yes':
            logger.info(validated_cluster.coherence_reasoning)
            return None
        
        # Calculate internal coherence score (simplified)
        coherence_score = 0.7  # Base score for validated clusters
        
        # Create cluster
        papers = [p for p, _, _ in papers_in_group]
        
        # Aggregate deviation analysis (use first paper's as template)
        first_deviation = list(papers_in_group[0][1].values())[0]
        aggregated_deviation = DeviationAnalysis(
            baseline_node_path=first_deviation.baseline_node_path,
            deviation_dimensions=common_dimensions,
            deviation_description=validated_cluster.cluster_summary,
            deviation_score=np.mean([avg_dev for _, _, avg_dev in papers_in_group]),
            expert_perspectives={},
        )
        
        # Create evidence list
        evidence = []
        for paper, _, _ in papers_in_group:
            for claim in paper.core_claims[:2]:  # Top 2 claims per paper
                evidence.append(Evidence(
                    paper=paper,
                    claim=claim,
                    supporting_text=paper.abstract[:200],
                    confidence_score=0.8,
                ))
        
        innovation_dims = validated_cluster.innovation_dimensions
        
        cluster = InnovationCluster(
            cluster_id=f"cluster_{len(papers)}_{hash(tuple(p.url for p in papers)) % 10000}",
            name=validated_cluster.cluster_name,
            core_papers=papers,
            deviation_from_consensus=aggregated_deviation,
            internal_coherence_score=coherence_score,
            innovation_dimensions=innovation_dims,
            supporting_evidence=evidence,
            knowledge_path=[topic] + first_deviation.baseline_node_path,
            cluster_summary=validated_cluster.cluster_summary,
            potential_impact=validated_cluster.potential_impact,
        )
        
        return cluster


class InnovativeNonSelfIdentificationModule:
    """
    Main module for Phase 2: Innovative Non-self Identification.
    
    Orchestrates the process of:
    1. Retrieving frontier research papers
    2. Analyzing papers from multiple expert perspectives
    3. Identifying innovation clusters
    4. Updating mind map with evolution states
    """
    
    def __init__(
        self,
        retriever: Retriever,
        analysis_lm: dspy.LM,
        top_k_papers: int = 30,
        min_cluster_size: int = 2,
        deviation_threshold: float = 0.5,
    ):
        self.paper_retriever = FrontierPaperRetriever(retriever, top_k=top_k_papers)
        self.expert_generator = ExpertPerspectiveGenerator(analysis_lm)
        self.deviation_analyzer = DifferenceAwareAnalyzer(analysis_lm)
        self.cluster_identifier = InnovationClusterIdentifier(analysis_lm)
        self.min_cluster_size = min_cluster_size
        self.deviation_threshold = deviation_threshold
    
    def identify_innovative_nonself(
        self,
        topic: str,
        cognitive_baseline: CognitiveBaseline,
    ) -> Tuple[List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]], List[InnovationCluster]]:
        """
        Execute Phase 2: Identify innovation gaps by analyzing frontier papers.
        
        Args:
            topic: Research topic
            cognitive_baseline: The cognitive baseline from Phase 1
            
        Returns:
            Tuple of (papers_with_deviations, innovation_clusters)
        """
        logger.info(f"=== Phase 2: Innovative Non-self Identification for '{topic}' ===")
        
        # Step 1: Retrieve frontier papers
        logger.info("Step 1: Retrieving frontier research papers...")
        paper_infos = self.paper_retriever.retrieve_frontier_papers(
            topic,
            cognitive_baseline.temporal_coverage.end if cognitive_baseline.temporal_coverage.end else None,
        )
        logger.info(f"Retrieved {len(paper_infos)} frontier papers")
        
        if not paper_infos:
            logger.warning("No frontier papers found.")
            return [], []
        
        # Step 2: Generate expert perspectives
        logger.info("Step 2: Generating expert perspectives...")
        expert_perspectives = self.expert_generator.generate_expert_perspectives(topic, cognitive_baseline)
        logger.info(f"Generated {len(expert_perspectives)} expert perspectives")
        
        # Step 3: Analyze papers from multiple perspectives
        logger.info("Step 3: Analyzing papers with difference-aware reasoning...")
        papers_with_deviations = []
        for i, paper_info in enumerate(paper_infos, 1):
            try:
                logger.info(f"  Analyzing paper {i}/{len(paper_infos)}: {paper_info.title}")
                paper, deviations = self.deviation_analyzer.analyze_paper(
                    topic,
                    paper_info,
                    cognitive_baseline,
                    expert_perspectives,
                )
                papers_with_deviations.append((paper, deviations))
            except Exception as e:
                logger.error(f"Failed to analyze paper {paper_info.title}: {e}")
                continue
        
        logger.info(f"Successfully analyzed {len(papers_with_deviations)} papers")
        
        # Step 4: Identify innovation clusters
        logger.info("Step 4: Identifying innovation clusters...")
        innovation_clusters = self.cluster_identifier.identify_clusters(
            topic,
            papers_with_deviations,
            min_cluster_size=self.min_cluster_size,
            deviation_threshold=self.deviation_threshold,
        )
        logger.info(f"Identified {len(innovation_clusters)} innovation clusters")
        
        logger.info("=== Phase 2 Complete ===\n")
        return papers_with_deviations, innovation_clusters
