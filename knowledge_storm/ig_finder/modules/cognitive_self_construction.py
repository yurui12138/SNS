"""
Cognitive Self Construction Module

This module implements Phase 1 of IG-Finder: constructing the cognitive baseline
by extracting consensus from review papers.
"""

import dspy
import logging
from typing import List, Dict, Optional
from datetime import datetime

from ...interface import Retriever, Information
from ..dataclass import KnowledgeBase
from ..dataclass import (
    CognitiveBaseline,
    ReviewPaper,
    ResearchParadigm,
    Method,
    Boundary,
    TimeRange,
    ExtendedKnowledgeNode,
    EvolutionState,
)
from ..schemas import ConsensusExtractionResult
from ..parsing import safe_json_loads

logger = logging.getLogger(__name__)


class ReviewRetriever:
    """
    Retrieves high-quality review/survey papers for a given topic.
    
    Strategy:
    - Prioritize papers with "survey", "review", "overview" in title
    - Sort by citations and recency
    - Filter to ensure they are actual review papers, not research articles
    """
    
    def __init__(self, retriever: Retriever, top_k: int = 10):
        self.retriever = retriever
        self.top_k = top_k
    
    def retrieve_reviews(self, topic: str) -> List[Information]:
        """
        Retrieve review papers for the topic.
        
        Args:
            topic: The research topic
            
        Returns:
            List of Information objects containing review papers
        """
        # Construct queries to find review papers
        review_queries = [
            f"{topic} survey",
            f"{topic} review",
            f"{topic} overview",
            f"systematic review of {topic}",
            f"{topic} state of the art",
        ]
        
        logger.info(f"Retrieving review papers for topic: {topic}")
        
        all_results = []
        for query in review_queries:
            results = self.retriever.retrieve(query=query, exclude_urls=[])
            all_results.extend(results)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Filter to keep likely review papers (heuristic: longer descriptions, certain keywords)
        filtered_results = self._filter_review_papers(unique_results)
        
        # Sort by relevance (using a simple heuristic)
        sorted_results = self._sort_by_relevance(filtered_results, topic)
        
        # Return top_k results
        return sorted_results[:self.top_k]
    
    def _filter_review_papers(self, results: List[Information]) -> List[Information]:
        """Filter to keep likely review papers."""
        review_keywords = ['survey', 'review', 'overview', 'comprehensive', 'systematic', 'state-of-the-art']
        
        filtered = []
        for result in results:
            title_lower = result.title.lower()
            description_lower = result.description.lower()
            
            # Check if title contains review keywords
            has_review_keyword = any(keyword in title_lower for keyword in review_keywords)
            
            # Check if description is substantial (reviews typically have longer abstracts)
            is_substantial = len(result.description) > 200
            
            if has_review_keyword or is_substantial:
                filtered.append(result)
        
        return filtered
    
    def _sort_by_relevance(self, results: List[Information], topic: str) -> List[Information]:
        """
        Sort results by relevance score.
        Simple heuristic based on keyword matches and position.
        """
        def relevance_score(info: Information) -> float:
            score = 0.0
            title_lower = info.title.lower()
            topic_lower = topic.lower()
            
            # Title contains exact topic: +10
            if topic_lower in title_lower:
                score += 10.0
            
            # Review keywords in title: +5 each
            review_keywords = ['survey', 'review']
            for keyword in review_keywords:
                if keyword in title_lower:
                    score += 5.0
            
            # Length of description (indicates comprehensive coverage): +0.01 per char
            score += len(info.description) * 0.01
            
            return score
        
        return sorted(results, key=relevance_score, reverse=True)


class ExtractReviewMetadata(dspy.Signature):
    """Extract structured metadata from a review paper."""
    
    title = dspy.InputField(desc="Title of the review paper")
    abstract = dspy.InputField(desc="Abstract or description of the review paper")
    url = dspy.InputField(desc="URL of the review paper")
    
    year = dspy.OutputField(desc="Publication year (estimate if not explicitly stated)")
    authors = dspy.OutputField(desc="List of authors (comma-separated, or 'Unknown' if not available)")
    venue = dspy.OutputField(desc="Publication venue (conference/journal name, or 'Unknown' if not available)")
    key_contributions = dspy.OutputField(desc="List of key contributions mentioned in the abstract (comma-separated)")


class ExtractConsensusFromReview(dspy.Signature):
    """Extract consensus knowledge from a review paper."""
    
    topic = dspy.InputField(desc="The research topic")
    review_title = dspy.InputField(desc="Title of the review paper")
    review_content = dspy.InputField(desc="Content of the review paper (abstract and available snippets)")
    
    field_development_history = dspy.OutputField(desc="Brief history of the field's development")
    research_paradigms = dspy.OutputField(desc="Strict JSON array: [{\"name\":...,\"description\":...,\"time_period\":...}]")
    mainstream_methods = dspy.OutputField(desc="Strict JSON array: [{\"name\":...,\"description\":...,\"category\":...,\"advantages\":[],\"limitations\":[]}]")
    knowledge_boundaries = dspy.OutputField(desc="Strict JSON array: [{\"dimension\":...,\"description\":...,\"known_limits\":[],\"open_questions\":[]}]")
    key_concepts_hierarchy = dspy.OutputField(desc="Strict JSON object: {\"Concept\":{\"description\":...,\"subconcepts\":[...]}}")


class ConsensusExtractor:
    """
    Extracts structured consensus knowledge from review papers.
    Uses LLM to parse review content and identify:
    - Field development history
    - Research paradigms
    - Mainstream methodologies
    - Knowledge boundaries
    - Concept hierarchies
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.metadata_extractor = dspy.ChainOfThought(ExtractReviewMetadata)
        self.consensus_extractor = dspy.ChainOfThought(ExtractConsensusFromReview)
    
    def extract_from_review(self, topic: str, review_info: Information) -> ReviewPaper:
        """
        Extract consensus from a single review paper.
        
        Args:
            topic: The research topic
            review_info: Information object containing review paper data
            
        Returns:
            ReviewPaper object with extracted consensus
        """
        logger.info(f"Extracting consensus from review: {review_info.title}")
        
        # Extract metadata
        with dspy.context(lm=self.lm):
            metadata_result = self.metadata_extractor(
                title=review_info.title,
                abstract=review_info.description,
                url=review_info.url,
            )
        
        # Parse metadata
        try:
            year = int(metadata_result.year.strip())
        except:
            year = datetime.now().year
        
        authors = [a.strip() for a in metadata_result.authors.split(',')] if metadata_result.authors != 'Unknown' else []
        venue = metadata_result.venue if metadata_result.venue != 'Unknown' else ""
        key_contributions = [c.strip() for c in metadata_result.key_contributions.split(',')]
        
        # Extract consensus knowledge
        review_content = f"{review_info.description}\n\n" + "\n".join(review_info.snippets[:5])
        
        with dspy.context(lm=self.lm):
            consensus_result = self.consensus_extractor(
                topic=topic,
                review_title=review_info.title,
                review_content=review_content,
            )
        
        research_paradigms_data = safe_json_loads(consensus_result.research_paradigms)
        mainstream_methods_data = safe_json_loads(consensus_result.mainstream_methods)
        knowledge_boundaries_data = safe_json_loads(consensus_result.knowledge_boundaries)
        key_concepts_hierarchy = safe_json_loads(consensus_result.key_concepts_hierarchy)

        extracted_consensus = {
            "field_development_history": consensus_result.field_development_history,
            "research_paradigms": research_paradigms_data if isinstance(research_paradigms_data, list) else [],
            "mainstream_methods": mainstream_methods_data if isinstance(mainstream_methods_data, list) else [],
            "knowledge_boundaries": knowledge_boundaries_data if isinstance(knowledge_boundaries_data, list) else [],
            "key_concepts_hierarchy": key_concepts_hierarchy if isinstance(key_concepts_hierarchy, dict) else {},
        }

        validated = ConsensusExtractionResult(**extracted_consensus).model_dump()
        
        review_paper = ReviewPaper(
            title=review_info.title,
            authors=authors,
            year=year,
            url=review_info.url,
            abstract=review_info.description,
            citations=0,  # We don't have citation count from retrieval
            venue=venue,
            key_contributions=key_contributions,
            extracted_consensus=validated,
        )
        
        return review_paper
    
    def extract_from_reviews(self, topic: str, review_infos: List[Information]) -> List[ReviewPaper]:
        """Extract consensus from multiple review papers."""
        review_papers = []
        for review_info in review_infos:
            try:
                review_paper = self.extract_from_review(topic, review_info)
                review_papers.append(review_paper)
            except Exception as e:
                logger.error(f"Failed to extract consensus from {review_info.title}: {e}")
                continue
        
        return review_papers


class CognitiveBaselineBuilder:
    """
    Builds the cognitive baseline by:
    1. Aggregating extracted consensus from multiple reviews
    2. Organizing consensus into a dynamic mind map (KnowledgeBase)
    3. Marking all nodes with CONSENSUS evolution state
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
    
    def build_baseline(self, topic: str, review_papers: List[ReviewPaper]) -> CognitiveBaseline:
        """
        Build cognitive baseline from extracted review papers.
        
        Args:
            topic: The research topic
            review_papers: List of ReviewPaper objects with extracted consensus
            
        Returns:
            CognitiveBaseline object
        """
        logger.info(f"Building cognitive baseline for topic: {topic}")
        
        # Aggregate consensus from all reviews
        all_paradigms = []
        all_methods = []
        all_boundaries = {}
        timeline_events = []
        
        for review in review_papers:
            consensus = review.extracted_consensus
            
            # Collect paradigms
            for paradigm_data in consensus.get("research_paradigms", []):
                paradigm = ResearchParadigm(
                    name=paradigm_data.get("name", ""),
                    description=paradigm_data.get("description", ""),
                    representative_works=paradigm_data.get("representative_works", []),
                    time_period=None,  # Could parse from time_period field if available
                )
                all_paradigms.append(paradigm)
            
            # Collect methods
            for method_data in consensus.get("mainstream_methods", []):
                method = Method(
                    name=method_data.get("name", ""),
                    description=method_data.get("description", ""),
                    category=method_data.get("category", ""),
                    advantages=method_data.get("advantages", []) if isinstance(method_data.get("advantages"), list) else [],
                    limitations=method_data.get("limitations", []) if isinstance(method_data.get("limitations"), list) else [],
                )
                all_methods.append(method)
            
            # Collect boundaries
            for boundary_data in consensus.get("knowledge_boundaries", []):
                dimension = boundary_data.get("dimension", "Unknown")
                boundary = Boundary(
                    dimension=dimension,
                    description=boundary_data.get("description", ""),
                    known_limits=boundary_data.get("known_limits", []) if isinstance(boundary_data.get("known_limits"), list) else [],
                    open_questions=boundary_data.get("open_questions", []) if isinstance(boundary_data.get("open_questions"), list) else [],
                )
                all_boundaries[dimension] = boundary
        
        # Build mind map from concept hierarchies
        consensus_map = self._build_consensus_mind_map(topic, review_papers)
        
        # Determine temporal coverage
        if review_papers:
            min_year = min(r.year for r in review_papers)
            max_year = max(r.year for r in review_papers)
            temporal_coverage = TimeRange(
                start=datetime(min_year, 1, 1),
                end=datetime(max_year, 12, 31),
            )
        else:
            temporal_coverage = TimeRange()
        
        cognitive_baseline = CognitiveBaseline(
            topic=topic,
            review_papers=review_papers,
            consensus_map=consensus_map,
            research_paradigms=all_paradigms,
            mainstream_methods=all_methods,
            knowledge_boundaries=all_boundaries,
            temporal_coverage=temporal_coverage,
            field_evolution_timeline=timeline_events,
        )
        
        return cognitive_baseline
    
    def _build_consensus_mind_map(self, topic: str, review_papers: List[ReviewPaper]) -> KnowledgeBase:
        """
        Build a KnowledgeBase (mind map) from consensus extracted from reviews.
        All nodes are marked with CONSENSUS evolution state.
        """
        root = ExtendedKnowledgeNode(
            name=topic,
            evolution_state=EvolutionState.CONSENSUS,
            source_papers=[r.url for r in review_papers],
        )
        knowledge_base = KnowledgeBase(topic=topic)
        knowledge_base.root = root
        
        # Aggregate all concept hierarchies from reviews
        all_hierarchies = []
        for review in review_papers:
            hierarchy = review.extracted_consensus.get("key_concepts_hierarchy", {})
            if hierarchy:
                all_hierarchies.append((review.url, hierarchy))
        
        # Build tree structure
        for review_url, hierarchy in all_hierarchies:
            for concept_name, concept_data in hierarchy.items():
                # Add first-level concept
                if not isinstance(concept_data, dict):
                    continue
                
                description = concept_data.get("description", "")
                
                # Find or create node
                existing_node = None
                for child in root.children:
                    if child.name == concept_name:
                        existing_node = child
                        break
                
                if existing_node:
                    # Add source paper to existing node
                    if review_url not in existing_node.source_papers:
                        existing_node.source_papers.append(review_url)
                else:
                    # Create new node
                    new_node = ExtendedKnowledgeNode(
                        name=concept_name,
                        parent=root,
                        evolution_state=EvolutionState.CONSENSUS,
                        source_papers=[review_url],
                    )
                    new_node.synthesize_output = description
                    root.children.append(new_node)
                    existing_node = new_node
                
                # Add subconcepts
                subconcepts = concept_data.get("subconcepts", [])
                if isinstance(subconcepts, list):
                    for subconcept in subconcepts:
                        if isinstance(subconcept, str):
                            subconcept_name = subconcept
                            subconcept_desc = ""
                        elif isinstance(subconcept, dict):
                            subconcept_name = subconcept.get("name", "")
                            subconcept_desc = subconcept.get("description", "")
                        else:
                            continue
                        
                        # Check if subconcept already exists
                        sub_existing = None
                        for child in existing_node.children:
                            if child.name == subconcept_name:
                                sub_existing = child
                                break
                        
                        if not sub_existing:
                            sub_node = ExtendedKnowledgeNode(
                                name=subconcept_name,
                                parent=existing_node,
                                evolution_state=EvolutionState.CONSENSUS,
                                source_papers=[review_url],
                            )
                            sub_node.synthesize_output = subconcept_desc
                            existing_node.children.append(sub_node)
        
        return knowledge_base


class CognitiveSelfConstructionModule:
    """
    Main module for Phase 1: Cognitive Self Construction.
    
    Orchestrates the process of:
    1. Retrieving review papers
    2. Extracting consensus knowledge
    3. Building the cognitive baseline with dynamic mind map
    """
    
    def __init__(
        self,
        retriever: Retriever,
        consensus_extraction_lm: dspy.LM,
        top_k_reviews: int = 10,
    ):
        self.review_retriever = ReviewRetriever(retriever, top_k=top_k_reviews)
        self.consensus_extractor = ConsensusExtractor(consensus_extraction_lm)
        self.baseline_builder = CognitiveBaselineBuilder(consensus_extraction_lm)
    
    def construct_cognitive_self(self, topic: str) -> CognitiveBaseline:
        """
        Execute Phase 1: Construct cognitive baseline from review papers.
        
        Args:
            topic: The research topic
            
        Returns:
            CognitiveBaseline object representing the "self" in immune system metaphor
        """
        logger.info(f"=== Phase 1: Cognitive Self Construction for '{topic}' ===")
        
        # Step 1: Retrieve review papers
        logger.info("Step 1: Retrieving review papers...")
        review_infos = self.review_retriever.retrieve_reviews(topic)
        logger.info(f"Retrieved {len(review_infos)} review papers")
        
        if not review_infos:
            logger.warning("No review papers found. Creating empty baseline.")
            return CognitiveBaseline(
                topic=topic,
                review_papers=[],
                consensus_map=KnowledgeBase(topic=topic),
                research_paradigms=[],
                mainstream_methods=[],
                knowledge_boundaries={},
                temporal_coverage=TimeRange(),
                field_evolution_timeline=[],
            )
        
        # Step 2: Extract consensus from reviews
        logger.info("Step 2: Extracting consensus from reviews...")
        review_papers = self.consensus_extractor.extract_from_reviews(topic, review_infos)
        logger.info(f"Successfully extracted consensus from {len(review_papers)} reviews")
        
        # Step 3: Build cognitive baseline
        logger.info("Step 3: Building cognitive baseline...")
        cognitive_baseline = self.baseline_builder.build_baseline(topic, review_papers)
        logger.info(f"Cognitive baseline constructed with {len(cognitive_baseline.consensus_map.root.children)} top-level concepts")
        
        logger.info("=== Phase 1 Complete ===\n")
        return cognitive_baseline
