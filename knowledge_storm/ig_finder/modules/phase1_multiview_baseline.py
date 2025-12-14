"""
Phase 1: Multi-view Baseline Construction

This module implements the construction of multi-view cognitive baseline
from review papers.
"""
import logging
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
import math

from ...interface import Retriever, Information
from ..dataclass import ReviewPaper
from ..dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    TaxonomyTree,
    TaxonomyTreeNode,
    NodeDefinition,
    EvidenceSpan,
    FacetLabel,
)
from ..schemas_v2 import create_taxonomy_extractor, create_node_definition_builder
from ..parsing import safe_json_loads

logger = logging.getLogger(__name__)


class ReviewRetriever:
    """
    Retrieves high-quality review/survey papers for a given topic.
    
    Implementation follows the design document's heuristic rules:
    - Prioritize papers with "survey", "review", "overview" in title
    - Sort by quality proxies (citations, recency, venue)
    - Filter to ensure they are actual review papers
    """
    
    def __init__(self, retriever: Retriever, top_k: int = 15):
        self.retriever = retriever
        self.top_k = top_k
        self.review_keywords = ['survey', 'review', 'overview', 'comprehensive', 
                               'systematic', 'state-of-the-art', 'tutorial']
    
    def retrieve_reviews(self, topic: str) -> List[Information]:
        """
        Retrieve review papers for the topic.
        
        Args:
            topic: The research topic
            
        Returns:
            List of Information objects containing review papers
        """
        logger.info(f"Retrieving review papers for topic: {topic}")
        
        # Construct queries to find review papers
        review_queries = [
            f"{topic} survey",
            f"{topic} review",
            f"{topic} overview",
            f"systematic review of {topic}",
        ]
        
        all_results = []
        for query in review_queries:
            try:
                results = self.retriever.retrieve(query=query, exclude_urls=[])
                all_results.extend(results)
                logger.info(f"Query '{query}' returned {len(results)} results")
            except Exception as e:
                logger.warning(f"Error retrieving with query '{query}': {e}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        logger.info(f"Found {len(unique_results)} unique results")
        
        # Filter to keep likely review papers
        filtered_results = self._filter_review_papers(unique_results)
        logger.info(f"After filtering: {len(filtered_results)} review papers")
        
        # Sort by quality
        sorted_results = self._sort_by_quality(filtered_results, topic)
        
        # Return top_k results
        return sorted_results[:self.top_k]
    
    def _filter_review_papers(self, results: List[Information]) -> List[Information]:
        """
        Filter to keep likely review papers using heuristic rules.
        
        Rules:
        1. Title contains review keywords
        2. Abstract is substantial (> 120 words)
        3. Not a research paper (heuristic: no method-specific terms in title)
        """
        research_indicators = ['novel', 'new', 'improved', 'propose', 'we introduce']
        
        filtered = []
        for result in results:
            title_lower = result.title.lower()
            description_lower = result.description.lower()
            
            # Check if title contains review keywords
            has_review_keyword = any(keyword in title_lower for keyword in self.review_keywords)
            
            # Check if description is substantial
            word_count = len(description_lower.split())
            is_substantial = word_count > 120
            
            # Check if it's not a research paper (heuristic)
            has_research_indicator = any(ind in title_lower for ind in research_indicators)
            
            if has_review_keyword and is_substantial and not has_research_indicator:
                filtered.append(result)
        
        return filtered
    
    def _sort_by_quality(self, reviews: List[Information], topic: str) -> List[Information]:
        """
        Sort reviews by quality proxy.
        
        Quality score = 0.4 * recency + 0.4 * citation_proxy + 0.2 * relevance
        
        Since we may not have citation counts, we use heuristic proxies:
        - Recency: Prefer recent papers (extract year from snippets)
        - Citation proxy: Presence in high-quality venues or well-formatted citations
        - Relevance: Topic keywords in title and abstract
        """
        scored_reviews = []
        
        for review in reviews:
            # Recency score
            year = self._extract_year(review.snippets)
            if year:
                current_year = datetime.now().year
                years_old = current_year - year
                recency_score = math.exp(-0.15 * years_old)  # α = 0.15 as per design doc
            else:
                recency_score = 0.5  # Default if year not found
            
            # Citation proxy (heuristic)
            citation_score = 0.5  # Default
            # Could be enhanced with actual citation data if available
            
            # Relevance score (keyword overlap)
            topic_keywords = set(topic.lower().split())
            title_keywords = set(review.title.lower().split())
            desc_keywords = set(review.description.lower().split())
            
            title_overlap = len(topic_keywords & title_keywords) / max(len(topic_keywords), 1)
            desc_overlap = len(topic_keywords & desc_keywords) / max(len(topic_keywords), 1)
            relevance_score = 0.7 * title_overlap + 0.3 * desc_overlap
            
            # Combined score
            quality_score = (
                0.4 * recency_score +
                0.4 * citation_score +
                0.2 * relevance_score
            )
            
            scored_reviews.append((review, quality_score))
        
        # Sort by quality score descending
        scored_reviews.sort(key=lambda x: x[1], reverse=True)
        
        return [review for review, _ in scored_reviews]
    
    def _extract_year(self, snippets: List[str]) -> Optional[int]:
        """Extract publication year from snippets."""
        for snippet in snippets:
            # Look for 4-digit years between 1990 and current year
            match = re.search(r'\b(19|20)\d{2}\b', snippet)
            if match:
                year = int(match.group())
                current_year = datetime.now().year
                if 1990 <= year <= current_year:
                    return year
        return None


class TaxonomyViewExtractor:
    """
    Extracts taxonomy view from a review paper.
    
    Uses LLM with fixed JSON schema (temperature=0) for reproducibility.
    """
    
    def __init__(self, lm):
        self.lm = lm
        self.extractor = create_taxonomy_extractor(lm)
    
    def extract_view(self, review: Information, view_id: str) -> Optional[TaxonomyView]:
        """
        Extract taxonomy view from a review paper.
        
        Args:
            review: Information object containing review paper
            view_id: Unique ID for this view (e.g., "T1", "T2")
            
        Returns:
            TaxonomyView object or None if extraction fails
        """
        logger.info(f"Extracting taxonomy view {view_id} from review: {review.title}")
        
        try:
            # Prepare input (limit text to avoid token limits)
            review_text = " ".join(review.snippets)[:10000]
            
            # Call LLM with temperature=0
            result = self.extractor(
                review_title=review.title,
                review_abstract=review.description,
                review_text=review_text
            )
            
            # Parse JSON output
            taxonomy_data = safe_json_loads(result.taxonomy_json)
            
            if not taxonomy_data:
                logger.warning(f"Failed to parse JSON for view {view_id}")
                return None
            
            # Build TaxonomyView
            view = self._build_taxonomy_view(taxonomy_data, review, view_id)
            
            logger.info(f"Successfully extracted view {view_id}: {view.facet_label.value}")
            return view
            
        except Exception as e:
            logger.error(f"Error extracting view {view_id}: {e}", exc_info=True)
            return None
    
    def _build_taxonomy_view(self, data: Dict, review: Information, view_id: str) -> TaxonomyView:
        """Build TaxonomyView from parsed JSON data."""
        
        # Parse facet label
        try:
            facet_label = FacetLabel[data.get("facet_label", "OTHER")]
        except KeyError:
            facet_label = FacetLabel.OTHER
        
        # Build taxonomy tree
        tree = self._build_tree(data.get("taxonomy_tree", {"name": "ROOT", "children": []}))
        
        # Parse evidence spans
        evidence = [
            EvidenceSpan(
                claim=e.get("claim", ""),
                page=e.get("page", 0),
                section=e.get("section", ""),
                char_start=e.get("char_start", 0),
                char_end=e.get("char_end", 0),
                quote=e.get("quote", "")
            )
            for e in data.get("evidence_spans", [])
        ]
        
        return TaxonomyView(
            view_id=view_id,
            review_id=review.url,
            review_title=review.title,
            facet_label=facet_label,
            facet_rationale=data.get("facet_rationale", ""),
            tree=tree,
            node_definitions={},  # Will be filled by NodeDefinitionBuilder
            weight=1.0,  # Will be calculated later
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


class NodeDefinitionBuilder:
    """
    Builds testable definitions for each node in taxonomy tree.
    
    Uses LLM with fixed JSON schema (temperature=0).
    """
    
    def __init__(self, lm):
        self.lm = lm
        self.builder = create_node_definition_builder(lm)
    
    def build_definitions(self, view: TaxonomyView, review: Information) -> Dict[str, NodeDefinition]:
        """
        Build definitions for all nodes in the taxonomy tree.
        
        Args:
            view: TaxonomyView to build definitions for
            review: Original review paper for context
            
        Returns:
            Dictionary mapping node_path -> NodeDefinition
        """
        logger.info(f"Building node definitions for view {view.view_id}")
        
        node_definitions = {}
        review_text = " ".join(review.snippets)
        
        # Build definitions for all nodes (including internal nodes)
        for node_path, node in view.tree.nodes.items():
            if node.name == "ROOT":
                continue  # Skip root
            
            try:
                definition = self._build_node_definition(node, view.tree, review_text)
                if definition:
                    node_definitions[node_path] = definition
            except Exception as e:
                logger.warning(f"Failed to build definition for node {node_path}: {e}")
        
        logger.info(f"Built {len(node_definitions)} node definitions for view {view.view_id}")
        return node_definitions
    
    def _build_node_definition(
        self, 
        node: TaxonomyTreeNode, 
        tree: TaxonomyTree,
        review_text: str
    ) -> Optional[NodeDefinition]:
        """Build definition for a single node."""
        
        # Get parent definition if exists
        parent_def = ""
        if node.parent and node.parent in tree.nodes:
            parent_node = tree.nodes[node.parent]
            parent_def = f"Parent: {parent_node.name}"
        
        # Extract relevant context from review (heuristic: search for node name)
        context = self._extract_context(node.name, review_text)
        
        try:
            # Call LLM
            result = self.builder(
                node_name=node.name,
                node_path=node.path,
                review_context=context,
                parent_definition=parent_def
            )
            
            # Parse JSON
            def_data = safe_json_loads(result.definition_json)
            
            if not def_data:
                return None
            
            # Build NodeDefinition
            evidence_spans = [
                EvidenceSpan(
                    claim=e.get("claim", ""),
                    page=e.get("page", 0),
                    section=e.get("section", ""),
                    char_start=e.get("char_start", 0),
                    char_end=e.get("char_end", 0),
                    quote=e.get("quote", "")
                )
                for e in def_data.get("evidence_spans", [])
            ]
            
            return NodeDefinition(
                node_path=def_data.get("node_path", node.path),
                definition=def_data.get("definition", ""),
                inclusion_criteria=def_data.get("inclusion_criteria", []),
                exclusion_criteria=def_data.get("exclusion_criteria", []),
                canonical_keywords=def_data.get("canonical_keywords", []),
                boundary_statements=def_data.get("boundary_statements", []),
                evidence_spans=evidence_spans
            )
            
        except Exception as e:
            logger.error(f"Error building definition for {node.path}: {e}")
            return None
    
    def _extract_context(self, node_name: str, review_text: str, window: int = 500) -> str:
        """
        Extract relevant context mentioning the node name.
        
        Simple heuristic: find sentences containing node name and surrounding text.
        """
        # Find all occurrences of node name (case insensitive)
        node_name_lower = node_name.lower()
        text_lower = review_text.lower()
        
        contexts = []
        pos = 0
        while True:
            pos = text_lower.find(node_name_lower, pos)
            if pos == -1:
                break
            
            # Extract window around match
            start = max(0, pos - window)
            end = min(len(review_text), pos + len(node_name) + window)
            contexts.append(review_text[start:end])
            
            pos += len(node_name)
        
        if contexts:
            return " ... ".join(contexts[:3])  # Max 3 contexts
        else:
            # Fallback: return first part of review
            return review_text[:1000]


class MultiViewBaselineBuilder:
    """
    Builds multi-view baseline by aggregating views and calculating weights.
    
    Implements weight calculation from design doc:
    w_i ∝ Quality(r_i) · Recency(r_i) · Coverage(r_i)
    """
    
    def __init__(self):
        pass
    
    def build_baseline(
        self, 
        topic: str, 
        views: List[TaxonomyView], 
        reviews: List[Information]
    ) -> MultiViewBaseline:
        """
        Build multi-view baseline from extracted views.
        
        Args:
            topic: Research topic
            views: List of TaxonomyView objects
            reviews: Original review papers (for weight calculation)
            
        Returns:
            MultiViewBaseline object
        """
        logger.info(f"Building multi-view baseline for topic: {topic}")
        logger.info(f"Number of views: {len(views)}")
        
        # Calculate weights for each view
        review_map = {r.url: r for r in reviews}
        
        for view in views:
            review = review_map.get(view.review_id)
            if review:
                view.weight = self._calculate_weight(review, view)
            else:
                view.weight = 1.0
        
        # Create baseline (normalization happens in __post_init__)
        baseline = MultiViewBaseline(
            topic=topic,
            views=views
        )
        
        logger.info(f"Multi-view baseline created with {len(baseline.views)} views")
        logger.info(f"View weights: {[f'{v.view_id}={v.weight:.3f}' for v in baseline.views]}")
        
        return baseline
    
    def _calculate_weight(self, review: Information, view: TaxonomyView) -> float:
        """
        Calculate weight for a view based on review quality.
        
        Formula: w ∝ Quality · Recency · Coverage
        - Recency: exp(-0.15 * (Y_now - Y_i))
        - Quality: log(1 + citations) (heuristic if citations not available)
        - Coverage: number of leaf nodes / 50 (normalized)
        """
        # Recency
        year = self._extract_year_from_snippets(review.snippets)
        if year:
            current_year = datetime.now().year
            years_old = current_year - year
            recency = math.exp(-0.15 * years_old)
        else:
            recency = 0.5  # Default
        
        # Quality (heuristic: use length as proxy if citations not available)
        # In real implementation, would use citation data from Semantic Scholar API
        quality = min(1.0, len(review.description) / 500.0)
        
        # Coverage
        num_leaves = len(view.tree.get_leaf_nodes())
        coverage = min(1.0, num_leaves / 50.0)
        
        # Combined weight
        raw_weight = recency * quality * coverage
        
        return raw_weight
    
    def _extract_year_from_snippets(self, snippets: List[str]) -> Optional[int]:
        """Extract year from snippets."""
        for snippet in snippets:
            match = re.search(r'\b(19|20)\d{2}\b', snippet)
            if match:
                year = int(match.group())
                current_year = datetime.now().year
                if 1990 <= year <= current_year:
                    return year
        return None


# ============================================================================
# Main Phase 1 Pipeline
# ============================================================================

class Phase1Pipeline:
    """
    Complete Phase 1 pipeline: Multi-view Baseline Construction.
    """
    
    def __init__(self, retriever: Retriever, lm, top_k_reviews: int = 15):
        self.review_retriever = ReviewRetriever(retriever, top_k=top_k_reviews)
        self.taxonomy_extractor = TaxonomyViewExtractor(lm)
        self.node_builder = NodeDefinitionBuilder(lm)
        self.baseline_builder = MultiViewBaselineBuilder()
    
    def run(self, topic: str) -> MultiViewBaseline:
        """
        Run complete Phase 1 pipeline.
        
        Args:
            topic: Research topic
            
        Returns:
            MultiViewBaseline object
        """
        logger.info("="*80)
        logger.info("PHASE 1: Multi-view Baseline Construction")
        logger.info("="*80)
        
        # Step 1: Retrieve reviews
        logger.info("Step 1: Retrieving review papers...")
        reviews = self.review_retriever.retrieve_reviews(topic)
        logger.info(f"Retrieved {len(reviews)} review papers")
        
        # Step 2: Extract taxonomy views
        logger.info("Step 2: Extracting taxonomy views...")
        views = []
        for i, review in enumerate(reviews):
            view_id = f"T{i+1}"
            view = self.taxonomy_extractor.extract_view(review, view_id)
            if view:
                # Step 3: Build node definitions for this view
                logger.info(f"Step 3: Building node definitions for view {view_id}...")
                node_defs = self.node_builder.build_definitions(view, review)
                view.node_definitions = node_defs
                views.append(view)
        
        logger.info(f"Successfully extracted {len(views)} views")
        
        # Step 4: Build multi-view baseline
        logger.info("Step 4: Building multi-view baseline...")
        baseline = self.baseline_builder.build_baseline(topic, views, reviews)
        
        logger.info("Phase 1 completed successfully!")
        logger.info("="*80)
        
        return baseline
