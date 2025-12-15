"""
Compensatory View Inducer

Induces additional taxonomy views from research papers when baseline quality is insufficient.
Implements the compensatory view strategy from the SNS method specification.
"""
import logging
import numpy as np
from typing import List, Optional, Dict, Set
from collections import Counter

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available. Install with: pip install hdbscan")

from ...interface import Information
from ..dataclass_v2 import (
    MultiViewBaseline, TaxonomyView, TaxonomyTree, TaxonomyTreeNode,
    NodeDefinition, EvidenceSpan, FacetLabel
)

logger = logging.getLogger(__name__)


class CompensatoryViewInducer:
    """
    Induces compensatory views when baseline quality is insufficient.
    
    Strategy:
    1. Cluster research papers using embeddings (HDBSCAN)
    2. Generate theme labels for each cluster
    3. Build flat taxonomy tree (root + leaf nodes)
    4. Select unique facet label (avoid conflicts with existing views)
    5. Create TaxonomyView with lower weight (0.5)
    
    Triggers when:
    - unique(facet) < min_facet_count (default 2)
    - OR dominant facet > max_dominant_ratio (default 0.6)
    """
    
    def __init__(self, embedder, lm, min_cluster_size: int = 3):
        """
        Initialize compensatory view inducer.
        
        Args:
            embedder: Embedding model for clustering papers
            lm: Language model for generating labels
            min_cluster_size: Minimum cluster size for HDBSCAN
        """
        self.embedder = embedder
        self.lm = lm
        self.min_cluster_size = min_cluster_size
    
    def should_induce(
        self, 
        baseline: MultiViewBaseline, 
        min_facet_count: int = 2,
        max_dominant_ratio: float = 0.6
    ) -> bool:
        """
        Check if compensatory view induction is needed.
        
        Args:
            baseline: Current multi-view baseline
            min_facet_count: Minimum required unique facets
            max_dominant_ratio: Maximum allowed ratio for dominant facet
            
        Returns:
            True if induction is needed
        """
        if not baseline.views:
            return True
        
        facet_counts = Counter([v.facet_label for v in baseline.views])
        
        # Condition 1: Not enough unique facets
        if len(facet_counts) < min_facet_count:
            logger.info(f"Compensatory view needed: only {len(facet_counts)} unique facets (min: {min_facet_count})")
            return True
        
        # Condition 2: Dominant facet
        for facet, count in facet_counts.items():
            ratio = count / len(baseline.views)
            if ratio > max_dominant_ratio:
                logger.info(f"Compensatory view needed: facet {facet.value} dominates with {ratio:.1%} (max: {max_dominant_ratio:.0%})")
                return True
        
        return False
    
    def induce_view(
        self,
        baseline: MultiViewBaseline,
        papers: List[Information],
        topic: str
    ) -> Optional[TaxonomyView]:
        """
        Induce a compensatory view from research papers.
        
        Args:
            baseline: Current baseline
            papers: Research papers to cluster
            topic: Research topic
            
        Returns:
            TaxonomyView or None if induction fails
        """
        if not papers:
            logger.warning("No papers provided for compensatory view induction")
            return None
        
        logger.info(f"Inducing compensatory view from {len(papers)} papers...")
        
        # Step 1: Cluster papers
        clusters = self._cluster_papers(papers)
        
        if not clusters:
            logger.warning("No clusters found, cannot induce view")
            return None
        
        logger.info(f"Found {len(clusters)} clusters")
        
        # Step 2: Generate labels for each cluster
        cluster_labels = self._generate_cluster_labels(clusters, topic)
        
        # Step 3: Build induced tree
        induced_tree = self._build_induced_tree(cluster_labels)
        
        # Step 4: Select unique facet
        used_facets = {v.facet_label for v in baseline.views}
        new_facet = self._select_unique_facet(used_facets)
        
        # Step 5: Create view
        view_id = f"T_induced_{len(baseline.views) + 1}"
        
        compensatory_view = TaxonomyView(
            view_id=view_id,
            review_id="INDUCED_FROM_PAPERS",
            review_title=f"Induced View: {new_facet.value} (from {len(papers)} papers)",
            facet_label=new_facet,
            facet_rationale=(
                f"Compensatory view induced from paper clustering to ensure baseline quality. "
                f"Represents emerging {new_facet.value.lower()} dimension discovered in {len(clusters)} clusters."
            ),
            tree=induced_tree,
            node_definitions={},  # Will be populated next
            weight=0.5,  # Slightly lower than normal reviews
            evidence=[]
        )
        
        # Step 6: Build node definitions
        compensatory_view.node_definitions = self._build_node_definitions(
            induced_tree, clusters, cluster_labels
        )
        
        logger.info(f"Successfully induced view {view_id} with {len(cluster_labels)} categories")
        
        return compensatory_view
    
    def _cluster_papers(self, papers: List[Information]) -> List[List[Information]]:
        """
        Cluster papers using HDBSCAN on embeddings.
        
        Args:
            papers: List of papers
            
        Returns:
            List of clusters (each cluster is a list of papers)
        """
        logger.info("Clustering papers using embeddings...")
        
        # Extract text for embedding
        texts = [f"{p.title} {p.description}" for p in papers]
        
        # Generate embeddings
        try:
            embeddings = self.embedder.encode(texts)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Fallback: simple split
            n_per_cluster = max(self.min_cluster_size, len(papers) // 3)
            return [papers[i:i+n_per_cluster] for i in range(0, len(papers), n_per_cluster)]
        
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, using simple fallback clustering")
            # Simple fallback: split into 3 groups
            n_per_cluster = max(self.min_cluster_size, len(papers) // 3)
            clusters = [papers[i:i+n_per_cluster] for i in range(0, len(papers), n_per_cluster)]
            return [c for c in clusters if len(c) >= self.min_cluster_size]
        
        # HDBSCAN clustering
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            labels = clusterer.fit_predict(embeddings)
            
            # Group by cluster
            cluster_map = {}
            for paper, label in zip(papers, labels):
                if label != -1:  # Ignore noise
                    if label not in cluster_map:
                        cluster_map[label] = []
                    cluster_map[label].append(paper)
            
            clusters = list(cluster_map.values())
            logger.info(f"HDBSCAN found {len(clusters)} clusters (noise filtered)")
            
            return clusters
            
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}, using fallback")
            # Fallback
            n_per_cluster = max(self.min_cluster_size, len(papers) // 3)
            clusters = [papers[i:i+n_per_cluster] for i in range(0, len(papers), n_per_cluster)]
            return [c for c in clusters if len(c) >= self.min_cluster_size]
    
    def _generate_cluster_labels(
        self, 
        clusters: List[List[Information]],
        topic: str
    ) -> List[str]:
        """
        Generate descriptive labels for each cluster.
        
        Uses keyword extraction as a simple approach.
        In production, could use LLM for better labels.
        
        Args:
            clusters: List of paper clusters
            topic: Research topic
            
        Returns:
            List of cluster labels
        """
        labels = []
        
        for i, cluster in enumerate(clusters):
            # Extract keywords from cluster papers
            keywords = self._extract_cluster_keywords(cluster)
            
            if keywords:
                # Use top keywords as label
                label = "_".join(keywords[:2])
            else:
                label = f"Category_{i+1}"
            
            labels.append(label)
        
        return labels
    
    def _extract_cluster_keywords(self, cluster: List[Information]) -> List[str]:
        """
        Extract representative keywords from a cluster.
        
        Args:
            cluster: List of papers
            
        Returns:
            List of keywords
        """
        # Collect all words from titles
        words = []
        for paper in cluster:
            words.extend(paper.title.lower().split())
        
        # Filter stop words (simple list)
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 
            'or', 'with', 'from', 'by', 'using', 'based', 'via', 'through'
        }
        
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(10)]
    
    def _build_induced_tree(self, cluster_labels: List[str]) -> TaxonomyTree:
        """
        Build flat taxonomy tree (root + leaf nodes).
        
        Args:
            cluster_labels: Labels for each cluster
            
        Returns:
            TaxonomyTree
        """
        # Create root
        root = TaxonomyTreeNode(
            name="Induced_Root",
            path="Induced_Root",
            parent=None,
            children=[],
            is_leaf=False
        )
        
        tree = TaxonomyTree(root=root)
        
        # Add leaf nodes for each cluster
        for label in cluster_labels:
            leaf_path = f"Induced_Root/{label}"
            leaf_node = TaxonomyTreeNode(
                name=label,
                path=leaf_path,
                parent="Induced_Root",
                children=[],
                is_leaf=True
            )
            tree.add_node(leaf_node)
        
        return tree
    
    def _select_unique_facet(self, used_facets: Set[FacetLabel]) -> FacetLabel:
        """
        Select a facet label not used in baseline.
        
        Args:
            used_facets: Set of already used facets
            
        Returns:
            FacetLabel
        """
        all_facets = list(FacetLabel)
        
        # Try to find unused facet (excluding OTHER)
        for facet in all_facets:
            if facet not in used_facets and facet != FacetLabel.OTHER:
                return facet
        
        # If all used, return OTHER or least used
        if FacetLabel.OTHER not in used_facets:
            return FacetLabel.OTHER
        
        # Return any (will be duplicate, but better than nothing)
        return all_facets[0]
    
    def _build_node_definitions(
        self,
        tree: TaxonomyTree,
        clusters: List[List[Information]],
        labels: List[str]
    ) -> Dict[str, NodeDefinition]:
        """
        Build node definitions for induced tree.
        
        Args:
            tree: Induced taxonomy tree
            clusters: Paper clusters
            labels: Cluster labels
            
        Returns:
            Dictionary of node_path -> NodeDefinition
        """
        node_defs = {}
        
        for label, cluster in zip(labels, clusters):
            node_path = f"Induced_Root/{label}"
            
            # Extract keywords
            keywords = self._extract_cluster_keywords(cluster)
            
            # Build definition
            definition = f"Papers related to {label.replace('_', ' ')}"
            
            # Generate inclusion criteria (at least 3)
            inclusion = []
            for kw in keywords[:5]:
                inclusion.append(f"Papers focusing on {kw}")
            
            # Pad to at least 3
            while len(inclusion) < 3:
                inclusion.append(f"Papers in this thematic area")
            
            # Generate exclusion criteria (at least 2)
            exclusion = [
                f"Papers not related to {label.replace('_', ' ')}",
                "Papers outside this thematic cluster"
            ]
            
            # Extract evidence from representative papers
            evidence = []
            for paper in cluster[:3]:  # First 3 as evidence
                evidence.append(EvidenceSpan(
                    claim=f"Representative paper: {paper.title}",
                    page=0,
                    section="Induced",
                    char_start=0,
                    char_end=len(paper.description) if paper.description else 0,
                    quote=paper.description[:200] if paper.description else ""
                ))
            
            # Create node definition
            node_def = NodeDefinition(
                node_path=node_path,
                definition=definition,
                inclusion_criteria=inclusion[:5],  # At most 5
                exclusion_criteria=exclusion,
                canonical_keywords=keywords[:10],
                boundary_statements=[],
                evidence_spans=evidence
            )
            
            node_defs[node_path] = node_def
        
        return node_defs
