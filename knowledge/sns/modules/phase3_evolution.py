"""
Phase 3: Stress Clustering & Minimal Evolution

This module implements clustering of stressed papers and proposes minimal
necessary structure updates to the taxonomy.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import dspy

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available. Install with: pip install hdbscan")

from ..dataclass import ResearchPaper
from ..dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    FitVector,
    FitLabel,
    StressCluster,
    ClusterType,
    EvolutionOperation,
    AddNodeOperation,
    SplitNodeOperation,
    RenameNodeOperation,
    EvolutionProposal,
    NewNodeProposal,
    EvidenceSpan,
    ViewReconstructionScore,
    WritingMode,
)
from ..schemas_v2 import (
    create_new_node_generator,
    create_subnode_generator,
    create_node_renamer,
)
from ..parsing import safe_json_loads

logger = logging.getLogger(__name__)


class StressClusterer:
    """
    Clusters papers based on their failure signatures.
    
    Uses HDBSCAN for clustering (no need to specify K).
    """
    
    def __init__(self, min_cluster_size: int = 3, embedding_model=None):
        self.min_cluster_size = min_cluster_size
        self.embedding_model = embedding_model
    
    def cluster_stressed_papers(
        self,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper],
        baseline: MultiViewBaseline,
        stress_threshold: float = 0.3
    ) -> List[StressCluster]:
        """
        Cluster papers with stress > threshold.
        
        Args:
            fit_vectors: Fit vectors for all papers
            papers: Original paper objects
            baseline: Multi-view baseline
            stress_threshold: Minimum stress score to consider
            
        Returns:
            List of StressCluster objects
        """
        logger.info("Clustering stressed papers...")
        
        # Filter papers with stress
        stressed_papers = []
        stressed_vectors = []
        
        paper_map = {p.url: p for p in papers}
        
        for fv in fit_vectors:
            if fv.stress_score > stress_threshold:
                if fv.paper_id in paper_map:
                    stressed_papers.append(paper_map[fv.paper_id])
                    stressed_vectors.append(fv)
        
        logger.info(f"Found {len(stressed_papers)} stressed papers (stress > {stress_threshold})")
        
        if len(stressed_papers) < self.min_cluster_size:
            logger.warning(f"Too few stressed papers for clustering")
            return []
        
        # Construct failure signatures
        signatures = []
        for fv in stressed_vectors:
            sig = self._construct_failure_signature(fv, baseline)
            signatures.append(sig)
        
        # Cluster using HDBSCAN
        cluster_labels = self._cluster_signatures(signatures)
        
        # Group papers by cluster
        clusters = self._group_by_cluster(
            stressed_papers,
            stressed_vectors,
            cluster_labels,
            baseline
        )
        
        logger.info(f"Identified {len(clusters)} stress clusters")
        
        return clusters
    
    def _construct_failure_signature(
        self,
        fit_vector: FitVector,
        baseline: MultiViewBaseline
    ) -> str:
        """
        Construct failure signature for a paper.
        
        Includes:
        - Facet labels where failure occurred
        - Best leaf paths
        - Lost novelty text
        """
        sig_parts = []
        
        for report in fit_vector.fit_reports:
            if report.label != FitLabel.FIT:
                sig_parts.append(f"{report.facet_label.value}:{report.best_leaf_path}")
                
                # Add lost novelty
                for lost in report.lost_novelty:
                    sig_parts.append(lost.bullet)
        
        return " ".join(sig_parts)
    
    def _cluster_signatures(self, signatures: List[str]) -> np.ndarray:
        """
        Cluster signatures using HDBSCAN.
        
        If HDBSCAN not available, falls back to simple keyword-based clustering.
        """
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, using fallback clustering")
            return self._fallback_clustering(signatures)
        
        # Convert signatures to vectors (simple bag-of-words for now)
        # In production, use actual embeddings
        vectors = self._signatures_to_vectors(signatures)
        
        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(vectors)
        
        return cluster_labels
    
    def _signatures_to_vectors(self, signatures: List[str]) -> np.ndarray:
        """
        Convert signatures to vectors.
        
        Simple bag-of-words for now. In production, use embeddings.
        """
        # Build vocabulary
        vocab = set()
        for sig in signatures:
            vocab.update(sig.lower().split())
        
        vocab = sorted(list(vocab))
        vocab_map = {word: i for i, word in enumerate(vocab)}
        
        # Convert to vectors
        vectors = np.zeros((len(signatures), len(vocab)))
        for i, sig in enumerate(signatures):
            words = sig.lower().split()
            for word in words:
                if word in vocab_map:
                    vectors[i, vocab_map[word]] += 1
        
        return vectors
    
    def _fallback_clustering(self, signatures: List[str]) -> np.ndarray:
        """
        Fallback clustering when HDBSCAN not available.
        
        Simple approach: group by first word (facet label).
        """
        clusters = {}
        cluster_id = 0
        
        labels = np.full(len(signatures), -1, dtype=int)
        
        for i, sig in enumerate(signatures):
            first_word = sig.split()[0] if sig.split() else "unknown"
            
            if first_word not in clusters:
                clusters[first_word] = cluster_id
                cluster_id += 1
            
            labels[i] = clusters[first_word]
        
        return labels
    
    def _group_by_cluster(
        self,
        papers: List[ResearchPaper],
        fit_vectors: List[FitVector],
        cluster_labels: np.ndarray,
        baseline: MultiViewBaseline
    ) -> List[StressCluster]:
        """Group papers into StressCluster objects."""
        
        cluster_map = defaultdict(list)
        vector_map = defaultdict(list)
        
        for paper, fv, label in zip(papers, fit_vectors, cluster_labels):
            if label != -1:  # -1 = noise in HDBSCAN
                cluster_map[label].append(paper)
                vector_map[label].append(fv)
        
        clusters = []
        for cluster_id, cluster_papers in cluster_map.items():
            cluster_vectors = vector_map[cluster_id]
            
            # Calculate cluster-level metrics
            stress_score = np.mean([fv.stress_score for fv in cluster_vectors])
            unfittable_score = np.mean([fv.unfittable_score for fv in cluster_vectors])
            
            # Calculate view failure rates
            view_failure_rates = self._calculate_view_failure_rates(
                cluster_vectors,
                baseline
            )
            
            # Determine cluster type
            cluster_type = self._determine_cluster_type(
                view_failure_rates,
                unfittable_score,
                baseline
            )
            
            # Construct failure signature
            failure_sig = " | ".join([
                self._construct_failure_signature(fv, baseline)
                for fv in cluster_vectors[:3]  # First 3 as representative
            ])
            
            cluster = StressCluster(
                cluster_id=f"C{cluster_id}",
                papers=cluster_papers,
                cluster_type=cluster_type,
                stress_score=stress_score,
                unfittable_score=unfittable_score,
                view_failure_rates=view_failure_rates,
                failure_signature=failure_sig
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_view_failure_rates(
        self,
        fit_vectors: List[FitVector],
        baseline: MultiViewBaseline
    ) -> Dict[str, float]:
        """Calculate failure rate for each view in the cluster."""
        
        view_failures = defaultdict(int)
        
        for fv in fit_vectors:
            for report in fv.fit_reports:
                if report.label != FitLabel.FIT:
                    view_failures[report.view_id] += 1
        
        view_failure_rates = {}
        for view in baseline.views:
            rate = view_failures[view.view_id] / len(fit_vectors)
            view_failure_rates[view.view_id] = rate
        
        return view_failure_rates
    
    def _determine_cluster_type(
        self,
        view_failure_rates: Dict[str, float],
        unfittable_score: float,
        baseline: MultiViewBaseline
    ) -> ClusterType:
        """
        Determine cluster type based on failure patterns.
        
        Rules from design doc:
        - Strong Shift: U(C) > 0.55 and ≥2 high-weight views fail
        - Facet-dependent: Some high-weight views fail, others fit
        - Stable: Most high-weight views fit
        """
        # Get high-weight views (top 30%)
        weights = [v.weight for v in baseline.views]
        threshold = np.percentile(weights, 70)
        high_weight_views = [v for v in baseline.views if v.weight > threshold]
        
        # Count high-weight view failures
        high_weight_failures = sum(
            1 for v in high_weight_views
            if view_failure_rates.get(v.view_id, 0) > 0.6
        )
        
        high_weight_fits = sum(
            1 for v in high_weight_views
            if view_failure_rates.get(v.view_id, 0) < 0.2
        )
        
        # Rule 1: Strong Shift
        if unfittable_score > 0.55 and high_weight_failures >= 2:
            return ClusterType.STRONG_SHIFT
        
        # Rule 2: Facet-dependent
        if high_weight_failures > 0 and high_weight_fits > 0:
            return ClusterType.FACET_DEPENDENT
        
        # Rule 3: Stable (default)
        return ClusterType.STABLE


class EvolutionPlanner:
    """
    Plans minimal necessary evolution operations for the taxonomy.
    
    Proposes ADD_NODE, SPLIT_NODE, RENAME_NODE operations.
    """
    
    def __init__(self, lm):
        self.lm = lm
        self.new_node_generator = create_new_node_generator(lm)
        self.subnode_generator = create_subnode_generator(lm)
        self.node_renamer = create_node_renamer(lm)
    
    def plan_evolution(
        self,
        clusters: List[StressCluster],
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector],
        lambda_reg: float = 0.8
    ) -> EvolutionProposal:
        """
        Plan minimal necessary evolution operations.
        
        Args:
            clusters: Stress clusters
            baseline: Multi-view baseline
            fit_vectors: Original fit vectors
            lambda_reg: Regularization parameter for edit cost
            
        Returns:
            EvolutionProposal with selected operations
        """
        logger.info("Planning taxonomy evolution...")
        
        all_operations = []
        
        for cluster in clusters:
            if cluster.cluster_type in [ClusterType.STRONG_SHIFT, ClusterType.FACET_DEPENDENT]:
                logger.info(f"Planning operations for cluster {cluster.cluster_id} ({cluster.cluster_type.value})")
                
                # Generate candidate operations
                candidates = []
                
                # Try ADD_NODE for each view with high failure rate
                for view_id, failure_rate in cluster.view_failure_rates.items():
                    if failure_rate > 0.6:
                        view = baseline.get_view_by_id(view_id)
                        if view:
                            add_op = self._propose_add_node(cluster, view, fit_vectors)
                            if add_op:
                                candidates.append(add_op)
                
                # Try SPLIT_NODE for overcrowded nodes
                for view_id, failure_rate in cluster.view_failure_rates.items():
                    if failure_rate > 0.5:  # Moderate failure, might need splitting
                        view = baseline.get_view_by_id(view_id)
                        if view:
                            split_op = self._propose_split_node(cluster, view, fit_vectors)
                            if split_op:
                                candidates.append(split_op)
                
                # Try RENAME_NODE for drifted nodes
                for view_id, failure_rate in cluster.view_failure_rates.items():
                    if failure_rate > 0.4:  # Semantic drift indication
                        view = baseline.get_view_by_id(view_id)
                        if view:
                            rename_op = self._propose_rename_node(cluster, view, fit_vectors)
                            if rename_op:
                                candidates.append(rename_op)
                
                # Select best operation
                if candidates:
                    best_op = max(candidates, key=lambda op: op.fit_gain - lambda_reg * op.edit_cost)
                    
                    objective = best_op.fit_gain - lambda_reg * best_op.edit_cost
                    if objective > 0:
                        all_operations.append(best_op)
                        logger.info(f"  Selected {best_op.operation_type.value} with objective={objective:.3f}")
        
        # Calculate total metrics
        total_fit_gain = sum(op.fit_gain for op in all_operations)
        total_edit_cost = sum(op.edit_cost for op in all_operations)
        objective_value = total_fit_gain - lambda_reg * total_edit_cost
        
        proposal = EvolutionProposal(
            operations=all_operations,
            total_fit_gain=total_fit_gain,
            total_edit_cost=total_edit_cost,
            objective_value=objective_value
        )
        
        logger.info(f"Evolution plan: {len(all_operations)} operations, objective={objective_value:.3f}")
        
        return proposal
    
    def compute_all_views_reconstruction(
        self,
        clusters: List[StressCluster],
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector],
        lambda_reg: float = 0.8
    ) -> List[ViewReconstructionScore]:
        """
        Compute reconstruction scores for ALL views.
        
        This implements the 'reconstruct-then-select' design principle:
        For each view, plan evolution operations and calculate:
        - FitGain: How much stressed papers would improve
        - Stress: Reduction in average stress score
        - Coverage: Taxonomy richness (normalized)
        - EditCost: Total cost of operations
        
        Args:
            clusters: Stress clusters
            baseline: Multi-view baseline
            fit_vectors: Original fit vectors
            lambda_reg: Regularization parameter
            
        Returns:
            List of ViewReconstructionScore for all views
        """
        logger.info("Computing reconstruction for all views...")
        
        reconstruction_scores = []
        
        for view in baseline.views:
            logger.info(f"  Reconstructing view {view.view_id} ({view.facet_label.value})...")
            
            # Plan operations for this view
            view_operations = []
            
            for cluster in clusters:
                if cluster.cluster_type in [ClusterType.STRONG_SHIFT, ClusterType.FACET_DEPENDENT]:
                    failure_rate = cluster.view_failure_rates.get(view.view_id, 0.0)
                    
                    if failure_rate > 0.6:  # This view struggles with this cluster
                        # Propose ADD_NODE
                        add_op = self._propose_add_node(cluster, view, fit_vectors)
                        if add_op:
                            view_operations.append(add_op)
            
            # Calculate metrics for this view
            if view_operations:
                # FitGain: Total improvement from all operations
                fit_gain = sum(op.fit_gain for op in view_operations)
                
                # EditCost: Total cost
                edit_cost = sum(op.edit_cost for op in view_operations)
                
                # Stress reduction: Calculate how much stress would reduce
                # (Simplified: assume proportional to fit_gain)
                stress_reduction = fit_gain / max(len(fit_vectors), 1) * 0.3
                
            else:
                # No operations needed - view is already good
                fit_gain = 0.0
                edit_cost = 0.0
                stress_reduction = 0.0
            
            # Coverage: Taxonomy richness (normalized by max leaves = 50)
            num_leaves = len(view.tree.get_leaf_nodes())
            coverage = min(1.0, num_leaves / 50.0)
            
            # Create score object (combined_score computed in __init__)
            score = ViewReconstructionScore(
                view_id=view.view_id,
                facet_label=view.facet_label,
                fit_gain=fit_gain,
                stress_reduction=stress_reduction,
                coverage=coverage,
                edit_cost=edit_cost
            )
            
            reconstruction_scores.append(score)
            
            logger.info(f"    FitGain={fit_gain:.3f}, Stress={stress_reduction:.3f}, "
                       f"Coverage={coverage:.3f}, EditCost={edit_cost:.3f}, "
                       f"CombinedScore={score.combined_score:.3f}")
        
        # Sort by combined score (highest first)
        reconstruction_scores.sort(key=lambda s: s.combined_score, reverse=True)
        
        if reconstruction_scores:
            logger.info(f"Reconstruction complete. Best view: {reconstruction_scores[0].view_id} "
                       f"(score={reconstruction_scores[0].combined_score:.3f})")
        else:
            logger.info("Reconstruction complete. No views to reconstruct.")
        
        return reconstruction_scores
    
    def _propose_add_node(
        self,
        cluster: StressCluster,
        view: TaxonomyView,
        fit_vectors: List[FitVector]
    ) -> Optional[AddNodeOperation]:
        """
        Propose adding a new node to accommodate the cluster.
        """
        logger.info(f"  Proposing ADD_NODE for view {view.view_id}")
        
        # Find best parent node (simplified: use root for now)
        parent_path = view.tree.root.path
        parent_node = view.tree.root
        
        # Prepare cluster information for LLM
        cluster_papers_text = self._format_cluster_papers(cluster)
        cluster_innovations_text = self._extract_cluster_innovations(cluster, fit_vectors, view)
        
        try:
            # Call LLM to generate new node (wrap in dspy context)
            with dspy.context(lm=self.lm):
                result = self.new_node_generator(
                    parent_node_name=parent_node.name,
                    parent_definition=view.node_definitions.get(parent_path, None),
                    cluster_papers=cluster_papers_text,
                    cluster_innovations=cluster_innovations_text
                )
            
            # Parse result
            node_data = safe_json_loads(result.new_node_json)
            
            if not node_data:
                return None
            
            new_node = NewNodeProposal(
                name=node_data.get("name", "New Category"),
                definition=node_data.get("definition", ""),
                inclusion_criteria=node_data.get("inclusion_criteria", []),
                exclusion_criteria=node_data.get("exclusion_criteria", []),
                keywords=node_data.get("keywords", [])
            )
            
            # Calculate fit gain (simplified: assume 50% improvement for cluster papers)
            fit_gain = len(cluster.papers) * 0.5
            
            # Extract evidence
            evidence = self._extract_cluster_evidence(cluster)
            
            operation = AddNodeOperation(
                view_id=view.view_id,
                parent_path=parent_path,
                new_node=new_node,
                evidence=evidence,
                fit_gain=fit_gain,
                improvement_rate=0.5
            )
            
            return operation
            
        except Exception as e:
            logger.error(f"Error proposing ADD_NODE: {e}")
            return None
    
    def _format_cluster_papers(self, cluster: StressCluster) -> str:
        """Format cluster papers for LLM."""
        lines = []
        for i, paper in enumerate(cluster.papers[:5], 1):  # First 5
            lines.append(f"{i}. {paper.title}")
            if paper.abstract:
                lines.append(f"   Abstract: {paper.abstract[:200]}...")
        
        return "\n".join(lines)
    
    def _extract_cluster_innovations(
        self,
        cluster: StressCluster,
        fit_vectors: List[FitVector],
        view: TaxonomyView
    ) -> str:
        """Extract key innovations from cluster papers."""
        innovations = []
        
        # Find fit vectors for cluster papers
        paper_ids = {p.url for p in cluster.papers}
        cluster_vectors = [fv for fv in fit_vectors if fv.paper_id in paper_ids]
        
        for fv in cluster_vectors[:5]:  # First 5
            for report in fv.fit_reports:
                if report.view_id == view.view_id:
                    for lost in report.lost_novelty:
                        innovations.append(lost.bullet)
        
        return "\n".join(f"- {inn}" for inn in innovations[:10])  # Top 10
    
    def _extract_cluster_evidence(self, cluster: StressCluster) -> List[EvidenceSpan]:
        """Extract evidence spans from cluster papers."""
        evidence = []
        
        for paper in cluster.papers[:3]:  # First 3
            evidence.append(EvidenceSpan(
                claim=f"Paper '{paper.title}' exhibits this pattern",
                page=0,
                section="Abstract",
                char_start=0,
                char_end=len(paper.abstract) if paper.abstract else 0,
                quote=paper.abstract[:200] if paper.abstract else ""
            ))
        
        return evidence
    
    def _propose_split_node(
        self,
        cluster: StressCluster,
        view: TaxonomyView,
        fit_vectors: List[FitVector]
    ) -> Optional[SplitNodeOperation]:
        """
        Propose splitting an overcrowded node into sub-nodes.
        
        This addresses cases where an existing category has become too broad
        and needs to be subdivided to better accommodate papers.
        """
        logger.info(f"  Proposing SPLIT_NODE for view {view.view_id}")
        
        # Identify candidate nodes for splitting (nodes with many children or high coverage)
        candidate_nodes = self._find_split_candidates(view, cluster, fit_vectors)
        
        if not candidate_nodes:
            return None
        
        # Select best candidate (node with highest coverage + conflict)
        target_node = candidate_nodes[0]
        target_path = target_node.path
        
        # Prepare cluster information for LLM
        cluster_papers_text = self._format_cluster_papers(cluster)
        cluster_innovations_text = self._extract_cluster_innovations(cluster, fit_vectors, view)
        
        # Get current node definition
        current_definition = view.node_definitions.get(target_path)
        
        try:
            # Call LLM to generate sub-nodes (wrap in dspy context)
            with dspy.context(lm=self.lm):
                result = self.subnode_generator(
                    parent_node_name=target_node.name,
                    parent_definition=current_definition.definition if current_definition else "",
                    cluster_papers=cluster_papers_text,
                    cluster_innovations=cluster_innovations_text
                )
            
            # Parse result
            subnode_data = safe_json_loads(result.subnodes_json)
            
            if not subnode_data or not subnode_data.get("subnodes"):
                return None
            
            # Build NewNodeProposal objects (for subnodes)
            subnodes = []
            for sn in subnode_data.get("subnodes", []):
                subnodes.append(NewNodeProposal(
                    name=sn.get("name", "Subnode"),
                    definition=sn.get("definition", ""),
                    inclusion_criteria=sn.get("inclusion_criteria", []),
                    exclusion_criteria=sn.get("exclusion_criteria", []),
                    keywords=sn.get("keywords", [])
                ))
            
            if len(subnodes) < 2:
                # Not worth splitting if less than 2 subnodes
                return None
            
            # Calculate fit gain (moderate improvement expected)
            fit_gain = len(cluster.papers) * 0.3
            
            # Extract evidence
            evidence = self._extract_cluster_evidence(cluster)
            
            operation = SplitNodeOperation(
                view_id=view.view_id,
                node_path=target_path,
                sub_nodes=subnodes,
                evidence=evidence,
                fit_gain=fit_gain
            )
            
            return operation
            
        except Exception as e:
            logger.error(f"Error proposing SPLIT_NODE: {e}")
            return None
    
    def _find_split_candidates(
        self,
        view: TaxonomyView,
        cluster: StressCluster,
        fit_vectors: List[FitVector]
    ) -> List[TaxonomyTreeNode]:
        """
        Find nodes that are candidates for splitting.
        
        Criteria:
        1. Has many children (>= 5) OR is a leaf with high coverage
        2. High coverage rate in cluster papers
        3. Has some conflict (indicating overcrowding)
        """
        candidates = []
        
        # Find fit vectors for cluster papers
        paper_ids = {p.url for p in cluster.papers}
        cluster_vectors = [fv for fv in fit_vectors if fv.paper_id in paper_ids]
        
        # Analyze each node
        for node_path, node in view.tree.nodes.items():
            if node.name == "ROOT":
                continue
            
            # Criterion 1: Many children or leaf
            has_many_children = len(node.children) >= 5
            is_leaf = node.is_leaf
            
            if not (has_many_children or is_leaf):
                continue
            
            # Criterion 2 & 3: Check coverage and conflict in cluster
            coverage_count = 0
            conflict_count = 0
            
            for fv in cluster_vectors[:10]:  # Sample first 10
                for report in fv.fit_reports:
                    if report.view_id == view.view_id:
                        for match in report.node_matches:
                            if match.node_path == node_path:
                                if match.coverage_score > 0.5:
                                    coverage_count += 1
                                if match.conflict_score > 0.3:
                                    conflict_count += 1
            
            if coverage_count >= 3 and conflict_count >= 1:
                # This node is a good candidate
                candidates.append((node, coverage_count, conflict_count))
        
        # Sort by combined score (coverage + conflict)
        candidates.sort(key=lambda x: (x[1] + x[2]), reverse=True)
        
        return [node for node, _, _ in candidates]
    
    def _propose_rename_node(
        self,
        cluster: StressCluster,
        view: TaxonomyView,
        fit_vectors: List[FitVector]
    ) -> Optional[RenameNodeOperation]:
        """
        Propose renaming a node that has undergone semantic drift.
        
        This addresses cases where the terminology/definition of a category
        has shifted and needs to be updated to reflect current usage.
        """
        logger.info(f"  Proposing RENAME_NODE for view {view.view_id}")
        
        # Identify candidate nodes for renaming (nodes with residual but some coverage)
        candidate_nodes = self._find_rename_candidates(view, cluster, fit_vectors)
        
        if not candidate_nodes:
            return None
        
        # Select best candidate (node with highest residual score)
        target_node = candidate_nodes[0]
        target_path = target_node.path
        
        # Prepare cluster information for LLM
        cluster_papers_text = self._format_cluster_papers(cluster)
        cluster_innovations_text = self._extract_cluster_innovations(cluster, fit_vectors, view)
        
        # Get current node definition
        current_definition = view.node_definitions.get(target_path)
        
        try:
            # Call LLM to propose new name and definition (wrap in dspy context)
            with dspy.context(lm=self.lm):
                result = self.node_renamer(
                    current_node_name=target_node.name,
                    current_definition=current_definition.definition if current_definition else "",
                    cluster_papers=cluster_papers_text,
                    cluster_innovations=cluster_innovations_text
                )
            
            # Parse result
            rename_data = safe_json_loads(result.rename_json)
            
            if not rename_data:
                return None
            
            new_name = rename_data.get("new_name", target_node.name)
            new_definition = rename_data.get("new_definition", "")
            rationale = rename_data.get("rationale", "")
            
            # Skip if name didn't change
            if new_name == target_node.name:
                return None
            
            # Calculate fit gain (small improvement expected)
            fit_gain = len(cluster.papers) * 0.2
            
            # Extract evidence
            evidence = self._extract_cluster_evidence(cluster)
            
            operation = RenameNodeOperation(
                view_id=view.view_id,
                node_path=target_path,
                old_name=target_node.name,
                new_name=new_name,
                new_definition=new_definition,
                drift_score=0.5,  # Can be calculated from residual scores
                evidence=evidence,
                fit_gain=fit_gain
            )
            
            return operation
            
        except Exception as e:
            logger.error(f"Error proposing RENAME_NODE: {e}")
            return None
    
    def _find_rename_candidates(
        self,
        view: TaxonomyView,
        cluster: StressCluster,
        fit_vectors: List[FitVector]
    ) -> List[TaxonomyTreeNode]:
        """
        Find nodes that are candidates for renaming.
        
        Criteria:
        1. Has moderate coverage (papers partially match)
        2. Has high residual (papers have unaddressed content)
        3. Low conflict (not contradicting, just incomplete)
        """
        candidates = []
        
        # Find fit vectors for cluster papers
        paper_ids = {p.url for p in cluster.papers}
        cluster_vectors = [fv for fv in fit_vectors if fv.paper_id in paper_ids]
        
        # Analyze each node
        for node_path, node in view.tree.nodes.items():
            if node.name == "ROOT":
                continue
            
            # Check coverage, residual, and conflict in cluster
            coverage_scores = []
            residual_scores = []
            conflict_scores = []
            
            for fv in cluster_vectors[:10]:  # Sample first 10
                for report in fv.fit_reports:
                    if report.view_id == view.view_id:
                        for match in report.node_matches:
                            if match.node_path == node_path:
                                coverage_scores.append(match.coverage_score)
                                residual_scores.append(match.residual_score)
                                conflict_scores.append(match.conflict_score)
            
            if not coverage_scores:
                continue
            
            avg_coverage = sum(coverage_scores) / len(coverage_scores)
            avg_residual = sum(residual_scores) / len(residual_scores)
            avg_conflict = sum(conflict_scores) / len(conflict_scores)
            
            # Criterion: moderate coverage + high residual + low conflict
            if 0.3 < avg_coverage < 0.7 and avg_residual > 0.5 and avg_conflict < 0.4:
                # This node is a good candidate for renaming
                candidates.append((node, avg_residual, avg_coverage))
        
        # Sort by residual score (higher = more drift)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _, _ in candidates]


# ============================================================================
# Main Phase 3 Pipeline
# ============================================================================

class Phase3Pipeline:
    """
    Complete Phase 3 pipeline: Stress Clustering & Minimal Evolution.
    """
    
    def __init__(self, lm, min_cluster_size: int = 3, lambda_reg: float = 0.8):
        self.clusterer = StressClusterer(min_cluster_size=min_cluster_size)
        self.planner = EvolutionPlanner(lm)
        self.lambda_reg = lambda_reg
    
    def run(
        self,
        fit_vectors: List[FitVector],
        papers: List[ResearchPaper],
        baseline: MultiViewBaseline
    ) -> Tuple[List[StressCluster], EvolutionProposal]:
        """
        Run complete Phase 3 pipeline.
        
        Args:
            fit_vectors: Fit vectors from Phase 2
            papers: Original research papers
            baseline: Multi-view baseline from Phase 1
            
        Returns:
            Tuple of (stress_clusters, evolution_proposal)
        """
        logger.info("="*80)
        logger.info("PHASE 3: Stress Clustering & Minimal Evolution")
        logger.info("="*80)
        
        # Step 1: Cluster stressed papers
        logger.info("Step 1: Clustering stressed papers...")
        clusters = self.clusterer.cluster_stressed_papers(
            fit_vectors,
            papers,
            baseline,
            stress_threshold=0.3
        )
        
        if not clusters:
            logger.warning("No stress clusters found")
            empty_proposal = EvolutionProposal(
                operations=[],
                total_fit_gain=0.0,
                total_edit_cost=0.0,
                objective_value=0.0
            )
            return clusters, empty_proposal
        
        # Log cluster summary
        for cluster in clusters:
            logger.info(f"  {cluster.cluster_id}: {len(cluster.papers)} papers, "
                       f"type={cluster.cluster_type.value}, "
                       f"stress={cluster.stress_score:.3f}")
        
        # Step 2: Plan evolution
        logger.info("Step 2: Planning minimal evolution...")
        proposal = self.planner.plan_evolution(
            clusters,
            baseline,
            fit_vectors,
            lambda_reg=self.lambda_reg
        )
        
        logger.info(f"Phase 3 completed: {len(clusters)} clusters, {len(proposal.operations)} operations")
        logger.info("="*80)
        
        return clusters, proposal
    
    def compute_reconstruction_scores(
        self,
        clusters: List[StressCluster],
        baseline: MultiViewBaseline,
        fit_vectors: List[FitVector]
    ) -> List[ViewReconstructionScore]:
        """
        Compute reconstruction scores for all views.
        
        This is the key method for Phase 4's 'reconstruct-then-select' approach.
        
        Args:
            clusters: Stress clusters from Phase 3
            baseline: Multi-view baseline
            fit_vectors: Fit vectors from Phase 2
            
        Returns:
            List of ViewReconstructionScore sorted by combined score
        """
        return self.planner.compute_all_views_reconstruction(
            clusters,
            baseline,
            fit_vectors,
            lambda_reg=self.lambda_reg
        )
        return clusters, proposal
