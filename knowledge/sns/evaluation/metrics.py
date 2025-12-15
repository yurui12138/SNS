"""
Comprehensive evaluation metrics for IG-Finder 2.0.

Includes Branch Hit@K, Edit Distance, and other quantitative metrics.
"""
import logging
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from ..dataclass_v2 import (
    TaxonomyTree,
    EvolutionOperation,
    EvolutionProposal,
    FitVector,
    FitLabel
)

logger = logging.getLogger(__name__)


class BranchHitAtK:
    """
    Branch Hit@K metric for evaluating evolution predictions.
    
    Measures whether predicted operations target the correct branches
    (parent paths) in the taxonomy.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
    
    def compute(
        self,
        predicted_operations: List[EvolutionOperation],
        ground_truth_branches: List[str]
    ) -> Dict[int, float]:
        """
        Compute Hit@K for different K values.
        
        Args:
            predicted_operations: Predicted evolution operations (ranked)
            ground_truth_branches: Ground truth parent paths that should be modified
            
        Returns:
            Dictionary mapping K to Hit@K score
        """
        results = {}
        
        # Extract predicted branches
        predicted_branches = [op.parent_path for op in predicted_operations]
        
        for k in self.k_values:
            top_k_branches = set(predicted_branches[:k])
            
            # Count how many ground truth branches are in top-K
            hits = sum(1 for gt_branch in ground_truth_branches 
                      if gt_branch in top_k_branches)
            
            hit_rate = hits / len(ground_truth_branches) if ground_truth_branches else 0.0
            results[k] = hit_rate
        
        return results


class TaxonomyEditDistance:
    """
    Compute edit distance between two taxonomy trees.
    
    Measures the structural difference between taxonomies.
    """
    
    def __init__(self):
        pass
    
    def compute(self, tree1: TaxonomyTree, tree2: TaxonomyTree) -> int:
        """
        Compute tree edit distance.
        
        Simplified version: count node additions, deletions, and renames.
        
        Args:
            tree1: First taxonomy tree
            tree2: Second taxonomy tree
            
        Returns:
            Edit distance (number of operations)
        """
        paths1 = set(tree1.nodes.keys())
        paths2 = set(tree2.nodes.keys())
        
        # Count additions and deletions
        additions = len(paths2 - paths1)
        deletions = len(paths1 - paths2)
        
        # Count renames (nodes at same path but different name)
        renames = 0
        common_paths = paths1 & paths2
        
        for path in common_paths:
            if tree1.nodes[path].name != tree2.nodes[path].name:
                renames += 1
        
        edit_distance = additions + deletions + renames
        
        return edit_distance
    
    def compute_normalized(self, tree1: TaxonomyTree, tree2: TaxonomyTree) -> float:
        """
        Compute normalized edit distance in [0, 1].
        
        Normalized by the maximum possible distance (sum of tree sizes).
        """
        edit_dist = self.compute(tree1, tree2)
        max_dist = len(tree1.nodes) + len(tree2.nodes)
        
        if max_dist == 0:
            return 0.0
        
        return edit_dist / max_dist


class FitScoreMetrics:
    """
    Metrics for evaluating fit scores and stress detection.
    """
    
    def __init__(self):
        pass
    
    def compute_stress_distribution(
        self,
        fit_vectors: List[FitVector]
    ) -> Dict[str, float]:
        """
        Compute distribution of stress scores.
        
        Returns statistics: mean, std, median, percentiles.
        """
        stress_scores = [fv.stress_score for fv in fit_vectors]
        
        if not stress_scores:
            return {}
        
        return {
            "mean": float(np.mean(stress_scores)),
            "std": float(np.std(stress_scores)),
            "median": float(np.median(stress_scores)),
            "p25": float(np.percentile(stress_scores, 25)),
            "p75": float(np.percentile(stress_scores, 75)),
            "p90": float(np.percentile(stress_scores, 90)),
            "p95": float(np.percentile(stress_scores, 95)),
        }
    
    def compute_fit_label_distribution(
        self,
        fit_vectors: List[FitVector]
    ) -> Dict[str, float]:
        """
        Compute distribution of fit labels across all views.
        
        Returns percentage for FIT, FORCE_FIT, UNFITTABLE.
        """
        label_counts = defaultdict(int)
        total = 0
        
        for fv in fit_vectors:
            for report in fv.fit_reports:
                label_counts[report.label.value] += 1
                total += 1
        
        if total == 0:
            return {}
        
        return {
            label: count / total
            for label, count in label_counts.items()
        }
    
    def compute_view_fit_rates(
        self,
        fit_vectors: List[FitVector]
    ) -> Dict[str, float]:
        """
        Compute FIT rate for each view.
        
        Returns dictionary mapping view_id to FIT rate.
        """
        view_stats = defaultdict(lambda: {"fit": 0, "total": 0})
        
        for fv in fit_vectors:
            for report in fv.fit_reports:
                view_stats[report.view_id]["total"] += 1
                if report.label == FitLabel.FIT:
                    view_stats[report.view_id]["fit"] += 1
        
        fit_rates = {}
        for view_id, stats in view_stats.items():
            fit_rates[view_id] = stats["fit"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return fit_rates


class EvolutionMetrics:
    """
    Metrics for evaluating evolution proposals.
    """
    
    def __init__(self):
        pass
    
    def compute_operation_distribution(
        self,
        proposal: EvolutionProposal
    ) -> Dict[str, int]:
        """
        Count operations by type.
        
        Returns distribution of ADD, SPLIT, RENAME operations.
        """
        op_counts = defaultdict(int)
        
        for op in proposal.operations:
            op_counts[op.operation_type.value] += 1
        
        return dict(op_counts)
    
    def compute_average_fit_gain(
        self,
        proposal: EvolutionProposal
    ) -> float:
        """
        Compute average fit gain per operation.
        """
        if not proposal.operations:
            return 0.0
        
        return proposal.total_fit_gain / len(proposal.operations)
    
    def compute_average_edit_cost(
        self,
        proposal: EvolutionProposal
    ) -> float:
        """
        Compute average edit cost per operation.
        """
        if not proposal.operations:
            return 0.0
        
        return proposal.total_edit_cost / len(proposal.operations)
    
    def compute_efficiency_ratio(
        self,
        proposal: EvolutionProposal
    ) -> float:
        """
        Compute efficiency ratio: fit_gain / edit_cost.
        
        Higher is better (more gain per unit cost).
        """
        if proposal.total_edit_cost == 0:
            return float('inf') if proposal.total_fit_gain > 0 else 0.0
        
        return proposal.total_fit_gain / proposal.total_edit_cost


def compute_all_metrics(
    fit_vectors: List[FitVector],
    evolution_proposal: EvolutionProposal,
    original_tree: Optional[TaxonomyTree] = None,
    evolved_tree: Optional[TaxonomyTree] = None
) -> Dict[str, any]:
    """
    Compute all evaluation metrics for a complete IG-Finder 2.0 run.
    
    Args:
        fit_vectors: All fit vectors from Phase 2
        evolution_proposal: Evolution proposal from Phase 3
        original_tree: Original taxonomy tree (optional)
        evolved_tree: Evolved taxonomy tree (optional)
        
    Returns:
        Comprehensive dictionary of all metrics
    """
    logger.info("Computing comprehensive evaluation metrics...")
    
    metrics = {}
    
    # Fit score metrics
    fit_metrics = FitScoreMetrics()
    metrics["stress_distribution"] = fit_metrics.compute_stress_distribution(fit_vectors)
    metrics["fit_label_distribution"] = fit_metrics.compute_fit_label_distribution(fit_vectors)
    metrics["view_fit_rates"] = fit_metrics.compute_view_fit_rates(fit_vectors)
    
    # Evolution metrics
    evo_metrics = EvolutionMetrics()
    metrics["operation_distribution"] = evo_metrics.compute_operation_distribution(evolution_proposal)
    metrics["average_fit_gain"] = evo_metrics.compute_average_fit_gain(evolution_proposal)
    metrics["average_edit_cost"] = evo_metrics.compute_average_edit_cost(evolution_proposal)
    metrics["efficiency_ratio"] = evo_metrics.compute_efficiency_ratio(evolution_proposal)
    
    # Tree edit distance (if both trees provided)
    if original_tree and evolved_tree:
        tree_dist = TaxonomyEditDistance()
        metrics["tree_edit_distance"] = tree_dist.compute(original_tree, evolved_tree)
        metrics["normalized_tree_distance"] = tree_dist.compute_normalized(original_tree, evolved_tree)
    
    # Summary statistics
    metrics["summary"] = {
        "total_papers": len(fit_vectors),
        "total_operations": len(evolution_proposal.operations),
        "total_fit_gain": evolution_proposal.total_fit_gain,
        "total_edit_cost": evolution_proposal.total_edit_cost,
        "objective_value": evolution_proposal.objective_value
    }
    
    logger.info("Metrics computation completed")
    
    return metrics


def print_metrics_report(metrics: Dict[str, any]):
    """
    Print a formatted report of evaluation metrics.
    
    Args:
        metrics: Metrics dictionary from compute_all_metrics
    """
    print("\n" + "=" * 80)
    print("IG-FINDER 2.0 EVALUATION METRICS REPORT")
    print("=" * 80)
    
    # Summary
    if "summary" in metrics:
        print("\n📊 SUMMARY")
        print("-" * 40)
        for key, value in metrics["summary"].items():
            print(f"  {key}: {value}")
    
    # Stress distribution
    if "stress_distribution" in metrics:
        print("\n📈 STRESS SCORE DISTRIBUTION")
        print("-" * 40)
        for key, value in metrics["stress_distribution"].items():
            print(f"  {key}: {value:.3f}")
    
    # Fit label distribution
    if "fit_label_distribution" in metrics:
        print("\n🏷️  FIT LABEL DISTRIBUTION")
        print("-" * 40)
        for label, percentage in metrics["fit_label_distribution"].items():
            print(f"  {label}: {percentage*100:.1f}%")
    
    # View fit rates
    if "view_fit_rates" in metrics:
        print("\n👁️  VIEW FIT RATES")
        print("-" * 40)
        for view_id, rate in sorted(metrics["view_fit_rates"].items(), key=lambda x: -x[1]):
            print(f"  {view_id}: {rate*100:.1f}%")
    
    # Operation distribution
    if "operation_distribution" in metrics:
        print("\n🔧 OPERATION DISTRIBUTION")
        print("-" * 40)
        for op_type, count in metrics["operation_distribution"].items():
            print(f"  {op_type}: {count}")
    
    # Evolution efficiency
    print("\n⚡ EVOLUTION EFFICIENCY")
    print("-" * 40)
    if "average_fit_gain" in metrics:
        print(f"  Average Fit Gain: {metrics['average_fit_gain']:.3f}")
    if "average_edit_cost" in metrics:
        print(f"  Average Edit Cost: {metrics['average_edit_cost']:.3f}")
    if "efficiency_ratio" in metrics:
        ratio = metrics['efficiency_ratio']
        print(f"  Efficiency Ratio: {ratio:.3f}" if ratio != float('inf') else "  Efficiency Ratio: ∞")
    
    # Tree distance
    if "tree_edit_distance" in metrics:
        print("\n🌳 TAXONOMY EVOLUTION")
        print("-" * 40)
        print(f"  Tree Edit Distance: {metrics['tree_edit_distance']}")
        if "normalized_tree_distance" in metrics:
            print(f"  Normalized Distance: {metrics['normalized_tree_distance']:.3f}")
    
    print("\n" + "=" * 80 + "\n")
