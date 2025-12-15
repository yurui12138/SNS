"""
Time-slice Taxonomy Shift Evaluation

Constructs datasets by time-slicing literature (e.g., 2020 vs 2023)
and evaluates whether the system can predict actual taxonomy evolution.
"""
import logging
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from ..dataclass import ReviewPaper, ResearchPaper
from ..dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyTree,
    EvolutionOperation,
    EvolutionProposal
)

logger = logging.getLogger(__name__)


@dataclass
class TimeSlice:
    """A time slice of literature."""
    start_year: int
    end_year: int
    reviews: List[ReviewPaper]
    research_papers: List[ResearchPaper]
    label: str  # e.g., "T0" (baseline) or "T1" (future)


@dataclass
class TaxonomyShift:
    """Documented shift in taxonomy between two time periods."""
    view_id: str
    parent_path: str
    operation_type: str  # "ADD", "SPLIT", "RENAME"
    new_node_name: Optional[str]
    split_node_path: Optional[str]
    split_into: Optional[List[str]]
    renamed_from: Optional[str]
    renamed_to: Optional[str]
    evidence_papers: List[str]  # Paper IDs
    justification: str


class TimeSliceDataset:
    """
    Dataset for time-slice evaluation.
    
    Constructs (T0, T1) pairs where:
    - T0: Reviews up to year Y, research papers up to Y
    - T1: Reviews up to year Y+3, research papers Y to Y+3
    
    Ground truth: Taxonomy changes extracted from T1 reviews.
    """
    
    def __init__(self):
        self.time_slices: List[TimeSlice] = []
        self.ground_truth_shifts: List[TaxonomyShift] = []
    
    def create_from_papers(
        self,
        topic: str,
        all_reviews: List[ReviewPaper],
        all_research: List[ResearchPaper],
        t0_end_year: int,
        t1_end_year: int,
        gap_years: int = 3
    ) -> Tuple[TimeSlice, TimeSlice]:
        """
        Create T0 and T1 time slices.
        
        Args:
            topic: Research topic
            all_reviews: All available review papers
            all_research: All available research papers
            t0_end_year: End year for T0 (baseline)
            t1_end_year: End year for T1 (future)
            gap_years: Gap between T0 and T1 research papers
            
        Returns:
            Tuple of (T0, T1) TimeSlice objects
        """
        logger.info(f"Creating time slices: T0 (≤{t0_end_year}) vs T1 (≤{t1_end_year})")
        
        # T0: Baseline period
        t0_reviews = [r for r in all_reviews if r.year <= t0_end_year]
        t0_research = [p for p in all_research if p.year <= t0_end_year]
        
        t0 = TimeSlice(
            start_year=min(r.year for r in t0_reviews) if t0_reviews else t0_end_year - 5,
            end_year=t0_end_year,
            reviews=t0_reviews,
            research_papers=t0_research,
            label="T0"
        )
        
        # T1: Future period (includes T0 + new papers)
        t1_reviews = [r for r in all_reviews if r.year <= t1_end_year]
        t1_research = [p for p in all_research 
                      if t0_end_year < p.year <= t1_end_year]
        
        t1 = TimeSlice(
            start_year=t0_end_year + 1,
            end_year=t1_end_year,
            reviews=t1_reviews,
            research_papers=t1_research,
            label="T1"
        )
        
        logger.info(f"T0: {len(t0_reviews)} reviews, {len(t0_research)} papers")
        logger.info(f"T1: {len(t1_reviews)} reviews, {len(t1_research)} research papers (new only)")
        
        self.time_slices = [t0, t1]
        
        return t0, t1
    
    def extract_ground_truth_shifts(
        self,
        t0_baseline: MultiViewBaseline,
        t1_baseline: MultiViewBaseline
    ) -> List[TaxonomyShift]:
        """
        Extract ground truth taxonomy shifts by comparing T0 and T1 baselines.
        
        This compares actual taxonomies from reviews at different time points
        to identify what changes occurred in the field's consensus.
        
        Args:
            t0_baseline: Baseline from T0 reviews
            t1_baseline: Baseline from T1 reviews
            
        Returns:
            List of detected TaxonomyShift objects
        """
        logger.info("Extracting ground truth taxonomy shifts...")
        
        shifts = []
        
        # Compare each view
        for t1_view in t1_baseline.views:
            # Find corresponding T0 view (same facet)
            t0_view = None
            for v in t0_baseline.views:
                if v.facet_label == t1_view.facet_label:
                    t0_view = v
                    break
            
            if not t0_view:
                logger.warning(f"No T0 view found for facet {t1_view.facet_label}")
                continue
            
            # Detect changes
            view_shifts = self._detect_tree_changes(
                t0_view.tree,
                t1_view.tree,
                t1_view.view_id
            )
            
            shifts.extend(view_shifts)
        
        logger.info(f"Detected {len(shifts)} ground truth shifts")
        
        self.ground_truth_shifts = shifts
        
        return shifts
    
    def _detect_tree_changes(
        self,
        t0_tree: TaxonomyTree,
        t1_tree: TaxonomyTree,
        view_id: str
    ) -> List[TaxonomyShift]:
        """Detect changes between two taxonomy trees."""
        
        shifts = []
        
        # Get all paths
        t0_paths = set(t0_tree.nodes.keys())
        t1_paths = set(t1_tree.nodes.keys())
        
        # New nodes (ADD operations)
        new_paths = t1_paths - t0_paths
        
        for path in new_paths:
            node = t1_tree.nodes[path]
            parent_path = path.rsplit('/', 1)[0] if '/' in path else ""
            
            shift = TaxonomyShift(
                view_id=view_id,
                parent_path=parent_path,
                operation_type="ADD",
                new_node_name=node.name,
                split_node_path=None,
                split_into=None,
                renamed_from=None,
                renamed_to=None,
                evidence_papers=[],
                justification=f"New node '{node.name}' appeared in T1 taxonomy"
            )
            
            shifts.append(shift)
        
        # Renamed nodes (approximate detection via name similarity)
        # Simplified: detect by checking if node names changed at same path
        common_paths = t0_paths & t1_paths
        
        for path in common_paths:
            t0_node = t0_tree.nodes[path]
            t1_node = t1_tree.nodes[path]
            
            if t0_node.name != t1_node.name:
                shift = TaxonomyShift(
                    view_id=view_id,
                    parent_path=path.rsplit('/', 1)[0] if '/' in path else "",
                    operation_type="RENAME",
                    new_node_name=None,
                    split_node_path=path,
                    split_into=None,
                    renamed_from=t0_node.name,
                    renamed_to=t1_node.name,
                    evidence_papers=[],
                    justification=f"Node renamed from '{t0_node.name}' to '{t1_node.name}'"
                )
                
                shifts.append(shift)
        
        # SPLIT operations (harder to detect automatically)
        # Simplified: if a T0 node disappears and multiple new nodes appear under its parent
        # (left for future enhancement)
        
        return shifts
    
    def save_to_file(self, filepath: str):
        """Save dataset to JSON file."""
        
        data = {
            "time_slices": [
                {
                    "start_year": ts.start_year,
                    "end_year": ts.end_year,
                    "label": ts.label,
                    "num_reviews": len(ts.reviews),
                    "num_research": len(ts.research_papers)
                }
                for ts in self.time_slices
            ],
            "ground_truth_shifts": [
                {
                    "view_id": shift.view_id,
                    "operation_type": shift.operation_type,
                    "parent_path": shift.parent_path,
                    "new_node_name": shift.new_node_name,
                    "split_node_path": shift.split_node_path,
                    "split_into": shift.split_into,
                    "renamed_from": shift.renamed_from,
                    "renamed_to": shift.renamed_to,
                    "justification": shift.justification
                }
                for shift in self.ground_truth_shifts
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved time-slice dataset to {filepath}")


class TimeSliceEvaluator:
    """
    Evaluates IG-Finder 2.0 predictions against ground truth time-slice data.
    """
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        predicted_proposal: EvolutionProposal,
        ground_truth_shifts: List[TaxonomyShift],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate predicted evolution against ground truth.
        
        Metrics:
        - Branch Hit@K: % of ground truth shifts where predicted operations
          target the correct branch (parent path)
        - Operation Type Accuracy
        - Node Name Match (for ADD operations)
        
        Args:
            predicted_proposal: Predicted evolution operations
            ground_truth_shifts: Ground truth shifts
            k_values: K values for Hit@K metric
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating predicted evolution against ground truth...")
        
        if not ground_truth_shifts:
            logger.warning("No ground truth shifts to evaluate against")
            return {}
        
        metrics = {}
        
        # Branch Hit@K
        for k in k_values:
            hit_at_k = self._compute_branch_hit_at_k(
                predicted_proposal.operations[:k],
                ground_truth_shifts
            )
            metrics[f"branch_hit@{k}"] = hit_at_k
        
        # Operation type accuracy
        op_type_acc = self._compute_operation_type_accuracy(
            predicted_proposal.operations,
            ground_truth_shifts
        )
        metrics["operation_type_accuracy"] = op_type_acc
        
        # Node name match (for ADD operations)
        node_name_match = self._compute_node_name_match(
            predicted_proposal.operations,
            ground_truth_shifts
        )
        metrics["node_name_match"] = node_name_match
        
        logger.info("Evaluation results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.3f}")
        
        return metrics
    
    def _compute_branch_hit_at_k(
        self,
        predicted_ops: List[EvolutionOperation],
        ground_truth: List[TaxonomyShift]
    ) -> float:
        """
        Compute Branch Hit@K metric.
        
        For each ground truth shift, check if any of the top-K predictions
        targets the correct parent path.
        """
        if not ground_truth:
            return 0.0
        
        hits = 0
        
        for gt_shift in ground_truth:
            # Check if any predicted operation matches this ground truth
            for pred_op in predicted_ops:
                # Check view and parent path
                if (pred_op.view_id == gt_shift.view_id and
                    pred_op.parent_path == gt_shift.parent_path):
                    hits += 1
                    break  # Count at most once per ground truth
        
        return hits / len(ground_truth)
    
    def _compute_operation_type_accuracy(
        self,
        predicted_ops: List[EvolutionOperation],
        ground_truth: List[TaxonomyShift]
    ) -> float:
        """
        Compute operation type accuracy.
        
        Among correctly located operations, what % have correct type?
        """
        if not ground_truth:
            return 0.0
        
        correct_type = 0
        total_matched = 0
        
        for gt_shift in ground_truth:
            for pred_op in predicted_ops:
                # Check if this prediction matches the location
                if (pred_op.view_id == gt_shift.view_id and
                    pred_op.parent_path == gt_shift.parent_path):
                    total_matched += 1
                    
                    # Check if type also matches
                    if pred_op.operation_type.value.upper() == gt_shift.operation_type.upper():
                        correct_type += 1
                    
                    break
        
        return correct_type / total_matched if total_matched > 0 else 0.0
    
    def _compute_node_name_match(
        self,
        predicted_ops: List[EvolutionOperation],
        ground_truth: List[TaxonomyShift]
    ) -> float:
        """
        Compute node name match for ADD operations.
        
        For ADD operations at correct location, what % have matching node name?
        (Using fuzzy matching)
        """
        if not ground_truth:
            return 0.0
        
        correct_name = 0
        total_add_matched = 0
        
        for gt_shift in ground_truth:
            if gt_shift.operation_type != "ADD":
                continue
            
            for pred_op in predicted_ops:
                if (pred_op.operation_type.value.upper() == "ADD" and
                    pred_op.view_id == gt_shift.view_id and
                    pred_op.parent_path == gt_shift.parent_path):
                    
                    total_add_matched += 1
                    
                    # Check name match (case-insensitive, fuzzy)
                    if hasattr(pred_op, 'new_node') and pred_op.new_node:
                        pred_name = pred_op.new_node.name.lower()
                        gt_name = gt_shift.new_node_name.lower() if gt_shift.new_node_name else ""
                        
                        # Simple fuzzy match: check if core words overlap
                        pred_words = set(pred_name.split())
                        gt_words = set(gt_name.split())
                        
                        overlap = len(pred_words & gt_words)
                        if overlap > 0:
                            correct_name += 1
                    
                    break
        
        return correct_name / total_add_matched if total_add_matched > 0 else 0.0
