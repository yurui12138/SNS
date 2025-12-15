"""
Human evaluation interface for IG-Finder 2.0.

Provides structured tools for collecting human judgments on:
1. Definition Quality
2. Evidence Sufficiency
3. Evolution Necessity
4. Guidance Usefulness
"""
import logging
import json
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

from ..dataclass_v2 import (
    NodeDefinition,
    EvolutionOperation,
    DeltaAwareGuidance,
    EvidenceSpan
)

logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    """Four evaluation dimensions from design doc."""
    DEFINITION_QUALITY = "definition_quality"
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"
    EVOLUTION_NECESSITY = "evolution_necessity"
    GUIDANCE_USEFULNESS = "guidance_usefulness"


@dataclass
class HumanRating:
    """A single human rating."""
    evaluator_id: str
    dimension: EvaluationDimension
    target_type: str  # "node_definition", "evolution_operation", "guidance"
    target_id: str
    score: int  # 1-5 scale
    comment: Optional[str]
    timestamp: str


@dataclass
class EvaluationTask:
    """A task for human evaluation."""
    task_id: str
    dimension: EvaluationDimension
    target_type: str
    target_id: str
    context: Dict  # Relevant context for evaluation
    question: str
    rating_scale: Dict[int, str]  # Mapping score to description


class HumanEvaluationInterface:
    """
    Interface for collecting and managing human evaluations.
    
    Supports 4 evaluation dimensions:
    1. Definition Quality (D): Clarity, testability of node definitions
    2. Evidence Sufficiency (E): Adequacy of supporting evidence
    3. Evolution Necessity (N): Justification for proposed changes
    4. Guidance Usefulness (U): Value of generated writing guidance
    """
    
    def __init__(self, output_dir: str = "./human_eval"):
        self.output_dir = output_dir
        self.tasks: List[EvaluationTask] = []
        self.ratings: List[HumanRating] = []
        
        # Rating scales for each dimension
        self.rating_scales = {
            EvaluationDimension.DEFINITION_QUALITY: {
                1: "Very unclear, not testable",
                2: "Somewhat unclear, hard to apply",
                3: "Acceptable clarity, moderately testable",
                4: "Clear and testable with minor issues",
                5: "Excellent clarity, highly testable"
            },
            EvaluationDimension.EVIDENCE_SUFFICIENCY: {
                1: "No supporting evidence",
                2: "Minimal evidence, unconvincing",
                3: "Some evidence, partially convincing",
                4: "Good evidence, mostly convincing",
                5: "Strong evidence, fully convincing"
            },
            EvaluationDimension.EVOLUTION_NECESSITY: {
                1: "Unnecessary change, no justification",
                2: "Weak justification, questionable necessity",
                3: "Moderate justification, somewhat necessary",
                4: "Good justification, likely necessary",
                5: "Strong justification, clearly necessary"
            },
            EvaluationDimension.GUIDANCE_USEFULNESS: {
                1: "Not useful, confusing or irrelevant",
                2: "Minimally useful, hard to follow",
                3: "Moderately useful, decent structure",
                4: "Quite useful, good structure and clarity",
                5: "Extremely useful, excellent guidance"
            }
        }
    
    def create_definition_quality_tasks(
        self,
        view_id: str,
        node_definitions: Dict[str, NodeDefinition]
    ) -> List[EvaluationTask]:
        """
        Create evaluation tasks for node definition quality.
        
        Args:
            view_id: Taxonomy view ID
            node_definitions: Map of node_path to NodeDefinition
            
        Returns:
            List of evaluation tasks
        """
        tasks = []
        
        for node_path, definition in node_definitions.items():
            task = EvaluationTask(
                task_id=f"def_quality_{view_id}_{node_path}",
                dimension=EvaluationDimension.DEFINITION_QUALITY,
                target_type="node_definition",
                target_id=f"{view_id}::{node_path}",
                context={
                    "view_id": view_id,
                    "node_path": node_path,
                    "definition": definition.definition,
                    "inclusion_criteria": definition.inclusion_criteria,
                    "exclusion_criteria": definition.exclusion_criteria,
                    "keywords": definition.canonical_keywords
                },
                question=(
                    f"Evaluate the quality of this node definition for '{node_path}':\n\n"
                    f"Definition: {definition.definition}\n\n"
                    f"Inclusion criteria: {', '.join(definition.inclusion_criteria)}\n"
                    f"Exclusion criteria: {', '.join(definition.exclusion_criteria)}\n\n"
                    "How clear and testable is this definition?"
                ),
                rating_scale=self.rating_scales[EvaluationDimension.DEFINITION_QUALITY]
            )
            
            tasks.append(task)
        
        self.tasks.extend(tasks)
        return tasks
    
    def create_evidence_sufficiency_tasks(
        self,
        operations: List[EvolutionOperation]
    ) -> List[EvaluationTask]:
        """
        Create evaluation tasks for evidence sufficiency.
        
        Args:
            operations: List of evolution operations
            
        Returns:
            List of evaluation tasks
        """
        tasks = []
        
        for i, operation in enumerate(operations):
            # Extract evidence
            evidence_text = self._format_evidence(operation.evidence)
            
            task = EvaluationTask(
                task_id=f"evidence_{operation.operation_type.value}_{i}",
                dimension=EvaluationDimension.EVIDENCE_SUFFICIENCY,
                target_type="evolution_operation",
                target_id=f"{operation.view_id}::{operation.parent_path}",
                context={
                    "operation_type": operation.operation_type.value,
                    "view_id": operation.view_id,
                    "parent_path": operation.parent_path,
                    "evidence": evidence_text
                },
                question=(
                    f"Evaluate the evidence for this {operation.operation_type.value} operation:\n\n"
                    f"Location: {operation.view_id} > {operation.parent_path}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    "Is the evidence sufficient to support this operation?"
                ),
                rating_scale=self.rating_scales[EvaluationDimension.EVIDENCE_SUFFICIENCY]
            )
            
            tasks.append(task)
        
        self.tasks.extend(tasks)
        return tasks
    
    def create_evolution_necessity_tasks(
        self,
        operations: List[EvolutionOperation]
    ) -> List[EvaluationTask]:
        """
        Create evaluation tasks for evolution necessity.
        
        Args:
            operations: List of evolution operations
            
        Returns:
            List of evaluation tasks
        """
        tasks = []
        
        for i, operation in enumerate(operations):
            task = EvaluationTask(
                task_id=f"necessity_{operation.operation_type.value}_{i}",
                dimension=EvaluationDimension.EVOLUTION_NECESSITY,
                target_type="evolution_operation",
                target_id=f"{operation.view_id}::{operation.parent_path}",
                context={
                    "operation_type": operation.operation_type.value,
                    "view_id": operation.view_id,
                    "parent_path": operation.parent_path,
                    "fit_gain": operation.fit_gain,
                    "edit_cost": operation.edit_cost
                },
                question=(
                    f"Evaluate the necessity of this {operation.operation_type.value} operation:\n\n"
                    f"Location: {operation.view_id} > {operation.parent_path}\n"
                    f"Expected Fit Gain: {operation.fit_gain:.3f}\n"
                    f"Edit Cost: {operation.edit_cost:.3f}\n\n"
                    "Is this structural change necessary?"
                ),
                rating_scale=self.rating_scales[EvaluationDimension.EVOLUTION_NECESSITY]
            )
            
            tasks.append(task)
        
        self.tasks.extend(tasks)
        return tasks
    
    def create_guidance_usefulness_tasks(
        self,
        guidance: DeltaAwareGuidance
    ) -> List[EvaluationTask]:
        """
        Create evaluation tasks for guidance usefulness.
        
        Args:
            guidance: Delta-aware guidance from Phase 4
            
        Returns:
            List of evaluation tasks
        """
        tasks = []
        
        # Overall guidance
        task = EvaluationTask(
            task_id=f"guidance_overall",
            dimension=EvaluationDimension.GUIDANCE_USEFULNESS,
            target_type="guidance",
            target_id="overall",
            context={
                "topic": guidance.topic,
                "main_axis": guidance.main_axis.facet_label.value,
                "aux_axis": guidance.aux_axis.facet_label.value if guidance.aux_axis else None,
                "num_sections": len(guidance.outline),
                "num_questions": len(guidance.must_answer_questions)
            },
            question=(
                f"Evaluate the usefulness of this survey writing guidance:\n\n"
                f"Topic: {guidance.topic}\n"
                f"Main Axis: {guidance.main_axis.facet_label.value}\n"
                f"Sections: {len(guidance.outline)}\n"
                f"Must-Answer Questions: {len(guidance.must_answer_questions)}\n\n"
                "How useful would this guidance be for writing a survey paper?"
            ),
            rating_scale=self.rating_scales[EvaluationDimension.GUIDANCE_USEFULNESS]
        )
        
        tasks.append(task)
        self.tasks.extend(tasks)
        
        return tasks
    
    def _format_evidence(self, evidence: List[EvidenceSpan]) -> str:
        """Format evidence spans for display."""
        if not evidence:
            return "No evidence provided"
        
        lines = []
        for i, span in enumerate(evidence[:5], 1):  # Show first 5
            lines.append(f"{i}. {span.claim}")
            if span.quote:
                lines.append(f"   Quote: \"{span.quote[:150]}...\"")
        
        return "\n".join(lines)
    
    def submit_rating(
        self,
        evaluator_id: str,
        task_id: str,
        score: int,
        comment: Optional[str] = None
    ):
        """
        Submit a rating for a task.
        
        Args:
            evaluator_id: ID of the human evaluator
            task_id: ID of the task being rated
            score: Rating score (1-5)
            comment: Optional comment
        """
        # Find task
        task = None
        for t in self.tasks:
            if t.task_id == task_id:
                task = t
                break
        
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        # Validate score
        if score not in [1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid score: {score}. Must be 1-5.")
        
        # Create rating
        rating = HumanRating(
            evaluator_id=evaluator_id,
            dimension=task.dimension,
            target_type=task.target_type,
            target_id=task.target_id,
            score=score,
            comment=comment,
            timestamp=datetime.now().isoformat()
        )
        
        self.ratings.append(rating)
        logger.info(f"Rating submitted: {evaluator_id} rated {task_id} = {score}")
    
    def export_tasks_to_file(self, filepath: str):
        """Export evaluation tasks to JSON file for distribution."""
        
        tasks_data = [
            {
                "task_id": task.task_id,
                "dimension": task.dimension.value,
                "target_type": task.target_type,
                "target_id": task.target_id,
                "question": task.question,
                "context": task.context,
                "rating_scale": task.rating_scale
            }
            for task in self.tasks
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(tasks_data)} evaluation tasks to {filepath}")
    
    def export_ratings_to_file(self, filepath: str):
        """Export collected ratings to JSON file."""
        
        ratings_data = [asdict(rating) for rating in self.ratings]
        
        # Convert enum to string
        for rating_dict in ratings_data:
            rating_dict['dimension'] = rating_dict['dimension'].value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ratings_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(ratings_data)} ratings to {filepath}")
    
    def compute_inter_rater_agreement(self) -> Dict[str, float]:
        """
        Compute inter-rater agreement metrics.
        
        Returns:
            Dictionary with agreement statistics
        """
        if len(self.ratings) < 2:
            logger.warning("Need at least 2 ratings to compute agreement")
            return {}
        
        # Group ratings by target
        target_ratings = {}
        for rating in self.ratings:
            key = rating.target_id
            if key not in target_ratings:
                target_ratings[key] = []
            target_ratings[key].append(rating.score)
        
        # Compute average pairwise agreement
        agreements = []
        
        for target_id, scores in target_ratings.items():
            if len(scores) < 2:
                continue
            
            # Pairwise exact agreement
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    if abs(scores[i] - scores[j]) <= 1:  # Allow 1-point difference
                        agreements.append(1.0)
                    else:
                        agreements.append(0.0)
        
        if not agreements:
            return {}
        
        return {
            "pairwise_agreement": sum(agreements) / len(agreements),
            "num_rated_targets": len(target_ratings),
            "total_ratings": len(self.ratings)
        }
    
    def get_dimension_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each evaluation dimension.
        
        Returns:
            Dictionary mapping dimension to statistics
        """
        import numpy as np
        
        dimension_scores = {}
        
        for rating in self.ratings:
            dim = rating.dimension.value
            if dim not in dimension_scores:
                dimension_scores[dim] = []
            dimension_scores[dim].append(rating.score)
        
        statistics = {}
        
        for dim, scores in dimension_scores.items():
            statistics[dim] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "count": len(scores)
            }
        
        return statistics
