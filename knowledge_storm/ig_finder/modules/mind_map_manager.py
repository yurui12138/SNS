"""
Dynamic Mind Map Manager

Manages the evolution of the mind map throughout the IG-Finder process,
tracking evolution states and deviation metrics.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from ..dataclass import KnowledgeBase
from ..dataclass import (
    CognitiveBaseline,
    EvolutionState,
    ExtendedKnowledgeNode,
    InnovationCluster,
    ResearchPaper,
    DeviationAnalysis,
)

logger = logging.getLogger(__name__)


class EvolutionStateAnnotator:
    """
    Annotates knowledge nodes with evolution states based on analysis results.
    """
    
    def annotate_with_frontier_analysis(
        self,
        mind_map: KnowledgeBase,
        papers_with_deviations: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]],
        innovation_clusters: List[InnovationCluster],
    ):
        """
        Annotate mind map nodes based on frontier paper analysis.
        
        Args:
            mind_map: The cognitive baseline mind map
            papers_with_deviations: Papers and their deviation analyses
            innovation_clusters: Identified innovation clusters
        """
        logger.info("Annotating mind map with evolution states...")
        
        # Create a map of cluster papers for quick lookup
        cluster_paper_urls = set()
        for cluster in innovation_clusters:
            for paper in cluster.core_papers:
                cluster_paper_urls.add(paper.url)
        
        # Process each paper
        for paper, deviations in papers_with_deviations:
            # Calculate average deviation score
            avg_deviation = sum(d.deviation_score for d in deviations.values()) / len(deviations)
            
            # Determine evolution state
            if paper.url in cluster_paper_urls:
                evolution_state = EvolutionState.INNOVATION
            elif avg_deviation > 0.7:
                evolution_state = EvolutionState.DEVIATION
            elif avg_deviation > 0.3:
                evolution_state = EvolutionState.POTENTIAL_GAP
            else:
                evolution_state = EvolutionState.CONTINUATION
            
            # Get matched baseline concepts from first expert's analysis
            first_deviation = list(deviations.values())[0]
            matched_concepts = first_deviation.baseline_node_path
            
            # Find or create nodes in mind map
            self._add_or_update_node(
                mind_map,
                paper,
                matched_concepts,
                evolution_state,
                first_deviation,
            )
        
        logger.info("Mind map annotation complete")
    
    def _add_or_update_node(
        self,
        mind_map: KnowledgeBase,
        paper: ResearchPaper,
        concept_path: List[str],
        evolution_state: EvolutionState,
        deviation: DeviationAnalysis,
    ):
        """Add or update a node in the mind map."""
        root = mind_map.root
        
        if not concept_path:
            return
        
        # Navigate to the matched concept or create path
        current_node = root
        for concept_name in concept_path:
            # Find child with this name
            found_child = None
            for child in current_node.children:
                if child.name.lower() == concept_name.lower():
                    found_child = child
                    break
            
            if found_child:
                current_node = found_child
            else:
                # Create new node
                new_node = ExtendedKnowledgeNode(
                    name=concept_name,
                    parent=current_node,
                    evolution_state=EvolutionState.CONSENSUS,
                    source_papers=[],
                )
                current_node.children.append(new_node)
                current_node = new_node
        
        # Now current_node is where we should add the paper's contribution
        # Create a child node for the specific paper insight
        paper_node_name = f"{paper.title[:50]}..." if len(paper.title) > 50 else paper.title
        
        paper_node = ExtendedKnowledgeNode(
            name=paper_node_name,
            parent=current_node,
            evolution_state=evolution_state,
            source_papers=[paper.url],
            deviation_metrics={
                "deviation_score": deviation.deviation_score,
                "deviation_dimensions": deviation.deviation_dimensions,
                "deviation_description": deviation.deviation_description,
            },
        )
        paper_node.synthesize_output = deviation.deviation_description
        current_node.children.append(paper_node)


class DynamicMindMapManager:
    """
    Manages the dynamic evolution of the mind map throughout IG-Finder process.
    """
    
    def __init__(self):
        self.annotator = EvolutionStateAnnotator()
    
    def update_with_phase2_results(
        self,
        cognitive_baseline: CognitiveBaseline,
        papers_with_deviations: List[Tuple[ResearchPaper, Dict[str, DeviationAnalysis]]],
        innovation_clusters: List[InnovationCluster],
    ) -> KnowledgeBase:
        """
        Update the mind map with Phase 2 analysis results.
        
        Args:
            cognitive_baseline: The cognitive baseline with consensus mind map
            papers_with_deviations: Papers and their deviation analyses
            innovation_clusters: Identified innovation clusters
            
        Returns:
            Updated KnowledgeBase with evolution states
        """
        logger.info("Updating mind map with Phase 2 results...")
        
        # Get the consensus map
        mind_map = cognitive_baseline.consensus_map
        
        # Annotate with frontier analysis
        self.annotator.annotate_with_frontier_analysis(
            mind_map,
            papers_with_deviations,
            innovation_clusters,
        )
        
        logger.info("Mind map update complete")
        return mind_map
    
    def identify_innovation_paths(self, mind_map: KnowledgeBase) -> List[List[str]]:
        """
        Identify paths in the mind map that lead to INNOVATION nodes.
        
        Args:
            mind_map: The updated mind map
            
        Returns:
            List of paths (each path is a list of node names from root to innovation node)
        """
        innovation_paths = []
        
        def traverse(node: ExtendedKnowledgeNode, current_path: List[str]):
            current_path = current_path + [node.name]
            
            if hasattr(node, 'evolution_state') and node.evolution_state == EvolutionState.INNOVATION:
                innovation_paths.append(current_path)
            
            for child in node.children:
                traverse(child, current_path)
        
        traverse(mind_map.root, [])
        return innovation_paths
    
    def get_evolution_state_distribution(self, mind_map: KnowledgeBase) -> Dict[str, int]:
        """
        Get distribution of evolution states in the mind map.
        
        Returns:
            Dictionary mapping state names to counts
        """
        state_counts = {state.value: 0 for state in EvolutionState}
        
        def count_states(node: ExtendedKnowledgeNode):
            if hasattr(node, 'evolution_state'):
                state_counts[node.evolution_state.value] += 1
            
            for child in node.children:
                count_states(child)
        
        count_states(mind_map.root)
        return state_counts
    
    def export_visualization_data(self, mind_map: KnowledgeBase) -> Dict:
        """
        Export mind map data for visualization.
        
        Returns:
            Dictionary with visualization-ready data
        """
        def node_to_dict(node: ExtendedKnowledgeNode) -> Dict:
            node_dict = {
                "name": node.name,
                "evolution_state": node.evolution_state.value if hasattr(node, 'evolution_state') else "consensus",
                "children": [node_to_dict(child) for child in node.children],
            }
            
            if hasattr(node, 'deviation_metrics') and node.deviation_metrics:
                node_dict["deviation_score"] = node.deviation_metrics.get("deviation_score", 0)
                node_dict["deviation_dimensions"] = node.deviation_metrics.get("deviation_dimensions", [])
            
            if hasattr(node, 'source_papers') and node.source_papers:
                node_dict["source_papers"] = node.source_papers
            
            return node_dict
        
        visualization_data = {
            "root": node_to_dict(mind_map.root),
            "statistics": self.get_evolution_state_distribution(mind_map),
            "innovation_paths": self.identify_innovation_paths(mind_map),
        }
        
        return visualization_data
