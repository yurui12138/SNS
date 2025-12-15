"""
Taxonomy Evolution Applier

This module implements the application of evolution operations to taxonomy views,
creating evolved versions that incorporate structural updates (ADD_NODE, SPLIT_NODE, RENAME_NODE).

This is critical for Phase 4 to generate the final updated taxonomy (taxonomy_v2)
that reflects the structural changes proposed in Phase 3.

Reference: SNS Method Specification - Phase 3 Structural Updates & Phase 4 Output
"""

import logging
import copy
from typing import List, Dict, Optional
from dataclasses import replace

from ..dataclass_v2 import (
    TaxonomyView,
    TaxonomyTree,
    TaxonomyTreeNode,
    NodeDefinition,
    EvolutionOperation,
    AddNodeOperation,
    SplitNodeOperation,
    RenameNodeOperation,
    OperationType,
    NewNodeProposal,
    EvidenceSpan,
)

logger = logging.getLogger(__name__)


class TaxonomyEvolutionApplier:
    """
    Applies evolution operations to a taxonomy view to create an evolved version.
    
    Supports:
    1. ADD_NODE: Add new category nodes to taxonomy
    2. SPLIT_NODE: Split overcrowded nodes into sub-nodes
    3. RENAME_NODE: Rename nodes that have undergone semantic drift
    """
    
    def __init__(self):
        pass
    
    def apply_evolution(
        self,
        base_view: TaxonomyView,
        operations: List[EvolutionOperation]
    ) -> TaxonomyView:
        """
        Apply evolution operations to create an evolved taxonomy view.
        
        Args:
            base_view: Original taxonomy view
            operations: List of evolution operations to apply
            
        Returns:
            Evolved TaxonomyView with structural updates
        """
        logger.info(f"Applying {len(operations)} evolution operations to view {base_view.view_id}")
        
        # Deep copy to avoid modifying original
        evolved_view = self._deep_copy_view(base_view)
        
        # Filter operations for this view
        view_operations = [op for op in operations if op.view_id == base_view.view_id]
        
        if not view_operations:
            logger.info(f"No operations for view {base_view.view_id}, returning unchanged")
            return evolved_view
        
        logger.info(f"Applying {len(view_operations)} operation(s) to view {base_view.view_id}")
        
        # Apply operations in order: RENAME -> SPLIT -> ADD
        # This order minimizes conflicts
        rename_ops = [op for op in view_operations if op.operation_type == OperationType.RENAME_NODE]
        split_ops = [op for op in view_operations if op.operation_type == OperationType.SPLIT_NODE]
        add_ops = [op for op in view_operations if op.operation_type == OperationType.ADD_NODE]
        
        # Apply RENAME operations first
        for op in rename_ops:
            self._apply_rename(evolved_view, op)
        
        # Apply SPLIT operations second
        for op in split_ops:
            self._apply_split(evolved_view, op)
        
        # Apply ADD operations last
        for op in add_ops:
            self._apply_add(evolved_view, op)
        
        logger.info(f"Evolution complete. View {evolved_view.view_id} now has "
                   f"{len(evolved_view.tree.nodes)} nodes "
                   f"({len(evolved_view.tree.get_leaf_nodes())} leaves)")
        
        return evolved_view
    
    def _deep_copy_view(self, view: TaxonomyView) -> TaxonomyView:
        """Create a deep copy of the taxonomy view."""
        # Deep copy tree
        new_tree = self._deep_copy_tree(view.tree)
        
        # Deep copy node definitions
        new_definitions = {
            path: self._deep_copy_definition(defn)
            for path, defn in view.node_definitions.items()
        }
        
        # Create new view (most fields are immutable or can be shallow copied)
        new_view = TaxonomyView(
            view_id=view.view_id,
            review_id=view.review_id,
            review_title=view.review_title,
            facet_label=view.facet_label,
            facet_rationale=view.facet_rationale,
            tree=new_tree,
            node_definitions=new_definitions,
            weight=view.weight,
            evidence=view.evidence.copy()
        )
        
        return new_view
    
    def _deep_copy_tree(self, tree: TaxonomyTree) -> TaxonomyTree:
        """Deep copy taxonomy tree."""
        new_root = TaxonomyTreeNode(
            name=tree.root.name,
            path=tree.root.path,
            parent=tree.root.parent,
            children=tree.root.children.copy(),
            is_leaf=tree.root.is_leaf
        )
        
        new_tree = TaxonomyTree(root=new_root)
        
        # Copy all nodes
        for path, node in tree.nodes.items():
            if path == tree.root.path:
                continue  # Root already added
            
            new_node = TaxonomyTreeNode(
                name=node.name,
                path=node.path,
                parent=node.parent,
                children=node.children.copy(),
                is_leaf=node.is_leaf
            )
            
            new_tree.nodes[path] = new_node
        
        return new_tree
    
    def _deep_copy_definition(self, defn: NodeDefinition) -> NodeDefinition:
        """Deep copy node definition."""
        return NodeDefinition(
            node_path=defn.node_path,
            definition=defn.definition,
            inclusion_criteria=defn.inclusion_criteria.copy(),
            exclusion_criteria=defn.exclusion_criteria.copy(),
            canonical_keywords=defn.canonical_keywords.copy(),
            boundary_statements=defn.boundary_statements.copy(),
            evidence_spans=defn.evidence_spans.copy()
        )
    
    def _apply_add(self, view: TaxonomyView, operation: AddNodeOperation):
        """
        Apply ADD_NODE operation.
        
        Adds a new node under the specified parent.
        """
        logger.info(f"  Applying ADD_NODE: {operation.new_node.name} under {operation.parent_path}")
        
        # Verify parent exists
        if operation.parent_path not in view.tree.nodes:
            logger.warning(f"  Parent node {operation.parent_path} not found, skipping")
            return
        
        # Create new node path
        new_node_path = f"{operation.parent_path}/{operation.new_node.name}"
        
        # Check if node already exists (avoid duplicates)
        if new_node_path in view.tree.nodes:
            logger.warning(f"  Node {new_node_path} already exists, skipping")
            return
        
        # Create new node
        new_node = TaxonomyTreeNode(
            name=operation.new_node.name,
            path=new_node_path,
            parent=operation.parent_path,
            children=[],
            is_leaf=True
        )
        
        # Add to tree
        view.tree.add_node(new_node)
        
        # Update parent's is_leaf status
        parent_node = view.tree.nodes[operation.parent_path]
        parent_node.is_leaf = False
        
        # Create node definition
        definition = NodeDefinition(
            node_path=new_node_path,
            definition=operation.new_node.definition,
            inclusion_criteria=operation.new_node.inclusion_criteria,
            exclusion_criteria=operation.new_node.exclusion_criteria,
            canonical_keywords=operation.new_node.keywords,
            boundary_statements=[],
            evidence_spans=operation.evidence
        )
        
        view.node_definitions[new_node_path] = definition
        
        logger.info(f"  ✓ Added node {new_node_path}")
    
    def _apply_split(self, view: TaxonomyView, operation: SplitNodeOperation):
        """
        Apply SPLIT_NODE operation.
        
        Splits an existing node into sub-nodes.
        """
        logger.info(f"  Applying SPLIT_NODE: {operation.node_path} into {len(operation.sub_nodes)} sub-nodes")
        
        # Verify target node exists
        if operation.node_path not in view.tree.nodes:
            logger.warning(f"  Target node {operation.node_path} not found, skipping")
            return
        
        target_node = view.tree.nodes[operation.node_path]
        
        # Add each sub-node as a child
        for sub_node_proposal in operation.sub_nodes:
            sub_node_path = f"{operation.node_path}/{sub_node_proposal.name}"
            
            # Check if sub-node already exists
            if sub_node_path in view.tree.nodes:
                logger.warning(f"  Sub-node {sub_node_path} already exists, skipping")
                continue
            
            # Create sub-node
            sub_node = TaxonomyTreeNode(
                name=sub_node_proposal.name,
                path=sub_node_path,
                parent=operation.node_path,
                children=[],
                is_leaf=True
            )
            
            # Add to tree
            view.tree.add_node(sub_node)
            
            # Create node definition
            definition = NodeDefinition(
                node_path=sub_node_path,
                definition=sub_node_proposal.definition,
                inclusion_criteria=sub_node_proposal.inclusion_criteria,
                exclusion_criteria=sub_node_proposal.exclusion_criteria,
                canonical_keywords=sub_node_proposal.keywords,
                boundary_statements=[],
                evidence_spans=operation.evidence
            )
            
            view.node_definitions[sub_node_path] = definition
            
            logger.info(f"    ✓ Added sub-node {sub_node_path}")
        
        # Update target node's is_leaf status
        target_node.is_leaf = False
        
        logger.info(f"  ✓ Split {operation.node_path} into {len(operation.sub_nodes)} sub-nodes")
    
    def _apply_rename(self, view: TaxonomyView, operation: RenameNodeOperation):
        """
        Apply RENAME_NODE operation.
        
        Renames an existing node and updates its definition.
        """
        logger.info(f"  Applying RENAME_NODE: {operation.node_path} from '{operation.old_name}' to '{operation.new_name}'")
        
        # Verify target node exists
        if operation.node_path not in view.tree.nodes:
            logger.warning(f"  Target node {operation.node_path} not found, skipping")
            return
        
        target_node = view.tree.nodes[operation.node_path]
        
        # Verify old name matches
        if target_node.name != operation.old_name:
            logger.warning(f"  Old name mismatch: expected '{operation.old_name}', found '{target_node.name}', skipping")
            return
        
        # Compute new path
        parent_path = target_node.parent
        new_node_path = f"{parent_path}/{operation.new_name}" if parent_path else operation.new_name
        
        # Check if new path already exists (avoid collisions)
        if new_node_path in view.tree.nodes and new_node_path != operation.node_path:
            logger.warning(f"  New path {new_node_path} already exists, skipping")
            return
        
        # Update node name and path
        old_path = operation.node_path
        target_node.name = operation.new_name
        target_node.path = new_node_path
        
        # Update tree nodes dict
        del view.tree.nodes[old_path]
        view.tree.nodes[new_node_path] = target_node
        
        # Update parent's children list
        if parent_path and parent_path in view.tree.nodes:
            parent_node = view.tree.nodes[parent_path]
            if old_path in parent_node.children:
                parent_node.children.remove(old_path)
                parent_node.children.append(new_node_path)
        
        # Update root if necessary
        if view.tree.root.path == old_path:
            view.tree.root = target_node
        
        # Update children's parent references
        for child_path in target_node.children.copy():
            if child_path in view.tree.nodes:
                child_node = view.tree.nodes[child_path]
                child_node.parent = new_node_path
                
                # Update child path
                old_child_path = child_path
                new_child_path = child_path.replace(old_path, new_node_path, 1)
                child_node.path = new_child_path
                
                # Update tree nodes dict
                del view.tree.nodes[old_child_path]
                view.tree.nodes[new_child_path] = child_node
                
                # Update target node's children list
                target_node.children.remove(old_child_path)
                target_node.children.append(new_child_path)
                
                # Recursively update descendants
                self._update_descendant_paths(view.tree, child_node, old_child_path, new_child_path)
        
        # Update node definition
        if old_path in view.node_definitions:
            old_definition = view.node_definitions[old_path]
            new_definition = NodeDefinition(
                node_path=new_node_path,
                definition=operation.new_definition,
                inclusion_criteria=old_definition.inclusion_criteria,
                exclusion_criteria=old_definition.exclusion_criteria,
                canonical_keywords=old_definition.canonical_keywords,
                boundary_statements=old_definition.boundary_statements,
                evidence_spans=operation.evidence
            )
            
            del view.node_definitions[old_path]
            view.node_definitions[new_node_path] = new_definition
        
        logger.info(f"  ✓ Renamed {old_path} to {new_node_path}")
    
    def _update_descendant_paths(
        self,
        tree: TaxonomyTree,
        node: TaxonomyTreeNode,
        old_prefix: str,
        new_prefix: str
    ):
        """
        Recursively update paths of all descendants after a rename.
        """
        for child_path in node.children.copy():
            if child_path in tree.nodes:
                child_node = tree.nodes[child_path]
                
                # Update child path
                old_child_path = child_path
                new_child_path = child_path.replace(old_prefix, new_prefix, 1)
                child_node.path = new_child_path
                
                # Update tree nodes dict
                del tree.nodes[old_child_path]
                tree.nodes[new_child_path] = child_node
                
                # Update parent's children list
                node.children.remove(old_child_path)
                node.children.append(new_child_path)
                
                # Recurse for grandchildren
                self._update_descendant_paths(tree, child_node, old_child_path, new_child_path)
