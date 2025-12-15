"""
DSPy schemas for IG-Finder 2.0 LLM tasks.

All LLM tasks have fixed JSON schemas with temperature=0 for reproducibility.
"""
import dspy
from typing import List, Optional


# ============================================================================
# Phase 1: Multi-view Baseline Schemas
# ============================================================================

class TaxonomyExtractionSignature(dspy.Signature):
    """
    Extract taxonomy structure from a review paper.
    
    Input: Review paper (title, abstract, full text)
    Output: JSON with taxonomy tree, facet label, evidence spans
    """
    review_title = dspy.InputField(desc="Title of the review paper")
    review_abstract = dspy.InputField(desc="Abstract of the review paper")
    review_text = dspy.InputField(desc="Full text of the review paper (first 10000 chars)")
    
    taxonomy_json = dspy.OutputField(
        desc="""JSON object with the following structure:
{
  "review_id": "arxiv:xxxx.xxxxx or url",
  "facet_label": "MODEL_ARCHITECTURE | TRAINING_PARADIGM | TASK_SETTING | THEORY | EVALUATION_PROTOCOL | APPLICATION_DOMAIN | DATA_PARADIGM | OTHER",
  "facet_rationale": "One sentence explaining the primary organizational dimension",
  "taxonomy_tree": {
    "name": "ROOT",
    "children": [
      {
        "name": "Category1",
        "children": [
          {"name": "Subcategory1", "children": []},
          {"name": "Subcategory2", "children": []}
        ]
      }
    ]
  },
  "evidence_spans": [
    {
      "claim": "The survey organizes methods by...",
      "page": 2,
      "section": "Introduction",
      "char_start": 0,
      "char_end": 100,
      "quote": "actual text from paper"
    }
  ]
}

REQUIREMENTS:
- facet_label must be one of the listed values
- taxonomy_tree must be hierarchical (at least 2 levels)
- evidence_spans must cite actual text from the review
- Output ONLY valid JSON, no other text
"""
    )


class NodeDefinitionSignature(dspy.Signature):
    """
    Generate testable definition for a taxonomy node.
    
    Input: Node name, node path, surrounding context from review
    Output: JSON with definition, inclusion/exclusion criteria, keywords, boundaries
    """
    node_name = dspy.InputField(desc="Name of the taxonomy node")
    node_path = dspy.InputField(desc="Full path from root (e.g., ROOT/CNN-based/ResNet)")
    review_context = dspy.InputField(desc="Relevant text from the review about this node")
    parent_definition = dspy.InputField(desc="Definition of the parent node (if any)", default="")
    
    definition_json = dspy.OutputField(
        desc="""JSON object with the following structure:
{
  "node_path": "ROOT/CNN-based",
  "definition": "Clear definition of what papers belong in this category",
  "inclusion_criteria": [
    "Criterion 1: Papers that...",
    "Criterion 2: Methods that..."
  ],
  "exclusion_criteria": [
    "Excludes papers that...",
    "Not suitable for..."
  ],
  "canonical_keywords": ["keyword1", "keyword2", "keyword3"],
  "boundary_statements": [
    "Limitation: Not suitable when...",
    "Edge case: May overlap with X when..."
  ],
  "evidence_spans": [
    {
      "claim": "This category includes...",
      "page": 3,
      "section": "Taxonomy",
      "char_start": 0,
      "char_end": 100,
      "quote": "actual text"
    }
  ]
}

REQUIREMENTS:
- Definition must be clear and testable
- Include at least 2 inclusion criteria and 2 exclusion criteria
- canonical_keywords: 5-10 key terms
- boundary_statements: describe limitations and edge cases
- evidence_spans must reference actual text from review
- Output ONLY valid JSON
"""
    )


# ============================================================================
# Phase 2: Stress Test Schemas
# ============================================================================

class PaperClaimExtractionSignature(dspy.Signature):
    """
    Extract structured claims from a research paper.
    
    Input: Paper title, abstract, (ideally full text)
    Output: JSON with problem, core idea, mechanism, novelty bullets, keywords
    """
    paper_title = dspy.InputField(desc="Title of the research paper")
    paper_abstract = dspy.InputField(desc="Abstract of the research paper")
    paper_text = dspy.InputField(desc="Full text of the paper (if available, first 15000 chars)")
    
    claims_json = dspy.OutputField(
        desc="""JSON object with the following structure:
{
  "paper_id": "arxiv:xxxx or url",
  "claims": {
    "problem": {
      "text": "The paper addresses the problem of...",
      "evidence": [{"page": 1, "section": "Introduction", "quote": "..."}]
    },
    "core_idea": [
      {
        "text": "The key idea is...",
        "evidence": [{"page": 2, "section": "Method", "quote": "..."}]
      }
    ],
    "mechanism": [
      {
        "text": "The method works by...",
        "evidence": [{"page": 3, "section": "Method", "quote": "..."}]
      }
    ],
    "training": [
      {
        "text": "Training procedure...",
        "evidence": [{"page": 4, "section": "Experiments", "quote": "..."}]
      }
    ],
    "evaluation": [
      {
        "text": "Evaluated on...",
        "evidence": [{"page": 5, "section": "Experiments", "quote": "..."}]
      }
    ],
    "novelty_bullets": [
      {"text": "Novel contribution 1", "evidence": [...]},
      {"text": "Novel contribution 2", "evidence": [...]},
      {"text": "Novel contribution 3", "evidence": [...]}
    ]
  },
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "tasks_datasets": ["task1", "dataset1"],
  "methods_components": ["component1", "component2"]
}

REQUIREMENTS:
- novelty_bullets must have EXACTLY 3 items
- All evidence must reference actual text spans from paper
- keywords: 5-10 key terms
- Output ONLY valid JSON
"""
    )


# ============================================================================
# Phase 3: Evolution Schemas
# ============================================================================

class NewNodeGenerationSignature(dspy.Signature):
    """
    Generate a new taxonomy node for a cluster of papers.
    
    Input: Parent node, cluster papers, cluster innovations
    Output: JSON with new node name, definition, criteria, keywords
    """
    parent_node_name = dspy.InputField(desc="Name of the parent node")
    parent_definition = dspy.InputField(desc="Definition of the parent node")
    cluster_papers = dspy.InputField(desc="Titles and key claims of papers in cluster")
    cluster_innovations = dspy.InputField(desc="Key innovations extracted from cluster")
    
    new_node_json = dspy.OutputField(
        desc="""JSON object with the following structure:
{
  "name": "New node name (concise, descriptive)",
  "definition": "Clear definition of what this new category represents",
  "inclusion_criteria": [
    "Papers that...",
    "Methods that..."
  ],
  "exclusion_criteria": [
    "Excludes papers that...",
    "Not suitable for..."
  ],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}

REQUIREMENTS:
- Name should be concise (2-4 words)
- Definition should clearly distinguish from parent and siblings
- At least 2 inclusion and 2 exclusion criteria
- 5-10 keywords
- Output ONLY valid JSON
"""
    )


class SubNodeGenerationSignature(dspy.Signature):
    """
    Generate sub-nodes for splitting an overcrowded node.
    
    Input: Original node, papers assigned to it, grouping
    Output: JSON array of sub-node definitions
    """
    original_node_name = dspy.InputField(desc="Name of the node to split")
    original_definition = dspy.InputField(desc="Original node definition")
    paper_group_descriptions = dspy.InputField(desc="Description of paper sub-groups")
    
    subnodes_json = dspy.OutputField(
        desc="""JSON array with the following structure:
[
  {
    "name": "Sub-category 1",
    "definition": "More specific definition",
    "inclusion_criteria": ["..."],
    "exclusion_criteria": ["..."],
    "keywords": ["..."]
  },
  {
    "name": "Sub-category 2",
    "definition": "More specific definition",
    "inclusion_criteria": ["..."],
    "exclusion_criteria": ["..."],
    "keywords": ["..."]
  }
]

REQUIREMENTS:
- Generate 2-3 sub-nodes
- Each should be more specific than the original
- Sub-nodes should be mutually exclusive
- Output ONLY valid JSON array
"""
    )


class NodeRenameSignature(dspy.Signature):
    """
    Rename a node due to semantic drift.
    
    Input: Old node name, old definition, recent papers, new keywords
    Output: JSON with new name and updated definition
    """
    old_name = dspy.InputField(desc="Current node name")
    old_definition = dspy.InputField(desc="Current node definition")
    old_keywords = dspy.InputField(desc="Original keywords")
    recent_papers = dspy.InputField(desc="Recent papers assigned to this node")
    new_keywords = dspy.InputField(desc="New keywords from recent papers")
    drift_description = dspy.InputField(desc="Description of semantic drift")
    
    rename_json = dspy.OutputField(
        desc="""JSON object with the following structure:
{
  "new_name": "Updated node name reflecting current usage",
  "new_definition": "Updated definition incorporating recent developments",
  "rationale": "Explanation of why the rename is necessary"
}

REQUIREMENTS:
- New name should reflect semantic drift
- New definition should subsume old definition + new developments
- Rationale should cite specific changes
- Output ONLY valid JSON
"""
    )


# ============================================================================
# Helper function to create Chain-of-Thought modules
# ============================================================================

def create_taxonomy_extractor(lm: Optional[dspy.LM] = None):
    """Create a Chain-of-Thought module for taxonomy extraction."""
    if lm:
        with dspy.context(lm=lm):
            return dspy.ChainOfThought(TaxonomyExtractionSignature)
    return dspy.ChainOfThought(TaxonomyExtractionSignature)


def create_node_definition_builder(lm: Optional[dspy.LM] = None):
    """Create a Chain-of-Thought module for node definition generation."""
    if lm:
        with dspy.context(lm=lm):
            return dspy.ChainOfThought(NodeDefinitionSignature)
    return dspy.ChainOfThought(NodeDefinitionSignature)


def create_paper_claim_extractor(lm: Optional[dspy.LM] = None):
    """Create a Chain-of-Thought module for paper claim extraction."""
    if lm:
        with dspy.context(lm=lm):
            return dspy.ChainOfThought(PaperClaimExtractionSignature)
    return dspy.ChainOfThought(PaperClaimExtractionSignature)


def create_new_node_generator(lm: Optional[dspy.LM] = None):
    """Create a Chain-of-Thought module for new node generation."""
    if lm:
        with dspy.context(lm=lm):
            return dspy.ChainOfThought(NewNodeGenerationSignature)
    return dspy.ChainOfThought(NewNodeGenerationSignature)


def create_subnode_generator(lm: Optional[dspy.LM] = None):
    """Create a Chain-of-Thought module for sub-node generation."""
    if lm:
        with dspy.context(lm=lm):
            return dspy.ChainOfThought(SubNodeGenerationSignature)
    return dspy.ChainOfThought(SubNodeGenerationSignature)


def create_node_renamer(lm: Optional[dspy.LM] = None):
    """Create a Chain-of-Thought module for node renaming."""
    if lm:
        with dspy.context(lm=lm):
            return dspy.ChainOfThought(NodeRenameSignature)
    return dspy.ChainOfThought(NodeRenameSignature)
