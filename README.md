# SNS: Self-Nonself Modeling for Automatic Survey Systems

> **Mitigating Cognitive Lag in Automatic Survey Systems via Self–Nonself Modeling**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-ICML_submission-b31b1b.svg)](https://arxiv.org)

## Table of Contents

- [Overview](#overview)
- [Core Innovation: Self-Nonself Modeling](#core-innovation-self-nonself-modeling)
- [Problem Statement](#problem-statement)
- [Method Architecture](#method-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Methodology](#detailed-methodology)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Overview

**SNS (Self-Nonself)** is a novel framework that addresses the **cognitive lag problem** in automatic survey generation systems. Instead of treating all research papers uniformly, SNS models the **self** (existing consensus knowledge represented in review papers) and **nonself** (new research that challenges or extends this consensus), then generates delta-aware guidance for survey writing.

### Key Features

- 🧬 **Self-Nonself Modeling**: Constructs multi-view cognitive baselines from existing reviews (self) and stress-tests them with new research papers (nonself)
- 🎯 **Auditable Fit Testing**: Deterministic, evidence-backed classification instead of black-box LLM scoring
- 🌳 **Minimal Evolution**: Proposes minimal necessary structural updates based on stress analysis
- 📝 **Delta-aware Guidance**: Generates machine-executable writing constraints with two modes (Delta-first vs Anchor+Delta)
- 🔬 **Comprehensive Evaluation**: Time-slice taxonomy shift analysis, branch hit@K metrics, and human evaluation interface

---

## Core Innovation: Self-Nonself Modeling

### The Analogy

Drawing inspiration from immunology, SNS treats the research literature ecosystem as an adaptive system:

| Immunology | SNS Framework |
|------------|---------------|
| **Self** | Existing review papers & their taxonomies (established consensus) |
| **Nonself** | New research papers that don't fit existing structures |
| **Immune Response** | Structural evolution proposals (ADD_NODE, SPLIT_NODE, RENAME_NODE) |
| **Adaptation** | Updated taxonomy + writing guidance |

### Why Self-Nonself?

Traditional survey generation systems suffer from **cognitive lag**:
- They adopt existing review structures unchanged
- New papers are force-fitted into inadequate categories
- Genuine structural shifts are missed or buried

SNS addresses this by:
1. **Formalizing "self"**: Multi-view taxonomy atlas with testable node definitions
2. **Identifying "nonself"**: Papers that exhibit structural stress (conflict, residual novelty)
3. **Proposing adaptation**: Minimal necessary structural updates with evidence
4. **Generating guidance**: Mode-aware writing constraints (Delta-first when structure collapses, Anchor+Delta when stable)

---

## Problem Statement

### The Cognitive Lag Problem

Existing automatic survey systems produce **lagging reviews**:
- Adopt existing taxonomies without validation
- Perform simple paper classification + summarization
- Miss genuine cognitive deltas (structural innovations)

### Our Goal

Not "which papers are innovative" (playing judge), but:
1. **Formalize existing consensus** as testable multi-view baselines
2. **Stress test** with new papers to identify structural failure points
3. **Propose minimal updates** with evidence (ADD/SPLIT/RENAME nodes)
4. **Generate executable guidance** for downstream survey writing

### Key Transformation

**From**: Innovation scoring → **To**: Structure failure detection + adaptation

---

## Method Architecture

SNS consists of four phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Multi-view Baseline (Self Construction)               │
│  ─────────────────────────────────────────────────────────────  │
│  Input:  Topic query + Review papers R                          │
│  Output: Multi-view taxonomy atlas T (weighted views)           │
│  Method: Extract taxonomy trees with testable node definitions  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Multi-view Stress Test (Nonself Identification)       │
│  ─────────────────────────────────────────────────────────────  │
│  Input:  Research papers P + Baseline T                         │
│  Output: Fit vectors v(p) with deterministic labels             │
│  Method: Coverage / Conflict / Residual scoring                 │
│          Label ∈ {FIT, FORCE_FIT, UNFITTABLE}                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Stress Clustering & Evolution (Adaptation)            │
│  ─────────────────────────────────────────────────────────────  │
│  Input:  Stressed papers + Baseline T                           │
│  Output: Evolution proposal (operations + evidence)             │
│  Method: HDBSCAN clustering → Reconstruct all views             │
│          Select operations: Objective = FitGain - λ×EditCost    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Delta-aware Guidance (Writing Mode Selection)         │
│  ─────────────────────────────────────────────────────────────  │
│  Input:  Baseline + Evolution + Reconstruction scores           │
│  Output: Structured guidance + writing rules                    │
│  Method: Reconstruct-then-select main axis                      │
│          Determine mode: DELTA_FIRST vs ANCHOR_PLUS_DELTA       │
│          Generate do/dont writing constraints                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.8+
- OpenAI API key (or compatible LLM endpoint)

### Install from Source

```bash
# Clone repository
git clone https://github.com/yurui12138/IG-Finder.git
cd IG-Finder

# Install dependencies
pip install -e .

# Install optional dependencies for clustering and NLI
pip install hdbscan transformers torch scikit-learn
```

### Dependencies

Core:
- `dspy_ai`: LLM interface and prompting
- `sentence-transformers`: Embedding models (SPECTER2, SciNCL)
- `numpy`: Numerical computations

Optional:
- `hdbscan`: Stress clustering (Phase 3)
- `transformers`, `torch`: NLI models (DeBERTa-MNLI for conflict detection)
- `scikit-learn`: Evaluation metrics

---

## Quick Start

### Basic Usage

```python
from knowledge_storm.sns import SNSRunner, SNSArguments
import dspy

# Configure LLM
lm = dspy.OpenAI(model="gpt-4", api_key="your-api-key")

# Initialize SNS
args = SNSArguments(
    topic="transformer neural networks",
    output_dir="./output",
    top_k_reviews=15,              # Number of review papers to retrieve
    top_k_research_papers=100,     # Number of research papers to test
    min_cluster_size=3,            # Minimum cluster size for HDBSCAN
    lambda_regularization=0.8,     # Regularization for evolution objective
)

runner = SNSRunner(args, lm)

# Run all phases
results = runner.run(
    do_phase1=True,  # Multi-view baseline construction
    do_phase2=True,  # Stress test
    do_phase3=True,  # Evolution planning
    do_phase4=True,  # Guidance generation
)

# Access results
print(f"Baseline views: {len(results.multiview_baseline.views)}")
print(f"Stress clusters: {len(results.stress_clusters)}")
print(f"Evolution operations: {len(results.evolution_proposal.operations)}")
print(f"Writing mode: {results.delta_aware_guidance.main_axis_mode.value}")
```

### Output Files

SNS generates two primary outputs and several intermediate files:

#### 📊 Primary Outputs (Required)

1. **`audit_report.md`** - Human-readable audit report
   - Executive summary of Self-Nonself analysis
   - Detailed statistics and visualizations
   - Multi-view baseline description
   - Stress test results with evidence
   - Evolution proposals with justifications
   - **Purpose**: For researchers to understand the analysis

2. **`guidance_pack.json`** - Machine-readable guidance pack
   - Structured taxonomy (main_axis, aux_axis)
   - Writing mode determination (DELTA_FIRST vs ANCHOR_PLUS_DELTA)
   - Executable writing rules (do/dont constraints)
   - Structured outline with evidence cards
   - Evolution summary with operations
   - Must-answer questions
   - **Purpose**: For downstream automated survey generation systems

#### 📁 Intermediate Files (Optional, for debugging)

- `multiview_baseline.json`: Self model (taxonomy atlas)
- `fit_vectors.json`: Nonself identification results
- `stress_clusters.json`: Clustered structural failures
- `evolution_proposal.json`: Proposed adaptations
- `delta_guidance.json`: Complete DeltaAwareGuidance object
- `sns_results.json`: Complete results bundle

---

## Detailed Methodology

### Phase 1: Multi-view Baseline (Self Construction)

**Goal**: Formalize existing consensus as testable multi-view taxonomy atlas.

#### Components

1. **ReviewRetriever**: Retrieve top-K review papers
2. **TaxonomyViewExtractor**: Extract taxonomy tree from each review
   - LLM-based extraction with JSON schema
   - Each node has: name, definition, parent
3. **NodeDefinitionBuilder**: Build testable definitions
   - Inclusion/exclusion criteria
   - Canonical keywords
   - Boundary statements
   - Evidence spans (anchored to source text)
4. **BaselineBuilder**: Aggregate views with weights
   - Weight: w ∝ Quality × Recency × Coverage
   - Quality gate: Check for diversity (≥2 facets, no dominant >60%)

#### Output: Multi-view Baseline T

```python
MultiViewBaseline(
    topic: str,
    views: List[TaxonomyView],  # Each view has:
                                 # - tree: Taxonomy tree
                                 # - facet_label: Organizing dimension
                                 # - node_definitions: Testable criteria
                                 # - weight: Normalized importance
)
```

### Phase 2: Multi-view Stress Test (Nonself Identification)

**Goal**: Identify papers that don't fit existing structures.

#### Tri-factor Scoring

For each paper p and taxonomy view T_i:

1. **Coverage** (semantic + lexical fit):
   ```
   Coverage = 0.7 × cos(emb(paper), emb(node)) + 0.3 × Jaccard(keywords)
   ```

2. **Conflict** (with exclusion criteria):
   ```
   Conflict = max_{h ∈ Exclusion} P_NLI(contradiction | claim, h)
   ```
   - Uses DeBERTa-MNLI for entailment detection
   - Checks paper claims against exclusion criteria

3. **Residual** (unaccounted novelty):
   ```
   Residual = 1 - max_{b ∈ NoveltyBullets} cos(emb(b), emb(best_leaf))
   ```

#### Combined FitScore

```
FitScore = Coverage - 0.8×Conflict - 0.4×Residual
```

#### Deterministic Labeling

```python
if Coverage < 0.45 or Conflict > 0.55:
    label = UNFITTABLE  # Clear structural failure
elif Residual > 0.45:
    label = FORCE_FIT   # Fits but loses significant novelty
else:
    label = FIT         # Good fit
```

#### Output: Fit Vectors

```python
FitVector(
    paper_id: str,
    fit_reports: List[FitReport],  # One per view
    stress_score: float,           # Weighted average of failures
    unfittable_score: float,       # Proportion of UNFITTABLE
)
```

### Phase 3: Stress Clustering & Evolution (Adaptation)

**Goal**: Propose minimal necessary structural updates.

#### Step 1: Stress Clustering

1. **Filter** papers with stress_score > 0.3
2. **Construct failure signatures**:
   - Facet labels of failed views
   - Best leaf paths
   - Lost novelty bullets
3. **Cluster** using HDBSCAN (no need to specify K)
4. **Classify** cluster types:
   - `STRONG_SHIFT`: U(C) > 0.55, ≥2 high-weight views fail
   - `FACET_DEPENDENT`: Mixed failure/fit across high-weight views
   - `STABLE`: Most high-weight views fit

#### Step 2: View Reconstruction (NEW)

**Critical innovation**: Reconstruct ALL views before selecting main axis.

For each view T_i:
1. Plan evolution operations for stress clusters
2. Calculate metrics:
   - **FitGain**: How much stressed papers would improve
   - **StressReduction**: Reduction in average stress
   - **Coverage**: Taxonomy richness (leaves/50)
   - **EditCost**: Total cost of operations

3. **Combined score**:
   ```
   Score_i = 0.4×FitGain + 0.3×StressReduction + 0.2×Coverage - 0.1×EditCost
   ```

#### Step 3: Operation Selection

For each stress cluster:
1. **Candidate operations**:
   - `ADD_NODE`: Add new child node (cost = 1.0)
   - `SPLIT_NODE`: Split overcrowded node (cost = 2.0)
   - `RENAME_NODE`: Rename drifted node (cost = 0.5)

2. **Selection criterion**:
   ```
   Objective = FitGain - λ×EditCost  (λ = 0.8)
   ```

3. **Greedy selection**: Pick operation with highest objective > 0

#### Output: Evolution Proposal

```python
EvolutionProposal(
    operations: List[EvolutionOperation],  # Sequence of tree edits
    total_fit_gain: float,
    total_edit_cost: float,
    objective_value: float,
)
```

### Phase 4: Delta-aware Guidance (Writing Mode Selection)

**Goal**: Generate machine-executable writing guidance.

#### Step 1: Main Axis Selection with Mode (NEW)

**Reconstruct-then-select** (not select-then-reconstruct!):

1. Use reconstruction scores from Phase 3
2. Select view with highest combined score
3. **Determine writing mode**:
   ```python
   if EditCost > 3.0 or FitGain > 10.0:
       mode = DELTA_FIRST      # Heavy reconstruction needed
   else:
       mode = ANCHOR_PLUS_DELTA  # Stable structure
   ```

#### Step 2: Auxiliary Axis Selection

Select based on discriminativeness:
```
Discriminativeness = Var(failure_rates_across_clusters)
```

Choose view with highest variance (best at distinguishing clusters).

#### Step 3: Writing Rules Generation (NEW)

**DELTA_FIRST mode**:
```python
DO:
- Lead with emerging trends and structural shifts
- Organize by innovation clusters and stress points
- Emphasize papers that don't fit existing taxonomies

DONT:
- Don't force-fit new work into inadequate categories
- Don't assume existing structure is still valid
```

**ANCHOR_PLUS_DELTA mode**:
```python
DO:
- Use main axis structure as foundation
- Integrate new papers where they fit
- Clearly mark structural updates

DONT:
- Don't ignore evolution and stress points
- Don't present taxonomy as static
```

#### Output: Delta-aware Guidance

```python
DeltaAwareGuidance(
    topic: str,
    main_axis: TaxonomyView,
    aux_axis: Optional[TaxonomyView],
    main_axis_mode: WritingMode,           # NEW
    outline: List[Section],                # Structured with evidence
    evolution_summary: List[EvolutionOp],  # What changed and why
    must_answer_questions: List[str],      # Required questions
    writing_rules: WritingRules,           # NEW: do/dont constraints
    reconstruction_scores: List[Score],    # NEW: Full transparency
)
```

### Guidance Pack Format

The **`guidance_pack.json`** is the primary machine-readable output for downstream systems:

```json
{
  "topic": "deepfake detection",
  "generation_date": "2025-12-15T10:30:00",
  "schema_version": "2.0",
  
  // Writing strategy
  "writing_mode": "DELTA_FIRST",  // or "ANCHOR_PLUS_DELTA"
  "writing_rules": {
    "do": [
      "Lead with emerging trends and structural shifts",
      "Organize by innovation clusters and stress points"
    ],
    "dont": [
      "Don't force-fit new work into inadequate categories"
    ]
  },
  
  // Taxonomy structure (with evolution applied)
  "taxonomy": {
    "main_axis": {
      "facet": "DETECTION_METHOD",
      "tree": { /* full tree structure */ },
      "weight": 0.35
    },
    "aux_axis": { /* optional */ }
  },
  
  // Structured outline with constraints
  "outline": [
    {
      "section": "Emerging Detection Methods",
      "subsections": [
        {
          "subsection": "Deep Learning Approaches",
          "required_nodes": ["/detection_method/deep_learning"],
          "required_citations": ["Smith2024", "Jones2023"],
          "must_answer": ["What are the latest architectures?"],
          "evidence_cards": [
            {
              "text": "Novel architecture achieves 95% accuracy...",
              "citation": "Smith2024",
              "page": 5
            }
          ]
        }
      ]
    }
  ],
  
  // Evolution context
  "evolution_summary": [
    {
      "operation": "ADD_NODE",
      "view": "DETECTION_METHOD",
      "parent": "/detection_method",
      "new_node": "transformer_based",
      "justification": "10 papers use this approach..."
    }
  ],
  
  // Questions to address
  "must_answer_questions": [
    "What are the emerging detection paradigms?",
    "How do new methods compare to traditional approaches?"
  ],
  
  // Transparency
  "reconstruction_scores": [
    {
      "view_id": "review_001/DETECTION_METHOD",
      "fit_gain": 12.5,
      "stress_reduction": 0.45,
      "edit_cost": 2.0,
      "combined_score": 8.3
    }
  ]
}
```

**Key Features**:
- ✅ **Machine-parseable**: JSON format with strict schema
- ✅ **Executable constraints**: `writing_rules` provide clear do/dont guidance
- ✅ **Evidence-grounded**: Each subsection links to specific evidence cards
- ✅ **Traceable**: `evolution_summary` explains all structural changes
- ✅ **Transparent**: `reconstruction_scores` show decision rationale

---

## API Reference

### Core Classes

#### SNSRunner

Main execution engine for SNS pipeline.

```python
class SNSRunner:
    def __init__(self, args: SNSArguments, lm: dspy.LM):
        """
        Args:
            args: Configuration parameters
            lm: Language model for extraction tasks
        """
    
    def run(
        self,
        do_phase1: bool = True,
        do_phase2: bool = True,
        do_phase3: bool = True,
        do_phase4: bool = True,
        save_intermediate: bool = True
    ) -> SNSResults:
        """Run SNS pipeline with specified phases."""
```

#### SNSArguments

Configuration dataclass.

```python
@dataclass
class SNSArguments:
    topic: str                          # Research topic
    output_dir: str = "./output"        # Output directory
    top_k_reviews: int = 15             # Reviews to retrieve
    top_k_research_papers: int = 100    # Papers to test
    min_cluster_size: int = 3           # HDBSCAN parameter
    lambda_regularization: float = 0.8  # Evolution objective λ
    embedding_model: str = "specter2"   # Embedding model choice
    save_intermediate_results: bool = True
```

### Key Data Structures

#### MultiViewBaseline (Self)

```python
@dataclass
class MultiViewBaseline:
    topic: str
    views: List[TaxonomyView]
    
    def get_view_by_id(self, view_id: str) -> Optional[TaxonomyView]
    def get_dominant_facets(self) -> List[FacetLabel]
```

#### FitVector (Nonself Identification)

```python
@dataclass
class FitVector:
    paper_id: str
    fit_reports: List[FitReport]     # One per view
    stress_score: float              # Weighted failure rate
    unfittable_score: float          # Proportion UNFITTABLE
```

#### ViewReconstructionScore (NEW)

```python
@dataclass
class ViewReconstructionScore:
    view_id: str
    fit_gain: float
    stress_reduction: float
    coverage: float
    edit_cost: float
    combined_score: float  # Auto-calculated
```

#### WritingMode (NEW)

```python
class WritingMode(Enum):
    DELTA_FIRST = "DELTA_FIRST"           # Emphasize evolution
    ANCHOR_PLUS_DELTA = "ANCHOR_PLUS_DELTA"  # Use existing + highlight changes
```

---

## Evaluation

SNS includes comprehensive evaluation frameworks:

### 1. Time-slice Taxonomy Shift

Evaluate on historical data (T0 → T1):
- Use T0 reviews as baseline
- Stress test with T1 papers
- Compare proposed evolution with actual T1 reviews
- **Metrics**: Branch Hit@K, operation type accuracy

```python
from knowledge_storm.sns.evaluation import TimeSliceEvaluator

evaluator = TimeSliceEvaluator(dataset_path="data/time_slice/")
results = evaluator.evaluate(sns_system)
print(f"Branch Hit@1: {results.branch_hit_at_1}")
print(f"Branch Hit@5: {results.branch_hit_at_5}")
```

### 2. Branch Hit@K

Measures whether proposed new nodes align with actual field evolution:
- For each proposed operation, check if it matches T1 review structure
- **Hit@K**: Proportion of proposals that appear in top-K similar branches

### 3. Human Evaluation Interface

4-dimension assessment (1-5 scale):
1. **Structural Accuracy**: Do proposals match field consensus?
2. **Evidence Quality**: Are justifications well-supported?
3. **Minimality**: Are changes necessary and minimal?
4. **Guidance Utility**: Is the writing guidance actionable?

```python
from knowledge_storm.sns.evaluation import HumanEvaluationInterface

interface = HumanEvaluationInterface()
interface.load_case(results)
interface.collect_ratings(expert_id="expert1")
interface.compute_agreement()  # Inter-rater reliability
```

### 4. Comprehensive Metrics

```python
from knowledge_storm.sns.evaluation import compute_comprehensive_metrics

metrics = compute_comprehensive_metrics(
    baseline=results.multiview_baseline,
    fit_vectors=results.fit_vectors,
    evolution=results.evolution_proposal
)

# Outputs:
# - Stress distribution (FIT/FORCE_FIT/UNFITTABLE rates)
# - Fit score statistics
# - Edit distance (total cost, operation breakdown)
# - Cluster quality (silhouette score, cluster sizes)
```

---

## Citation

If you use SNS in your research, please cite:

```bibtex
@inproceedings{sns2024,
  title={SNS: Mitigating Cognitive Lag in Automatic Survey Systems via Self--Nonself Modeling},
  author={[Authors]},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

---

## Project Structure

```
IG-Finder/
├── knowledge_storm/
│   └── sns/                        # Main SNS package (formerly ig_finder)
│       ├── __init__.py
│       ├── dataclass_v2.py         # Core data structures
│       ├── schemas_v2.py           # LLM prompts and schemas
│       ├── engine_v2.py            # SNSRunner implementation
│       ├── modules/
│       │   ├── phase1_multiview_baseline.py
│       │   ├── phase2_stress_test.py
│       │   ├── phase3_evolution.py
│       │   ├── phase4_guidance.py
│       ├── infrastructure/
│       │   ├── embeddings.py       # SPECTER2, SciNCL, etc.
│       │   ├── nli.py              # DeBERTa-MNLI for conflict
│       └── evaluation/
│           ├── time_slice.py       # Time-slice evaluation
│           ├── metrics.py          # Branch Hit@K, etc.
│           └── human_eval.py       # Human evaluation interface
├── examples/
│   ├── run_sns_complete.py         # End-to-end example
│   └── run_sns_example.py          # Quick start example
├── tests/                          # Unit tests
├── requirements.txt
├── setup.py
└── README.md                       # This file
```

---

## Design Principles

### 1. Auditable over Black-box

- LLMs only for extraction and evidence localization
- Critical decisions (FIT/FORCE_FIT/UNFITTABLE) by deterministic rules
- All classifications come with evidence (quotes, page numbers)

### 2. Reconstruct-then-Select

- Evaluate ALL views after reconstruction
- Select main axis based on reconstruction potential (not current FIT rate)
- Prevents premature optimization bias

### 3. Minimal Evolution

- Objective function balances FitGain and EditCost
- Greedy selection ensures only necessary changes
- Each operation backed by cluster evidence

### 4. Delta-aware Mode Selection

- DELTA_FIRST when structure needs heavy reconstruction
- ANCHOR_PLUS_DELTA when structure is stable
- Data-driven choice, not fixed strategy

### 5. Machine-executable Guidance

- Writing rules as do/dont lists (not just suggestions)
- Structured outline with required nodes and citations
- Full transparency with reconstruction scores

---

## Future Work

- [ ] **Compensatory view induction**: Auto-generate views when baseline quality is low
- [ ] **Validation step**: Verify guidance completeness before output
- [ ] **LLM-enhanced writing rules**: Use LLM to generate domain-specific constraints
- [ ] **Incremental updates**: Support continuous monitoring and adaptation
- [ ] **Multi-modal support**: Extend to figures, equations, and code

---

## Acknowledgments

This work builds on STORM (Writing STORies with information retrieved from the external world) and extends it with self-nonself modeling principles.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [https://github.com/yurui12138/IG-Finder/issues](https://github.com/yurui12138/IG-Finder/issues)
- Email: [contact information]

---

**SNS**: Bridging the cognitive lag in automatic survey generation through principled self-nonself modeling. 🧬📚
