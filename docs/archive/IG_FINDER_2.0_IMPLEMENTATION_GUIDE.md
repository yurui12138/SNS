

# IG-Finder 2.0 Implementation Guide

**Complete Reference for Phase 3-4, Infrastructure, and Evaluation**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Phase 3: Stress Clustering & Minimal Evolution](#phase-3)
4. [Phase 4: Delta-aware Guidance Generation](#phase-4)
5. [Infrastructure Enhancements](#infrastructure)
6. [Evaluation Framework](#evaluation)
7. [Installation & Setup](#installation)
8. [Usage Examples](#usage)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

IG-Finder 2.0 introduces a **multi-view taxonomy atlas stress test** paradigm shift. Instead of scoring innovations, it:

1. **Constructs multi-view cognitive baseline** from existing reviews
2. **Performs auditable fit tests** on emerging research papers
3. **Identifies structural pressure clusters** via minimal tree edits
4. **Proposes evidence-backed taxonomy updates** with main/auxiliary axis guidance

**Key Principles:**
- LLMs for extraction & evidence localization only
- Critical decisions via deterministic, verifiable rules
- All judgments auditable with evidence spans
- Minimal necessary structural evolution

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     IG-Finder 2.0                        │
└──────────────────────────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───▼────┐            ┌─────▼─────┐           ┌────▼────┐
│ Phase 1│            │  Phase 2  │           │ Phase 3 │
│Multi-  │───────────▶│  Stress   │──────────▶│Evolution│
│view    │            │   Test    │           │Planning │
│Baseline│            │           │           │         │
└────────┘            └───────────┘           └─────┬───┘
                                                    │
                                              ┌─────▼────┐
                                              │ Phase 4  │
                                              │Guidance  │
                                              │Generation│
                                              └──────────┘
                                                    │
                                              ┌─────▼────┐
                                              │Evaluation│
                                              │Framework │
                                              └──────────┘
```

### Module Structure

```
knowledge_storm/ig_finder/
├── embeddings.py              # SPECTER2, SciNCL, Sentence-BERT
├── nli.py                     # DeBERTa-MNLI for conflict detection
├── modules/
│   ├── phase1_multiview_baseline.py
│   ├── phase2_stress_test.py
│   ├── phase3_evolution.py    # NEW: Clustering & evolution planning
│   └── phase4_guidance.py     # NEW: Axis selection & guidance
├── evaluation/
│   ├── time_slice.py          # NEW: Time-slice dataset & evaluator
│   ├── metrics.py             # NEW: Branch Hit@K, edit distance
│   └── human_eval.py          # NEW: Human evaluation interface
└── engine_v2.py               # Orchestrator
```

---

## Phase 3: Stress Clustering & Minimal Evolution

### Objective
Cluster stressed papers and propose **minimal necessary** structural updates to the taxonomy.

### Components

#### 3.1 StressClusterer

**Purpose:** Cluster papers based on failure signatures using HDBSCAN.

**Failure Signature:**
```
signature = concat([
    f"{facet}:{best_leaf_path}",
    lost_novelty_bullets
])
```

**Cluster Types:**
- **Strong Shift**: `U(C) > 0.55` and `≥2 high-weight views fail`
- **Facet-dependent**: Some high-weight views fail, others fit
- **Stable**: Most high-weight views fit well

**Code Example:**
```python
from knowledge_storm.ig_finder.modules.phase3_evolution import StressClusterer

clusterer = StressClusterer(min_cluster_size=3)
clusters = clusterer.cluster_stressed_papers(
    fit_vectors=fit_vectors,
    papers=research_papers,
    baseline=multiview_baseline,
    stress_threshold=0.3
)

for cluster in clusters:
    print(f"Cluster {cluster.cluster_id}:")
    print(f"  Type: {cluster.cluster_type.value}")
    print(f"  Papers: {len(cluster.papers)}")
    print(f"  Stress: {cluster.stress_score:.3f}")
```

#### 3.2 EvolutionPlanner

**Purpose:** Plan minimal necessary evolution operations.

**Operations:**
1. **ADD_NODE**: Add new category when residual > 0.6
2. **SPLIT_NODE**: Split overcrowded node when child count > 15
3. **RENAME_NODE**: Rename when semantic drift > threshold

**Objective Function:**
```
Objective = FitGain - λ * EditCost
```

**Code Example:**
```python
from knowledge_storm.ig_finder.modules.phase3_evolution import EvolutionPlanner

planner = EvolutionPlanner(lm=language_model)
proposal = planner.plan_evolution(
    clusters=stress_clusters,
    baseline=multiview_baseline,
    fit_vectors=fit_vectors,
    lambda_reg=0.8  # Prefer fewer, higher-gain operations
)

print(f"Proposed {len(proposal.operations)} operations")
print(f"Total fit gain: {proposal.total_fit_gain:.3f}")
print(f"Total edit cost: {proposal.total_edit_cost:.3f}")
print(f"Objective value: {proposal.objective_value:.3f}")
```

#### 3.3 Complete Phase 3 Pipeline

```python
from knowledge_storm.ig_finder.modules.phase3_evolution import Phase3Pipeline

phase3 = Phase3Pipeline(
    lm=language_model,
    min_cluster_size=3,
    lambda_reg=0.8
)

clusters, proposal = phase3.run(
    fit_vectors=fit_vectors,
    papers=research_papers,
    baseline=multiview_baseline
)
```

**Output:**
- `stress_clusters.json`: List of stress clusters
- `evolution_proposal.json`: Proposed operations with evidence

---

## Phase 4: Delta-aware Guidance Generation

### Objective
Select main/auxiliary axes and generate structured guidance for downstream survey generation.

### Components

#### 4.1 AxisSelector

**Main Axis Selection:**
```
score = 0.6 * FIT_rate + 0.3 * Stability + 0.1 * Coverage
main_axis = argmax(score)
```

**Auxiliary Axis Selection:**
```
discriminativeness = Var(failure_rates_across_clusters)
aux_axis = argmax(discriminativeness)
```

**Code Example:**
```python
from knowledge_storm.ig_finder.modules.phase4_guidance import AxisSelector

selector = AxisSelector()

# Select main axis
main_axis = selector.select_main_axis(
    baseline=multiview_baseline,
    fit_vectors=fit_vectors
)

# Select auxiliary axis
aux_axis = selector.select_aux_axis(
    baseline=multiview_baseline,
    clusters=stress_clusters,
    main_axis=main_axis
)

print(f"Main axis: {main_axis.facet_label.value}")
if aux_axis:
    print(f"Aux axis: {aux_axis.facet_label.value}")
```

#### 4.2 GuidanceGenerator

**Purpose:** Generate structured outline with required nodes, citations, and questions.

**Output Structure:**
```json
{
  "topic": "...",
  "main_axis": { ... },
  "aux_axis": { ... },
  "outline": [
    {
      "section": "Section Name",
      "subsections": [
        {
          "subsection": "Subsection Name",
          "required_nodes": ["path1", "path2"],
          "required_citations": ["paper_id1", "paper_id2"],
          "must_answer": ["Question 1", "Question 2"],
          "evidence_cards": [ ... ]
        }
      ]
    }
  ],
  "evolution_summary": [ ... ],
  "must_answer_questions": [ ... ]
}
```

**Code Example:**
```python
from knowledge_storm.ig_finder.modules.phase4_guidance import GuidanceGenerator

generator = GuidanceGenerator()

guidance = generator.generate_guidance(
    topic="transformer neural networks",
    main_axis=main_axis,
    aux_axis=aux_axis,
    clusters=stress_clusters,
    evolution_proposal=evolution_proposal,
    fit_vectors=fit_vectors,
    papers=research_papers
)

# Access outline
for section in guidance.outline:
    print(f"\n{section.section}")
    for subsection in section.subsections:
        print(f"  - {subsection.subsection}")
        print(f"    Required citations: {len(subsection.required_citations)}")
```

#### 4.3 Complete Phase 4 Pipeline

```python
from knowledge_storm.ig_finder.modules.phase4_guidance import Phase4Pipeline

phase4 = Phase4Pipeline()

guidance = phase4.run(
    topic="transformer neural networks",
    baseline=multiview_baseline,
    fit_vectors=fit_vectors,
    papers=research_papers,
    clusters=stress_clusters,
    evolution_proposal=evolution_proposal
)
```

**Output:**
- `delta_guidance.json`: Complete delta-aware writing guidance

---

## Infrastructure Enhancements

### Embedding Models

#### Supported Models
1. **SPECTER2** (Recommended): `allenai/specter2_base`
2. **SciNCL**: `malteos/scincl`
3. **Sentence-BERT**: `sentence-transformers/all-MiniLM-L6-v2`
4. **Fallback**: TF-IDF (no neural models)

#### Usage

```python
from knowledge_storm.ig_finder.embeddings import create_embedding_model

# Create embedding model
embedding_model = create_embedding_model(
    model_type="specter2",  # or "scincl", "sbert", "fallback"
    device="cuda"  # or "cpu"
)

# Encode texts
texts = ["Paper abstract 1", "Paper abstract 2"]
embeddings = embedding_model.encode(texts)

# Compute similarity
sim = embedding_model.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {sim:.3f}")

# Hybrid similarity (0.7 semantic + 0.3 lexical)
from knowledge_storm.ig_finder.embeddings import compute_hybrid_similarity

hybrid_sim = compute_hybrid_similarity(
    text1="Paper abstract",
    text2="Node definition",
    embedding_model=embedding_model,
    semantic_weight=0.7,
    lexical_weight=0.3
)
```

#### Top-K Retrieval

```python
from knowledge_storm.ig_finder.embeddings import compute_top_k_matches

# Find top-5 most similar candidates
query_emb = embedding_model.encode(["Query text"])[0]
candidate_embs = embedding_model.encode(candidate_texts)

top_k_indices = compute_top_k_matches(
    query_embedding=query_emb,
    candidate_embeddings=candidate_embs,
    k=5
)
```

### NLI Models

#### Supported Models
1. **DeBERTa-v3-large-mnli** (Recommended): `microsoft/deberta-v3-large-mnli`
2. **RoBERTa-large-mnli**: `roberta-large-mnli`
3. **Fallback**: Keyword-based heuristics

#### Usage

```python
from knowledge_storm.ig_finder.nli import create_nli_model

# Create NLI model
nli_model = create_nli_model(
    model_type="deberta",  # or "roberta", "fallback"
    device="cuda"
)

# Predict entailment/neutral/contradiction
premise = "Node definition: Methods using attention mechanisms"
hypothesis = "Paper claim: We propose a fully convolutional approach"

label, confidence = nli_model.predict(premise, hypothesis)
print(f"Label: {label.value}, Confidence: {confidence:.3f}")

# Compute contradiction score
contradiction_score = nli_model.compute_contradiction_score(premise, hypothesis)
print(f"Contradiction score: {contradiction_score:.3f}")
```

#### Batch Processing

```python
# Batch prediction for efficiency
premises = [definition1, definition2, definition3]
hypotheses = [claim1, claim2, claim3]

results = nli_model.predict_batch(premises, hypotheses)
for (label, conf) in results:
    print(f"{label.value}: {conf:.3f}")

# Batch contradiction scores
scores = nli_model.compute_contradiction_scores_batch(premises, hypotheses)
```

#### Integration in Phase 2

```python
from knowledge_storm.ig_finder.nli import compute_max_conflict_score

conflict_score = compute_max_conflict_score(
    claim_text="Paper's main claim",
    node_definition_text="Node definition",
    exclusion_criteria=["Criterion 1", "Criterion 2"],
    nli_model=nli_model
)

# Used in FitScore calculation:
# Conflict = max NLI_contradiction(claim, excl_i)
```

---

## Evaluation Framework

### Time-slice Taxonomy Shift Evaluation

**Concept:** Construct (T0, T1) dataset pairs and evaluate if system predicts actual taxonomy evolution.

#### Dataset Construction

```python
from knowledge_storm.ig_finder.evaluation import TimeSliceDataset

dataset = TimeSliceDataset()

# Create time slices
t0, t1 = dataset.create_from_papers(
    topic="deep learning",
    all_reviews=all_review_papers,
    all_research=all_research_papers,
    t0_end_year=2020,    # Baseline: up to 2020
    t1_end_year=2023,    # Future: up to 2023
    gap_years=3
)

# Extract ground truth shifts
# (Compare taxonomies from T0 reviews vs T1 reviews)
ground_truth_shifts = dataset.extract_ground_truth_shifts(
    t0_baseline=baseline_from_t0_reviews,
    t1_baseline=baseline_from_t1_reviews
)

# Save dataset
dataset.save_to_file("time_slice_dataset.json")
```

#### Evaluation

```python
from knowledge_storm.ig_finder.evaluation import TimeSliceEvaluator

evaluator = TimeSliceEvaluator()

metrics = evaluator.evaluate(
    predicted_proposal=evolution_proposal,
    ground_truth_shifts=ground_truth_shifts,
    k_values=[1, 3, 5]
)

print(f"Branch Hit@1: {metrics['branch_hit@1']:.3f}")
print(f"Branch Hit@3: {metrics['branch_hit@3']:.3f}")
print(f"Branch Hit@5: {metrics['branch_hit@5']:.3f}")
print(f"Operation type accuracy: {metrics['operation_type_accuracy']:.3f}")
print(f"Node name match: {metrics['node_name_match']:.3f}")
```

### Comprehensive Metrics

```python
from knowledge_storm.ig_finder.evaluation import compute_all_metrics, print_metrics_report

metrics = compute_all_metrics(
    fit_vectors=fit_vectors,
    evolution_proposal=evolution_proposal,
    original_tree=baseline.views[0].tree,
    evolved_tree=evolved_tree  # After applying operations
)

# Print formatted report
print_metrics_report(metrics)
```

**Metrics Included:**
- Stress distribution (mean, std, percentiles)
- Fit label distribution (FIT, FORCE_FIT, UNFITTABLE %)
- View fit rates
- Operation distribution (ADD, SPLIT, RENAME counts)
- Evolution efficiency (fit_gain, edit_cost, efficiency ratio)
- Tree edit distance

### Human Evaluation Interface

#### Creating Evaluation Tasks

```python
from knowledge_storm.ig_finder.evaluation import HumanEvaluationInterface

interface = HumanEvaluationInterface(output_dir="./human_eval")

# 1. Definition Quality tasks
def_tasks = interface.create_definition_quality_tasks(
    view_id="view_1",
    node_definitions=baseline.views[0].node_definitions
)

# 2. Evidence Sufficiency tasks
evidence_tasks = interface.create_evidence_sufficiency_tasks(
    operations=evolution_proposal.operations
)

# 3. Evolution Necessity tasks
necessity_tasks = interface.create_evolution_necessity_tasks(
    operations=evolution_proposal.operations
)

# 4. Guidance Usefulness tasks
guidance_tasks = interface.create_guidance_usefulness_tasks(
    guidance=delta_aware_guidance
)

# Export tasks for annotators
interface.export_tasks_to_file("evaluation_tasks.json")
```

#### Submitting Ratings

```python
# Evaluator submits rating
interface.submit_rating(
    evaluator_id="annotator_1",
    task_id="def_quality_view_1_root/methods",
    score=4,  # 1-5 scale
    comment="Clear definition but inclusion criteria could be more specific"
)

# Export ratings
interface.export_ratings_to_file("evaluation_ratings.json")
```

#### Analyzing Results

```python
# Inter-rater agreement
agreement = interface.compute_inter_rater_agreement()
print(f"Pairwise agreement: {agreement['pairwise_agreement']:.3f}")

# Dimension statistics
stats = interface.get_dimension_statistics()
for dimension, dimension_stats in stats.items():
    print(f"\n{dimension}:")
    print(f"  Mean: {dimension_stats['mean']:.2f}")
    print(f"  Std: {dimension_stats['std']:.2f}")
```

---

## Installation & Setup

### Requirements

```bash
# Core dependencies
pip install dspy_ai sentence-transformers transformers torch hdbscan scikit-learn

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### From Source

```bash
git clone https://github.com/yurui12138/IG-Finder.git
cd IG-Finder
pip install -e .
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional
```

---

## Usage Examples

### Basic End-to-End Run

```bash
python run_igfinder2_complete.py \
  --topic "transformer neural networks" \
  --output-dir ./output \
  --top-k-reviews 15 \
  --top-k-research 30 \
  --embedding-model specter2 \
  --nli-model deberta \
  --device cuda \
  --run-evaluation
```

### Phase-by-Phase Execution

```python
from knowledge_storm.ig_finder.engine_v2 import IGFinder2Runner, IGFinder2Arguments
from knowledge_storm.interface import LMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import ArxivRM

# Setup
lm_configs = LMConfigs()
for attr in ['consensus_extraction_lm', 'deviation_analysis_lm', 
             'cluster_validation_lm', 'report_generation_lm']:
    setattr(lm_configs, attr, LitellmModel(model='gpt-4o', temperature=0))

args = IGFinder2Arguments(
    topic="graph neural networks",
    output_dir="./output",
    top_k_reviews=10,
    top_k_research_papers=20,
    embedding_model="specter2",
    lambda_regularization=0.8
)

runner = IGFinder2Runner(args, lm_configs, ArxivRM())

# Run all phases
results = runner.run(
    do_phase1=True,
    do_phase2=True,
    do_phase3=True,
    do_phase4=True
)

# Access results
print(f"Baseline views: {len(results.multiview_baseline.views)}")
print(f"Stressed papers: {len([fv for fv in results.fit_vectors if fv.stress_score > 0.5])}")
print(f"Stress clusters: {len(results.stress_clusters)}")
print(f"Proposed operations: {len(results.evolution_proposal.operations)}")
```

### Using Custom Embedding/NLI Models

```python
from knowledge_storm.ig_finder.embeddings import create_embedding_model
from knowledge_storm.ig_finder.nli import create_nli_model

# Custom embedding model
embedding_model = create_embedding_model(
    model_type="specter2",
    model_name="allenai/specter2_base",
    device="cuda"
)

# Custom NLI model
nli_model = create_nli_model(
    model_type="deberta",
    model_name="microsoft/deberta-v3-large-mnli",
    device="cuda"
)

# Use in Phase 2
from knowledge_storm.ig_finder.modules.phase2_stress_test import Phase2Pipeline

phase2 = Phase2Pipeline(
    lm=language_model,
    embedding_model=embedding_model,
    nli_model=nli_model
)
```

---

## API Reference

### Core Classes

#### `IGFinder2Runner`
Main orchestrator for all phases.

**Methods:**
- `run(do_phase1, do_phase2, do_phase3, do_phase4)`: Execute pipeline
- `summary()`: Print results summary

#### `Phase3Pipeline`
Stress clustering and evolution planning.

**Methods:**
- `run(fit_vectors, papers, baseline)`: Execute Phase 3

**Returns:** `(stress_clusters, evolution_proposal)`

#### `Phase4Pipeline`
Delta-aware guidance generation.

**Methods:**
- `run(topic, baseline, fit_vectors, papers, clusters, evolution_proposal)`: Execute Phase 4

**Returns:** `DeltaAwareGuidance`

### Data Structures

See `dataclass_v2.py` for complete definitions:
- `MultiViewBaseline`: Multi-view taxonomy atlas
- `FitVector`: Fit test results for a paper
- `StressCluster`: Cluster of stressed papers
- `EvolutionProposal`: Proposed taxonomy updates
- `DeltaAwareGuidance`: Writing guidance structure

---

## Troubleshooting

### Common Issues

#### 1. HDBSCAN Installation
```bash
# If pip install hdbscan fails:
conda install -c conda-forge hdbscan
```

#### 2. CUDA Out of Memory
```python
# Use CPU or reduce batch size
embedding_model = create_embedding_model(model_type="specter2", device="cpu")
nli_model = create_nli_model(model_type="deberta", device="cpu")
```

#### 3. Model Download Timeout
```python
# Set cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
```

#### 4. API Rate Limiting
```python
# Add delays between LLM calls
import time
time.sleep(1)  # In LLM call loop
```

### Performance Optimization

1. **Use GPU** for embedding/NLI models
2. **Batch processing** for embeddings and NLI
3. **Cache intermediate results** with `save_intermediate_results=True`
4. **Limit paper counts** with `top_k_reviews` and `top_k_research_papers`

---

## Citation

```bibtex
@software{igfinder2,
  title={IG-Finder 2.0: Multi-view Atlas Stress Test for Delta-aware Survey Writing},
  author={[Author Name]},
  year={2024},
  url={https://github.com/yurui12138/IG-Finder}
}
```

---

## Support

- **GitHub Issues**: https://github.com/yurui12138/IG-Finder/issues
- **Documentation**: See `IG_FINDER_2.0_DESIGN.md` for design details
- **Examples**: See `run_igfinder2_complete.py` for usage

---

*Last Updated: 2024-12-14*
