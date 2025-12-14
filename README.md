# IG-Finder 2.0: Multi-view Atlas Stress Test

**A framework for identifying structural taxonomy gaps through multi-view stress testing and minimal necessary evolution planning.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

IG-Finder 2.0 introduces a **paradigm shift** from innovation scoring to **structure failure detection**. Instead of scoring individual papers, it:

1. **Constructs multi-view cognitive baseline** from existing review papers
2. **Performs auditable fit tests** on emerging research papers  
3. **Identifies structural pressure clusters** through failure signature analysis
4. **Proposes minimal necessary taxonomy updates** with evidence-backed justifications
5. **Generates delta-aware writing guidance** for survey generation

### Key Innovations

- ✨ **Multi-view Taxonomy Atlas**: Aggregates perspectives from multiple reviews
- ✨ **Auditable Classification**: All decisions backed by verifiable metrics
- ✨ **Minimal Evolution**: Objective-driven with `FitGain - λ×EditCost`
- ✨ **Delta-aware Writing**: Main/aux axes with evolution-conscious guidance
- ✨ **Comprehensive Evaluation**: Time-slice prediction + 4D human evaluation

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Optional: CUDA for GPU acceleration

### Install from Source

```bash
git clone https://github.com/yurui12138/IG-Finder.git
cd IG-Finder
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- `dspy_ai`: LLM framework
- `sentence-transformers`: Embedding models
- `transformers`: NLI models
- `torch`: Deep learning backend
- `hdbscan`: Clustering algorithm
- `scikit-learn`: ML utilities

---

## ⚡ Quick Start

### Basic Usage

```bash
python run_igfinder2_complete.py \
  --topic "transformer neural networks" \
  --output-dir ./output \
  --top-k-reviews 15 \
  --top-k-research 30
```

### With Advanced Features

```bash
python run_igfinder2_complete.py \
  --topic "graph neural networks" \
  --output-dir ./output \
  --embedding-model specter2 \
  --nli-model deberta \
  --device cuda \
  --run-evaluation
```

### Python API

```python
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import ArxivRM
from knowledge_storm.ig_finder import IGFinder2Runner, IGFinder2Arguments
from knowledge_storm.interface import LMConfigs

# Configure LLMs
lm_configs = LMConfigs()
for attr in ['consensus_extraction_lm', 'deviation_analysis_lm', 
             'cluster_validation_lm', 'report_generation_lm']:
    setattr(lm_configs, attr, LitellmModel(
        model='gpt-4o',
        temperature=0,
        api_key="your-api-key"
    ))

# Create runner
args = IGFinder2Arguments(
    topic="deep learning",
    output_dir="./output",
    top_k_reviews=15,
    top_k_research_papers=30,
    embedding_model="specter2",
    lambda_regularization=0.8
)

runner = IGFinder2Runner(args, lm_configs, ArxivRM())

# Run complete pipeline
results = runner.run(
    do_phase1=True,  # Multi-view baseline
    do_phase2=True,  # Stress test
    do_phase3=True,  # Evolution planning
    do_phase4=True   # Guidance generation
)

# Access results
print(f"Views: {len(results.multiview_baseline.views)}")
print(f"Stressed papers: {len([fv for fv in results.fit_vectors if fv.stress_score > 0.5])}")
print(f"Proposed operations: {len(results.evolution_proposal.operations)}")
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    IG-Finder 2.0                        │
└─────────────────────────────────────────────────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
  ┌───▼────┐        ┌─────▼─────┐       ┌────▼────┐
  │Phase 1 │───────▶│  Phase 2  │──────▶│ Phase 3 │
  │Multi-  │        │  Stress   │       │Evolution│
  │view    │        │   Test    │       │Planning │
  │Baseline│        └───────────┘       └─────┬───┘
  └────────┘                                   │
                                         ┌─────▼────┐
                                         │ Phase 4  │
                                         │Guidance  │
                                         └──────────┘
```

### Components

- **Phase 1**: Multi-view Baseline Construction
  - ReviewRetriever, TaxonomyViewExtractor, NodeDefinitionBuilder
- **Phase 2**: Multi-view Stress Test
  - PaperClaimExtractor, FitTester (tri-factor scoring)
- **Phase 3**: Stress Clustering & Minimal Evolution
  - StressClusterer (HDBSCAN), EvolutionPlanner
- **Phase 4**: Delta-aware Guidance Generation
  - AxisSelector, GuidanceGenerator

---

## 📖 Usage Examples

### Example 1: Basic Pipeline

```python
from knowledge_storm.ig_finder import IGFinder2Runner, IGFinder2Arguments

args = IGFinder2Arguments(
    topic="attention mechanisms",
    output_dir="./output"
)

runner = IGFinder2Runner(args, lm_configs, retriever)
results = runner.run()
runner.summary()
```

### Example 2: Evaluation Framework

```python
from knowledge_storm.ig_finder.evaluation import (
    compute_all_metrics,
    print_metrics_report
)

metrics = compute_all_metrics(
    fit_vectors=results.fit_vectors,
    evolution_proposal=results.evolution_proposal
)

print_metrics_report(metrics)
```

### Example 3: Human Evaluation

```python
from knowledge_storm.ig_finder.evaluation import HumanEvaluationInterface

interface = HumanEvaluationInterface()

# Create evaluation tasks
interface.create_definition_quality_tasks(
    view_id="view_1",
    node_definitions=baseline.views[0].node_definitions
)

# Export for annotators
interface.export_tasks_to_file("eval_tasks.json")
```

---

## 📚 Documentation

### Comprehensive Guides

- **[Design Document](IG_FINDER_2.0_DESIGN.md)** (46.5KB)
  - Formal problem definition
  - Mathematical formulations
  - Detailed phase specifications
  
- **[Implementation Guide](IG_FINDER_2.0_IMPLEMENTATION_GUIDE.md)** (20KB)
  - Architecture overview
  - Phase-by-phase tutorials
  - API reference
  - Troubleshooting

- **[Summary Document](IG_FINDER_2.0_SUMMARY.md)** (16KB)
  - Implementation overview
  - Code statistics
  - Future roadmap

### Quick References

- **Phase 1**: Multi-view baseline construction with weight calculation
- **Phase 2**: Tri-factor fit scoring (Coverage, Conflict, Residual)
- **Phase 3**: HDBSCAN clustering + ADD/SPLIT/RENAME operations
- **Phase 4**: Main/aux axis selection + structured guidance

---

## 🔬 Evaluation

### Time-slice Taxonomy Shift

```python
from knowledge_storm.ig_finder.evaluation import TimeSliceDataset, TimeSliceEvaluator

# Create dataset
dataset = TimeSliceDataset()
t0, t1 = dataset.create_from_papers(
    topic="deep learning",
    all_reviews=reviews,
    all_research=papers,
    t0_end_year=2020,
    t1_end_year=2023
)

# Extract ground truth
ground_truth = dataset.extract_ground_truth_shifts(t0_baseline, t1_baseline)

# Evaluate predictions
evaluator = TimeSliceEvaluator()
metrics = evaluator.evaluate(predicted_proposal, ground_truth, k_values=[1,3,5])
```

### Metrics

- **Branch Hit@K**: Prediction accuracy at top-K operations
- **Operation Type Accuracy**: Correct operation type percentage
- **Taxonomy Edit Distance**: Structural difference measure
- **Comprehensive Metrics**: Stress distribution, fit rates, efficiency

---

## 📊 Output Files

Running IG-Finder 2.0 generates:

```
output/
├── multiview_baseline.json      # Phase 1: Multi-view taxonomy atlas
├── fit_vectors.json             # Phase 2: Fit test results
├── stress_clusters.json         # Phase 3: Clustered stressed papers
├── evolution_proposal.json      # Phase 3: Proposed operations
├── delta_guidance.json          # Phase 4: Writing guidance
├── igfinder2_results.json       # Complete results
├── igfinder2_report.md          # Human-readable report
└── evaluation_metrics.json      # Evaluation metrics (if enabled)
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines.

### Development Setup

```bash
git clone https://github.com/yurui12138/IG-Finder.git
cd IG-Finder
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

- **GitHub Issues**: https://github.com/yurui12138/IG-Finder/issues
- **Pull Requests**: https://github.com/yurui12138/IG-Finder/pulls

---

## 🎓 Citation

If you use IG-Finder 2.0 in your research, please cite:

```bibtex
@software{igfinder2,
  title={IG-Finder 2.0: Multi-view Atlas Stress Test for Delta-aware Survey Writing},
  author={[Author Name]},
  year={2024},
  url={https://github.com/yurui12138/IG-Finder}
}
```

---

## 🙏 Acknowledgments

Built on top of the STORM framework for knowledge curation.

---

**Version**: 2.0.0  
**Last Updated**: 2024-12-14
