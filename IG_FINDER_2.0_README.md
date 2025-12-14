# IG-Finder 2.0: Multi-view Atlas Stress Test & Minimal Evolution

## 🎯 Overview

IG-Finder 2.0 transforms the approach to automatic survey generation from "innovation scoring" to **"structure failure detection + structure update proposal"**.

### The Problem
Existing automatic survey systems generate "lagging reviews" that:
- Reuse existing survey structures and consensus boundaries
- Only add and categorize new papers
- Fail to produce cognitive delta (认知增量) relative to existing surveys

### Our Solution
IG-Finder 2.0:
1. Formalizes "existing survey organization" as a testable multi-view baseline
2. Stress-tests new papers against this structure to identify failure points
3. Proposes minimal necessary structure updates based on evidence
4. Generates delta-aware writing guidance (main/aux axes + outline) for downstream survey systems

### Key Transformation
**From**: "Judge which papers are innovative (be the referee)"  
**To**: "Detect structure failures + propose minimal updates (be the diagnostician)"

---

## 🏗️ Architecture

### Phase 1: Multi-view Baseline Construction
Extract multiple taxonomy views from review papers:
```
T = {T₁, T₂, ..., Tₖ}
```

Each view Tᵢ includes:
- **Tree structure**: Hierarchical organization
- **Node definitions**: Testable inclusion/exclusion criteria
- **Facet label**: Organizational dimension (MODEL_ARCHITECTURE, TRAINING_PARADIGM, etc.)
- **Weight**: Quality metric (wᵢ ∝ Quality · Recency · Coverage)
- **Evidence spans**: Citations from original reviews

### Phase 2: Multi-view Stress Test
Test each paper against all views:
```
f(p, Tᵢ) → (yᵢ, ℓᵢ, eᵢ)
```

Where:
- `yᵢ ∈ {FIT, FORCE_FIT, UNFITTABLE}`: Fit label
- `ℓᵢ`: Best matching leaf node
- `eᵢ`: Evidence (spans + failure reasons)

**Fit Score Calculation**:
```
Coverage = 0.7 · semantic_similarity + 0.3 · lexical_overlap
Conflict = max NLI(contradiction | claim, boundary)
Residual = 1 - max cos(novelty_bullet, leaf_vector)

FitScore = Coverage - 0.8 · Conflict - 0.4 · Residual
```

**Label Determination** (deterministic thresholds):
- If `Coverage < 0.45` or `Conflict > 0.55` → **UNFITTABLE**
- Else if `Residual > 0.45` → **FORCE_FIT**
- Else → **FIT**

### Phase 3: Stress Clustering & Minimal Evolution (Planned)
1. Cluster papers by failure signature (HDBSCAN)
2. Determine cluster type: Strong Shift / Facet-dependent / Stable
3. Propose minimal operations: ADD_NODE / SPLIT_NODE / RENAME_NODE
4. Optimize: `T' = argmax (FitGain - λ·EditCost)`

### Phase 4: Delta-aware Guidance (Planned)
Generate structured guidance for downstream survey generation:
- **Main Axis**: Most stable organizational dimension
- **Aux Axis**: Best discriminator for stress clusters
- **Outline**: Required nodes + citations + questions
- **Evolution Summary**: Structure updates with evidence

---

## 🚀 Quick Start

### Installation

```bash
cd /home/user/webapp
pip install -r requirements.txt
```

### Basic Usage

```bash
python run_igfinder2_example.py \
    --topic "transformer models in natural language processing" \
    --output-dir ./igfinder2_output \
    --top-k-reviews 5 \
    --top-k-research 10
```

### Python API

```python
from knowledge_storm.ig_finder import (
    IGFinder2Runner,
    IGFinder2Arguments,
    IGFinderLMConfigs,
)
from knowledge_storm.rm import ArxivSearchRM
from knowledge_storm.lm import LitellmModel
from knowledge_storm.interface import Retriever

# Setup language models
lm_configs = IGFinderLMConfigs()
openai_kwargs = {
    'api_key': 'your_api_key',
    'api_base': 'https://api.yunwu.ai/v1',
    'temperature': 0.0,  # For reproducibility
}
consensus_lm = LitellmModel(model='gpt-4o', max_tokens=3000, **openai_kwargs)
lm_configs.set_consensus_extraction_lm(consensus_lm)
# ... set other LMs

# Setup retriever
rm = Retriever(rm=ArxivSearchRM(k=10))

# Create arguments
args = IGFinder2Arguments(
    topic="your research topic",
    output_dir="./output",
    top_k_reviews=5,
    top_k_research_papers=10,
)

# Run pipeline
runner = IGFinder2Runner(args, lm_configs, rm)
results = runner.run()

# Access results
print(f"Extracted {len(results.multiview_baseline.views)} taxonomy views")
print(f"Tested {len(results.fit_vectors)} papers")
print(f"Average stress score: {results.statistics['avg_stress_score']:.3f}")
```

---

## 📊 Output

IG-Finder 2.0 generates:

### 1. Multi-view Baseline (`multiview_baseline.json`)
```json
{
  "topic": "...",
  "views": [
    {
      "view_id": "T1",
      "facet_label": "MODEL_ARCHITECTURE",
      "tree": {...},
      "node_definitions": {...},
      "weight": 0.35
    }
  ]
}
```

### 2. Fit Vectors (`fit_vectors.json`)
```json
[
  {
    "paper_id": "arxiv:2345.67890",
    "fit_reports": [
      {
        "view_id": "T1",
        "label": "FORCE_FIT",
        "scores": {
          "coverage": 0.62,
          "conflict": 0.10,
          "residual": 0.58,
          "fit_score": 0.29
        },
        "lost_novelty": [...]
      }
    ],
    "stress_score": 0.67,
    "unfittable_score": 0.23
  }
]
```

### 3. Complete Results (`igfinder2_results.json`)
Includes baseline, fit vectors, clusters, evolution proposal, and guidance.

### 4. Human-readable Report (`igfinder2_report.md`)
Markdown summary with:
- Statistics
- Multi-view baseline description
- High-stress papers
- Delta-aware guidance

---

## 🔬 Key Design Principles

### 1. LLM Role Boundaries
**LLM only does**:
- Extraction (claims, definitions, facet labels)
- Evidence locating (text spans)
- Candidate generation (proposed nodes, explanations)

**Deterministic algorithms do**:
- Candidate retrieval (embedding similarity)
- Scoring (Coverage/Conflict/Residual formulas)
- Label determination (threshold-based)
- Clustering (HDBSCAN)
- Evolution selection (argmax objective)

### 2. Reproducibility
- **Temperature = 0** for all LLM calls
- **Fixed JSON schemas** for all outputs
- **Evidence anchoring**: All claims must cite original text
- **Deterministic thresholds**: No learned parameters

### 3. Auditable Evidence
Every decision is traceable:
- Why FORCE_FIT? → High residual + lost novelty bullets
- Why UNFITTABLE? → Low coverage or high conflict + evidence
- Why new node? → Cluster evidence + improvement rate

---

## 📈 Evaluation (Planned)

### Time-slice Taxonomy Shift
```
T₀: Old reviews → Build T
T₁: New papers → Stress test
T₂: Future reviews → Ground truth for structure evolution
```

**Metrics**:
- **Branch Hit@K**: How many proposed branches appear in T₂
- **ForceFit/Unfit Reduction**: Structure improvement after updates
- **Evidence Sufficiency**: Support for each proposed structure

### Human Evaluation
Compare surveys generated **with vs. without** IG-Finder 2.0 guidance:
- **Frontierability** (前沿性): Avoids lag, proposes new organization
- **Explanatory Power** (解释力): Provides new cognitive lens
- **Evidence Grounding** (证据锚定): Citations support claims
- **Usefulness** (有用性): Helps survey writing / topic selection

---

## 🛠️ Implementation Status

### ✅ Completed (v0.1)
- [x] Complete design document with formulas and algorithms
- [x] Data structures (TaxonomyView, FitReport, StressCluster, etc.)
- [x] Phase 1: Multi-view Baseline Construction
  - [x] ReviewRetriever with heuristic filtering
  - [x] TaxonomyViewExtractor with fixed JSON schema
  - [x] NodeDefinitionBuilder for testable definitions
  - [x] Weight calculation (Quality·Recency·Coverage)
- [x] Phase 2: Multi-view Stress Test
  - [x] PaperClaimExtractor (enforces 3 novelty bullets)
  - [x] EmbeddingBasedRetriever (Top-K candidates)
  - [x] FitTester (Coverage/Conflict/Residual scoring)
  - [x] Label determination (threshold-based)
  - [x] FitReport generation with evidence
- [x] Engine v2 with pipeline orchestration
- [x] Example running script

### 🚧 In Progress
- [ ] Phase 3: Stress Clustering & Minimal Evolution
  - [ ] HDBSCAN clustering
  - [ ] Cluster type determination
  - [ ] ADD_NODE / SPLIT_NODE / RENAME_NODE operations
  - [ ] Minimal evolution optimization
- [ ] Phase 4: Main/Aux Axis Selection
  - [ ] Main axis selection (FIT rate + stability + coverage)
  - [ ] Aux axis selection (variance of failure rates)
  - [ ] Delta-aware guidance generation
- [ ] Evaluation framework
  - [ ] Time-slice dataset construction
  - [ ] Branch Hit@K implementation
  - [ ] Human evaluation interface

### 📝 TODO (Next Steps)
1. Integrate actual embedding models (SPECTER2, SciNCL, or e5-base)
2. Integrate NLI model for conflict detection (DeBERTa-MNLI)
3. Implement complete Phase 3 and Phase 4
4. Add comprehensive unit tests
5. Run experiments on 3-5 different domains
6. Write evaluation scripts
7. Prepare paper submission

---

## 🎓 Citation

If you use IG-Finder 2.0 in your research, please cite:

```bibtex
@software{igfinder2_2024,
  title={IG-Finder 2.0: Multi-view Atlas Stress Test for Delta-aware Survey Writing},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-repo/ig-finder-2.0}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

This work builds upon:
- **STORM** (Shao et al., NAACL 2024): Foundation framework
- **Co-STORM** (Jiang et al., EMNLP 2024): Dynamic mind map inspiration
- **DSPy**: LLM programming paradigm

---

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com].

---

**IG-Finder 2.0: From "innovation scoring" to "structure failure detection"** 🚀
