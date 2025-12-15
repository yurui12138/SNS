# IG-Finder 2.0: Multi-view Atlas Stress Test & Minimal Evolution for Delta-aware Survey Writing

## Executive Summary

**IG-Finder 2.0 将多篇综述的组织结构建模为多视角 taxonomy 图集 T，对每篇前沿论文在每个视角上做"可审计的归类适配测试"，得到适配向量 v(p)，再以跨视角一致性规则识别结构压力簇，并通过最小必要树编辑（ADD/SPLIT/RENAME）提出结构更新与主轴+辅轴写作约束。LLM 只负责抽取与证据定位；关键判定由可复验规则完成。**

---

## 0. Problem Definition

### The Problem
现有自动综述系统常生成"滞后性综述"（lagging reviews）：
- 沿用既有综述的组织结构与共识边界
- 只做新论文的补录与归类
- 难以产生相对既有综述的认知增量（cognitive delta）

### Our Goal
不是"判定哪些论文是创新（当裁判）"，而是：
1. 将"既有综述的组织结构"形式化为可测试的基线
2. 用新论文对该结构做压力测试，识别结构失效点（何处出现强行归类/无法归类）
3. 基于证据提出最小必要结构更新（minimal necessary change）
4. 生成可用于下游综述写作的结构化指导（主轴/辅轴或新增分支）

### Key Transformation
从"创新打分"转为"结构失效检测 + 结构更新提案"

---

## 1. Inputs & Outputs

### Inputs
1. **主题查询** q (e.g., "transformer models in NLP")
2. **综述集合** R = {r₁, ..., rₘ} (review/survey/tutorial papers)
3. **前沿论文集合** P = {p₁, ..., pₙ} (non-review papers, recent time window)

### Outputs (核心交付物)
1. **Multi-view Cognitive Baseline** T: 多视角 taxonomy 图集
2. **Stress Test Results**: 每篇论文的多视角适配向量 v(p) 与证据
3. **Evolution Proposal**: 最小必要结构更新（树编辑操作序列 + 证据）
4. **Delta-aware Writing Guidance**: 下游综述生成的结构约束
   - 主轴/辅轴（main/aux axes）
   - 大纲（outline with required nodes）
   - 每章必须回答的问题与引用证据

---

## 2. Core Concepts & Formal Definitions

### 2.1 Multi-view Atlas (多视角 Taxonomy 图集)

从多篇综述中抽取得到 taxonomy 集合：

```
T = {T₁, T₂, ..., Tₖ}
```

每个视角 Tᵢ 包含：

#### Tree Structure
- 节点集合与父子关系: Gᵢ = (Vᵢ, Eᵢ)
- 每个节点 v 包含：
  - `name`: 节点名称
  - `definition`: 节点定义文本
  - `inclusion_criteria`: 纳入标准列表
  - `exclusion_criteria`: 排除标准列表
  - `boundary_statements`: 边界条件列表
  - `canonical_keywords`: 关键词列表
  - `evidence_spans`: 证据来源（来自原综述）

#### Facet Label
- `facet(Tᵢ) ∈ {MODEL_ARCHITECTURE, TRAINING_PARADIGM, TASK_SETTING, THEORY, EVALUATION_PROTOCOL, ...}`

#### Weight
- `wᵢ`: 视角权重（基于综述质量、一致性、时间衰减等可复验启发式）

**Weight Calculation Formula:**
```
wᵢ ∝ Quality(rᵢ) · Recency(rᵢ) · Coverage(rᵢ)

where:
- Recency(rᵢ) = exp(-α · (Yₙₒw - Yᵢ)), α = 0.15
- Quality(rᵢ) = log(1 + citation_count) (normalized)
- Coverage(rᵢ) = number of leaf nodes (normalized)

Final normalization: wᵢ = wᵢ / Σⱼ wⱼ
```

### 2.2 Fit Test Function & Fit Vector

定义适配测试函数：

```
f(p, Tᵢ) → (yᵢ, ℓᵢ, eᵢ)
```

其中：
- `yᵢ ∈ {FIT, FORCE_FIT, UNFITTABLE}`: 适配标签
- `ℓᵢ`: 最合适叶节点（或 None）
- `eᵢ`: 证据（evidence spans + 不适配原因）

于是每篇论文得到多视角适配向量：

```
v(p) = (y₁, ..., yₖ)
```

### 2.3 Cross-view Consistency Determination

用权重加权汇总不适配强度：

```
S(p) = Σᵢ₌₁ᵏ wᵢ · 1[yᵢ ≠ FIT]         # Stress score
U(p) = Σᵢ₌₁ᵏ wᵢ · 1[yᵢ = UNFITTABLE]  # Unfittable score
```

在簇级别做判定更稳（降低噪声）：

```
S(C) = (1/|C|) Σₚ∈C S(p)
U(C) = (1/|C|) Σₚ∈C U(p)
```

#### 输出三类结构压力类型：

1. **Strong Shift（跨视角结构压力）**
   - 高权重视角普遍 FORCE/UNFIT
   - 条件: U(C) 高（例如 > 0.55）

2. **Facet-dependent Shift（视角依赖压力）**
   - 部分高权重视角失效、部分仍 FIT
   - 条件: 存在视角 A 失败率 > 0.6，同时存在视角 B 失败率 < 0.2

3. **Stable/Incremental（结构稳定）**
   - 多数高权重视角 FIT

### 2.4 Minimal Necessary Change

将 taxonomy 更新表示为一组可复验编辑操作：

#### Operation Types
- `ADD_NODE(parent, new_node, definition, evidence)`
- `SPLIT_NODE(node, sub_nodes, definitions, evidence)`
- `RENAME_NODE(node, new_name, new_definition, evidence)`

#### Edit Cost Function
```
EditCost(T', T) = Σ operations (cost_of_operation)

where:
- ADD_NODE: cost = 1.0
- SPLIT_NODE: cost = 2.0  
- RENAME_NODE: cost = 0.5
```

#### Fit Gain Function
```
FitGain(T', P) = Δ(#FIT - #FORCE_FIT - 2·#UNFITTABLE)
```

#### Optimization Objective
```
T' = argmax_T' (FitGain(T', P) - λ·EditCost(T', T))

where λ = 0.8 (default regularization parameter)
```

---

## 3. Algorithm Pipeline (可复现实现)

### Phase 1: Multi-view Cognitive Baseline Construction

#### 1.1 ReviewRetriever

**Input**: topic query q  
**Output**: reviews R = {r₁, ..., rₘ}

**Algorithm**:
```python
def retrieve_reviews(q, k=15):
    queries = [
        f"{q} survey",
        f"{q} review", 
        f"{q} overview",
        f"systematic review of {q}"
    ]
    
    results = []
    for query in queries:
        results += search_engine.retrieve(query)
    
    # Remove duplicates by URL
    results = deduplicate(results)
    
    # Filter: heuristic rules
    filtered = []
    for r in results:
        if has_review_keyword(r.title):
            if len(r.abstract) > 120:  # Substantial abstract
                if not is_research_paper(r):  # Exclude research
                    filtered.append(r)
    
    # Sort by quality proxy
    sorted_results = sort_by_quality(filtered)
    
    return sorted_results[:k]

def sort_by_quality(reviews):
    # Simple heuristic: prefer recent + high citation
    scores = []
    for r in reviews:
        score = (
            0.4 * recency_score(r.year) +
            0.4 * log(1 + r.citations) +
            0.2 * has_high_quality_venue(r.venue)
        )
        scores.append((r, score))
    
    return [r for r, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
```

#### 1.2 TaxonomyViewExtractor

**Input**: review paper r  
**Output**: TaxonomyView {tree, facet_label, node_defs, evidence, ...}

**LLM Schema (固定 JSON)**:
```json
{
  "review_id": "arxiv:xxxx.xxxxx",
  "facet_label": "MODEL_ARCHITECTURE | TRAINING_PARADIGM | ...",
  "facet_rationale": "一句话说明该综述主要按什么维度组织",
  "taxonomy_tree": {
    "name": "ROOT",
    "children": [...]
  },
  "evidence_spans": [
    {
      "claim": "The survey organizes methods by...",
      "page": 2,
      "section": "Introduction",
      "char_start": 10234,
      "char_end": 10420,
      "quote": "..."
    }
  ]
}
```

**LLM Prompt Template**:
```
You are a scientific literature analyzer. Extract the taxonomy structure from this review paper.

Review Title: {title}
Review Abstract: {abstract}
Review Full Text: {text}

Task:
1. Identify the primary organizational dimension (facet) of this review
2. Extract the hierarchical taxonomy structure (as a tree)
3. Provide evidence spans from the original text

Output ONLY valid JSON following this schema:
{schema}

Requirements:
- facet_label must be one of: MODEL_ARCHITECTURE, TRAINING_PARADIGM, TASK_SETTING, THEORY, EVALUATION_PROTOCOL, OTHER
- evidence_spans must cite actual text from the review (with char positions)
- temperature=0 for reproducibility
```

#### 1.3 NodeDefinitionBuilder

**Input**: node v from tree, review text  
**Output**: NodeDefinition

**LLM Schema**:
```json
{
  "node_path": "ROOT/CNN-based",
  "definition": "This node covers methods whose primary...",
  "inclusion_criteria": [
    "Uses convolution as primary context aggregation",
    "Assumes local receptive fields"
  ],
  "exclusion_criteria": [
    "Uses recurrence as primary aggregator",
    "Uses global self-attention"
  ],
  "canonical_keywords": ["convolution", "receptive field", "local"],
  "boundary_statements": [
    "Not suitable when long-range dependency dominates..."
  ],
  "evidence_spans": [...]
}
```

**LLM Prompt**:
```
Given a taxonomy node from a review paper, generate testable definition.

Node Name: {node.name}
Node Path: {node.path}
Context from Review: {surrounding_text}

Generate:
1. A clear definition
2. Inclusion criteria (what papers belong here)
3. Exclusion criteria (what papers do NOT belong here)
4. Canonical keywords
5. Boundary statements (edge cases, limitations)

Must provide evidence spans from the original review text.

Output JSON only, temperature=0.
```

#### 1.4 Multi-view Baseline Aggregation

**Algorithm**:
```python
def build_multiview_baseline(reviews, topic):
    views = []
    
    for r in reviews:
        # Extract taxonomy view
        tree, facet, evidence = extract_taxonomy_view(r)
        
        # Build node definitions
        node_defs = {}
        for node in tree.all_nodes():
            node_def = build_node_definition(node, r.text)
            node_defs[node.path] = node_def
        
        # Calculate view weight
        weight = calculate_view_weight(r)
        
        view = TaxonomyView(
            view_id=f"T{len(views)+1}",
            review_id=r.id,
            tree=tree,
            facet_label=facet,
            node_defs=node_defs,
            weight=weight,
            evidence=evidence
        )
        views.append(view)
    
    return MultiViewBaseline(
        topic=topic,
        views=views
    )

def calculate_view_weight(review):
    # Recency
    current_year = datetime.now().year
    recency = exp(-0.15 * (current_year - review.year))
    
    # Quality (citation-based)
    quality = log(1 + review.citations) / 10.0  # normalized
    
    # Coverage (number of leaf nodes)
    tree = extract_tree(review)
    coverage = len(tree.leaf_nodes()) / 50.0  # normalized
    
    raw_weight = recency * quality * coverage
    return raw_weight  # Will be normalized across all views
```

---

### Phase 2: Multi-view Stress Test

#### 2.1 PaperClaimExtractor

**Input**: research paper p  
**Output**: PaperClaims

**LLM Schema**:
```json
{
  "paper_id": "arxiv:xxxx.xxxxx",
  "claims": {
    "problem": {
      "text": "The paper addresses...",
      "evidence": [{"page": 1, "section": "Introduction", "quote": "..."}]
    },
    "core_idea": [
      {"text": "The key idea is...", "evidence": [...]}
    ],
    "mechanism": [
      {"text": "The method works by...", "evidence": [...]}
    ],
    "training": [
      {"text": "Training procedure...", "evidence": [...]}
    ],
    "evaluation": [
      {"text": "Evaluated on...", "evidence": [...]}
    ],
    "novelty_bullets": [
      {"text": "Novel contribution 1", "evidence": [...]},
      {"text": "Novel contribution 2", "evidence": [...]},
      {"text": "Novel contribution 3", "evidence": [...]}
    ]
  },
  "keywords": ["attention", "transformer", "self-attention"],
  "tasks_datasets": ["machine translation", "WMT14"],
  "methods_components": ["multi-head attention", "positional encoding"]
}
```

**Requirements**:
- `novelty_bullets` 强制 3 条（稳定）
- All `evidence` must reference actual text spans from paper PDF
- temperature=0

#### 2.2 Candidate Leaf Retrieval (Top-K)

**不让 LLM 自由匹配，先用确定性检索缩小范围**

**Algorithm**:
```python
def retrieve_candidate_leaves(paper, view, k=5):
    # Step 1: Precompute embeddings
    paper_text = (
        paper.title + " " + 
        paper.abstract + " " + 
        " ".join([claim.text for claim in paper.novelty_bullets])
    )
    paper_vector = embedding_model.encode(paper_text)  # SPECTER2 or SciNCL
    
    # Step 2: Compute similarity to all leaf nodes
    similarities = []
    for leaf in view.tree.leaf_nodes():
        leaf_text = (
            leaf.name + " " + 
            leaf.definition + " " + 
            " ".join(leaf.keywords)
        )
        leaf_vector = embedding_model.encode(leaf_text)
        sim = cosine_similarity(paper_vector, leaf_vector)
        similarities.append((leaf, sim))
    
    # Step 3: Return top-K
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    return [leaf for leaf, _ in top_k]
```

**使用现成科学 Embedding**:
- SPECTER2: `allenai/specter2`
- SciNCL: `malteos/scincl`
- e5-base: `intfloat/e5-base-v2`

选一个即可，无需训练。

#### 2.3 Fit Score Calculation (Coverage / Conflict / Residual)

对每个候选叶节点 ℓ ∈ Lᵢ(p)，计算三个分数：

##### A) Coverage (覆盖度)

**定义**: 叶节点定义能否覆盖论文核心内容

**公式**:
```
cov_sem = cosine(paper_vector, leaf_vector)
cov_lex = Jaccard(keywords_paper, keywords_leaf)

Coverage = 0.7 · cov_sem + 0.3 · cov_lex
```

**实现**:
```python
def calculate_coverage(paper, leaf):
    # Semantic coverage
    paper_vec = encode(paper.title + " " + paper.abstract + " " + 
                       " ".join(paper.novelty_bullets))
    leaf_vec = encode(leaf.name + " " + leaf.definition + " " + 
                     " ".join(leaf.keywords))
    cov_sem = cosine_similarity(paper_vec, leaf_vec)
    
    # Lexical coverage (Jaccard)
    paper_keywords = set(paper.keywords)
    leaf_keywords = set(leaf.canonical_keywords)
    if len(paper_keywords | leaf_keywords) == 0:
        cov_lex = 0.0
    else:
        cov_lex = len(paper_keywords & leaf_keywords) / len(paper_keywords | leaf_keywords)
    
    coverage = 0.7 * cov_sem + 0.3 * cov_lex
    return coverage
```

##### B) Conflict (边界冲突)

**定义**: 论文是否违反节点的排除标准/边界条件

**方法**: 使用现成 NLI 模型（无需训练）

**公式**:
```
Conflict = max_{h ∈ Exclusion+Boundary} P_NLI(contradiction | claim, h)
```

**实现**:
```python
def calculate_conflict(paper, leaf):
    # Use pretrained NLI model (e.g., DeBERTa-MNLI)
    nli_model = load_nli_model("microsoft/deberta-v3-large-mnli")
    
    max_contradiction = 0.0
    
    # Check against exclusion criteria
    for exclusion in leaf.exclusion_criteria:
        for claim in paper.claims.core_idea + paper.claims.novelty_bullets:
            result = nli_model.predict(
                premise=claim.text,
                hypothesis=exclusion
            )
            contradiction_prob = result['contradiction']
            max_contradiction = max(max_contradiction, contradiction_prob)
    
    # Check against boundary statements
    for boundary in leaf.boundary_statements:
        for claim in paper.claims.core_idea + paper.claims.mechanism:
            result = nli_model.predict(
                premise=claim.text,
                hypothesis=boundary
            )
            contradiction_prob = result['contradiction']
            max_contradiction = max(max_contradiction, contradiction_prob)
    
    return max_contradiction
```

**如果坚决不用 NLI 模型**（虽然不推荐）:
```python
def calculate_conflict_heuristic(paper, leaf):
    # Fallback: keyword-based conflict detection
    conflict_score = 0.0
    
    for exclusion_keyword in extract_keywords(leaf.exclusion_criteria):
        if exclusion_keyword in paper.keywords:
            conflict_score += 0.2
    
    # Check if paper keywords contradict leaf keywords
    paper_kw_set = set(paper.keywords)
    exclusion_kw_set = set(extract_keywords(leaf.exclusion_criteria))
    overlap = len(paper_kw_set & exclusion_kw_set)
    
    conflict_score = min(1.0, overlap / max(1, len(exclusion_kw_set)))
    return conflict_score
```

##### C) Residual (贡献丢失度)

**定义**: 论文关键贡献无法用该节点语言表达的程度

**公式**:
```
Residual = 1 - max_{b ∈ NoveltyBullets} cos(emb(b), leaf_vector)
```

**实现**:
```python
def calculate_residual(paper, leaf):
    leaf_vec = encode(leaf.name + " " + leaf.definition)
    
    max_similarity = 0.0
    for novelty in paper.novelty_bullets:
        novelty_vec = encode(novelty.text)
        sim = cosine_similarity(novelty_vec, leaf_vec)
        max_similarity = max(max_similarity, sim)
    
    residual = 1.0 - max_similarity
    return residual
```

**意义**: Residual 高表示"强行塞入会丢信息"

##### Combined Fit Score

**公式**:
```
FitScore = Coverage - 0.8 · Conflict - 0.4 · Residual
```

**选择最佳叶节点**:
```
ℓᵢ = argmax_ℓ FitScore(p, ℓ)
```

#### 2.4 Label Determination (确定性阈值)

**规则**:
```python
def determine_label(coverage, conflict, residual):
    if coverage < 0.45 or conflict > 0.55:
        return "UNFITTABLE"
    elif residual > 0.45:
        return "FORCE_FIT"
    else:
        return "FIT"
```

**阈值校准** (审稿友好):
- 初始使用上述默认值
- 在小验证集上通过网格搜索调优（不训练，只调参）
- 做 ablation：不同阈值敏感性分析

#### 2.5 Fit Report Output (可审计)

**Schema**:
```json
{
  "paper_id": "arxiv:2345.67890",
  "view_id": "T1",
  "facet_label": "MODEL_ARCHITECTURE",
  "best_leaf_path": "ROOT/RNN-based/LSTM",
  "label": "FORCE_FIT",
  "scores": {
    "coverage": 0.62,
    "conflict": 0.10,
    "residual": 0.58,
    "fit_score": 0.29
  },
  "lost_novelty": [
    {
      "bullet": "Uses global self-attention instead of recurrence",
      "evidence": [...],
      "similarity_to_leaf": 0.21
    }
  ],
  "conflict_evidence": [
    {
      "boundary": "excludes attention-based global aggregation",
      "nli_contradiction": 0.61,
      "paper_claim": "Our model uses multi-head self-attention..."
    }
  ]
}
```

**关键**: 所有判定必须可追溯到原文证据

#### 2.6 Fit Vector v(p)

```python
def compute_fit_vector(paper, multiview_baseline):
    fit_reports = []
    
    for view in multiview_baseline.views:
        # Get candidate leaves
        candidates = retrieve_candidate_leaves(paper, view, k=5)
        
        # Calculate fit scores for each candidate
        best_leaf = None
        best_score = -float('inf')
        best_report = None
        
        for leaf in candidates:
            coverage = calculate_coverage(paper, leaf)
            conflict = calculate_conflict(paper, leaf)
            residual = calculate_residual(paper, leaf)
            
            fit_score = coverage - 0.8 * conflict - 0.4 * residual
            
            if fit_score > best_score:
                best_score = fit_score
                best_leaf = leaf
                best_report = {
                    'coverage': coverage,
                    'conflict': conflict,
                    'residual': residual,
                    'fit_score': fit_score
                }
        
        # Determine label
        label = determine_label(
            best_report['coverage'],
            best_report['conflict'],
            best_report['residual']
        )
        
        # Generate full fit report
        fit_report = FitReport(
            paper_id=paper.id,
            view_id=view.view_id,
            facet_label=view.facet_label,
            best_leaf_path=best_leaf.path if best_leaf else None,
            label=label,
            scores=best_report,
            lost_novelty=extract_lost_novelty(paper, best_leaf, best_report['residual']),
            conflict_evidence=extract_conflict_evidence(paper, best_leaf, best_report['conflict'])
        )
        
        fit_reports.append(fit_report)
    
    # Compute fit vector
    v_p = [report.label for report in fit_reports]
    
    # Compute weighted scores
    S_p = sum(
        view.weight * (1 if report.label != "FIT" else 0)
        for view, report in zip(multiview_baseline.views, fit_reports)
    )
    
    U_p = sum(
        view.weight * (1 if report.label == "UNFITTABLE" else 0)
        for view, report in zip(multiview_baseline.views, fit_reports)
    )
    
    return {
        'fit_vector': v_p,
        'fit_reports': fit_reports,
        'stress_score': S_p,
        'unfittable_score': U_p
    }
```

---

### Phase 3: Stress Clustering & Minimal Evolution

#### 3.1 Stress Clustering

**目标**: 聚类出现 FORCE_FIT/UNFITTABLE 的论文

**Algorithm**:
```python
def stress_clustering(papers, fit_results, min_cluster_size=3):
    # Step 1: Filter papers with stress
    stressed_papers = [
        p for p in papers
        if fit_results[p.id]['stress_score'] > 0.3  # threshold
    ]
    
    if len(stressed_papers) < min_cluster_size:
        return []
    
    # Step 2: Construct failure signature for each paper
    signatures = []
    for p in stressed_papers:
        sig_text = construct_failure_signature(p, fit_results[p.id])
        sig_vec = embedding_model.encode(sig_text)
        signatures.append(sig_vec)
    
    # Step 3: Cluster using HDBSCAN (no need to specify K)
    from hdbscan import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='cosine'
    )
    cluster_labels = clusterer.fit_predict(np.array(signatures))
    
    # Step 4: Group papers by cluster
    clusters = defaultdict(list)
    for paper, label in zip(stressed_papers, cluster_labels):
        if label != -1:  # -1 = noise
            clusters[label].append(paper)
    
    return list(clusters.values())

def construct_failure_signature(paper, fit_result):
    # Combine failure information from all views
    sig_parts = []
    
    for report in fit_result['fit_reports']:
        if report.label != "FIT":
            sig_parts.append(f"{report.facet_label}:{report.best_leaf_path}")
            sig_parts.append(" ".join([nov.bullet for nov in report.lost_novelty]))
    
    sig_parts.extend(paper.keywords)
    
    return " ".join(sig_parts)
```

#### 3.2 Cluster Type Determination

**目标**: 判定簇属于 Strong Shift / Facet-dependent / Stable

**Algorithm**:
```python
def determine_cluster_type(cluster, fit_results, views):
    # Calculate per-view failure rate for this cluster
    view_failure_rates = {}
    
    for view in views:
        failures = 0
        total = len(cluster)
        
        for paper in cluster:
            report = get_report_for_view(fit_results[paper.id], view.view_id)
            if report.label != "FIT":
                failures += 1
        
        view_failure_rates[view.view_id] = failures / total
    
    # Calculate cluster-level unfittable score
    U_C = np.mean([
        fit_results[p.id]['unfittable_score']
        for p in cluster
    ])
    
    # Determine type based on rules
    
    # Rule 1: Strong Shift
    if U_C > 0.55:
        # Check if multiple high-weight views fail
        high_weight_views = [v for v in views if v.weight > np.percentile([v2.weight for v2 in views], 70)]
        high_weight_failures = sum(
            1 for v in high_weight_views
            if view_failure_rates[v.view_id] > 0.6
        )
        if high_weight_failures >= 2:
            return "STRONG_SHIFT"
    
    # Rule 2: Facet-dependent Shift
    high_failure_views = [
        v for v in views
        if v.weight > np.percentile([v2.weight for v2 in views], 70) and
           view_failure_rates[v.view_id] > 0.6
    ]
    
    low_failure_views = [
        v for v in views
        if v.weight > np.percentile([v2.weight for v2 in views], 70) and
           view_failure_rates[v.view_id] < 0.2
    ]
    
    if len(high_failure_views) > 0 and len(low_failure_views) > 0:
        return "FACET_DEPENDENT"
    
    # Rule 3: Stable (default)
    return "STABLE"
```

#### 3.3 Evolution Planner

**支持三种操作**:

##### Operation 1: ADD_NODE

**触发条件**: 簇在某视角大量 UNFITTABLE 或 FORCE_FIT

**Algorithm**:
```python
def propose_add_node(cluster, view, fit_reports):
    # Step 1: Find best parent node
    cluster_summary = summarize_cluster(cluster)
    cluster_vec = embedding_model.encode(cluster_summary)
    
    best_parent = None
    best_similarity = -1.0
    
    for node in view.tree.internal_nodes():
        node_vec = embedding_model.encode(
            node.name + " " + node.definition
        )
        sim = cosine_similarity(cluster_vec, node_vec)
        if sim > best_similarity:
            best_similarity = sim
            best_parent = node
    
    # Step 2: LLM generates new node name + definition
    new_node_proposal = llm.generate(
        prompt=f"""
        Generate a new taxonomy node for the following research cluster.
        
        Parent Node: {best_parent.name}
        Parent Definition: {best_parent.definition}
        
        Cluster Papers:
        {format_cluster_papers(cluster)}
        
        Cluster Key Innovations:
        {extract_cluster_innovations(cluster, fit_reports)}
        
        Generate:
        1. New node name
        2. Definition
        3. Inclusion criteria
        4. Exclusion criteria
        5. Keywords
        
        Output JSON only. Temperature=0.
        """,
        schema=NewNodeSchema
    )
    
    # Step 3: Validate by re-testing cluster papers
    temp_view = view.copy()
    temp_view.tree.add_child(best_parent, new_node_proposal)
    
    # Re-run fit test on cluster papers
    improved_count = 0
    for paper in cluster:
        old_report = get_report_for_view(fit_reports[paper.id], view.view_id)
        new_report = run_fit_test(paper, temp_view)
        
        if new_report.label == "FIT" and old_report.label != "FIT":
            improved_count += 1
    
    improvement_rate = improved_count / len(cluster)
    
    # Only accept if significant improvement
    if improvement_rate > 0.2:  # At least 20% improvement
        return AddNodeOperation(
            view_id=view.view_id,
            parent_path=best_parent.path,
            new_node=new_node_proposal,
            evidence=extract_evidence(cluster),
            improvement_rate=improvement_rate
        )
    else:
        return None  # Reject operation
```

##### Operation 2: SPLIT_NODE

**触发条件**: 很多论文都被分到同一个叶节点但 Residual 高

**Algorithm**:
```python
def propose_split_node(leaf, assigned_papers, fit_reports):
    # Step 1: Check if splitting is needed
    high_residual_count = sum(
        1 for p in assigned_papers
        if get_report_for_leaf(fit_reports[p.id], leaf.path).scores['residual'] > 0.45
    )
    
    if high_residual_count < 0.5 * len(assigned_papers):
        return None  # Not enough stress to warrant split
    
    # Step 2: Sub-cluster the assigned papers
    paper_vecs = [embedding_model.encode(paper_to_text(p)) for p in assigned_papers]
    
    from sklearn.cluster import KMeans
    n_subclusters = min(3, max(2, len(assigned_papers) // 3))
    kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
    subcluster_labels = kmeans.fit_predict(paper_vecs)
    
    subclusters = defaultdict(list)
    for paper, label in zip(assigned_papers, subcluster_labels):
        subclusters[label].append(paper)
    
    # Step 3: LLM generates sub-node definitions
    sub_nodes = []
    for sub_papers in subclusters.values():
        sub_node = llm.generate(
            prompt=f"""
            Generate a sub-category definition for the following papers,
            which are currently all classified under: {leaf.name}
            
            Original Node Definition: {leaf.definition}
            
            Papers in this sub-group:
            {format_papers(sub_papers)}
            
            Generate a more specific sub-category that captures these papers.
            Output JSON only. Temperature=0.
            """,
            schema=NewNodeSchema
        )
        sub_nodes.append(sub_node)
    
    # Step 4: Validate improvement
    fit_gain = calculate_fit_gain_after_split(leaf, sub_nodes, assigned_papers)
    
    if fit_gain > 0.3:  # Threshold
        return SplitNodeOperation(
            node_path=leaf.path,
            sub_nodes=sub_nodes,
            evidence=extract_evidence(assigned_papers),
            fit_gain=fit_gain
        )
    else:
        return None
```

##### Operation 3: RENAME_NODE

**触发条件**: 节点 Coverage 高、Conflict 低，但关键词分布明显变化

**Algorithm**:
```python
def propose_rename_node(node, recent_papers, fit_reports):
    # Step 1: Check keyword drift
    old_keywords = set(node.canonical_keywords)
    
    # Extract keywords from recent papers assigned to this node
    recent_keywords = set()
    for paper in recent_papers:
        if is_assigned_to_node(paper, node, fit_reports):
            recent_keywords.update(paper.keywords)
    
    # Calculate drift
    old_vec = embedding_model.encode(" ".join(old_keywords))
    new_vec = embedding_model.encode(" ".join(recent_keywords))
    drift = 1.0 - cosine_similarity(old_vec, new_vec)
    
    if drift < 0.35:
        return None  # Not enough drift
    
    # Check if recent papers dominate
    recent_count = len([p for p in recent_papers if p.year >= 2023])
    if recent_count < 0.6 * len(recent_papers):
        return None  # Not dominated by recent work
    
    # Step 2: LLM generates new name and definition
    renamed = llm.generate(
        prompt=f"""
        The taxonomy node "{node.name}" shows semantic drift.
        
        Original Definition: {node.definition}
        Original Keywords: {old_keywords}
        
        Recent Papers:
        {format_papers(recent_papers)}
        
        New Keywords: {recent_keywords}
        
        Suggest a revised name and definition that better reflects the current research.
        Output JSON only. Temperature=0.
        """,
        schema=RenameNodeSchema
    )
    
    return RenameNodeOperation(
        node_path=node.path,
        old_name=node.name,
        new_name=renamed.name,
        new_definition=renamed.definition,
        drift_score=drift,
        evidence=extract_evidence(recent_papers)
    )
```

#### 3.4 Minimal Evolution Optimization

**Objective Function**:
```
T' = argmax_T' (FitGain(T', P) - λ·EditCost(T', T))
```

**Greedy Algorithm** (sufficient for v1.0):
```python
def select_minimal_evolution(clusters, views, fit_results, lambda_reg=0.8):
    selected_operations = []
    
    for cluster in clusters:
        cluster_type = determine_cluster_type(cluster, fit_results, views)
        
        if cluster_type in ["STRONG_SHIFT", "FACET_DEPENDENT"]:
            # Generate candidate operations
            candidates = []
            
            for view in views:
                # Try ADD_NODE
                add_op = propose_add_node(cluster, view, fit_results)
                if add_op:
                    candidates.append(add_op)
                
                # Try SPLIT_NODE (for overcrowded nodes)
                for leaf in view.tree.leaf_nodes():
                    split_op = propose_split_node(leaf, cluster, fit_results)
                    if split_op:
                        candidates.append(split_op)
                
                # Try RENAME_NODE (for drifted nodes)
                for node in view.tree.all_nodes():
                    rename_op = propose_rename_node(node, cluster, fit_results)
                    if rename_op:
                        candidates.append(rename_op)
            
            # Select best operation (max objective)
            best_op = None
            best_objective = -float('inf')
            
            for op in candidates:
                fit_gain = calculate_fit_gain(op, cluster, views, fit_results)
                edit_cost = get_edit_cost(op)
                
                objective = fit_gain - lambda_reg * edit_cost
                
                if objective > best_objective:
                    best_objective = objective
                    best_op = op
            
            if best_op and best_objective > 0:
                selected_operations.append(best_op)
    
    return selected_operations

def get_edit_cost(operation):
    if isinstance(operation, AddNodeOperation):
        return 1.0
    elif isinstance(operation, SplitNodeOperation):
        return 2.0
    elif isinstance(operation, RenameNodeOperation):
        return 0.5
    else:
        return 0.0

def calculate_fit_gain(operation, cluster, views, fit_results):
    # Simulate applying the operation
    temp_views = apply_operation_temporarily(views, operation)
    
    # Re-run fit test on cluster papers
    old_fit_count = sum(
        1 for p in cluster
        if any(r.label == "FIT" for r in fit_results[p.id]['fit_reports'])
    )
    
    new_fit_count = 0
    for paper in cluster:
        new_reports = run_fit_test_multiview(paper, temp_views)
        if any(r.label == "FIT" for r in new_reports):
            new_fit_count += 1
    
    # FitGain = Δ(#FIT - #FORCE_FIT - 2·#UNFITTABLE)
    fit_gain = (new_fit_count - old_fit_count) / len(cluster)
    
    return fit_gain
```

---

### Phase 4: Main/Aux Axis Selection

#### 4.1 Main Axis Selection

**目标**: 选择覆盖面广且稳定的视角作为主轴

**Algorithm**:
```python
def select_main_axis(views, papers, fit_results):
    axis_scores = []
    
    for view in views:
        # Calculate FIT rate
        fit_count = 0
        total = 0
        for paper in papers:
            report = get_report_for_view(fit_results[paper.id], view.view_id)
            total += 1
            if report.label == "FIT":
                fit_count += 1
        
        fit_rate = fit_count / total if total > 0 else 0.0
        
        # Calculate stability (cross-review consistency)
        # Approximation: if multiple reviews have same facet, higher stability
        same_facet_count = sum(1 for v in views if v.facet_label == view.facet_label)
        stability = same_facet_count / len(views)
        
        # Calculate coverage (richness of taxonomy)
        coverage = len(view.tree.leaf_nodes()) / 50.0  # normalized
        coverage = min(1.0, coverage)
        
        # Combined score
        score = 0.6 * fit_rate + 0.3 * stability + 0.1 * coverage
        
        axis_scores.append((view, score))
    
    # Select main axis
    main_axis = max(axis_scores, key=lambda x: x[1])[0]
    
    return main_axis
```

#### 4.2 Auxiliary Axis Selection

**目标**: 选择能解释压力簇差异的视角作为辅轴

**Algorithm**:
```python
def select_aux_axis(views, clusters, fit_results, main_axis):
    # Calculate variance of failure rate across clusters for each view
    axis_discriminativeness = []
    
    for view in views:
        if view.view_id == main_axis.view_id:
            continue  # Skip main axis
        
        # Calculate failure rate for each cluster
        cluster_failure_rates = []
        for cluster in clusters:
            failures = 0
            for paper in cluster:
                report = get_report_for_view(fit_results[paper.id], view.view_id)
                if report.label != "FIT":
                    failures += 1
            
            failure_rate = failures / len(cluster)
            cluster_failure_rates.append(failure_rate)
        
        # Calculate variance (high variance = good discriminator)
        if len(cluster_failure_rates) > 1:
            variance = np.var(cluster_failure_rates)
        else:
            variance = 0.0
        
        axis_discriminativeness.append((view, variance))
    
    # Select aux axis with highest discriminativeness
    if len(axis_discriminativeness) > 0:
        aux_axis = max(axis_discriminativeness, key=lambda x: x[1])[0]
        return aux_axis
    else:
        return None
```

---

### Phase 5: Delta-aware Writing Guidance Generation

**目标**: 生成可直接用于下游综述生成的结构化指导

**Output Schema**:
```json
{
  "outline": [
    {
      "section": "Pretraining Paradigms (MAIN AXIS)",
      "subsections": [
        {
          "subsection": "Architecture Mechanisms (AUX AXIS)",
          "required_nodes": ["CNN", "RNN", "Self-Attention"],
          "required_citations": ["arxiv:1706.03762", "..."],
          "must_answer": [
            "Why old CNN/RNN structure taxonomy is insufficient (with evidence)",
            "Key trade-offs across mechanisms under pretraining"
          ],
          "evidence_cards": [
            {
              "paper_id": "arxiv:1706.03762",
              "claim": "Attention is all you need",
              "quote": "...",
              "page": 1
            }
          ]
        }
      ]
    }
  ],
  "evolution_summary": [
    {
      "operation": "ADD_NODE",
      "view": "MODEL_ARCHITECTURE",
      "parent": "ROOT",
      "new_node": "Self-Attention-based",
      "trigger_cluster": "C3",
      "justification_evidence": [...]
    }
  ]
}
```

**Algorithm**:
```python
def generate_delta_aware_guidance(
    main_axis,
    aux_axis,
    evolution_operations,
    clusters,
    fit_results
):
    # Step 1: Build outline structure
    outline = []
    
    # Use main axis as top-level structure
    for main_node in main_axis.tree.children_of_root():
        section = {
            "section": f"{main_node.name} (MAIN AXIS)",
            "subsections": []
        }
        
        # If aux_axis exists, cross-organize
        if aux_axis:
            for aux_node in aux_axis.tree.children_of_root():
                subsection = generate_subsection(
                    main_node,
                    aux_node,
                    clusters,
                    fit_results
                )
                section["subsections"].append(subsection)
        else:
            # No aux axis, just use main axis children
            for child in main_axis.tree.children(main_node):
                subsection = generate_subsection(
                    child,
                    None,
                    clusters,
                    fit_results
                )
                section["subsections"].append(subsection)
        
        outline.append(section)
    
    # Step 2: Summarize evolution operations
    evolution_summary = [
        {
            "operation": op.type,
            "view": op.view_id,
            "parent": op.parent_path if hasattr(op, 'parent_path') else None,
            "new_node": op.new_node.name if hasattr(op, 'new_node') else None,
            "trigger_cluster": op.cluster_id if hasattr(op, 'cluster_id') else None,
            "justification_evidence": op.evidence
        }
        for op in evolution_operations
    ]
    
    # Step 3: Generate must-answer questions
    must_answer_questions = generate_must_answer_questions(
        evolution_operations,
        clusters
    )
    
    return DeltaAwareGuidance(
        outline=outline,
        evolution_summary=evolution_summary,
        must_answer_questions=must_answer_questions
    )

def generate_subsection(main_node, aux_node, clusters, fit_results):
    # Find relevant clusters for this subsection
    relevant_clusters = find_clusters_for_nodes(main_node, aux_node, clusters, fit_results)
    
    # Extract required papers
    required_citations = []
    evidence_cards = []
    for cluster in relevant_clusters:
        for paper in cluster.core_papers[:3]:  # Top 3 per cluster
            required_citations.append(paper.id)
            evidence_cards.append({
                "paper_id": paper.id,
                "claim": paper.core_claims[0] if paper.core_claims else "",
                "quote": extract_key_quote(paper),
                "page": 1  # Simplified
            })
    
    # Generate must-answer questions
    must_answer = []
    if len(relevant_clusters) > 0:
        must_answer.append(
            f"Why is the existing taxonomy insufficient for {main_node.name}?"
        )
        must_answer.append(
            f"What are the key differences introduced by papers in this category?"
        )
    
    subsection_name = main_node.name
    if aux_node:
        subsection_name += f" × {aux_node.name}"
    
    return {
        "subsection": subsection_name,
        "required_nodes": [main_node.path, aux_node.path if aux_node else None],
        "required_citations": required_citations,
        "must_answer": must_answer,
        "evidence_cards": evidence_cards
    }
```

---

## 4. LLM Role Boundaries (审稿人关注点)

### LLM Only Does:
1. **Extraction**: Claims, node definitions, facet labels
2. **Evidence Locating**: Original text spans
3. **Candidate Generation**: Proposed node names, definitions, explanations

### Deterministic Algorithm Does:
1. **Candidate Retrieval**: Top-K leaves (embedding similarity)
2. **Scoring**: Coverage / Conflict / Residual (formulas)
3. **Label Determination**: FIT / FORCE_FIT / UNFITTABLE (thresholds)
4. **Clustering**: HDBSCAN on failure signatures
5. **Type Determination**: Strong / Facet-dependent / Stable (rules)
6. **Evolution Selection**: argmax (FitGain - λ·EditCost)
7. **Axis Selection**: argmax (weighted scores)

### Reproducibility Measures:
- **Temperature = 0** for all LLM calls
- **Fixed sampling strategy** (or deterministic decoding)
- **Multi-model consistency check** (optional, GPT-4 vs Claude vs Gemini)
- **All intermediate results cached** in JSON

---

## 5. Evaluation Framework

### 5.1 Time-slice Taxonomy Shift (主实验)

**Setup**:
```
T₀: Old reviews (build T)
T₁: New papers (stress test input)
T₂: Subsequent reviews/tutorials (ground truth for structure evolution)
```

**Metrics**:

#### Metric 1: Branch Hit@K
```
Branch Hit@K = (# of our top-K proposed branches that appear in T₂) / K

where K ∈ {3, 5, 10}
```

**Implementation**:
```python
def calculate_branch_hit_at_k(proposed_branches, t2_taxonomy, k):
    # Extract section/chapter titles from T₂
    t2_sections = extract_sections(t2_taxonomy)
    
    # For each proposed branch, check if synonymous section exists in T₂
    hits = 0
    for i, branch in enumerate(proposed_branches[:k]):
        for t2_sec in t2_sections:
            similarity = semantic_similarity(branch.name, t2_sec.name)
            if similarity > 0.75:  # Threshold for synonym match
                hits += 1
                break
    
    return hits / k
```

#### Metric 2: ForceFit/Unfit Reduction
```
Reduction = (Force+Unfit_before - Force+Unfit_after) / Force+Unfit_before
```

**Implementation**:
```python
def calculate_reduction(papers, baseline_before, baseline_after):
    force_unfit_before = sum(
        1 for p in papers
        if any(r.label in ["FORCE_FIT", "UNFITTABLE"] 
               for r in test_paper(p, baseline_before))
    )
    
    force_unfit_after = sum(
        1 for p in papers
        if any(r.label in ["FORCE_FIT", "UNFITTABLE"]
               for r in test_paper(p, baseline_after))
    )
    
    if force_unfit_before == 0:
        return 0.0
    
    return (force_unfit_before - force_unfit_after) / force_unfit_before
```

#### Metric 3: Evidence Sufficiency
```
Evidence Sufficiency = (# of proposed structures with ≥m papers + evidence) / (# of proposed structures)

where m = 3 (default)
```

### 5.2 Downstream Survey Quality (人评为主)

**Control Group**: 同一 baseline 综述生成器
- 检索预算一致
- 引用数量一致
- 长度一致
- 输出模板一致

**唯一差异**: 是否使用 IG-Finder 的 delta-aware guidance

**Human Evaluation Dimensions** (5-point Likert scale):
1. **Frontierability (前沿性)**: 是否避免滞后、是否提出新组织
2. **Explanatory Power (解释力)**: 是否给出新认知透镜/对比维度
3. **Evidence Grounding (证据锚定)**: 引用是否支持论断
4. **Usefulness (有用性)**: 对写综述/选题决策的帮助

**Important**: Allow three options:
- A is better
- B is better  
- About the same / Both not good

(避免强迫选择噪声)

### 5.3 Ablation Studies (必做)

```
1. No multi-view: 只用单一综述树
2. No stress test: 直接生成大纲/综述
3. No minimal evolution: 只报差异，不给结构更新
4. No evidence grounding: 去掉证据段落
5. Uniform weights: wᵢ = 1/k (不区分视角权重)
```

**Measure Impact**:
- Branch Hit@K
- ForceFit/Unfit Reduction
- Human eval scores
- Evidence sufficiency

---

## 6. Failure Modes & Boundaries (主动写入论文)

### Failure Mode 1: 综述本身滞后或带强学派偏置

**Mitigation**:
- 多视角 + 权重 + 跨综述一致性
- 报告视角间的争议区域（disagreement zones）

### Failure Mode 2: 前沿论文文本不完整（仅摘要）

**Mitigation**:
- 证据锚定能力下降，明确数据来源与限制
- 建议使用全文 PDF（grobid解析）

### Failure Mode 3: 新方向早期稀疏

**Mitigation**:
- 容易被当噪声
- 通过簇级判定与证据充分性阈值缓解
- 明确标注"潜在创新，需更多证据"

### Failure Mode 4: 维度不正交（结构与训练范式相关）

**Mitigation**:
- 不强称正交，只称 multi-view
- 在输出中允许主轴/辅轴交叉
- 明确说明视角可能有语义重叠

---

## 7. Implementation Checklist

### 阶段 1: 数据结构 & 基础设施
- [ ] 定义 `TaxonomyView` dataclass
- [ ] 定义 `NodeDefinition` dataclass
- [ ] 定义 `FitReport` dataclass
- [ ] 定义 `StressCluster` dataclass
- [ ] 定义 `EvolutionOperation` (ADD/SPLIT/RENAME)
- [ ] 定义 `DeltaAwareGuidance` dataclass
- [ ] PDF解析模块（PDF → text with spans）

### 阶段 2: Phase 1 实现
- [ ] `ReviewRetriever` (with filtering rules)
- [ ] `TaxonomyViewExtractor` (LLM + schema)
- [ ] `NodeDefinitionBuilder` (LLM + schema)
- [ ] `MultiViewBaselineBuilder` (aggregation + weight calculation)
- [ ] Weight calculation formula implementation

### 阶段 3: Phase 2 实现
- [ ] `PaperClaimExtractor` (LLM + schema)
- [ ] Embedding model setup (SPECTER2 or SciNCL)
- [ ] `retrieve_candidate_leaves` (Top-K retrieval)
- [ ] `calculate_coverage` (semantic + lexical)
- [ ] `calculate_conflict` (NLI-based or heuristic)
- [ ] `calculate_residual` (max similarity to novelty)
- [ ] `determine_label` (threshold-based)
- [ ] `FitReport` generation

### 阶段 4: Phase 3 实现
- [ ] `stress_clustering` (HDBSCAN)
- [ ] `determine_cluster_type` (Strong/Facet-dependent/Stable)
- [ ] `propose_add_node` (with validation)
- [ ] `propose_split_node` (with sub-clustering)
- [ ] `propose_rename_node` (with drift detection)
- [ ] `select_minimal_evolution` (greedy optimization)

### 阶段 5: Axis Selection & Guidance
- [ ] `select_main_axis` (FIT rate + stability + coverage)
- [ ] `select_aux_axis` (variance of failure rates)
- [ ] `generate_delta_aware_guidance` (outline + evidence cards)

### 阶段 6: 评估框架
- [ ] Time-slice dataset construction
- [ ] Branch Hit@K implementation
- [ ] ForceFit/Unfit Reduction calculation
- [ ] Evidence Sufficiency check
- [ ] Human evaluation interface

### 阶段 7: 文档 & 示例
- [ ] 更新 README with IG-Finder 2.0 说明
- [ ] 完整 API 文档
- [ ] 运行示例脚本
- [ ] Jupyter notebook tutorial

---

## 8. Timeline Estimate (For Reference)

| Phase | Task | Estimated Time |
|-------|------|---------------|
| 1 | Data structures & Infrastructure | 2-3 days |
| 2 | Phase 1 implementation | 3-4 days |
| 3 | Phase 2 implementation | 4-5 days |
| 4 | Phase 3 implementation | 3-4 days |
| 5 | Axis selection & Guidance | 2-3 days |
| 6 | Evaluation framework | 2-3 days |
| 7 | Testing & Bug fixing | 2-3 days |
| 8 | Documentation & Examples | 2 days |
| **Total** | | **20-27 days** |

---

## 9. Paper Writing Strategy

### Contribution Claims (顶会友好)
1. **Formalization**: 将综述结构形式化为可测试的 multi-view taxonomy atlas
2. **Stress Test**: 提出适配向量 v(p) 与跨视角一致性规则，将"滞后"操作化为结构失效模式
3. **Minimal Evolution**: 正则化的结构更新（FitGain - λ·EditCost），避免过拟合噪声
4. **Delta-aware Guidance**: 可直接服务综述生成的主轴/辅轴写作约束
5. **Validation**: Time-slice 实验验证能预测未来综述结构更新 + 人评验证认知增量

### Where to Submit
- **ICML** (Machine Learning Systems track)
- **NeurIPS** (Datasets & Benchmarks track)
- **IJCAI** (AI & Knowledge Representation)
- **EMNLP** (NLP Applications)

### Narrative Structure
```
1. Introduction: "滞后性综述"问题 + 我们的目标（结构失效检测）
2. Related Work: Survey generation, Taxonomy learning, Scientific innovation detection
3. Problem Formulation: Multi-view baseline, Fit test, Minimal evolution (形式化定义)
4. Algorithm: Phase 1-2-3-4 pipeline (伪代码级别)
5. Experiments: Time-slice + Human eval + Ablation
6. Analysis: Failure modes, Case studies
7. Conclusion: 贡献总结 + 未来工作
```

---

## 10. FAQ (预期审稿意见)

### Q1: "这不就是让 LLM 评价创新吗？主观性太强"

**A**: 我们不让 LLM 当裁判。LLM 只做抽取（claims, definitions）与证据定位。关键判定（FIT/FORCE_FIT/UNFITTABLE、聚类、演化选择）由可复验规则和确定性算法完成。我们在 §4 明确了 LLM 职责边界，并在 ablation 中报告了多模型一致性。

### Q2: "多视角 taxonomy 如何保证正交性？"

**A**: 我们不强称正交。视角本身可能有语义重叠（例如 MODEL_ARCHITECTURE 和 TRAINING_PARADIGM 相关），这是真实综述的特点。我们通过权重与跨视角一致性规则来综合利用多视角，而不是强行融合成单一 taxonomy。

### Q3: "如何评估'创新'识别的准确性？"

**A**: 我们不评估"创新"本身，而是评估"结构预测"。Time-slice 实验中，我们预测 T₁ 时期的论文会导致哪些结构更新，然后与 T₂ 时期真实出现的综述结构对比（Branch Hit@K）。这避免了"谁定义创新"的哲学陷阱。

### Q4: "阈值太多（0.45, 0.55, 0.8, ...），如何设置？"

**A**: 初始阈值基于小验证集网格搜索（不训练，只调参）。我们在 ablation 中报告了不同阈值的敏感性分析。未来工作可用少量人工标注做更精细的校准。

### Q5: "computational cost 是否太高？"

**A**: 主要成本在 LLM 调用（抽取、定义生成）和 embedding 计算。我们通过 (1) 缓存所有中间结果，(2) Top-K candidate retrieval 减少 LLM 调用，(3) 使用现成预训练模型（无需训练）来控制成本。实验 section 会报告具体 API calls 和运行时间。

---

## 11. Next Steps

1. **Implement** 按照 Implementation Checklist 逐步实现
2. **Test** 在 3-5 个不同领域上测试（NLP, CV, RL, etc.）
3. **Evaluate** Time-slice 实验 + Human eval（至少 2 个 domains）
4. **Write** 论文初稿（按 Narrative Structure）
5. **Iterate** 根据实验结果调整算法与阈值
6. **Submit** 选择合适 venue（建议 ICML 或 EMNLP）

---

**END OF DESIGN DOCUMENT**

此文档提供了 IG-Finder 2.0 的完整实现细节、公式、算法和评估方案，可直接用于工程实现和论文写作。所有"启发式"都已具体化为可复现的规则和公式。
