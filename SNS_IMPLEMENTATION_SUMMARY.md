# SNS方法实现分析与改进总结

## 执行概要

**分析日期**: 2025-12-15  
**项目**: SNS (Self-Nonself) for Automatic Survey Generation  
**代码库路径**: `/home/user/webapp`

---

## 1. 核心发现

### ✅ 已实现的优秀部分

1. **完整的数据结构体系** (`dataclass_v2.py`)
   - 所有核心数据类已定义: `MultiViewBaseline`, `FitVector`, `StressCluster`, `EvolutionProposal`, `DeltaAwareGuidance`
   - 包含方法说明要求的所有字段
   - 支持序列化/反序列化

2. **Phase 1-4 Pipeline架构** (`engine_v2.py`, `phase1-4_*.py`)
   - 4个Phase的流程框架完整
   - Pipeline orchestration清晰
   - 中间结果保存机制完善

3. **Embeddings基础设施** (`embeddings.py`) ✅ **已存在**
   - SPECTER2, SciNCL, Sentence-BERT实现
   - Fallback机制 (TF-IDF)
   - Hybrid similarity (semantic + lexical)

4. **NLI基础设施** (`nli.py`) ✅ **已存在**
   - DeBERTa-MNLI实现
   - Batch processing支持
   - Fallback机制 (keyword-based)

5. **关键设计决策**
   - Reconstruct-then-select: ✅ 完全实现
   - Writing Mode determination: ✅ 符合方法说明
   - Multi-view Atlas: ✅ 架构正确
   - Evidence anchoring: ✅ 数据结构支持

### ⚠️ 需要修复的关键问题

#### 🔴 Critical Issue 1: Phase 2未使用真实Embeddings和NLI

**现状**: `phase2_stress_test.py` 使用placeholder实现
- `EmbeddingBasedRetriever`: keyword overlap (line 214-223)
- `FitTester._calculate_conflict()`: keyword-based (line 335-347)

**问题**: 导致Coverage和Conflict分数不准确,FIT判定可能错误

**解决方案**: 已有`embeddings.py`和`nli.py`,需要集成到Phase 2

#### 🔴 Critical Issue 2: 补视角策略未实现

**现状**: `phase1_multiview_baseline.py` 只warning,没有补救 (line 537-538)

**问题**: baseline质量不足时无法自动恢复

**解决方案**: 需要实现`CompensatoryViewInducer`类

#### 🔴 Critical Issue 3: SPLIT_NODE和RENAME_NODE未实现

**现状**: `phase3_evolution.py` 只有TODO注释 (line 389-392)

**问题**: 只能处理ADD场景,无法处理overcrowding和semantic drift

**解决方案**: 需要实现这两个operation类型

#### 🔴 Critical Issue 4: Taxonomy_v2未应用Evolution

**现状**: `guidance_pack.json`输出的taxonomy是原始版本,未包含evolution operations

**问题**: 下游系统看不到结构更新

**解决方案**: 在Phase 4前应用operations到taxonomy tree

---

## 2. 详细分析 - 按Phase

### Phase 1: Multi-view Baseline Construction

**实现度**: 85%

| 功能 | 方法要求 | 代码实现 | 状态 | 优先级 |
|-----|---------|---------|------|--------|
| 综述检索 | review/survey/tutorial关键词 | ✅ ReviewRetriever | ✅ 完成 | - |
| Taxonomy抽取 | LLM JSON schema, temperature=0 | ✅ TaxonomyViewExtractor | ✅ 完成 | - |
| Facet标签 | 受控枚举集 | ✅ FacetLabel enum | ✅ 完成 | - |
| 节点定义 | inclusion≥3, exclusion≥2, evidence_spans | ✅ NodeDefinitionBuilder | ⚠️ 无验证 | 🟡 High |
| 权重计算 | w_i ∝ Recency×Quality×Coverage | ✅ _calculate_weight() | ✅ 完成 | - |
| 质量闸门 | unique(facet)<2 检查 | ✅ _check_baseline_quality() | ✅ 完成 | - |
| **补视角策略** | **诱导T_extra** | **❌ 只warning** | **❌ 缺失** | **🔴 Critical** |

**关键代码位置**:
- `knowledge/sns/modules/phase1_multiview_baseline.py`
- Lines 515-552: quality gate有检查但无补救
- **需要添加**: `CompensatoryViewInducer` 类

### Phase 2: Multi-view Stress Test

**实现度**: 70%

| 功能 | 方法要求 | 代码实现 | 状态 | 优先级 |
|-----|---------|---------|------|--------|
| Claim抽取 | problem, core_idea, mechanism, training, evaluation, novelty_bullets=3 | ✅ PaperClaimExtractor | ✅ 完成 | - |
| 候选召回 | Embedding相似度 + Top-K | ⚠️ keyword overlap | ❌ Placeholder | 🔴 Critical |
| Coverage计算 | 0.7×cos(emb) + 0.3×Jaccard | ⚠️ keyword-based semantic | ❌ Placeholder | 🔴 Critical |
| **Conflict计算** | **max P_NLI(contradiction)** | **❌ keyword overlap** | **❌ Placeholder** | **🔴 Critical** |
| Residual计算 | 1 - max cos(novelty, leaf) | ⚠️ keyword-based | ❌ Placeholder | 🔴 Critical |
| 标签判定 | 阈值规则 (0.45, 0.55, 0.45) | ✅ _determine_label() | ✅ 完成 | - |
| Evidence提取 | lost_novelty, conflict_evidence | ✅ 数据结构 | ✅ 完成 | - |

**关键代码位置**:
- `knowledge/sns/modules/phase2_stress_test.py`
- Lines 146-224: `EmbeddingBasedRetriever` - 需要集成真实embeddings
- Lines 309-347: `_calculate_conflict()` - 需要集成NLI
- **已有基础设施**: `knowledge/sns/embeddings.py`, `knowledge/sns/nli.py`

**需要做的修改**:
1. `EmbeddingBasedRetriever.__init__()`: 使用`create_embedding_model()`
2. `EmbeddingBasedRetriever._compute_similarity()`: 使用`embedder.similarity()`
3. `FitTester.__init__()`: 添加`nli_model`参数
4. `FitTester._calculate_conflict()`: 使用`nli_model.compute_max_conflict_score()`

### Phase 3: Stress Clustering & Evolution

**实现度**: 75%

| 功能 | 方法要求 | 代码实现 | 状态 | 优先级 |
|-----|---------|---------|------|--------|
| 压力筛选 | stress_score > 0.3 | ✅ StressClusterer | ✅ 完成 | - |
| 失败签名 | facet + leaf_path + lost_novelty | ✅ _construct_failure_signature() | ✅ 完成 | - |
| HDBSCAN聚类 | 无需指定K | ✅ _cluster_signatures() | ✅ 完成 | - |
| ClusterType判定 | STRONG_SHIFT, FACET_DEPENDENT, STABLE | ✅ _determine_cluster_type() | ✅ 完成 | - |
| ADD_NODE操作 | cost=1.0 | ✅ _propose_add_node() | ✅ 完成 | - |
| **SPLIT_NODE操作** | **cost=2.0** | **❌ TODO注释** | **❌ 缺失** | **🔴 Critical** |
| **RENAME_NODE操作** | **cost=0.5** | **❌ TODO注释** | **❌ 缺失** | **🔴 Critical** |
| Objective选择 | FitGain - λ·EditCost | ✅ Greedy selection | ✅ 完成 | - |
| **Reconstruction scores** | **对所有视角计算** | **✅ compute_all_views_reconstruction()** | **✅ 完成** | - |

**关键代码位置**:
- `knowledge/sns/modules/phase3_evolution.py`
- Lines 389-392: SPLIT/RENAME的TODO注释
- Lines 419-512: **重要**: `compute_all_views_reconstruction()` 已实现 (支持新设计)

**需要做的修改**:
1. 添加`_propose_split_node()` 方法到`EvolutionPlanner`
2. 添加`_propose_rename_node()` 方法到`EvolutionPlanner`
3. 在`plan_evolution()`中调用这两个方法

### Phase 4: Delta-aware Guidance

**实现度**: 90%

| 功能 | 方法要求 | 代码实现 | 状态 | 优先级 |
|-----|---------|---------|------|--------|
| Reconstruct-then-select | 先重构再选择主轴 | ✅ select_main_axis_with_mode() | ✅ 完成 | - |
| Writing Mode判定 | EditCost>3.0 or FitGain>10.0 → DELTA_FIRST | ✅ 阈值规则 | ✅ 完成 | - |
| Aux axis选择 | discriminativeness (variance) | ✅ select_aux_axis() | ✅ 完成 | - |
| Writing rules生成 | Mode-specific do/dont | ✅ _generate_writing_rules() | ✅ 完成 | - |
| Outline生成 | Cross-organize with main/aux | ✅ _generate_outline() | ✅ 完成 | - |
| **Taxonomy_v2** | **应用evolution到tree** | **❌ 使用原始tree** | **❌ 缺失** | **🔴 Critical** |
| Must-answer questions | 具体问题 | ⚠️ 通用问题 | ⚠️ 质量低 | 🟡 High |
| Evidence cards | 精确quotes | ⚠️ 只有abstract | ⚠️ 质量低 | 🟡 High |
| `guidance_pack.json` | 机器可读 | ✅ _save_guidance_pack() | ✅ 完成 | - |
| `audit_report.md` | 人类可读 | ✅ _generate_markdown_report() | ✅ 完成 | - |

**关键代码位置**:
- `knowledge/sns/modules/phase4_guidance.py`
- `knowledge/sns/engine_v2.py` lines 442-606 (输出生成)

**需要做的修改**:
1. 添加`apply_evolution_to_taxonomy()` 函数
2. 在`Phase4Pipeline.run()`开始时应用evolution
3. 增强`_generate_must_answer_questions()` 生成evolution-specific问题
4. 增强`_create_subsection()` 从fit_reports提取精确evidence

---

## 3. 改进实施计划

### 优先级1: 集成真实Embeddings和NLI到Phase 2 (1-2 days)

**目标**: 提升FIT判定准确性

**文件**: `knowledge/sns/modules/phase2_stress_test.py`

**修改点**:

#### 修改1: EmbeddingBasedRetriever集成真实embeddings

```python
# 在文件顶部import
from ..embeddings import create_embedding_model

class EmbeddingBasedRetriever:
    def __init__(self, embedding_model_name: str = "specter2"):
        # 使用真实embedding模型
        self.embedder = create_embedding_model(
            model_type=embedding_model_name,
            device="cpu"  # 或 "cuda" if available
        )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        # 使用真实embedding计算
        emb1 = self.embedder.encode([text1])[0]
        emb2 = self.embedder.encode([text2])[0]
        return self.embedder.similarity(emb1, emb2)
```

#### 修改2: FitTester集成NLI

```python
# 在文件顶部import
from ..nli import create_nli_model, compute_max_conflict_score

class FitTester:
    def __init__(self, retriever: EmbeddingBasedRetriever, nli_model=None):
        self.retriever = retriever
        # 初始化NLI模型 (可选,如果不提供则fallback)
        if nli_model is None:
            try:
                self.nli_model = create_nli_model(model_type="deberta", device="cpu")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}. Using fallback.")
                self.nli_model = create_nli_model(model_type="fallback")
        else:
            self.nli_model = nli_model
    
    def _calculate_conflict(self, claims: PaperClaims, node_def: NodeDefinition) -> float:
        # 使用真实NLI计算
        all_claims_text = " ".join([
            c.text for claim_list in [claims.core_idea, claims.mechanism, claims.novelty_bullets]
            for c in claim_list
        ])
        
        return compute_max_conflict_score(
            claim_text=all_claims_text,
            node_definition_text=node_def.definition,
            exclusion_criteria=node_def.exclusion_criteria + node_def.boundary_statements,
            nli_model=self.nli_model
        )
```

#### 修改3: 更新Phase2Pipeline初始化

```python
class Phase2Pipeline:
    def __init__(self, lm, embedding_model: str = "specter2", nli_model_type: str = "deberta"):
        self.claim_extractor = PaperClaimExtractor(lm)
        self.retriever = EmbeddingBasedRetriever(embedding_model)  # 真实模型
        
        # 初始化NLI
        try:
            nli_model = create_nli_model(model_type=nli_model_type, device="cpu")
        except Exception:
            logger.warning("Using fallback NLI")
            nli_model = None
        
        self.stress_tester = MultiViewStressTester(self.retriever, nli_model)
```

**预期效果**:
- Coverage分数准确 (基于SPECTER2)
- Conflict分数准确 (基于DeBERTa-MNLI)
- FIT/FORCE_FIT/UNFITTABLE判定更可靠

---

### 优先级2: 实现补视角策略 (2-3 days)

**目标**: 当baseline质量不足时自动补救

**文件**: 新建 `knowledge/sns/modules/compensatory_view.py`

**核心逻辑**:

```python
from collections import Counter
import numpy as np
from typing import List, Optional
import logging

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from ..dataclass_v2 import (
    MultiViewBaseline, TaxonomyView, TaxonomyTree, TaxonomyTreeNode,
    NodeDefinition, EvidenceSpan, FacetLabel
)
from ...interface import Information

logger = logging.getLogger(__name__)


class CompensatoryViewInducer:
    """
    当baseline质量不足时,从前沿论文诱导补视角。
    
    策略:
    1. 对前沿论文聚类 (基于title+abstract embeddings)
    2. 为每个簇生成主题标签 (使用LLM)
    3. 构建induced taxonomy tree
    4. 分配新的facet_label (避免与现有facet冲突)
    """
    
    def __init__(self, embedder, lm, min_cluster_size: int = 3):
        self.embedder = embedder
        self.lm = lm
        self.min_cluster_size = min_cluster_size
    
    def should_induce(self, baseline: MultiViewBaseline, min_facet_count: int = 2) -> bool:
        """检查是否需要补视角"""
        facet_counts = Counter([v.facet_label for v in baseline.views])
        
        # 条件1: unique facets < min_facet_count
        if len(facet_counts) < min_facet_count:
            return True
        
        # 条件2: dominant facet > 60%
        if baseline.views:
            for facet, count in facet_counts.items():
                if count / len(baseline.views) > 0.6:
                    return True
        
        return False
    
    def induce_view(
        self,
        baseline: MultiViewBaseline,
        papers: List[Information],
        topic: str
    ) -> Optional[TaxonomyView]:
        """诱导一个补视角"""
        
        if not papers:
            logger.warning("No papers provided for compensatory view induction")
            return None
        
        logger.info(f"Inducing compensatory view from {len(papers)} papers...")
        
        # 1. 对论文聚类
        clusters = self._cluster_papers(papers)
        
        if not clusters:
            logger.warning("No clusters found, cannot induce view")
            return None
        
        # 2. 为每个簇生成标签
        cluster_labels = self._generate_cluster_labels(clusters)
        
        # 3. 构建induced tree
        induced_tree = self._build_induced_tree(cluster_labels)
        
        # 4. 选择unique facet
        used_facets = {v.facet_label for v in baseline.views}
        new_facet = self._select_unique_facet(used_facets)
        
        # 5. 创建view
        view_id = f"T_induced_{len(baseline.views) + 1}"
        
        compensatory_view = TaxonomyView(
            view_id=view_id,
            review_id="INDUCED_FROM_PAPERS",
            review_title=f"Induced View: {new_facet.value} (from {len(papers)} papers)",
            facet_label=new_facet,
            facet_rationale=(
                f"Compensatory view induced from paper clustering to ensure baseline quality. "
                f"Represents emerging {new_facet.value.lower()} dimension."
            ),
            tree=induced_tree,
            node_definitions={},  # Will be populated
            weight=0.5,  # Slightly lower than normal reviews
            evidence=[]
        )
        
        # 6. 构建节点定义
        compensatory_view.node_definitions = self._build_node_definitions(
            induced_tree, clusters, cluster_labels
        )
        
        logger.info(f"Successfully induced view {view_id} with {len(cluster_labels)} categories")
        
        return compensatory_view
    
    def _cluster_papers(self, papers: List[Information]) -> List[List[Information]]:
        """使用HDBSCAN对论文聚类"""
        # 提取text
        texts = [f"{p.title} {p.description}" for p in papers]
        
        # 生成embeddings
        embeddings = self.embedder.encode(texts)
        
        if not HDBSCAN_AVAILABLE:
            # Fallback: simple k-means or just split by keywords
            logger.warning("HDBSCAN not available, using simple fallback clustering")
            # 简单分为3组
            n_per_cluster = len(papers) // 3 + 1
            return [papers[i:i+n_per_cluster] for i in range(0, len(papers), n_per_cluster)]
        
        # HDBSCAN聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # 分组
        clusters = {}
        for paper, label in zip(papers, labels):
            if label != -1:  # 忽略噪声
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(paper)
        
        return list(clusters.values())
    
    def _generate_cluster_labels(self, clusters: List[List[Information]]) -> List[str]:
        """使用LLM为每个簇生成标签"""
        labels = []
        
        for i, cluster in enumerate(clusters):
            # 构建cluster摘要
            titles = [p.title for p in cluster[:5]]  # 前5篇
            summary = f"Cluster {i+1} papers:\n" + "\n".join(f"- {t}" for t in titles)
            
            # 调用LLM生成标签 (简化:使用关键词提取)
            # 实际应该用LLM: "Generate a short category name for these papers"
            # 这里用简化版本
            from collections import Counter
            words = []
            for p in cluster:
                words.extend(p.title.lower().split())
            
            common_words = Counter(words).most_common(3)
            label = "_".join([w for w, _ in common_words if len(w) > 3])
            
            labels.append(label or f"Category_{i+1}")
        
        return labels
    
    def _build_induced_tree(self, cluster_labels: List[str]) -> TaxonomyTree:
        """构建induced taxonomy tree (flat: root + leaves)"""
        # 创建root
        root = TaxonomyTreeNode(
            name="Induced_Root",
            path="Induced_Root",
            parent=None,
            children=[],
            is_leaf=False
        )
        
        tree = TaxonomyTree(root=root)
        
        # 添加leaf节点
        for label in cluster_labels:
            leaf_path = f"Induced_Root/{label}"
            leaf_node = TaxonomyTreeNode(
                name=label,
                path=leaf_path,
                parent="Induced_Root",
                children=[],
                is_leaf=True
            )
            tree.add_node(leaf_node)
        
        return tree
    
    def _select_unique_facet(self, used_facets: set) -> FacetLabel:
        """选择未使用的facet"""
        all_facets = list(FacetLabel)
        
        for facet in all_facets:
            if facet not in used_facets and facet != FacetLabel.OTHER:
                return facet
        
        # 如果都用了,返回OTHER或重用
        return FacetLabel.OTHER
    
    def _build_node_definitions(
        self,
        tree: TaxonomyTree,
        clusters: List[List[Information]],
        labels: List[str]
    ) -> dict:
        """为induced tree的每个节点构建定义"""
        node_defs = {}
        
        for label, cluster in zip(labels, clusters):
            node_path = f"Induced_Root/{label}"
            
            # 提取关键词
            keywords = self._extract_keywords(cluster)
            
            # 构建定义
            definition = f"Papers related to {label.replace('_', ' ')}"
            
            # 生成inclusion/exclusion criteria (simplified)
            inclusion = [
                f"Papers about {kw}" for kw in keywords[:3]
            ]
            exclusion = ["Papers outside this theme"]
            
            # Evidence spans (从前3篇论文)
            evidence = []
            for p in cluster[:3]:
                evidence.append(EvidenceSpan(
                    claim=f"Representative paper: {p.title}",
                    page=0,
                    section="Induced",
                    char_start=0,
                    char_end=len(p.description) if p.description else 0,
                    quote=p.description[:200] if p.description else ""
                ))
            
            node_def = NodeDefinition(
                node_path=node_path,
                definition=definition,
                inclusion_criteria=inclusion,
                exclusion_criteria=exclusion,
                canonical_keywords=keywords,
                boundary_statements=[],
                evidence_spans=evidence
            )
            
            node_defs[node_path] = node_def
        
        return node_defs
    
    def _extract_keywords(self, cluster: List[Information]) -> List[str]:
        """从cluster提取关键词"""
        from collections import Counter
        
        words = []
        for p in cluster:
            words.extend(p.title.lower().split())
        
        # 过滤停用词 (简化版)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        common = Counter(words).most_common(10)
        
        return [w for w, _ in common]
```

**集成到Phase 1**:

```python
# 在 phase1_multiview_baseline.py

def _check_baseline_quality(self, baseline: MultiViewBaseline) -> None:
    """... 现有代码 ..."""
    
    # 新增: 补视角触发
    if self.compensatory_inducer:
        if self.compensatory_inducer.should_induce(baseline):
            logger.warning("Baseline quality insufficient, triggering compensatory view induction")
            
            # 需要前沿论文作为输入
            # (这需要在Phase1Pipeline中提前检索一些前沿论文)
            # compensatory_view = self.compensatory_inducer.induce_view(
            #     baseline, self.cached_papers, self.topic
            # )
            # if compensatory_view:
            #     baseline.views.append(compensatory_view)
            #     # 重新归一化权重
            #     baseline.__post_init__()
```

**预期效果**:
- baseline始终满足质量要求 (≥2 unique facets, no dominant >60%)
- 当综述数据不足时自动补救

---

### 优先级3: 实现SPLIT_NODE和RENAME_NODE (2-3 days)

**目标**: 完整支持3种evolution操作

**文件**: `knowledge/sns/modules/phase3_evolution.py`

**添加方法** (详见分析文档Section 4.3):
1. `_propose_split_node()` - 处理overcrowded节点
2. `_propose_rename_node()` - 处理semantic drift

**集成到`plan_evolution()`**:

```python
def plan_evolution(...):
    # ... 现有ADD_NODE逻辑 ...
    
    # 新增: Try SPLIT_NODE
    split_op = self._propose_split_node(cluster, view, fit_vectors)
    if split_op:
        candidates.append(split_op)
    
    # 新增: Try RENAME_NODE
    rename_op = self._propose_rename_node(cluster, view, fit_vectors)
    if rename_op:
        candidates.append(rename_op)
    
    # ... 现有selection逻辑 ...
```

**预期效果**:
- 可以处理节点过度拥挤场景
- 可以处理语义漂移场景
- Evolution proposal更全面

---

### 优先级4: 应用Evolution到Taxonomy生成v2 (1-2 days)

**目标**: 下游系统看到演化后的taxonomy

**文件**: 新建 `knowledge/sns/modules/taxonomy_evolution_applier.py`

**核心函数**:

```python
import copy
from ..dataclass_v2 import (
    TaxonomyView, EvolutionOperation, AddNodeOperation,
    SplitNodeOperation, RenameNodeOperation,
    TaxonomyTreeNode, NodeDefinition
)


def apply_evolution_to_taxonomy(
    view: TaxonomyView,
    operations: List[EvolutionOperation]
) -> TaxonomyView:
    """
    将accepted evolution operations应用到taxonomy tree,生成taxonomy_v2。
    """
    view_v2 = copy.deepcopy(view)
    
    for op in operations:
        if op.view_id != view.view_id:
            continue  # 只应用到对应view
        
        if isinstance(op, AddNodeOperation):
            _apply_add_node(view_v2, op)
        
        elif isinstance(op, SplitNodeOperation):
            _apply_split_node(view_v2, op)
        
        elif isinstance(op, RenameNodeOperation):
            _apply_rename_node(view_v2, op)
    
    return view_v2
```

**集成到Phase 4**:

```python
# 在 phase4_guidance.py

class Phase4Pipeline:
    def run(self, ...):
        # ... 现有代码 ...
        
        # **新增: 在axis selection之前,应用evolution**
        from ..modules.taxonomy_evolution_applier import apply_evolution_to_taxonomy
        
        # 为所有视角应用evolution
        baseline_v2 = copy.deepcopy(baseline)
        for i, view in enumerate(baseline_v2.views):
            view_operations = [op for op in evolution_proposal.operations if op.view_id == view.view_id]
            if view_operations:
                baseline_v2.views[i] = apply_evolution_to_taxonomy(view, view_operations)
        
        # 使用evolved baseline进行axis selection
        main_axis, main_axis_mode = self.axis_selector.select_main_axis_with_mode(
            reconstruction_scores,
            baseline_v2  # 使用v2
        )
        
        # ...
```

**预期效果**:
- `guidance_pack.json` 中的taxonomy是演化后的版本
- 下游系统可以看到ADD_NODE/SPLIT_NODE/RENAME_NODE的结果

---

## 4. 修改文件清单

| 文件 | 修改类型 | 优先级 | 预估工作量 |
|-----|---------|--------|-----------|
| `phase2_stress_test.py` | 修改 (集成embeddings+NLI) | 🔴 Critical | 4-6 hours |
| `compensatory_view.py` | 新建 | 🔴 Critical | 1-2 days |
| `phase1_multiview_baseline.py` | 修改 (集成compensatory) | 🔴 Critical | 2-4 hours |
| `phase3_evolution.py` | 添加SPLIT/RENAME方法 | 🔴 Critical | 1-2 days |
| `taxonomy_evolution_applier.py` | 新建 | 🔴 Critical | 1-2 days |
| `phase4_guidance.py` | 修改 (应用evolution, 增强questions) | 🟡 High | 4-8 hours |
| `engine_v2.py` | 小修改 (pipeline参数) | 🟢 Low | 1-2 hours |

**总预估**: 5-7个工作日

---

## 5. 测试计划

### 单元测试

1. **Embeddings集成测试**:
   - 测试`EmbeddingBasedRetriever`使用真实SPECTER2
   - 验证相似度分数范围 [0, 1]

2. **NLI集成测试**:
   - 测试`FitTester`使用真实DeBERTa-MNLI
   - 验证冲突检测准确性

3. **补视角测试**:
   - 测试`CompensatoryViewInducer.should_induce()`
   - 测试诱导视角的tree结构

4. **Evolution应用测试**:
   - 测试`apply_evolution_to_taxonomy()`
   - 验证tree结构更新正确

### 集成测试

1. **End-to-end Pipeline**:
   - 运行完整Phase 1-4
   - 验证输出文件格式
   - 检查`guidance_pack.json` schema

2. **质量验证**:
   - 检查FIT/FORCE_FIT/UNFITTABLE分布合理性
   - 验证evolution operations有证据支持
   - 验证writing_rules符合mode

---

## 6. 部署建议

### Phase 1: 内部测试 (Week 1-2)

1. 完成优先级1-2修改
2. 在小规模数据上测试
3. 调整阈值和参数

### Phase 2: Beta测试 (Week 3-4)

1. 完成优先级3-4修改
2. 在真实数据上测试
3. 收集用户反馈

### Phase 3: 生产部署 (Week 5+)

1. 性能优化 (embedding缓存, batch processing)
2. 文档补充
3. 发布v2.0

---

## 7. 最终评估

### 当前实现完整度

| Phase | 完整度 | 关键缺失 |
|-------|--------|---------|
| Phase 1 | 85% | 补视角策略 |
| Phase 2 | 70% | 真实Embeddings+NLI |
| Phase 3 | 75% | SPLIT/RENAME操作 |
| Phase 4 | 90% | Taxonomy_v2应用 |
| **总体** | **80%** | **4个Critical issues** |

### 方法说明对齐度

- **设计原则**: ✅ 100% 对齐
- **数据结构**: ✅ 100% 对齐
- **Pipeline流程**: ✅ 95% 对齐
- **功能完整性**: ⚠️ 80% 完整

### 代码质量评价

- **架构设计**: ⭐⭐⭐⭐⭐ 优秀
- **可扩展性**: ⭐⭐⭐⭐⭐ 优秀
- **文档质量**: ⭐⭐⭐⭐☆ 良好
- **测试覆盖**: ⭐⭐⭐☆☆ 中等
- **生产就绪**: ⭐⭐⭐☆☆ 需要补齐Critical issues

---

## 8. 结论与建议

### 结论

1. **已有坚实基础**: 代码架构优秀,核心framework完整
2. **关键组件已实现**: Embeddings和NLI模块已存在,只需集成
3. **主要是集成工作**: 大部分缺失功能是"连接现有组件"而非"从零实现"
4. **可快速补齐**: 预计5-7个工作日可完成所有Critical issues

### 建议优先顺序

1. **Week 1**: 集成真实Embeddings+NLI到Phase 2 (最大ROI)
2. **Week 2**: 实现补视角策略 (保证baseline质量)
3. **Week 3**: 实现SPLIT/RENAME + 应用Evolution
4. **Week 4**: 质量提升 (questions, evidence cards, 测试)

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|-----|-----|---------|
| Embedding模型加载失败 | 中 | 高 | Fallback机制已实现 |
| NLI模型太慢 | 中 | 中 | Batch processing, 缓存 |
| 补视角质量差 | 低 | 中 | LLM生成label, 人工review |
| Evolution应用bug | 低 | 高 | 充分测试, tree validation |

---

**文档版本**: 1.0  
**最后更新**: 2025-12-15  
**作者**: Claude (AI Code Assistant)
