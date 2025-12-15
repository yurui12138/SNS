# SNS方法实现对比分析报告

## 执行概要

本报告详细分析了SNS (Self-Nonself) 方法说明文档与当前代码实现的对齐情况,识别缺失功能并提出改进方案。

**日期**: 2025-12-15  
**分析范围**: Phase 1-4 完整流程

---

## 1. 总体对齐情况

### ✅ 已实现的核心功能

| 方法组件 | 实现状态 | 文件位置 |
|---------|---------|---------|
| Phase 1: Multi-view Baseline | ✅ 基本实现 | `phase1_multiview_baseline.py` |
| Phase 2: Stress Testing | ✅ 基本实现 | `phase2_stress_test.py` |
| Phase 3: Evolution Planning | ✅ 基本实现 | `phase3_evolution.py` |
| Phase 4: Guidance Generation | ✅ 基本实现 | `phase4_guidance.py` |
| 核心数据结构 | ✅ 完整实现 | `dataclass_v2.py` |

### ⚠️ 需要改进的关键功能

## 2. Phase 1: Multi-view Baseline Construction - 详细分析

### 2.1 方法说明要求

**目标**: 将目标领域既有综述形成的"自我（Self）认知结构"显式建模为多视角组织基线

**关键步骤**:
1. 综述检索与筛选 (按review/survey/tutorial关键词)
2. 抽取taxonomy结构 (从目录/章节/引言)
3. 视角标签识别 (从受控枚举集选择facet)
4. 节点定义构建 (definition, inclusion_criteria ≥3, exclusion_criteria ≥2, keywords, evidence_spans)
5. 多视角闸门与补视角 (unique(facet) < 2时触发补视角策略)
6. 权重计算: w_i ∝ Recency × Quality × Coverage

### 2.2 当前代码实现

#### ✅ 已实现功能

1. **ReviewRetriever** (lines 32-189):
   - ✅ 基于关键词检索综述 (`survey`, `review`, `overview` 等)
   - ✅ 启发式筛选 (title关键词, 摘要长度 >120 words)
   - ✅ 质量评分 `0.4*recency + 0.4*citation + 0.2*relevance`
   - ✅ 年份提取 (从snippets)

2. **TaxonomyViewExtractor** (lines 191-325):
   - ✅ 使用LLM提取taxonomy tree (JSON schema, temperature=0)
   - ✅ 解析facet_label (FacetLabel枚举)
   - ✅ 递归构建TaxonomyTree

3. **NodeDefinitionBuilder** (lines 327-458):
   - ✅ 为每个节点构建定义
   - ✅ 包含 definition, inclusion_criteria, exclusion_criteria, canonical_keywords, boundary_statements
   - ✅ 绑定evidence_spans到源文本

4. **MultiViewBaselineBuilder** (lines 460-595):
   - ✅ 权重计算 `recency * quality * coverage`
   - ✅ 权重归一化 (在MultiViewBaseline.__post_init__)
   - ✅ 质量闸门检查:
     - ✅ unique facets < 2 时发出警告
     - ✅ dominant facet > 60% 时发出警告

#### ❌ 缺失功能

**CRITICAL**: **补视角策略未实现**

方法说明要求:
> 若 `unique(facet) < 2` 或 facet 分布退化,触发补视角策略:  
> 从前沿论文集合(或综述语料)诱导构建 `T_extra`(以聚类得到的主题簇命名形成树),使图集具备至少两类视角。

**当前代码**:
```python
# phase1_multiview_baseline.py, lines 515-552
def _check_baseline_quality(self, baseline: MultiViewBaseline) -> None:
    # ...
    if num_unique_facets < 2:
        logger.warning("⚠️ QUALITY GATE WARNING: Only {num_unique_facets} unique facets")
        logger.warning("   Consider retrieving more diverse reviews or inducing additional views")
    # ❌ 没有实际的补视角实现！只是警告
```

**影响**: 当综述数据不足或质量低时,系统无法自动补救,导致baseline质量不达标。

#### ⚠️ 需要增强的功能

1. **节点定义质量**:
   - 方法要求: `inclusion_criteria ≥ 3`, `exclusion_criteria ≥ 2`
   - 当前实现: 没有硬性验证,可能生成不足3/2的标准
   - **建议**: 添加schema验证和LLM prompt约束

2. **Evidence Spans质量**:
   - 方法强调: "所有节点定义、适配原因与新增结构均绑定原文spans"
   - 当前实现: 依赖LLM生成,没有验证char_start/char_end准确性
   - **建议**: 添加evidence完整性检查

### 2.3 改进建议

#### **改进1: 实现补视角策略 (CompensatoryViewInducer)**

```python
class CompensatoryViewInducer:
    """
    当baseline质量不足时,从前沿论文诱导补视角。
    
    策略:
    1. 使用HDBSCAN对前沿论文聚类 (基于title+abstract embeddings)
    2. 为每个簇生成主题标签 (使用LLM)
    3. 构建induced taxonomy tree (以簇标签为叶节点)
    4. 分配新的facet_label (避免与现有facet冲突)
    """
    
    def induce_compensatory_view(
        self,
        baseline: MultiViewBaseline,
        papers: List[Information],  # 前沿论文
        min_facet_count: int = 2
    ) -> Optional[TaxonomyView]:
        """诱导补视角以满足多样性要求"""
        
        # 检查是否需要补视角
        facet_counts = Counter([v.facet_label for v in baseline.views])
        if len(facet_counts) >= min_facet_count:
            return None  # 质量达标,无需补救
        
        # 1. 对论文聚类
        clusters = self._cluster_papers(papers)
        
        # 2. 为每个簇生成标签
        cluster_labels = self._generate_cluster_labels(clusters)
        
        # 3. 构建induced tree
        induced_tree = self._build_induced_tree(cluster_labels)
        
        # 4. 创建新视角
        compensatory_view = TaxonomyView(
            view_id=f"T_induced_{len(baseline.views)+1}",
            review_id="INDUCED_FROM_PAPERS",
            review_title="Induced View from Research Papers",
            facet_label=self._select_unique_facet(baseline),  # 选择未使用的facet
            facet_rationale="Compensatory view induced from paper clustering",
            tree=induced_tree,
            node_definitions={},
            weight=0.5,  # 略低于正常综述
            evidence=[]
        )
        
        return compensatory_view
```

**集成位置**: `Phase1Pipeline.run()` 的Step 4之后

#### **改进2: 增强节点定义验证**

```python
def _validate_node_definition(self, node_def: NodeDefinition) -> bool:
    """验证节点定义质量"""
    errors = []
    
    # 规则1: inclusion_criteria ≥ 3
    if len(node_def.inclusion_criteria) < 3:
        errors.append(f"inclusion_criteria count={len(node_def.inclusion_criteria)} < 3")
    
    # 规则2: exclusion_criteria ≥ 2
    if len(node_def.exclusion_criteria) < 2:
        errors.append(f"exclusion_criteria count={len(node_def.exclusion_criteria)} < 2")
    
    # 规则3: 必须有evidence_spans
    if not node_def.evidence_spans:
        errors.append("Missing evidence_spans")
    
    if errors:
        logger.warning(f"Node {node_def.node_path} validation failed: {errors}")
        return False
    
    return True
```

---

## 3. Phase 2: Multi-view Stress Test - 详细分析

### 3.1 方法说明要求

**目标**: 将最新非综述论文视为潜在"非我（Nonself）"输入,通过跨视角适配测试识别结构压力

**关键步骤**:
1. **论文主张抽取**: 抽取problem, core_idea, mechanism, training, evaluation, novelty_bullets (恰好3条)
2. **候选叶节点召回**: Embedding相似度 + 关键词匹配, Top-K候选
3. **适配打分 (Tri-factor)**:
   - **Coverage**: `0.7 × cos(emb) + 0.3 × Jaccard(keywords)`
   - **Conflict**: `max_{h∈Exclusion} P_NLI(contradiction|claim,h)` - 需要NLI模型
   - **Residual**: `1 - max_{b∈NoveltyBullets} cos(emb(b), leaf_vector)`
4. **标签判定**:
   ```
   if Coverage < 0.45 or Conflict > 0.55: UNFITTABLE
   elif Residual > 0.45: FORCE_FIT
   else: FIT
   ```
5. **证据输出**: lost_novelty, conflict_evidence, spans

### 3.2 当前代码实现

#### ✅ 已实现功能

1. **PaperClaimExtractor** (lines 36-138):
   - ✅ 使用LLM提取结构化claims (JSON schema)
   - ✅ 强制novelty_bullets = 3 (padding/trimming)
   - ✅ 绑定evidence到PaperClaim

2. **EmbeddingBasedRetriever** (lines 140-224):
   - ✅ Top-K候选召回 (基于相似度排序)
   - ⚠️ **Placeholder实现**: 使用简单keyword overlap,不是真实embeddings

3. **FitTester** (lines 226-435):
   - ✅ Coverage计算 `0.7*semantic + 0.3*lexical`
   - ⚠️ **Placeholder Conflict**: 使用keyword overlap,不是NLI模型
   - ✅ Residual计算 `1 - max(novelty_sim)`
   - ✅ 阈值判定规则 (0.45, 0.55, 0.45)
   - ✅ 提取lost_novelty和conflict_evidence

4. **MultiViewStressTester** (lines 437-552):
   - ✅ 对所有视角测试
   - ✅ 加权stress_score和unfittable_score

#### ❌ 缺失功能

**CRITICAL 1**: **NLI冲突检测未实现**

方法说明明确要求:
> **Conflict**: `max_{h ∈ Exclusion} P_NLI(contradiction | claim, h)`
> - 使用DeBERTa-MNLI模型检测entailment

**当前代码**:
```python
# phase2_stress_test.py, lines 335-347
def _keyword_conflict_score(self, claim: str, exclusion: str) -> float:
    """
    Placeholder conflict detection based on keywords.
    
    In production: Replace with NLI model prediction.  # ❌ 说明这是TODO
    """
    # 简单keyword overlap
```

**影响**: Conflict分数不准确,导致FIT/UNFITTABLE判定可能错误。

**CRITICAL 2**: **真实Embedding模型未集成**

**当前代码**:
```python
# phase2_stress_test.py, lines 146-149
def __init__(self, embedding_model_name: str = "dummy"):
    self.model_name = embedding_model_name
    # In production, load actual model:
    # from sentence_transformers import SentenceTransformer
    # self.model = SentenceTransformer('allenai/specter2')  # ❌ 未实际加载
```

**影响**: 
- Coverage中的semantic分数不准确
- Residual中的novelty相似度不准确
- 候选召回质量差

#### ⚠️ 需要增强的功能

1. **Evidence Span绑定质量**:
   - lost_novelty和conflict_evidence都有evidence字段
   - 但当前实现可能丢失精确的char_start/end
   
2. **阈值可校准**:
   - 方法说明: "用小规模标注或敏感性分析确定阈值"
   - 当前: 硬编码 (0.45, 0.55, 0.45)
   - **建议**: 添加threshold配置接口

### 3.3 改进建议

#### **改进1: 集成NLI冲突检测**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NLIConflictDetector:
    """
    使用DeBERTa-MNLI检测claim与exclusion的冲突。
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def detect_conflict(self, claim: str, exclusion: str) -> float:
        """
        计算P_NLI(contradiction | claim, exclusion)
        
        Returns:
            contradiction概率 [0.0, 1.0]
        """
        inputs = self.tokenizer(
            claim, 
            exclusion,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # DeBERTa-MNLI输出: [entailment, neutral, contradiction]
        contradiction_prob = probs[0, 2].item()
        
        return contradiction_prob
```

**集成到FitTester**:
```python
class FitTester:
    def __init__(self, retriever, nli_detector: Optional[NLIConflictDetector] = None):
        self.retriever = retriever
        self.nli_detector = nli_detector  # 新增
    
    def _calculate_conflict(self, claims: PaperClaims, node_def: NodeDefinition) -> float:
        if self.nli_detector:
            # 使用真实NLI模型
            max_conflict = 0.0
            for claim in all_claims:
                for exclusion in all_exclusions:
                    conflict_score = self.nli_detector.detect_conflict(claim, exclusion)
                    max_conflict = max(max_conflict, conflict_score)
            return max_conflict
        else:
            # Fallback to keyword-based
            return self._keyword_conflict_score(...)
```

#### **改进2: 集成SPECTER2 Embeddings**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class ScientificEmbedder:
    """
    使用SPECTER2或SciNCL生成科学论文embeddings。
    """
    
    def __init__(self, model_name: str = "allenai/specter2"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}  # 缓存embeddings
    
    def encode(self, text: str) -> np.ndarray:
        """生成embedding"""
        if text in self.cache:
            return self.cache[text]
        
        emb = self.model.encode(text, show_progress_bar=False)
        self.cache[text] = emb
        return emb
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """计算余弦相似度"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
```

**更新EmbeddingBasedRetriever**:
```python
class EmbeddingBasedRetriever:
    def __init__(self, embedding_model_name: str = "allenai/specter2"):
        self.embedder = ScientificEmbedder(embedding_model_name)
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        return self.embedder.cosine_similarity(text1, text2)
```

---

## 4. Phase 3: Stress Clustering & Evolution - 详细分析

### 4.1 方法说明要求

**目标**: 通过跨视角适配测试、结构压力聚合与最小必要结构更新,生成可审计的认知增量

**关键步骤**:
1. **压力论文筛选**: stress_score > threshold (如0.3)
2. **失败签名构建**: facet + best_leaf_path + lost_novelty + 关键术语
3. **压力簇聚类**: 使用HDBSCAN (无需指定K)
4. **跨视角一致性判定**: 计算 U(C), S(C), ρ_i(C)
   - `STRONG_SHIFT`: U(C) > 0.55 且 ≥2个高权重视角失败
   - `FACET_DEPENDENT`: 混合失败/适配
   - `STABLE`: 多数高权重视角适配
5. **候选结构更新生成**: ADD/SPLIT/RENAME,绑定证据卡
6. **最小必要更新选择**: `Objective = FitGain - λ·EditCost` (λ=0.8)
7. **主轴/辅轴组织方案确定 (Evolution-first)**:
   - **先重构再选择**: 对候选视角执行最小必要重构得到T_i'
   - 计算 `Score_i = α·FitGain + β·Stress + γ·Coverage − λ·EditCost`
   - 选择两种模式之一:
     - **Delta-first**: 重构后的高压力视角作为主轴
     - **Anchor+Delta**: 覆盖稳定视角作为锚定主轴,重构视角作为贯穿辅轴

### 4.2 当前代码实现

#### ✅ 已实现功能

1. **StressClusterer** (lines 48-334):
   - ✅ 筛选 stress_score > 0.3
   - ✅ 构建failure_signature (facet + leaf_path + lost_novelty)
   - ✅ HDBSCAN聚类 (with fallback)
   - ✅ 计算view_failure_rates
   - ✅ ClusterType判定 (STRONG_SHIFT, FACET_DEPENDENT, STABLE)

2. **EvolutionPlanner** (lines 336-624):
   - ✅ 为每个cluster提议ADD_NODE操作
   - ✅ 使用LLM生成NewNodeProposal
   - ✅ 计算fit_gain (simplified as 0.5*cluster_size)
   - ✅ Greedy selection: `objective = fit_gain - lambda*edit_cost`
   - ✅ **compute_all_views_reconstruction()** (lines 419-512):
     - ✅ 对所有视角计算reconstruction metrics
     - ✅ Combined score: `0.4*FitGain + 0.3*StressRed + 0.2*Coverage - 0.1*EditCost`
     - ✅ 排序输出ViewReconstructionScore列表

3. **ViewReconstructionScore** (dataclass_v2.py, lines 720-778):
   - ✅ 定义完整,包含 fit_gain, stress_reduction, coverage, edit_cost, combined_score
   - ✅ 自动计算combined_score (在__init__)

#### ❌ 缺失功能

**CRITICAL**: **SPLIT_NODE和RENAME_NODE操作未实现**

方法说明要求三种操作:
- `ADD_NODE`: 新增分支 (cost = 1.0) ✅ 已实现
- `SPLIT_NODE`: 拆分节点 (cost = 2.0) ❌ 未实现
- `RENAME_NODE`: 节点语义漂移后的重命名 (cost = 0.5) ❌ 未实现

**当前代码**:
```python
# phase3_evolution.py, lines 389-392
# Try SPLIT_NODE for overcrowded nodes
# (simplified: skip for now, can add later)  # ❌ TODO comment

# Try RENAME_NODE for drifted nodes
# (simplified: skip for now, can add later)  # ❌ TODO comment
```

**影响**: 只能处理"新增类别"场景,无法处理"节点过度拥挤"或"语义漂移"场景。

#### ⚠️ 需要增强的功能

1. **FitGain计算过于简化**:
   - 当前: `fit_gain = len(cluster.papers) * 0.5` (固定50%改进率)
   - 方法说明: 应基于实际re-fit测试计算改进
   - **建议**: 模拟应用operation后重新测试fit

2. **Evidence绑定不完整**:
   - `_extract_cluster_evidence()`只取前3篇论文的abstract
   - 应该从paper claims的evidence_spans提取

3. **Coverage计算**:
   - 当前: `min(1.0, num_leaves / 50.0)` - 硬编码50
   - **建议**: 使用动态基准 (如baseline的平均叶节点数)

### 4.3 改进建议

#### **改进1: 实现SPLIT_NODE操作**

```python
def _propose_split_node(
    self,
    cluster: StressCluster,
    view: TaxonomyView,
    fit_vectors: List[FitVector],
    overcrowding_threshold: int = 15  # 叶节点包含>15篇论文视为overcrowded
) -> Optional[SplitNodeOperation]:
    """
    提议拆分overcrowded节点。
    
    识别标准:
    1. 节点包含的论文数 > threshold
    2. 论文之间缺乏cohesion (内部相似度低)
    
    拆分策略:
    1. 对节点内论文聚类 (k=2或3)
    2. 为每个子簇生成新的sub_node定义
    """
    
    # 找到overcrowded的叶节点
    leaf_paper_count = defaultdict(list)
    for fv in fit_vectors:
        for report in fv.fit_reports:
            if report.view_id == view.view_id and report.best_leaf_path:
                leaf_paper_count[report.best_leaf_path].append(fv.paper_id)
    
    # 找到候选拆分节点
    for leaf_path, paper_ids in leaf_paper_count.items():
        if len(paper_ids) > overcrowding_threshold:
            # 检查cluster论文是否在这个叶节点中
            cluster_paper_ids = {p.url for p in cluster.papers}
            overlap = cluster_paper_ids & set(paper_ids)
            
            if len(overlap) >= 3:  # 至少3篇cluster论文在这个节点
                # 对节点内论文聚类
                sub_clusters = self._subcluster_papers(overlap)
                
                # 生成子节点定义
                sub_nodes = []
                for sub_cluster in sub_clusters:
                    sub_node = self._generate_subnode_definition(
                        parent_path=leaf_path,
                        papers=sub_cluster,
                        view=view
                    )
                    sub_nodes.append(sub_node)
                
                # 计算fit_gain
                fit_gain = self._estimate_split_fit_gain(leaf_path, sub_nodes, fit_vectors, view)
                
                operation = SplitNodeOperation(
                    view_id=view.view_id,
                    node_path=leaf_path,
                    sub_nodes=sub_nodes,
                    evidence=self._extract_cluster_evidence(cluster),
                    fit_gain=fit_gain
                )
                
                return operation
    
    return None
```

#### **改进2: 实现RENAME_NODE操作**

```python
def _propose_rename_node(
    self,
    cluster: StressCluster,
    view: TaxonomyView,
    fit_vectors: List[FitVector],
    drift_threshold: float = 0.3
) -> Optional[RenameNodeOperation]:
    """
    提议重命名semantic drift的节点。
    
    识别标准:
    1. 节点的现有定义与实际论文的semantic gap > threshold
    2. 大量FORCE_FIT论文 (定义过时但勉强归类)
    
    策略:
    1. 分析FORCE_FIT论文的lost_novelty
    2. 提取共同主题
    3. 生成新的节点名称和定义
    """
    
    # 找到FORCE_FIT率高的叶节点
    leaf_force_fit = defaultdict(lambda: {'total': 0, 'force_fit': 0, 'novelties': []})
    
    for fv in fit_vectors:
        for report in fv.fit_reports:
            if report.view_id == view.view_id and report.best_leaf_path:
                leaf_force_fit[report.best_leaf_path]['total'] += 1
                if report.label == FitLabel.FORCE_FIT:
                    leaf_force_fit[report.best_leaf_path]['force_fit'] += 1
                    leaf_force_fit[report.best_leaf_path]['novelties'].extend(
                        [ln.bullet for ln in report.lost_novelty]
                    )
    
    # 找到drift候选
    for leaf_path, stats in leaf_force_fit.items():
        if stats['total'] < 3:
            continue
        
        force_fit_rate = stats['force_fit'] / stats['total']
        
        if force_fit_rate > drift_threshold:
            # 这个节点有语义漂移
            old_def = view.node_definitions.get(leaf_path)
            if not old_def:
                continue
            
            # 分析lost_novelty,生成新定义
            new_name, new_def = self._generate_renamed_definition(
                old_def=old_def,
                lost_novelties=stats['novelties']
            )
            
            operation = RenameNodeOperation(
                view_id=view.view_id,
                node_path=leaf_path,
                old_name=old_def.node_path.split('/')[-1],
                new_name=new_name,
                new_definition=new_def,
                drift_score=force_fit_rate,
                evidence=self._extract_cluster_evidence(cluster),
                fit_gain=stats['force_fit'] * 0.3  # 假设30%改进
            )
            
            return operation
    
    return None
```

#### **改进3: 增强FitGain计算 (真实Re-fit测试)**

```python
def _estimate_operation_fit_gain_precise(
    self,
    operation: EvolutionOperation,
    cluster: StressCluster,
    view: TaxonomyView,
    fit_vectors: List[FitVector]
) -> float:
    """
    精确估算operation的FitGain: 模拟应用operation后重新fit测试。
    
    步骤:
    1. 克隆view并应用operation
    2. 对cluster论文重新fit测试
    3. 计算before/after的fit_score差异
    """
    
    # 克隆view
    view_copy = copy.deepcopy(view)
    
    # 应用operation到view_copy
    if isinstance(operation, AddNodeOperation):
        # 添加新节点到tree
        new_path = f"{operation.parent_path}/{operation.new_node.name}"
        new_tree_node = TaxonomyTreeNode(
            name=operation.new_node.name,
            path=new_path,
            parent=operation.parent_path,
            children=[],
            is_leaf=True
        )
        view_copy.tree.add_node(new_tree_node)
        
        # 添加节点定义
        new_node_def = NodeDefinition(
            node_path=new_path,
            definition=operation.new_node.definition,
            inclusion_criteria=operation.new_node.inclusion_criteria,
            exclusion_criteria=operation.new_node.exclusion_criteria,
            canonical_keywords=operation.new_node.keywords,
            boundary_statements=[],
            evidence_spans=operation.evidence
        )
        view_copy.node_definitions[new_path] = new_node_def
    
    # 对cluster论文重新fit测试
    fit_gain_sum = 0.0
    for paper in cluster.papers:
        # 找到原始fit_score
        original_fv = next((fv for fv in fit_vectors if fv.paper_id == paper.url), None)
        if not original_fv:
            continue
        
        original_report = next((r for r in original_fv.fit_reports if r.view_id == view.view_id), None)
        if not original_report:
            continue
        
        original_score = original_report.scores.fit_score
        
        # 用view_copy重新测试 (需要PaperClaims)
        # new_score = self._refit_paper(paper, view_copy)
        # fit_gain_sum += max(0, new_score - original_score)
        
        # Simplified: 假设新节点使UNFITTABLE→FIT (gain=1.0), FORCE_FIT→FIT (gain=0.5)
        if original_report.label == FitLabel.UNFITTABLE:
            fit_gain_sum += 1.0
        elif original_report.label == FitLabel.FORCE_FIT:
            fit_gain_sum += 0.5
    
    return fit_gain_sum
```

---

## 5. Phase 4: Delta-aware Guidance - 详细分析

### 5.1 方法说明要求

**目标**: 生成可审计的认知增量资产与可执行的写作约束,支持下游综述系统生成具有结构性新认知的综述内容

**关键步骤**:
1. **主轴/辅轴组织方案确定 (Evolution-first/Reconstruct-then-select)**:
   - 对候选视角执行最小必要重构得到 T_i'
   - 计算每视角的 `Score_i = α·FitGain + β·Stress + γ·Coverage − λ·EditCost`
   - 依据分数与覆盖性选择两种组织模式之一:
     - **Delta-first**: 重构后的高压力视角作为主轴 (EditCost > 3.0 或 FitGain > 10.0)
     - **Anchor+Delta**: 覆盖稳定视角作为锚定主轴,重构视角作为贯穿辅轴

2. **写作约束编译**:
   - **组织模式与轴**: `main_axis_mode`, `main_axis`, `aux_axes`
   - **taxonomy_v2**: 更新后的主轴树结构 + 演化操作序列
   - **outline_constraints**: 章节标题、必覆盖点、必答问题、必引论文列表
   - **writing_rules (Do/Don't)**: 明确禁止滞后写法,明确证据要求

3. **输出形态**:
   - `audit_report.md`: 审计报告 (人类可读)
   - `guidance_pack.json`: 机器可读约束包 (下游可执行)

### 5.2 当前代码实现

#### ✅ 已实现功能

1. **AxisSelector** (lines 32-223):
   - ✅ **select_main_axis_with_mode()** (NEW DESIGN, lines 45-112):
     - ✅ 基于reconstruction_scores选择main_axis
     - ✅ 确定writing_mode (DELTA_FIRST vs ANCHOR_PLUS_DELTA)
     - ✅ 阈值规则: `EditCost > 3.0 or FitGain > 10.0 → DELTA_FIRST`
   - ✅ **select_aux_axis()** (lines 165-223):
     - ✅ 基于discriminativeness (variance of failure rates)

2. **GuidanceGenerator** (lines 225-602):
   - ✅ **generate_guidance()** (lines 233-312):
     - ✅ 包含 main_axis_mode, writing_rules, reconstruction_scores
   - ✅ **_generate_outline()** (lines 314-384):
     - ✅ 基于main_axis tree结构生成sections
     - ✅ Cross-organize with aux_axis (subsections)
   - ✅ **_generate_writing_rules()** (lines 482-549):
     - ✅ Mode-specific rules:
       - DELTA_FIRST: "Lead with evolution", "Don't force-fit"
       - ANCHOR_PLUS_DELTA: "Use structure", "Mark updates"
   - ✅ **_generate_evolution_summary()** (lines 551-570)
   - ✅ **_generate_must_answer_questions()** (lines 572-601)

3. **SNSRunner._save_guidance_pack()** (engine_v2.py, lines 442-516):
   - ✅ 生成 `guidance_pack.json` (machine-readable)
   - ✅ 包含: writing_mode, writing_rules, taxonomy, outline, evolution_summary, must_answer_questions, reconstruction_scores

4. **SNSRunner._generate_markdown_report()** (engine_v2.py, lines 518-606):
   - ✅ 生成 `audit_report.md` (human-readable)

#### ⚠️ 需要增强的功能

1. **taxonomy_v2 (演化后的taxonomy)未明确输出**:
   - 方法要求: "taxonomy_v2: 更新后的主轴树结构与节点定义 + 演化操作序列"
   - 当前: `guidance_pack.json` 包含main_axis.tree,但**没有应用evolution operations**
   - **问题**: taxonomy仍然是Phase 1的原始版本,没有ADD_NODE/SPLIT_NODE的更新
   - **建议**: 在Phase 4开始前,将accepted operations应用到view.tree

2. **Outline constraints不够具体**:
   - 方法要求: "必覆盖点、必答问题、必引证据"
   - 当前: `must_answer`问题过于通用 ("What are the key approaches?")
   - **建议**: 从evolution操作和cluster分析生成更具体的问题

3. **Evidence Cards质量**:
   - 当前: Evidence cards只包含paper title和abstract前200字
   - 方法强调: "每项操作的触发簇与证据卡"
   - **建议**: 从fit_reports的lost_novelty和conflict_evidence提取精确quotes

### 5.3 改进建议

#### **改进1: 应用Evolution Operations到Taxonomy**

```python
def apply_evolution_to_taxonomy(
    view: TaxonomyView,
    operations: List[EvolutionOperation]
) -> TaxonomyView:
    """
    将accepted evolution operations应用到taxonomy tree,生成taxonomy_v2。
    
    支持:
    - ADD_NODE: 添加新叶节点
    - SPLIT_NODE: 拆分现有节点为多个子节点
    - RENAME_NODE: 重命名节点及更新定义
    """
    
    view_v2 = copy.deepcopy(view)
    
    for op in operations:
        if isinstance(op, AddNodeOperation):
            # 添加新节点
            new_path = f"{op.parent_path}/{op.new_node.name}"
            new_node = TaxonomyTreeNode(
                name=op.new_node.name,
                path=new_path,
                parent=op.parent_path,
                children=[],
                is_leaf=True
            )
            view_v2.tree.add_node(new_node)
            
            # 添加节点定义
            new_def = NodeDefinition(
                node_path=new_path,
                definition=op.new_node.definition,
                inclusion_criteria=op.new_node.inclusion_criteria,
                exclusion_criteria=op.new_node.exclusion_criteria,
                canonical_keywords=op.new_node.keywords,
                boundary_statements=[],
                evidence_spans=op.evidence
            )
            view_v2.node_definitions[new_path] = new_def
        
        elif isinstance(op, SplitNodeOperation):
            # 拆分节点
            parent_node = view_v2.tree.nodes[op.node_path]
            parent_node.is_leaf = False  # 变为内部节点
            
            for sub_node_prop in op.sub_nodes:
                sub_path = f"{op.node_path}/{sub_node_prop.name}"
                sub_node = TaxonomyTreeNode(
                    name=sub_node_prop.name,
                    path=sub_path,
                    parent=op.node_path,
                    children=[],
                    is_leaf=True
                )
                view_v2.tree.add_node(sub_node)
                
                # 添加定义
                sub_def = NodeDefinition(
                    node_path=sub_path,
                    definition=sub_node_prop.definition,
                    inclusion_criteria=sub_node_prop.inclusion_criteria,
                    exclusion_criteria=sub_node_prop.exclusion_criteria,
                    canonical_keywords=sub_node_prop.keywords,
                    boundary_statements=[],
                    evidence_spans=op.evidence
                )
                view_v2.node_definitions[sub_path] = sub_def
        
        elif isinstance(op, RenameNodeOperation):
            # 重命名节点
            node = view_v2.tree.nodes[op.node_path]
            old_name = node.name
            node.name = op.new_name
            
            # 更新path (需要递归更新子节点)
            # ... (实现path更新逻辑)
            
            # 更新定义
            if op.node_path in view_v2.node_definitions:
                view_v2.node_definitions[op.node_path].definition = op.new_definition
    
    return view_v2
```

**集成到Phase4Pipeline**:
```python
def run(self, ...):
    # ...
    
    # 在axis selection之前,应用evolution到main_axis
    main_axis_v2 = apply_evolution_to_taxonomy(main_axis, evolution_proposal.operations)
    
    guidance = self.guidance_generator.generate_guidance(
        # ...
        main_axis=main_axis_v2,  # 使用evolved版本
        # ...
    )
```

#### **改进2: 生成Delta-aware Must-answer Questions**

```python
def _generate_must_answer_questions_enhanced(
    self,
    main_axis: TaxonomyView,
    aux_axis: Optional[TaxonomyView],
    clusters: List[StressCluster],
    proposal: EvolutionProposal,
    baseline: MultiViewBaseline
) -> List[str]:
    """
    生成具体的必答问题,直接关联到evolution和stress points。
    """
    
    questions = []
    
    # 1. 基础结构问题
    questions.append(
        f"What are the main organizational dimensions in {main_axis.facet_label.value}?"
    )
    
    # 2. 演化操作相关问题 (每个operation一个问题)
    for op in proposal.operations:
        if isinstance(op, AddNodeOperation):
            questions.append(
                f"Why was the new category '{op.new_node.name}' needed in {op.view_id}? "
                f"What papers don't fit existing structure?"
            )
        elif isinstance(op, SplitNodeOperation):
            questions.append(
                f"Why was '{op.node_path}' split into subcategories? "
                f"What overcrowding or heterogeneity was observed?"
            )
        elif isinstance(op, RenameNodeOperation):
            questions.append(
                f"Why was '{op.old_name}' renamed to '{op.new_name}'? "
                f"What semantic drift occurred?"
            )
    
    # 3. Stress cluster相关问题
    for cluster in clusters:
        if cluster.cluster_type == ClusterType.STRONG_SHIFT:
            # 找到cluster中最高压力的视角
            max_failure_view_id = max(
                cluster.view_failure_rates.items(),
                key=lambda x: x[1]
            )[0]
            view = baseline.get_view_by_id(max_failure_view_id)
            
            questions.append(
                f"Cluster {cluster.cluster_id} shows strong structural shift: "
                f"How do these {len(cluster.papers)} papers challenge "
                f"{view.facet_label.value if view else 'existing'} organization? "
                f"What new patterns emerge?"
            )
    
    # 4. 旧结构不足证据问题 (关键!)
    questions.append(
        "What evidence demonstrates that existing taxonomies are insufficient "
        "for organizing recent research? Cite specific FORCE_FIT or UNFITTABLE cases."
    )
    
    # 5. 辅轴discriminativeness问题
    if aux_axis:
        questions.append(
            f"How does {aux_axis.facet_label.value} provide orthogonal perspective? "
            f"Which stress clusters does it help discriminate?"
        )
    
    return questions
```

---

## 6. 缺失功能总结与优先级

### 🔴 Critical (必须实现)

| 功能 | Phase | 当前状态 | 影响 |
|-----|-------|---------|------|
| **补视角策略 (CompensatoryViewInducer)** | Phase 1 | ❌ 未实现 | baseline质量不足时无法自动补救 |
| **NLI冲突检测 (NLIConflictDetector)** | Phase 2 | ❌ 未实现 | Conflict分数不准确,FIT判定可能错误 |
| **真实Embedding模型 (SPECTER2/SciNCL)** | Phase 2 | ❌ 未实现 | Coverage/Residual分数不准确 |
| **SPLIT_NODE操作** | Phase 3 | ❌ 未实现 | 无法处理节点overcrowding |
| **RENAME_NODE操作** | Phase 3 | ❌ 未实现 | 无法处理semantic drift |
| **应用Evolution到Taxonomy_v2** | Phase 4 | ⚠️ 部分实现 | 输出的taxonomy未包含结构更新 |

### 🟡 High Priority (应该实现)

| 功能 | Phase | 当前状态 | 影响 |
|-----|-------|---------|------|
| **节点定义质量验证** | Phase 1 | ⚠️ 无验证 | 可能生成不足3/2标准的定义 |
| **Evidence Span精确性验证** | Phase 1-4 | ⚠️ 依赖LLM | char_start/end可能不准确 |
| **阈值可校准接口** | Phase 2 | ⚠️ 硬编码 | 无法适配不同领域 |
| **FitGain精确计算 (re-fit测试)** | Phase 3 | ⚠️ 简化估算 | operation benefit估算不准 |
| **Delta-aware Must-answer Questions** | Phase 4 | ⚠️ 过于通用 | 问题缺乏针对性 |
| **Evidence Cards质量增强** | Phase 4 | ⚠️ 只有abstract | 缺乏精确quotes |

### 🟢 Medium Priority (建议实现)

| 功能 | Phase | 当前状态 | 影响 |
|-----|-------|---------|------|
| **权重计算的Quality因子** | Phase 1 | ⚠️ 简化 | 使用文本长度代理citations |
| **多facet补视角多样性** | Phase 1 | ⚠️ 单次补救 | 可能需要多轮补视角 |
| **Cluster内cohesion检查** | Phase 3 | ❌ 未实现 | 聚类质量未验证 |
| **动态Coverage基准** | Phase 3 | ⚠️ 硬编码50 | 不同领域taxonomy大小差异大 |

---

## 7. 实现改进路线图

### Phase 1: 核心功能补齐 (Week 1-2)

1. **集成真实Embedding模型**:
   - 添加 `ScientificEmbedder` 类 (SPECTER2)
   - 更新 `EmbeddingBasedRetriever`
   - 更新 `FitTester._calculate_coverage()` 和 `._calculate_residual()`

2. **集成NLI冲突检测**:
   - 添加 `NLIConflictDetector` 类 (DeBERTa-MNLI)
   - 更新 `FitTester._calculate_conflict()`
   - 添加fallback机制

3. **实现补视角策略**:
   - 添加 `CompensatoryViewInducer` 类
   - 集成到 `Phase1Pipeline.run()` 的quality gate后

### Phase 2: Evolution操作完整化 (Week 3)

4. **实现SPLIT_NODE**:
   - 添加 `_propose_split_node()` 到 `EvolutionPlanner`
   - 实现节点内论文聚类逻辑
   - 生成LLM prompt for subnode definition

5. **实现RENAME_NODE**:
   - 添加 `_propose_rename_node()` 到 `EvolutionPlanner`
   - 检测FORCE_FIT率高的节点
   - 分析lost_novelty生成新定义

6. **增强FitGain计算**:
   - 实现 `_estimate_operation_fit_gain_precise()`
   - 模拟应用operation后re-fit测试

### Phase 3: Guidance质量提升 (Week 4)

7. **应用Evolution到Taxonomy**:
   - 实现 `apply_evolution_to_taxonomy()`
   - 在Phase 4中使用taxonomy_v2

8. **增强Must-answer Questions**:
   - 实现 `_generate_must_answer_questions_enhanced()`
   - 每个operation生成具体问题
   - 每个STRONG_SHIFT cluster生成问题

9. **提升Evidence Cards质量**:
   - 从fit_reports提取lost_novelty quotes
   - 从conflict_evidence提取精确spans

### Phase 4: 质量保障与验证 (Week 5)

10. **添加节点定义验证**:
    - 实现 `_validate_node_definition()`
    - Schema验证 (≥3 inclusion, ≥2 exclusion)
    - Evidence spans完整性检查

11. **阈值可配置化**:
    - 将0.45, 0.55等阈值移到SNSArguments
    - 添加per-domain calibration接口

12. **集成测试**:
    - 端到端测试 (完整pipeline)
    - 验证guidance_pack.json格式
    - 验证audit_report.md可读性

---

## 8. 关键设计决策与对齐

### ✅ 设计决策对齐良好

1. **Reconstruct-then-select (Phase 3-4)**:
   - 方法要求: "先重构再选择"
   - 代码实现: ✅ `compute_all_views_reconstruction()` → `select_main_axis_with_mode()`

2. **Writing Mode Determination**:
   - 方法要求: `EditCost > 3.0 or FitGain > 10.0 → DELTA_FIRST`
   - 代码实现: ✅ 完全一致 (phase4_guidance.py, lines 94-101)

3. **Multi-view Atlas**:
   - 方法要求: "多视角图集,每个视角独立树结构"
   - 代码实现: ✅ `MultiViewBaseline` + `TaxonomyView` + `TaxonomyTree`

4. **Evidence Anchoring**:
   - 方法要求: "所有claim绑定原文spans"
   - 代码实现: ✅ `EvidenceSpan` 数据结构包含 char_start, char_end, quote

5. **Deterministic Scoring**:
   - 方法要求: "关键决策(FIT/FORCE_FIT/UNFITTABLE)由确定性规则"
   - 代码实现: ✅ 阈值规则 (0.45, 0.55) in `FitTester._determine_label()`

### ⚠️ 设计偏离 (需要解释或改进)

1. **LLM Temperature**:
   - 方法要求: "temperature=0 for reproducibility"
   - 代码实现: ❓ 使用 `dspy.context(lm=self.lm)` 但未显式设置temperature
   - **建议**: 在LM初始化时强制 `temperature=0`

2. **Novelty Bullets数量**:
   - 方法要求: "Must have exactly 3"
   - 代码实现: ✅ Enforced with padding/trimming (phase2, lines 83-91)

3. **Weight Normalization**:
   - 方法要求: w_i归一化
   - 代码实现: ✅ 在 `MultiViewBaseline.__post_init__` (lines 224-229)

---

## 9. 结论与建议

### 9.1 总体评估

**实现完整度**: **70%**

- ✅ **核心框架**: 完整实现 (Phase 1-4 pipeline, 数据结构)
- ✅ **关键设计**: Reconstruct-then-select, Writing mode, Multi-view atlas
- ❌ **缺失Critical功能**: 补视角、NLI、Embeddings、SPLIT/RENAME、Taxonomy_v2应用
- ⚠️ **质量待提升**: Evidence精确性、FitGain估算、Must-answer问题

### 9.2 优先改进建议

**立即着手 (Week 1)**:
1. 集成SPECTER2 embeddings (影响所有相似度计算)
2. 集成NLI冲突检测 (影响FIT判定准确性)
3. 实现补视角策略 (保证baseline质量)

**短期目标 (Week 2-3)**:
4. 实现SPLIT_NODE和RENAME_NODE
5. 增强FitGain计算 (re-fit测试)
6. 应用Evolution到Taxonomy_v2

**中期目标 (Week 4-5)**:
7. 提升Guidance质量 (questions, evidence cards)
8. 添加质量验证 (schema, thresholds)
9. 端到端测试与文档

### 9.3 方法说明适配性评估

**对方法说明的遵循度**: **Good (85%)**

- ✅ 所有核心概念都有对应实现
- ✅ 4个Phase流程完整
- ✅ 数据结构与方法说明一致
- ❌ 部分细节功能缺失 (但设计空间已预留)
- ⚠️ 一些实现为placeholder (但有清晰TODO注释)

**建议**: 当前代码架构良好,适合渐进式补齐功能,无需大规模重构。

---

## 附录: 文件修改清单

### 需要修改的文件

| 文件 | 修改类型 | 优先级 |
|-----|---------|--------|
| `phase1_multiview_baseline.py` | 添加CompensatoryViewInducer类 | 🔴 Critical |
| `phase2_stress_test.py` | 添加NLIConflictDetector和ScientificEmbedder | 🔴 Critical |
| `phase3_evolution.py` | 添加SPLIT/RENAME操作 | 🔴 Critical |
| `phase4_guidance.py` | 添加apply_evolution和enhanced questions | 🔴 Critical |
| `engine_v2.py` | 集成新组件到pipeline | 🔴 Critical |
| `dataclass_v2.py` | (可能需要新数据结构) | 🟡 High |
| `schemas_v2.py` | 添加LLM prompts for SPLIT/RENAME | 🟡 High |

### 需要新增的文件

| 文件 | 用途 | 优先级 |
|-----|-----|--------|
| `knowledge/sns/infrastructure/embeddings_real.py` | SPECTER2/SciNCL实现 | 🔴 Critical |
| `knowledge/sns/infrastructure/nli_real.py` | DeBERTa-MNLI实现 | 🔴 Critical |
| `knowledge/sns/modules/compensatory_view.py` | 补视角策略实现 | 🔴 Critical |

---

**报告生成日期**: 2025-12-15  
**分析人员**: Claude (AI Code Assistant)  
**项目路径**: `/home/user/webapp`
