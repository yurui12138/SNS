# SNS系统运行结果评估报告

## 执行概况

**测试主题**: deepfake  
**分析日期**: 2025-12-15  
**输入数据**: 
- Review papers: 2篇
- Research papers: 5篇
- Total fit tests: 10次（5篇论文 × 2个视角）

---

## 🔴 关键问题识别

### 1. 极端的失配率（100% UNFITTABLE）⚠️

**问题严重性**: 🔴 **CRITICAL**

**观察结果**:
```
- FIT: 0 (0.0%)
- FORCE_FIT: 0 (0.0%)
- UNFITTABLE: 10 (100.0%)
- Average Stress Score: 1.000
- Average Unfittable Score: 1.000
```

**问题分析**:

这是一个**极度异常**的结果。100%的UNFITTABLE率表明：

1. **Baseline质量问题**:
   - 只检索到2篇review papers（目标是15篇）
   - 两个view都是同一个facet（APPLICATION_DOMAIN）
   - 缺乏多样性：没有METHOD、DATASET、EVALUATION等其他维度

2. **Phase 2 Fit Test过于严格**:
   - Coverage分数极低：0.056（阈值0.45）
   - Residual分数极高：0.951（阈值0.45）
   - 所有论文都被判定为UNFITTABLE

3. **Taxonomy提取质量差**:
   ```
   View T1: 
   - ROOT/AI Music Generation
   - ROOT/Detection Methods
     - Foundation Models
     - ...
   
   View T2:
   - ROOT/DeepFake Generation
   - ROOT/DeepFake Detection  
   - ROOT/Evasion of DeepFake Detection
   ```
   
   问题：
   - T1聚焦"AI-Generated Music"（过于狭窄）
   - T2的taxonomy太粗糙（只有2-3层）
   - 缺少关键子类别

**根本原因**:
- 🔴 **Review检索失败**：只找到2篇，而不是15篇
- 🔴 **Taxonomy提取不完整**：树太浅，叶节点太少
- 🔴 **Fit scoring过于保守**：阈值设置不当

---

### 2. Phase 3完全失效（空结果）⚠️

**问题严重性**: 🔴 **CRITICAL**

**观察结果**:
```json
// stress_clusters.json
[]

// evolution_proposal.json
{
  "operations": [],
  "total_fit_gain": 0.0,
  "total_edit_cost": 0.0,
  "objective_value": 0.0
}
```

**问题分析**:

Phase 3（Stress Clustering & Minimal Evolution）完全没有输出：

1. **Stress Clustering失败**:
   - 输入：5篇高压力论文（stress_score=1.0）
   - 输出：空的cluster列表（`[]`）
   - 原因：HDBSCAN clustering可能失败或min_cluster_size设置不当

2. **Evolution Planning未触发**:
   - 因为没有clusters，所以没有提出任何evolution操作
   - `operations: []` 意味着系统认为不需要任何taxonomy更新

**根本原因**:
- 🔴 **Sample size太小**：5篇论文不足以形成有意义的clusters
- 🔴 **HDBSCAN参数不当**：`min_cluster_size=3` 对于5篇论文太大
- 🔴 **Feature representation问题**：embedding可能是"dummy"模式

---

### 3. Phase 4降级为简化模式⚠️

**问题严重性**: 🟡 **MEDIUM**

**观察结果**:
```python
# guidance_pack.json
"writing_mode": "ANCHOR_PLUS_DELTA"
"evolution_summary": []
"must_answer_questions": [
  "What are the main organizational dimensions in APPLICATION_DOMAIN?",
  "How has the field evolved beyond existing reviews?"
]
```

**问题分析**:

由于Phase 3失败，Phase 4退化为简化模式：

1. **Writing Mode选择不准确**:
   - 选择了`ANCHOR_PLUS_DELTA`模式
   - 但实际情况是100% UNFITTABLE，应该是`DELTA_FIRST`
   - reconstruction_scores缺失导致无法正确判定

2. **Guidance质量低**:
   - 没有evolution_summary（因为Phase 3失败）
   - must_answer_questions太泛化
   - 缺少具体的structural updates指导

3. **Outline过于简单**:
   - 直接使用main_axis的树结构
   - 没有cross-organization with aux_axis
   - 缺少evidence cards

---

## 📊 数据质量问题

### 问题4: Review Paper检索不足

**观察**:
- 目标：15篇review papers
- 实际：2篇review papers
- 达成率：13.3%

**可能原因**:
1. **检索策略过于保守**:
   ```python
   # 当前查询
   queries = [
       "deepfake survey",
       "deepfake review", 
       "deepfake overview",
       "systematic review of deepfake"
   ]
   ```
   - 可能arxiv上关于deepfake的survey确实较少
   - 需要扩展到其他数据源（Semantic Scholar, Google Scholar）

2. **过滤规则太严格**:
   ```python
   def _filter_review_papers(results):
       # 检查title中是否有review关键词
       # 检查abstract长度
       # 检查snippet数量
   ```
   - 可能过滤掉了一些相关但不完全符合的papers

3. **arXiv局限性**:
   - deepfake领域的survey可能发表在会议/期刊，而非arXiv
   - 需要多数据源支持

**影响**:
- ❌ Baseline质量差
- ❌ 多样性不足（只有1个facet）
- ❌ 导致后续所有phase失败

---

### 问题5: Research Paper数量过少

**观察**:
- 配置：`top_k_research_papers=10`（example脚本中）
- 实际：5篇papers
- 达成率：50%

**影响**:
- Clustering无法进行（需要至少`min_cluster_size * 2`篇论文）
- Evolution proposal无法触发
- 统计显著性不足

---

### 问题6: Embedding Model使用Dummy模式

**观察**:
```python
# run_sns_example.py
embedding_model="dummy"  # Using simple embedding for demo
```

**问题**:
- Dummy embedding无法捕捉语义相似度
- Coverage计算不准确
- Residual计算不准确
- Clustering效果差

**应该使用**:
- `allenai/specter2`（for scientific papers）
- `sentence-transformers/all-MiniLM-L6-v2`（通用）

---

## 🟡 算法参数问题

### 问题7: Fit Test阈值过于严格

**当前阈值**:
```python
# Phase 2: FitTester
coverage_threshold = 0.45
conflict_threshold = 0.55
residual_threshold = 0.45
```

**观察到的分数**:
```json
{
  "coverage": 0.056,      // << 0.45 (FAIL)
  "conflict": 0.071,      // < 0.55 (PASS)
  "residual": 0.951,      // >> 0.45 (FAIL)
  "fit_score": -0.382
}
```

**问题**:
- Coverage阈值0.45太高（实际只有0.056）
- Residual阈值0.45太低（实际是0.951）

**建议**:
- 降低coverage阈值：0.45 → 0.25
- 提高residual阈值：0.45 → 0.60
- 或者采用动态阈值（基于baseline质量）

---

### 问题8: HDBSCAN参数不当

**当前配置**:
```python
min_cluster_size = 3  # 在example脚本中设为2
```

**问题**:
- 5篇论文，min_cluster_size=3，最多只能形成1个cluster
- 无法发现多样的stress patterns
- 导致clustering失败

**建议**:
- 动态计算：`min_cluster_size = max(2, len(papers) // 3)`
- 对于小样本：`min_cluster_size = 2`
- 添加sample size检查：如果papers < 10，跳过clustering

---

## 🟢 系统输出正确性

### ✅ 正确的部分

1. **数据结构完整**:
   - ✅ `audit_report.md` 生成正确
   - ✅ `guidance_pack.json` 格式正确
   - ✅ 包含required fields：writing_mode, writing_rules, taxonomy

2. **Writing Rules有意义**:
   ```json
   "do": [
     "Use main axis structure as foundation",
     "Integrate new papers where they fit",
     "Clearly mark structural updates"
   ],
   "dont": [
     "Don't ignore evolution and stress points",
     "Don't present taxonomy as static"
   ]
   ```

3. **Guidance Pack机器可读**:
   - ✅ 严格的JSON格式
   - ✅ 包含taxonomy树结构
   - ✅ 包含writing_mode和rules

---

## 🎯 核心问题总结

### 关键失败链条

```
1. Review检索失败 (2/15)
   ↓
2. Baseline质量差 (单一facet, 浅树)
   ↓
3. Fit Test过于严格 (100% UNFITTABLE)
   ↓
4. Sample size太小 (5 papers)
   ↓
5. Clustering失败 (empty clusters)
   ↓
6. Evolution未触发 (no operations)
   ↓
7. Guidance降级 (simplified mode)
```

### 优先级排序

| 优先级 | 问题 | 影响范围 | 修复难度 |
|--------|------|----------|----------|
| 🔴 P0 | Review检索不足 | 整个pipeline | 中 |
| 🔴 P0 | Fit Test阈值过严 | Phase 2-4 | 低 |
| 🔴 P0 | Embedding使用dummy | Phase 2-3 | 低 |
| 🟡 P1 | Sample size太小 | Phase 3-4 | 低 |
| 🟡 P1 | HDBSCAN参数不当 | Phase 3 | 低 |
| 🟢 P2 | Taxonomy提取质量 | Phase 1 | 中 |
| 🟢 P2 | Guidance质量 | Phase 4 | 中 |

---

## 💡 改进建议

### 立即修复（P0）

#### 修复1: 改进Review检索策略

**文件**: `knowledge_storm/sns/modules/phase1_multiview_baseline.py`

```python
class ReviewRetriever:
    def retrieve_reviews(self, topic: str) -> List[Information]:
        # 扩展查询策略
        review_queries = [
            f"{topic} survey",
            f"{topic} review",
            f"{topic} comprehensive overview",
            f"{topic} state of the art",
            f"{topic} recent advances",
            f"survey on {topic}",  # NEW
            f"{topic} literature review",  # NEW
            f"{topic} systematic review",
            f"{topic} tutorial",  # NEW
        ]
        
        # 放宽过滤条件
        def _filter_review_papers(self, results):
            filtered = []
            for result in results:
                # 降低title关键词要求
                title_lower = result.title.lower()
                has_review_keyword = any(
                    kw in title_lower 
                    for kw in ['survey', 'review', 'overview', 
                              'comprehensive', 'systematic', 'tutorial',
                              'advances', 'state-of-the-art', 'progress']  # 扩展
                )
                
                # 放宽abstract长度要求
                has_long_abstract = len(result.description) > 300  # 降低从500
                
                if has_review_keyword or has_long_abstract:
                    filtered.append(result)
            
            return filtered
```

#### 修复2: 调整Fit Test阈值

**文件**: `knowledge_storm/sns/modules/phase2_stress_test.py`

```python
class FitTester:
    def _determine_label(self, coverage, conflict, residual):
        # 方案A: 降低阈值
        if coverage < 0.25 or conflict > 0.55:  # 0.45 → 0.25
            return FitLabel.UNFITTABLE
        elif residual > 0.60:  # 0.45 → 0.60
            return FitLabel.FORCE_FIT
        else:
            return FitLabel.FIT
        
        # 方案B: 动态阈值（更好）
        # 根据baseline质量调整阈值
        baseline_quality = self._assess_baseline_quality(baseline)
        if baseline_quality < 0.5:  # 低质量baseline
            coverage_threshold = 0.20
            residual_threshold = 0.65
        else:
            coverage_threshold = 0.35
            residual_threshold = 0.50
```

#### 修复3: 启用真实Embedding

**文件**: `run_sns_example.py`

```python
# 修改前
embedding_model="dummy"

# 修改后
embedding_model="allenai/specter2"  # 或 "sentence-transformers/all-MiniLM-L6-v2"
```

**同时更新Phase 2**:
```python
class Phase2Pipeline:
    def __init__(self, lm, embedding_model="allenai/specter2"):
        self.embedding_model = embedding_model
        # 实际加载embedding model
        if embedding_model != "dummy":
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model)
```

---

### 短期改进（P1）

#### 改进4: 动态调整Clustering参数

**文件**: `knowledge_storm/sns/modules/phase3_evolution.py`

```python
class Phase3Pipeline:
    def run(self, fit_vectors, papers, baseline):
        # 动态计算min_cluster_size
        num_stressed = sum(1 for fv in fit_vectors if fv.stress_score > 0.3)
        
        if num_stressed < 10:
            logger.warning(f"Only {num_stressed} stressed papers. "
                          f"Clustering may not be meaningful.")
            min_cluster_size = 2
        else:
            min_cluster_size = max(2, num_stressed // 4)
        
        # 如果样本太小，跳过clustering
        if num_stressed < 6:
            logger.info("Sample size too small for clustering. "
                       "Generating global evolution proposals instead.")
            return self._generate_global_proposals(fit_vectors, baseline)
        
        # 正常clustering
        clusters = self.clusterer.cluster_stressed_papers(
            fit_vectors=fit_vectors,
            papers=papers,
            baseline=baseline,
            stress_threshold=0.3,
            min_cluster_size=min_cluster_size  # 动态
        )
```

#### 改进5: 增加Research Paper数量

**文件**: `run_sns_example.py`

```python
# 修改前
top_k_research_papers=10

# 修改后
top_k_research_papers=30  # 增加到30篇

# 同时改进检索策略
def _retrieve_research_papers(self) -> List:
    queries = [
        f"{self.args.topic}",
        f"{self.args.topic} method",
        f"{self.args.topic} approach",
        f"{self.args.topic} deep learning",  # NEW
        f"{self.args.topic} detection",  # NEW
        f"{self.args.topic} generation",  # NEW
        f"recent work on {self.args.topic}",  # NEW
    ]
    # 每个query取更多结果
    # 去重后确保至少有top_k篇
```

---

### 中期改进（P2）

#### 改进6: 改善Taxonomy提取

**文件**: `knowledge_storm/sns/schemas_v2.py`

强化LLM prompt，要求更详细的taxonomy：

```python
class TaxonomyExtractionSignature(dspy.Signature):
    """Extract a hierarchical taxonomy from a review paper.
    
    REQUIREMENTS:
    - At least 3 levels deep (ROOT → Category → Subcategory → Leaf)
    - At least 8 leaf nodes
    - Clear inclusion/exclusion criteria for each node
    - Diverse facets (not just APPLICATION_DOMAIN)
    """
    
    review_title = dspy.InputField(...)
    review_abstract = dspy.InputField(...)
    review_text = dspy.InputField(...)
    
    taxonomy_json = dspy.OutputField(
        desc="JSON taxonomy with AT LEAST 3 levels and 8+ leaf nodes. "
             "Each node must have: name, parent, children. "
             "Ensure diversity: cover methods, datasets, evaluation, applications."
    )
```

#### 改进7: 增强Phase 4 Guidance生成

**文件**: `knowledge_storm/sns/modules/phase4_guidance.py`

```python
class GuidanceGenerator:
    def generate_guidance(self, ...):
        # 即使evolution为空，也生成有意义的guidance
        if not evolution_proposal.operations:
            # 基于high-stress papers生成"emerging topics"
            emerging_topics = self._extract_emerging_topics(
                stressed_papers=[fv for fv in fit_vectors if fv.stress_score > 0.7]
            )
            
            # 生成特定的must-answer questions
            must_answer = [
                f"What are the emerging trends in {topic}?",
                f"Which recent methods are not covered by existing taxonomy?",
                f"What are the limitations of current {main_axis.facet} organization?",
                *[f"How does {t.name} relate to existing categories?" 
                  for t in emerging_topics]
            ]
        
        # 增强evidence cards
        evidence_cards = self._generate_rich_evidence_cards(
            papers=stressed_papers,
            taxonomy=main_axis
        )
```

---

### 长期改进（P3）

#### 改进8: 多数据源支持

扩展检索能力：
- Semantic Scholar API
- Google Scholar (via SerpAPI)
- PubMed (for bio-medical topics)
- ACL Anthology (for NLP topics)

#### 改进9: 自适应阈值学习

基于历史运行结果，学习最优阈值：
- 收集每次运行的fit rate
- 如果fit rate < 10%，降低阈值
- 如果fit rate > 80%，提高阈值
- 使用bayesian optimization调优

#### 改进10: 增强的Clustering

- 使用multiple clustering algorithms（DBSCAN, HDBSCAN, Agglomerative）
- Ensemble clustering
- 基于multiple features（semantic, citation, temporal）

---

## 📋 验证清单

在修复后，使用以下清单验证：

### Phase 1 验证
- [ ] 检索到 ≥10 篇review papers
- [ ] Baseline包含 ≥3 个不同的facets
- [ ] 每个taxonomy ≥3层深度
- [ ] 总共 ≥15 个leaf nodes
- [ ] 通过baseline quality gate检查

### Phase 2 验证
- [ ] FIT rate: 20-60%（健康范围）
- [ ] FORCE_FIT rate: 20-40%
- [ ] UNFITTABLE rate: 20-40%
- [ ] Average stress score: 0.3-0.6
- [ ] Coverage分数分布合理

### Phase 3 验证
- [ ] 识别出 ≥2 个stress clusters
- [ ] 每个cluster ≥2 篇论文
- [ ] 提出 ≥1 个evolution operation
- [ ] Fit gain > 0
- [ ] Edit cost合理

### Phase 4 验证
- [ ] Writing mode判定合理
- [ ] Writing rules不为空
- [ ] Evolution summary包含具体操作
- [ ] Must-answer questions具体
- [ ] Evidence cards丰富

---

## 🔧 快速修复包

为了快速验证修复效果，创建一个修复脚本：

```python
# quick_fix.py
from knowledge_storm.sns import SNSRunner, SNSArguments, SNSLMConfigs

# 快速修复配置
args = SNSArguments(
    topic="deepfake",
    output_dir="./output_fixed",
    top_k_reviews=20,  # 增加 (从15)
    top_k_research_papers=30,  # 增加 (从10)
    min_cluster_size=2,  # 降低 (从3)
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 真实embedding
)

# 运行并检查结果
results = runner.run()

# 验证指标
print(f"Review papers: {len(results.multiview_baseline.views)}")
print(f"Unique facets: {len(set(v.facet_label for v in results.multiview_baseline.views))}")
print(f"FIT rate: {results.statistics['fit_rate']:.2%}")
print(f"Clusters: {len(results.stress_clusters)}")
print(f"Evolution ops: {len(results.evolution_proposal.operations)}")
```

---

## 总结

### 当前状态评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **Phase 1: Baseline** | 2/10 | Review检索严重不足 |
| **Phase 2: Stress Test** | 4/10 | 阈值过严，但逻辑正确 |
| **Phase 3: Evolution** | 0/10 | 完全失效 |
| **Phase 4: Guidance** | 5/10 | 降级但有输出 |
| **整体系统** | 3/10 | 需要重大改进 |

### 修复后预期

应用P0修复后：
- Phase 1: 2/10 → 7/10
- Phase 2: 4/10 → 8/10
- Phase 3: 0/10 → 6/10
- Phase 4: 5/10 → 8/10
- **整体**: 3/10 → 7/10

### 下一步行动

1. **立即**（今天）：
   - 修复Fit Test阈值
   - 启用真实embedding
   - 调整example脚本参数

2. **本周**：
   - 改进Review检索
   - 动态clustering参数
   - 添加sample size检查

3. **下周**：
   - 强化taxonomy提取
   - 增强guidance生成
   - 添加验证测试

---

**报告生成时间**: 2025-12-15  
**分析人员**: Claude (genspark-ai-developer)  
**状态**: ⚠️ 需要紧急修复
