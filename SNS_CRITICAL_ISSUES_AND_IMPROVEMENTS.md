# SNS 系统严重问题分析与改进建议

## 基于 Deepfake 主题运行结果的诊断报告

**分析日期**: 2025-12-15  
**测试主题**: deepfake  
**分析人员**: Claude (genspark-ai-developer)

---

## 🚨 发现的严重问题

### 问题 #1: **所有论文100%不适配** ⚠️⚠️⚠️

#### 现象
```
- Total Papers Analyzed: 5
- FIT: 0 (0.0%)
- FORCE_FIT: 0 (0.0%)
- UNFITTABLE: 10 (100.0%)  ← 所有测试都失败！
- Average Stress Score: 1.000
- Average Unfittable Score: 1.000
```

#### 具体数据
所有5篇论文在2个视角下的10次测试全部标记为 `UNFITTABLE`，FitScore 全部为负值：

| 论文 | View T1 | View T2 |
|------|---------|---------|
| 2306.00863v2 | -0.382 | -0.435 |
| 2305.06564v4 | -0.439 | -0.430 |
| 2505.18587v1 | -0.427 | -0.435 |
| 2511.10212v1 | -0.459 | -0.466 |
| 2412.09921v2 | -0.426 | -0.509 |

#### 根本原因分析

##### 1. **Coverage 分数异常低**
从 `fit_vectors.json` 分析：
```json
"scores": {
  "coverage": 0.056,    // ← 仅5.6%的覆盖率！
  "conflict": 0.071,
  "residual": 0.951,    // ← 95%的新颖性无法匹配
  "fit_score": -0.382
}
```

**问题**:
- Coverage 应该至少达到 0.3-0.5 才合理
- 当前只有 ~5-8% 的覆盖率
- 说明论文内容与分类节点几乎不匹配

**可能原因**:
- ✅ **语义相似度计算有问题**（使用 "dummy" embedding）
- ✅ **词汇相似度计算不足**
- ✅ **节点定义 (NodeDefinition) 质量差**
- ✅ **evidence retrieval 失败**

##### 2. **Residual 分数极高**
```json
"residual": 0.951  // ← 95%的新颖性无法匹配任何叶节点
```

**问题**:
- Residual = 1 - max(cos(novelty_bullet, leaf_embedding))
- 高residual意味着论文的创新点与所有叶节点都不相似
- 说明分类体系太粗糙或embedding质量差

##### 3. **FitScore 公式过于严格**
```
FitScore = coverage - 0.8 * conflict - 0.4 * residual
         = 0.056 - 0.8 * 0.071 - 0.4 * 0.951
         = 0.056 - 0.057 - 0.380
         = -0.381
```

**问题**:
- 当 coverage 很低时，residual 的惩罚（-0.4 * 0.95 = -0.38）占主导
- 导致几乎所有论文都会被标记为 UNFITTABLE

---

### 问题 #2: **Phase 3 没有生成任何演化提案** ⚠️⚠️

#### 现象
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

#### 根本原因

##### 1. **HDBSCAN 聚类失败**
```python
# 5篇论文太少，无法形成有意义的聚类
min_cluster_size = 3  # 默认值
total_stressed_papers = 5  # 所有论文都是stressed
```

**问题**:
- HDBSCAN 需要足够多的样本才能聚类
- 5篇论文可能全部被标记为噪声点（outliers）
- 导致 `stress_clusters = []`

##### 2. **演化规划器没有触发**
```python
# Phase3Pipeline.run()
if len(stress_clusters) == 0:
    logger.info("No stress clusters found, skipping evolution planning")
    return [], EvolutionProposal(operations=[], ...)
```

**问题**:
- 没有聚类 → 没有演化提案
- 即使所有论文都不适配，系统也不会提出结构更新

---

### 问题 #3: **基线质量差** ⚠️

#### 现象
```
- Total Taxonomy Views: 2
- View T1: APPLICATION_DOMAIN (weight=0.400, 4 leaf nodes)
- View T2: APPLICATION_DOMAIN (weight=0.600, 6 leaf nodes)
```

#### 问题

##### 1. **只有2个视角，且facet相同**
```
T1: APPLICATION_DOMAIN (Audio deepfake detection)
T2: APPLICATION_DOMAIN (Malicious deepfakes)
```

**问题**:
- 两个视角的 facet_label 相同（都是 APPLICATION_DOMAIN）
- 违反了"至少2个不同facet"的设计要求
- 缺乏多样性，无法覆盖不同的组织维度

##### 2. **叶节点太少**
```
T1: 4 leaf nodes
T2: 6 leaf nodes
Total: ~10 leaf nodes (去重后可能更少)
```

**问题**:
- 设计要求至少 20-50 个叶节点
- 当前只有 ~10 个节点
- 分类体系太粗糙，无法精细匹配论文

##### 3. **review papers 数量不足**
```
top_k_reviews = 5  // 只检索了5篇review论文
实际提取视角: 2个  // 只有2个成功提取
```

**问题**:
- 检索到的review太少
- 提取成功率低（5篇中只有2篇成功）
- 可能是因为LLM提取失败或review质量差

---

### 问题 #4: **Embedding 模型是 "dummy"** ⚠️⚠️⚠️

#### 现象
```python
# SNSArguments
embedding_model="dummy"  # 使用占位符！
```

#### 影响

这直接导致：
1. **Coverage 计算失败**
   - `coverage = 0.7 * cov_sem + 0.3 * cov_lex`
   - `cov_sem` 依赖 embedding 相似度
   - dummy embedding 会返回随机或固定值

2. **Residual 计算失败**
   - `residual = 1 - max(cos(novelty_bullet, leaf_embedding))`
   - dummy embedding 导致所有相似度都很低

3. **候选节点检索失败**
   - Phase 2 使用 embedding 检索候选叶节点
   - dummy embedding 导致检索到不相关的节点

**这是最严重的问题之一！**

---

### 问题 #5: **Delta Guidance 质量差** ⚠️

#### 现象
```json
// guidance_pack.json
{
  "writing_mode": "ANCHOR_PLUS_DELTA",
  "writing_rules": {
    "do": ["Use main axis structure as the organizational foundation", ...],
    "dont": ["Don't ignore evolution and stress points", ...]
  },
  "outline": [...],  // 只有基于main_axis的章节
  "evolution_summary": [],  // 空的！
  "must_answer_questions": [
    "What are the main organizational dimensions in APPLICATION_DOMAIN?",
    "How has the field evolved beyond existing reviews?"
  ]
}
```

#### 问题

##### 1. **Writing Mode 选择不当**
```
当前: ANCHOR_PLUS_DELTA (使用主轴作为基础)
期望: DELTA_FIRST (因为所有论文都不适配，应该强调新兴趋势)
```

**根据设计文档**:
```
DELTA_FIRST 条件:
- EditCost > 3.0  或
- FitGain > 10.0  或
- StressReduction > 0.6

当前情况:
- 100% 论文不适配
- Average stress = 1.0
- 应该选择 DELTA_FIRST
```

##### 2. **Evolution Summary 为空**
```json
"evolution_summary": []  // 没有任何演化操作！
```

**问题**:
- 由于 Phase 3 没有聚类，没有提出任何结构更新
- Guidance 无法告诉下游系统"需要添加哪些新节点"

##### 3. **Outline 缺乏约束**
```json
"outline": [
  {
    "section": "DeepFake Generation",
    "subsections": [
      {
        "subsection": "...",
        "required_nodes": [],      // 空的！
        "required_citations": [],  // 空的！
        "must_answer": [],         // 空的！
        "evidence_cards": []       // 空的！
      }
    ]
  }
]
```

**问题**:
- 缺少具体的约束和证据
- 下游系统无法知道每个小节应该包含什么

---

## 📋 优先级改进建议

### 🔴 P0: 立即修复（核心功能）

#### 1. **启用真实的 Embedding 模型**

**当前问题**: 使用 `embedding_model="dummy"`

**修复方案**:
```python
# 修改 run_sns_example.py
igfinder2_args = SNSArguments(
    topic=args.topic,
    output_dir=args.output_dir,
    top_k_reviews=args.top_k_reviews,
    top_k_research_papers=args.top_k_research,
    min_cluster_size=2,
    save_intermediate_results=True,
    embedding_model="allenai/specter2",  # ← 改为真实模型
    lambda_regularization=0.8,
)
```

**依赖安装**:
```bash
pip install sentence-transformers
```

**预期改进**:
- Coverage 分数提升至 0.3-0.5
- Residual 分数降低至 0.4-0.6
- FitScore 提升，部分论文可能变为 FIT 或 FORCE_FIT

---

#### 2. **调整 FitScore 阈值和公式**

**当前问题**: 阈值太严格，导致所有论文都 UNFITTABLE

**修复方案 A**: 放宽阈值
```python
# 当前阈值（dataclass_v2.py 或 phase2_stress_test.py）
UNFITTABLE: coverage < 0.45 or conflict > 0.55

# 建议调整
UNFITTABLE: coverage < 0.25 or conflict > 0.7  # 更宽松
FORCE_FIT: 0.25 <= coverage < 0.40 or residual > 0.5
FIT: coverage >= 0.40 and conflict < 0.7 and residual < 0.5
```

**修复方案 B**: 调整公式权重
```python
# 当前公式
fit_score = coverage - 0.8 * conflict - 0.4 * residual

# 建议调整（降低 residual 权重）
fit_score = coverage - 0.6 * conflict - 0.2 * residual
```

**理由**:
- 当 embedding 质量差时，residual 会异常高
- 降低 residual 权重可以让 coverage 和 conflict 发挥更大作用

---

#### 3. **增加 Review 和 Research Paper 数量**

**当前问题**: 样本太少

**修复方案**:
```python
# run_sns_example.py
parser.add_argument(
    '--top-k-reviews',
    type=int,
    default=15,  # ← 从5增加到15
    help='Number of review papers to retrieve'
)
parser.add_argument(
    '--top-k-research',
    type=int,
    default=30,  # ← 从10增加到30
    help='Number of research papers to retrieve'
)
```

**预期改进**:
- 更多视角（3-5个）
- 更多叶节点（20-50个）
- 足够的论文进行聚类（30篇）

---

#### 4. **降低 HDBSCAN min_cluster_size**

**当前问题**: min_cluster_size=3 对小数据集太大

**修复方案**:
```python
# SNSArguments
SNSArguments(
    ...,
    min_cluster_size=2,  # ← 从3降低到2
)
```

或者动态调整：
```python
# Phase3Pipeline.__init__
def __init__(self, lm, min_cluster_size=3, lambda_reg=0.8):
    self.clusterer = StressClusterer(
        min_cluster_size=max(2, min(min_cluster_size, len(papers) // 5))
        # 动态调整：至少2，最多不超过论文数的1/5
    )
```

**预期改进**:
- 能够形成至少1-2个聚类
- Phase 3 可以提出演化操作

---

### 🟡 P1: 重要优化（提升质量）

#### 5. **改进 NodeDefinition 质量**

**问题**: Coverage 低可能是因为节点定义质量差

**修复方案**:
```python
# schemas_v2.py - NodeDefinitionSignature
class NodeDefinitionSignature(dspy.Signature):
    """..."""
    
    # 增强 prompt
    definition = dspy.OutputField(
        desc="""Detailed definition of this node (50-100 words).
        
        REQUIREMENTS:
        - Use concrete technical terms and keywords
        - Include synonyms and related concepts
        - Mention representative papers or methods
        - Be specific, not generic
        
        GOOD example: "Convolutional Neural Networks (CNNs) for image deepfake 
        detection, including ResNet, VGG, and their variants. These methods use 
        2D convolutions to extract spatial features from image frames."
        
        BAD example: "Methods that use neural networks for detection."
        """
    )
```

**预期改进**:
- 更丰富的关键词
- 更高的词汇相似度（cov_lex）
- Coverage 提升 0.1-0.2

---

#### 6. **添加回退机制：强制聚类**

**问题**: HDBSCAN 可能返回空聚类

**修复方案**:
```python
# phase3_evolution.py - StressClusterer
def cluster_stressed_papers(self, fit_vectors, papers, baseline):
    # 尝试 HDBSCAN
    clusters = self._hdbscan_cluster(...)
    
    # 回退：如果没有聚类，使用简单的阈值聚类
    if len(clusters) == 0:
        logger.warning("HDBSCAN found no clusters, using fallback clustering")
        clusters = self._fallback_cluster(fit_vectors, papers, baseline)
    
    return clusters

def _fallback_cluster(self, fit_vectors, papers, baseline):
    """
    回退策略：基于 failure_signature 相似度聚类
    """
    # 按 failure_signature 分组
    # 至少保证每个高度stressed的论文都在某个"cluster"中
    ...
```

---

#### 7. **实现 Writing Mode 自动判定**

**问题**: 当前总是选择 ANCHOR_PLUS_DELTA

**修复方案**:
```python
# phase4_guidance.py - AxisSelector
def select_main_axis_with_mode(self, baseline, reconstruction_scores):
    # 1. 选择主轴（基于 reconstruction_score）
    best_view = max(reconstruction_scores, key=lambda s: s.combined_score)
    
    # 2. 判定写作模式
    if best_view.edit_cost > 3.0 or best_view.fit_gain > 10.0:
        mode = WritingMode.DELTA_FIRST
    elif best_view.stress_reduction > 0.6:
        mode = WritingMode.DELTA_FIRST
    else:
        mode = WritingMode.ANCHOR_PLUS_DELTA
    
    logger.info(f"Selected writing mode: {mode.value}")
    logger.info(f"  Reason: EditCost={best_view.edit_cost:.1f}, "
                f"FitGain={best_view.fit_gain:.1f}, "
                f"StressReduction={best_view.stress_reduction:.2f}")
    
    return best_view, mode
```

**当前问题**:
- reconstruction_scores 可能为空（因为 Phase 3 没有计算）
- 需要添加默认逻辑

---

#### 8. **生成具体的 Outline 约束**

**问题**: evidence_cards, required_citations 都是空的

**修复方案**:
```python
# phase4_guidance.py - GuidanceGenerator
def _generate_outline(self, main_axis, aux_axis, papers, fit_vectors):
    outline = []
    
    for child_node in main_axis.tree.get_children("ROOT"):
        section = Section(
            section=child_node.name,
            subsections=[]
        )
        
        # 找到匹配这个节点的论文
        relevant_papers = self._find_papers_for_node(
            child_node, papers, fit_vectors, main_axis
        )
        
        for paper in relevant_papers[:3]:  # 每个节点最多3篇
            # 提取 evidence card
            evidence_cards = self._extract_evidence_cards(
                paper, child_node, fit_vectors
            )
            
            subsection = Subsection(
                subsection=f"{child_node.name} - {paper.title[:50]}",
                required_nodes=[child_node.path],
                required_citations=[paper.paper_id],
                must_answer=[
                    f"How does {paper.title[:30]}... contribute to {child_node.name}?"
                ],
                evidence_cards=evidence_cards
            )
            section.subsections.append(subsection)
        
        outline.append(section)
    
    return outline
```

---

### 🟢 P2: 增强功能（长期优化）

#### 9. **添加质量检查和警告**

```python
# engine_v2.py - SNSRunner
def _validate_results(self):
    """验证结果质量并输出警告"""
    
    # 检查1: Fit rate
    fit_rate = sum(1 for fv in self.fit_vectors 
                   for fr in fv.fit_reports 
                   if fr.label == FitLabel.FIT) / len(self.fit_vectors)
    
    if fit_rate < 0.1:
        logger.warning(f"⚠️ Very low fit rate: {fit_rate:.1%}")
        logger.warning("  Possible causes:")
        logger.warning("  - Embedding model quality (check embedding_model parameter)")
        logger.warning("  - NodeDefinition quality (review extraction might have failed)")
        logger.warning("  - Baseline diversity (may need more review papers)")
    
    # 检查2: Cluster count
    if len(self.stress_clusters) == 0:
        logger.warning("⚠️ No stress clusters formed")
        logger.warning("  Possible causes:")
        logger.warning("  - Too few stressed papers (need at least 5-10)")
        logger.warning("  - min_cluster_size too large")
        logger.warning("  - Papers too dissimilar (no clear failure patterns)")
    
    # 检查3: Baseline quality
    unique_facets = len(set(v.facet_label for v in self.baseline.views))
    if unique_facets < 2:
        logger.warning(f"⚠️ Only {unique_facets} unique facets in baseline")
        logger.warning("  Recommendation: Increase top_k_reviews parameter")
```

---

#### 10. **添加中间结果可视化**

```python
# 新增工具函数
def visualize_fit_distribution(fit_vectors):
    """生成 fit score 分布图"""
    import matplotlib.pyplot as plt
    
    scores = [fr.scores.fit_score 
              for fv in fit_vectors 
              for fr in fv.fit_reports]
    
    plt.hist(scores, bins=20, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Fit Score')
    plt.ylabel('Count')
    plt.title('Fit Score Distribution')
    plt.legend()
    plt.savefig('fit_score_distribution.png')
    logger.info("Saved fit score distribution to fit_score_distribution.png")
```

---

#### 11. **支持增量更新**

```python
# 支持在已有baseline基础上增量添加论文
def incremental_run(self, new_papers: List[ResearchPaper]):
    """增量运行：只测试新论文，不重新构建baseline"""
    
    # Phase 2: 只测试新论文
    new_fit_vectors = self.phase2.run(new_papers, self.baseline)
    
    # 合并到现有结果
    self.fit_vectors.extend(new_fit_vectors)
    
    # Phase 3: 重新聚类（使用所有论文）
    self.stress_clusters, self.evolution_proposal = self.phase3.run(
        self.fit_vectors, self.research_papers + new_papers, self.baseline
    )
    
    # Phase 4: 更新guidance
    ...
```

---

## 🎯 推荐的修复顺序

### 第一步：立即修复（今天）
1. ✅ 启用 SPECTER2 embedding
2. ✅ 放宽 FitScore 阈值
3. ✅ 增加论文数量（top_k_reviews=15, top_k_research=30）
4. ✅ 降低 min_cluster_size=2

### 第二步：质量提升（本周）
5. ✅ 改进 NodeDefinition prompt
6. ✅ 添加聚类回退机制
7. ✅ 实现 Writing Mode 自动判定
8. ✅ 添加质量检查警告

### 第三步：功能增强（下周）
9. ✅ 生成具体的 Outline 约束
10. ✅ 添加可视化工具
11. ✅ 支持增量更新

---

## 📊 预期改善效果

### 修复前（当前状态）
```
✗ FIT: 0%
✗ FORCE_FIT: 0%
✗ UNFITTABLE: 100%
✗ Stress Clusters: 0
✗ Evolution Operations: 0
✗ Guidance Quality: Low
```

### 修复后（预期）
```
✓ FIT: 20-40%
✓ FORCE_FIT: 30-40%
✓ UNFITTABLE: 20-50%
✓ Stress Clusters: 1-3
✓ Evolution Operations: 2-5
✓ Guidance Quality: Medium-High
```

---

## 🔧 快速修复脚本

创建一个快速修复配置文件：

```python
# quick_fix_config.py
from knowledge_storm.sns import SNSArguments

def get_improved_args(topic, output_dir):
    """返回改进后的配置"""
    return SNSArguments(
        topic=topic,
        output_dir=output_dir,
        
        # 增加样本数量
        top_k_reviews=15,           # 从5增加到15
        top_k_research_papers=30,   # 从10增加到30
        
        # 降低聚类阈值
        min_cluster_size=2,         # 从3降低到2
        
        # 启用真实embedding
        embedding_model="allenai/specter2",  # 不再使用dummy
        
        # 其他参数
        save_intermediate_results=True,
        lambda_regularization=0.8,
    )
```

使用方式：
```python
# 在 run_sns_example.py 中
from quick_fix_config import get_improved_args

args = get_improved_args(args.topic, args.output_dir)
runner = SNSRunner(args=args, lm_configs=lm_configs, rm=rm)
```

---

## 📝 总结

### 当前最严重的问题
1. 🔴 **Embedding 模型是 dummy**（导致所有分数不准确）
2. 🔴 **FitScore 阈值太严格**（导致100%不适配）
3. 🔴 **样本数量太少**（导致无法聚类和演化）

### 最紧急的修复
1. 启用 SPECTER2
2. 调整阈值
3. 增加论文数量

### 预期改善
- FIT rate: 0% → 20-40%
- 聚类数量: 0 → 1-3
- 演化操作: 0 → 2-5
- 整体可用性: ❌ → ✅

---

**报告完成时间**: 2025-12-15  
**下一步行动**: 实施 P0 修复，重新测试系统
