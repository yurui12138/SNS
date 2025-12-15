# IG-Finder 2.0 代码改进方案

## 核心问题总结

根据您提供的完整设计方案，当前代码存在以下**关键性**偏离：

### 🔴 关键问题 1：Phase 3 主轴选择逻辑错误

**设计要求：**
```
先对每个视角做重构 → 计算重构后评分 → 根据评分选择写作模式
Score_i = α·FitGain_i + β·Stress_i + γ·Coverage_i - λ·EditCost_i
```

**当前实现（错误）：**
```python
# phase4_guidance.py 第88行
score = 0.6 * fit_rate + 0.3 * stability + 0.1 * coverage
```
这会**偏向未崩溃视角**，违背"Delta-first"的设计初衷。

**改进方案：**
1. 在 Phase 3 中为每个视角都计算重构方案
2. 计算每个重构方案的 FitGain、Stress、Coverage、EditCost
3. 使用新公式选择最优视角
4. 判定写作模式（Delta-first vs Anchor+Delta）

---

### 🔴 关键问题 2：缺少写作模式字段

**设计要求：**
- `main_axis_mode`: DELTA_FIRST 或 ANCHOR_PLUS_DELTA
- 不同模式生成不同的guidance结构

**当前实现：**
- DeltaAwareGuidance 缺少 `main_axis_mode` 字段
- 没有根据模式差异化生成逻辑

**改进方案：**
1. 在 dataclass_v2.py 中添加 WritingMode 枚举
2. 在 DeltaAwareGuidance 中添加 main_axis_mode 字段
3. 在 Phase 4 中根据模式生成不同结构的 guidance

---

### 🔴 关键问题 3：Phase 2 评分公式需确认

**设计要求（明确版）：**
```python
Coverage = 0.7 * semantic_sim + 0.3 * jaccard
Conflict = max(NLI_contradiction_probs)
Residual = 1 - max(cos_sim(novelty_bullet, leaf))
FitScore = Coverage - 0.8*Conflict - 0.4*Residual

# 标签判定
if Coverage < 0.45 or Conflict > 0.55:
    label = UNFITTABLE
elif Residual > 0.45:
    label = FORCE_FIT
else:
    label = FIT
```

**需要确认：**
- 当前代码中的权重是否与此一致
- 阈值是否准确

---

### 🟡 重要问题 4：缺少 writing_rules

**设计要求：**
```json
{
  "writing_rules": {
    "do": [
      "必须在开头明确说明本综述相对现有综述的新增认知维度",
      "对于新增节点，必须解释为何现有分类无法容纳",
      ...
    ],
    "dont": [
      "不要简单罗列新论文而不说明结构演化",
      "不要使用'近期研究表明'这类模糊表述",
      ...
    ]
  }
}
```

**当前实现：**
- 没有 writing_rules 字段

---

### 🟡 重要问题 5：Phase 1 缺少基线质量闸门

**设计要求：**
```python
unique_facets = len(set(view.facet_label for view in views))
max_facet_ratio = max(facet_counts.values()) / len(views)

if unique_facets < 2 or max_facet_ratio > 0.6:
    # 触发补视角策略
    extra_view = induce_view_from_papers(papers)
    views.append(extra_view)
```

**当前实现：**
- 没有质量闸门检查
- 没有补视角策略

---

## 详细改进步骤

### 步骤 1：更新数据结构（dataclass_v2.py）

```python
# 添加写作模式枚举
class WritingMode(Enum):
    DELTA_FIRST = "DELTA_FIRST"  # 崩溃视角重构后作主轴
    ANCHOR_PLUS_DELTA = "ANCHOR_PLUS_DELTA"  # 稳定视角主轴+崩溃视角辅轴

# 添加视角评分数据类
@dataclass
class ViewReconstructionScore:
    view_id: str
    fit_gain: float  # 重构后适配改善
    stress_score: float  # 该视角压力强度
    coverage: float  # 重构后覆盖度
    edit_cost: float  # 重构编辑代价
    total_score: float  # 综合评分
    operations: List[EvolutionOperation]  # 该视角的重构操作

# 添加 writing_rules
@dataclass
class WritingRules:
    do: List[str]
    dont: List[str]
    
    def to_dict(self) -> Dict:
        return {"do": self.do, "dont": self.dont}

# 更新 DeltaAwareGuidance
@dataclass
class DeltaAwareGuidance:
    topic: str
    main_axis_mode: WritingMode  # 新增
    main_axis: TaxonomyView
    aux_axis: Optional[TaxonomyView]
    main_axis_rationale: str  # 新增：为何选择此视角为主轴
    aux_axis_rationale: Optional[str]  # 新增
    outline: List[Section]
    evolution_summary: List[EvolutionSummaryItem]
    must_answer_questions: List[str]
    writing_rules: WritingRules  # 新增
```

### 步骤 2：修正 Phase 3 重构逻辑（phase3_evolution.py）

在 `EvolutionPlanner` 中添加：

```python
def compute_all_views_reconstruction(
    self,
    clusters: List[StressCluster],
    baseline: MultiViewBaseline,
    fit_vectors: List[FitVector],
    lambda_reg: float = 0.8
) -> Dict[str, ViewReconstructionScore]:
    """
    为每个视角计算重构方案和评分
    
    Returns:
        Dict mapping view_id to ViewReconstructionScore
    """
    view_scores = {}
    
    for view in baseline.views:
        # 计算该视角的压力强度
        stress_score = self._compute_view_stress(view, fit_vectors)
        
        # 为该视角生成重构操作
        operations = self._generate_view_operations(view, clusters, fit_vectors)
        
        # 计算 FitGain（模拟重构后的改善）
        fit_gain = sum(op.fit_gain for op in operations)
        
        # 计算编辑代价
        edit_cost = sum(op.edit_cost for op in operations)
        
        # 计算重构后覆盖度
        coverage = self._estimate_reconstructed_coverage(view, operations)
        
        # 综合评分（这才是正确的公式）
        alpha, beta, gamma = 0.4, 0.3, 0.2
        total_score = (alpha * fit_gain + 
                      beta * stress_score + 
                      gamma * coverage - 
                      lambda_reg * edit_cost)
        
        view_scores[view.view_id] = ViewReconstructionScore(
            view_id=view.view_id,
            fit_gain=fit_gain,
            stress_score=stress_score,
            coverage=coverage,
            edit_cost=edit_cost,
            total_score=total_score,
            operations=operations
        )
    
    return view_scores
```

### 步骤 3：修正 Phase 4 主轴选择（phase4_guidance.py）

```python
class AxisSelector:
    def select_main_axis_with_mode(
        self,
        baseline: MultiViewBaseline,
        view_scores: Dict[str, ViewReconstructionScore],
        min_coverage_threshold: float = 0.6
    ) -> Tuple[TaxonomyView, WritingMode, str]:
        """
        先重构再选择，并判定写作模式
        
        Returns:
            (main_axis, mode, rationale)
        """
        # 找到评分最高的视角
        best_view_id = max(view_scores.items(), 
                          key=lambda x: x[1].total_score)[0]
        best_score = view_scores[best_view_id]
        best_view = baseline.get_view_by_id(best_view_id)
        
        # 判定写作模式
        if best_score.stress_score > 0.5 and best_score.coverage >= min_coverage_threshold:
            # Delta-first：崩溃视角重构后仍能提供良好覆盖
            mode = WritingMode.DELTA_FIRST
            rationale = (
                f"视角 {best_view.facet_label.value} 虽有结构压力"
                f"(stress={best_score.stress_score:.2f})，但重构后"
                f"可获得最大认知增量(FitGain={best_score.fit_gain:.2f})"
                f"且覆盖度充足(coverage={best_score.coverage:.2f})，"
                f"因此采用此视角的重构版本作为主轴，优先展示新增认知。"
            )
        else:
            # Anchor+Delta：选择稳定视角做锚定
            # 找压力小、覆盖好的视角
            stable_views = [(vid, score) for vid, score in view_scores.items()
                           if score.stress_score < 0.3 and score.coverage >= min_coverage_threshold]
            
            if stable_views:
                stable_view_id = max(stable_views, key=lambda x: x[1].coverage)[0]
                best_view = baseline.get_view_by_id(stable_view_id)
                mode = WritingMode.ANCHOR_PLUS_DELTA
                rationale = (
                    f"视角 {best_view.facet_label.value} 结构稳定"
                    f"(stress={view_scores[stable_view_id].stress_score:.2f})"
                    f"且覆盖全面(coverage={view_scores[stable_view_id].coverage:.2f})，"
                    f"采用其作为锚定主轴，用崩溃视角作为贯穿辅轴提供批判视角。"
                )
            else:
                # Fallback
                mode = WritingMode.DELTA_FIRST
                rationale = "使用评分最高视角的重构版本。"
        
        return best_view, mode, rationale
```

### 步骤 4：Phase 2 确认评分公式（phase2_stress_test.py）

需要检查 FitTester 中的权重和阈值：

```python
# 确认这些值与设计一致
fit_score = coverage_score - 0.8 * conflict_score - 0.4 * residual_score

# 确认阈值
if coverage_score < 0.45 or conflict_score > 0.55:
    label = FitLabel.UNFITTABLE
elif residual_score > 0.45:
    label = FitLabel.FORCE_FIT
else:
    label = FitLabel.FIT
```

### 步骤 5：Phase 1 添加基线质量闸门（phase1_multiview_baseline.py）

在 `MultiViewBaseline` 构建后添加：

```python
def validate_baseline_quality(self, baseline: MultiViewBaseline) -> MultiViewBaseline:
    """检查基线质量，必要时补视角"""
    
    facet_counts = {}
    for view in baseline.views:
        facet_counts[view.facet_label] = facet_counts.get(view.facet_label, 0) + 1
    
    unique_facets = len(facet_counts)
    max_facet_ratio = max(facet_counts.values()) / len(baseline.views)
    
    if unique_facets < 2:
        logger.warning(f"基线质量不足：仅有 {unique_facets} 个独特视角")
        # 触发补视角（简化版：发出警告）
        logger.info("建议：增加更多不同facet的综述")
    
    if max_facet_ratio > 0.6:
        dominant_facet = max(facet_counts.items(), key=lambda x: x[1])[0]
        logger.warning(f"基线质量不足：{dominant_facet} 占比 {max_facet_ratio:.1%}")
        logger.info("建议：增加其他facet的综述以平衡视角")
    
    return baseline
```

### 步骤 6：添加 writing_rules 生成逻辑

```python
def generate_writing_rules(mode: WritingMode) -> WritingRules:
    """根据写作模式生成规则"""
    
    common_do = [
        "在开头明确说明本综述相对现有综述的新增认知维度",
        "对于新增分类节点，必须解释为何现有分类无法容纳这些工作",
        "引用论文时必须说明其在结构演化中的角色（是否触发重构）",
    ]
    
    common_dont = [
        "不要简单罗列新论文而不说明结构演化原因",
        "不要使用'近期研究表明'等模糊时间表述",
        "不要在未说明必要性的情况下引入新分类",
    ]
    
    if mode == WritingMode.DELTA_FIRST:
        do_rules = common_do + [
            "优先展示导致结构变化的新工作",
            "说明新分类如何反映领域认知的转变",
        ]
        dont_rules = common_dont + [
            "不要过度强调稳定不变的部分",
        ]
    else:  # ANCHOR_PLUS_DELTA
        do_rules = common_do + [
            "先建立稳定的组织框架，再引入变化",
            "在稳定视角下讨论新工作时，明确指出哪些符合、哪些挑战现有分类",
        ]
        dont_rules = common_dont + [
            "不要让新增内容割裂原有结构的连贯性",
        ]
    
    return WritingRules(do=do_rules, dont=dont_rules)
```

---

## 实施优先级

### 阶段 1：核心逻辑修正（必须完成）
1. ✅ 更新数据结构（添加 WritingMode, ViewReconstructionScore, WritingRules）
2. ✅ 修正 Phase 3 主轴选择逻辑
3. ✅ 更新 Phase 4 根据模式生成 guidance

### 阶段 2：质量保证（重要）
4. ✅ 确认 Phase 2 评分公式和阈值
5. ✅ 添加 Phase 1 基线质量闸门

### 阶段 3：完善功能（可选）
6. ⏸ 实现完整的补视角策略（较复杂，可后续优化）
7. ⏸ 添加 validation 步骤（可后续优化）

---

## 测试验证

完成改进后，需要验证：
1. Phase 3 确实为每个视角计算重构方案
2. 主轴选择基于重构后评分，而非原始 FIT rate
3. 能正确判定 Delta-first 和 Anchor+Delta 模式
4. guidance_pack.json 包含所有必需字段
5. 基线质量闸门能触发警告

---

## 预期效果

改进后的系统将：
- ✅ **正确实现"Delta-first"设计理念**：优先考虑有认知增量的崩溃视角
- ✅ **提供明确的写作模式指导**：下游系统知道用哪种策略组织综述
- ✅ **输出机器可执行的约束包**：包含结构化的 writing_rules
- ✅ **确保基线质量**：避免单一视角主导

这些改进将使 IG-Finder 2.0 **真正符合论文设计方案**，而不仅是实现了类似的功能。
