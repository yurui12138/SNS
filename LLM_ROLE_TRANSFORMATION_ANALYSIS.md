# IG-Finder 2.0 LLM角色转换分析表

## 设计哲学

**核心原则**: 由于LLM固有的幻觉和不可解释性，我们将原方法中所有LLM进行判定的步骤全部替换为可复现、可计算的客观指标进行衡量，而让LLM不再负责判断和推理，只负责信息抽取与整合。

---

## LLM职责边界

### ✅ LLM只负责（Information Extraction）:
1. **Extraction**: Claims, node definitions, facet labels
2. **Evidence Locating**: Original text spans
3. **Candidate Generation**: Proposed node names, definitions, explanations

### ✅ 确定性算法负责（Deterministic Judgment）:
1. **Candidate Retrieval**: Top-K leaves (embedding similarity)
2. **Scoring**: Coverage / Conflict / Residual (formulas)
3. **Label Determination**: FIT / FORCE_FIT / UNFITTABLE (thresholds)
4. **Clustering**: HDBSCAN on failure signatures
5. **Type Determination**: Strong / Facet-dependent / Stable (rules)
6. **Evolution Selection**: argmax (FitGain - λ·EditCost)
7. **Axis Selection**: argmax (weighted scores)

---

## 详细步骤对比表

| 步骤名称 | 改进前（LLM判断方式） | 改进后（客观指标计算方式） |
|---------|---------------------|------------------------|
| **Phase 1: 多视角基线构建** | | |
| 1.1 综述检索排序 | ❌ 让LLM评价综述质量和相关性<br>• "Rate this review's quality from 1-10"<br>• "Is this review relevant to the topic?"<br>• 主观性强，不可复现 | ✅ 可复现启发式公式<br>• `Quality = log(1 + citations)`<br>• `Recency = exp(-0.15 × (Year_now - Year_review))`<br>• `Coverage = num_leaf_nodes / 50`<br>• `Score = 0.4×Recency + 0.4×Quality + 0.2×Coverage`<br>• 完全客观，可审计 |
| 1.2 视角权重计算 | ❌ LLM对比多个综述权威性<br>• "Which review is more authoritative?"<br>• "Rate the importance of this perspective"<br>• 跨综述比较无标准 | ✅ 确定性归一化公式<br>• `w_raw = Recency(r) × Quality(r) × Coverage(r)`<br>• `w_i = w_raw_i / Σ_j w_raw_j`<br>• 基于客观指标的可复验权重 |
| 1.3 Facet标签分类 | ⚠️ LLM从枚举中选择（保留，但受约束）<br>• LLM输出: `facet_label` ∈ {MODEL_ARCHITECTURE, TRAINING_PARADIGM, ...}<br>• 但限制在固定枚举中<br>• Temperature=0确保稳定性 | ✅ 保留LLM分类（信息抽取角色）<br>• LLM仅做结构化信息抽取<br>• 输出必须符合预定义枚举<br>• 使用固定JSON schema验证<br>• **不涉及判断，只是结构化标注** |
| 1.4 Taxonomy树抽取 | ⚠️ LLM从综述文本中提取结构<br>• 自由生成层级结构<br>• 可能产生幻觉节点 | ✅ LLM受约束抽取（信息提取角色）<br>• 强制JSON schema<br>• 要求evidence spans追溯原文<br>• 节点定义必须有出处引用<br>• **LLM是抽取器，不是创造者** |
| **Phase 2: 多视角压力测试** | | |
| 2.1 候选节点匹配 | ❌ LLM自由判断论文属于哪个节点<br>• "Which taxonomy node best fits this paper?"<br>• 主观判断<br>• 不同模型结果差异大 | ✅ Top-K embedding检索（确定性）<br>• `sim = cosine(paper_vec, leaf_vec)`<br>• 使用SPECTER2/SciNCL预训练模型<br>• 返回Top-5候选节点<br>• **完全可复现的相似度排序** |
| 2.2 覆盖度(Coverage)评分 | ❌ LLM评价"这个节点能否覆盖论文内容"<br>• "Rate how well this node covers the paper (1-10)"<br>• 主观量化<br>• 无法审计 | ✅ 组合相似度公式<br>• `cov_sem = cosine(paper_vec, leaf_vec)`<br>• `cov_lex = Jaccard(keywords_paper, keywords_leaf)`<br>• `Coverage = 0.7 × cov_sem + 0.3 × cov_lex`<br>• **数值可追溯到向量和关键词** |
| 2.3 冲突度(Conflict)评分 | ❌ LLM判断"论文是否违反节点边界条件"<br>• "Does this paper violate the exclusion criteria?"<br>• Yes/No主观判断<br>• 幻觉风险高 | ✅ 预训练NLI模型（客观推理）<br>• 使用DeBERTa-MNLI等现成模型<br>• `Conflict = max(P_NLI(contradiction \| claim, exclusion))`<br>• 对每个exclusion_criteria计算矛盾概率<br>• **基于训练好的模型，可复现** |
| 2.4 残差(Residual)评分 | ❌ LLM判断"论文创新点是否被节点覆盖"<br>• "Is the novelty captured by this node?"<br>• 创新性判断极主观 | ✅ Embedding最大相似度公式<br>• `Residual = 1 - max(cos(novelty_vec, leaf_vec))`<br>• 遍历所有novelty bullets<br>• 取最大相似度作为"可表达度"<br>• **完全基于向量计算** |
| 2.5 FIT标签判定 | ❌ LLM综合判断"论文是否适配此节点"<br>• "Is this paper FIT / FORCE_FIT / UNFITTABLE?"<br>• 黑盒决策<br>• 无法解释 | ✅ 确定性阈值规则<br>• `if Coverage < 0.25 or Conflict > 0.70: UNFITTABLE`<br>• `elif Residual > 0.60: FORCE_FIT`<br>• `else: FIT`<br>• **可审计的条件分支** |
| 2.6 FitScore计算 | ❌ LLM打分"整体适配度(1-10)"<br>• 无公式<br>• 主观量化 | ✅ 加权组合公式<br>• `FitScore = Coverage - 0.8×Conflict - 0.4×Residual`<br>• 权重可调参（grid search）<br>• **数学可解释** |
| **Phase 3: 压力聚类与演化** | | |
| 3.1 压力论文聚类 | ❌ LLM分组"哪些论文失败模式相似"<br>• 要求LLM理解失败模式<br>• 分组不稳定 | ✅ HDBSCAN无监督聚类<br>• `failure_sig = facet:path + lost_novelty + keywords`<br>• `clusterer = HDBSCAN(min_cluster_size=2, metric='cosine')`<br>• 基于embedding自动分组<br>• **确定性算法** |
| 3.2 聚类类型判定 | ❌ LLM判断"这是Strong Shift还是Facet-dependent?"<br>• 要求LLM理解结构演化类型<br>• 定义模糊 | ✅ 跨视角一致性规则<br>• `U(C) = mean(unfittable_score_per_paper)`<br>• Strong Shift: `U(C) > 0.55`<br>• Facet-dependent: 存在视角失败率>0.6 且 其他<0.2<br>• **基于统计阈值** |
| 3.3 演化操作选择 | ❌ LLM提议"应该ADD/SPLIT/RENAME哪些节点"<br>• 结构更新完全由LLM决定<br>• 最小性无保证 | ✅ 优化目标函数<br>• `T' = argmax(FitGain - λ×EditCost)`<br>• FitGain = Δ(#FIT - #FORCE_FIT - 2×#UNFITTABLE)<br>• EditCost = Σ(ADD:1.0, SPLIT:2.0, RENAME:0.5)<br>• **可证明的最小必要变更** |
| 3.4 新节点命名 | ⚠️ LLM生成节点名称和定义<br>• "Suggest a name for a new node"<br>• 创造性任务 | ✅ LLM做候选生成（信息整合角色）<br>• LLM基于clustered papers提取共性<br>• 生成候选名称+定义<br>• 但**选择**由FitGain公式决定<br>• **LLM生成，算法筛选** |
| 3.5 重构评分计算 | ❌ LLM评价"这个重构方案好不好"<br>• 主观评价方案优劣 | ✅ ViewReconstructionScore公式<br>• `score = α×FitGain + β×StressReduction + γ×Coverage - λ×EditCost`<br>• α=0.4, β=0.3, γ=0.2, λ=0.8<br>• **多目标加权，可调参** |
| **Phase 4: Delta感知指导生成** | | |
| 4.1 主轴选择 | ❌ LLM判断"哪个视角应该作为主轴"<br>• "Which perspective is most suitable as main axis?"<br>• 主观性强 | ✅ Reconstruction Score排序<br>• `main_axis = argmax_view(reconstruction_score)`<br>• 基于Phase 3计算的各视角重构评分<br>• **数值化排序** |
| 4.2 写作模式判定 | ❌ LLM决定"应该用Delta-first还是Anchor+Delta"<br>• 要求LLM理解写作策略 | ✅ 确定性条件规则<br>• `if EditCost > 3.0 or FitGain > 10.0: DELTA_FIRST`<br>• `elif StressReduction > 0.6: DELTA_FIRST`<br>• `else: ANCHOR_PLUS_DELTA`<br>• **基于客观指标的分支判断** |
| 4.3 大纲生成 | ⚠️ LLM生成章节结构<br>• 创造性任务<br>• 结构自由度高 | ✅ LLM生成 + 算法约束（结构化角色）<br>• LLM基于main_axis tree生成outline<br>• 但必须包含required_nodes（从演化操作提取）<br>• 必须包含evidence_cards（从fit_vectors提取）<br>• **LLM创作，算法验证完整性** |
| 4.4 Writing Rules生成 | ⚠️ LLM生成写作规则<br>• "What writing guidelines should be followed?"<br>• 规则任意性高 | ✅ 模板化规则 + 模式匹配<br>• 根据writing_mode选择规则模板<br>• DELTA_FIRST模板: "必须在开头说明新增认知维度"<br>• ANCHOR_PLUS_DELTA模板: "使用主轴结构，标注更新"<br>• **预定义模板 + 参数填充** |
| **跨阶段：质量验证** | | |
| 5.1 基线质量检查 | ❌ LLM判断"baseline质量是否足够"<br>• 无标准 | ✅ 启发式阈值检查<br>• `if unique_facets < 2: WARNING`<br>• `if total_leaf_nodes < 20: WARNING`<br>• `if fit_rate < 0.1: WARNING`<br>• **可自动化的质量闸门** |
| 5.2 演化必要性判断 | ❌ LLM判断"是否需要更新结构"<br>• 主观决策 | ✅ 统计阈值规则<br>• `if operations == 0 and unfittable_rate > 0.5: WARNING`<br>• 基于实际数据触发警告<br>• **数据驱动的判断** |

---

## 实施保障措施

### 可复现性保证（Reproducibility）:
1. **Temperature = 0** for all LLM calls
2. **Fixed JSON schema** for all LLM outputs
3. **Multi-model consistency check** (optional, GPT-4 vs Claude vs Gemini)
4. **All intermediate results cached** in JSON

### 可审计性保证（Auditability）:
1. **Evidence tracing**: 所有节点定义/claims必须引用原文spans
2. **Score breakdown**: 所有分数可分解为向量/关键词/公式
3. **Threshold calibration**: 阈值通过验证集grid search调参，而非人工拍定
4. **Deterministic algorithms**: 聚类/排序/选择使用确定性算法

### 客观性保证（Objectivity）:
1. **No free-form rating**: 杜绝"rate 1-10"类主观打分
2. **No binary judgment**: 避免"yes/no"黑盒判断
3. **Formula-based**: 所有判定可追溯到数学公式或规则
4. **Ablation study**: 测试不同阈值/权重的敏感性

---

## 关键设计对比

### ❌ 传统方法（LLM判断）:
```
输入论文 → LLM判断"这是创新吗？" → 输出创新论文列表
         ↑ 黑盒、主观、不可复现
```

### ✅ IG-Finder 2.0（客观测量）:
```
输入论文 → 多视角适配测试 → 计算Coverage/Conflict/Residual
         ↓ 白盒、客观、可复现
         确定性阈值判定 → FIT/FORCE_FIT/UNFITTABLE标签
         ↓
         HDBSCAN聚类 → 识别失败模式
         ↓
         优化目标函数 → 最小必要结构更新
         ↓
         重构评分排序 → Delta感知写作指导
```

---

## LLM角色总结

| 角色类型 | 是否使用LLM | 职责范围 | 可复现性 |
|---------|------------|---------|---------|
| **信息抽取** | ✅ LLM | 从PDF提取claims, definitions, evidence spans | 高（固定schema + temp=0） |
| **结构化标注** | ✅ LLM | Facet分类, tree抽取（受约束） | 中高（枚举约束 + 证据追溯） |
| **候选生成** | ✅ LLM | 生成节点名称/定义候选 | 中（算法做最终筛选） |
| **数值判断** | ❌ 算法 | Coverage/Conflict/Residual评分 | 完全（数学公式） |
| **标签判定** | ❌ 算法 | FIT/FORCE_FIT/UNFITTABLE | 完全（阈值规则） |
| **聚类分组** | ❌ 算法 | HDBSCAN压力论文聚类 | 完全（确定性算法） |
| **优化选择** | ❌ 算法 | 演化操作选择、主轴选择 | 完全（目标函数优化） |
| **模式判定** | ❌ 算法 | 写作模式、聚类类型判定 | 完全（条件规则） |

---

## 结论

**IG-Finder 2.0的核心创新**在于将传统依赖LLM主观判断的步骤系统性地替换为：

1. **可复现的数学公式**（Coverage, Conflict, Residual, FitScore）
2. **确定性算法**（HDBSCAN, embedding retrieval, argmax优化）
3. **明确阈值规则**（FIT标签判定、写作模式选择）
4. **证据可追溯**（所有判定必须关联到原文spans或数值计算）

**LLM仅负责"不需要判断的信息抽取任务"**，而所有"需要判断和推理的步骤"由客观、可审计的算法完成。这确保了系统的**科学性、可复现性和可解释性**。
