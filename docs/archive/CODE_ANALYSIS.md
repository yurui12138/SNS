# IG-Finder 2.0 代码分析：与设计目标的差异

## 关键差异点分析

### 1. Phase 1: 多视角基线构建

**设计要求：**
- 基线质量闸门：unique(facet) < 2 或最大facet占比 > 0.6 时触发补视角
- 视角权重计算：w ∝ Recency × Quality × Coverage
- 必须避免单一视角主导

**当前实现：**
✅ 已实现视角权重计算
❌ 缺少基线质量闸门检查
❌ 缺少补视角策略

### 2. Phase 2: Fit Test 判定逻辑

**设计要求：**
- Coverage = 0.7×semantic + 0.3×lexical
- Conflict = max P(contradiction) 使用NLI
- Residual = 1 - max cos(novelty, leaf)
- FitScore = Coverage - 0.8×Conflict - 0.4×Residual
- 标签判定：Coverage<0.45 或 Conflict>0.55 → UNFITTABLE
              Residual>0.45 → FORCE_FIT
              否则 → FIT

**当前实现：**
✅ 三因子评分框架已实现
❌ 阈值与权重可能与设计不一致
❌ 需要验证确切公式

### 3. Phase 3: 主轴选择逻辑（关键问题）

**设计要求（纠正版）：**
- **先对所有视角做重构**，得到每个视角的重构版本 T_i'
- Score_i = α·FitGain + β·Stress + γ·Coverage - λ·EditCost
- 选择两种写作模式：
  * Delta-first：崩溃视角重构后Score最高 → 作为主轴
  * Anchor+Delta：稳定视角做主轴，崩溃视角做辅轴

**当前实现：**
❌ **严重问题**：当前实现是"先选主轴再重构"
❌ 主轴选择公式：0.6×FIT_rate + 0.3×Stability + 0.1×Coverage
   这会偏向未崩溃视角，违背设计目标
❌ 没有实现"Delta-first vs Anchor+Delta"两种模式选择

### 4. Phase 4: GuidanceCompiler 输出

**设计要求：**
- 输出 guidance_pack.json（机器可执行约束）
- 包含：main_axis_mode, taxonomy_v2, outline_constraints, writing_rules
- 三步闭环：Deterministic summary → LLM synthesis → Validation

**当前实现：**
✅ 有 DeltaAwareGuidance 数据结构
❌ 缺少 main_axis_mode（Delta-first / Anchor+Delta）
❌ 缺少 writing_rules.do/dont
❌ 缺少 validation 步骤

## 需要改进的优先级

### 🔴 高优先级（影响核心逻辑）
1. **Phase 3: 修正主轴选择为"先重构再选择"**
2. **Phase 3: 实现两种写作模式判定**
3. **Phase 2: 确认 FitScore 公式与阈值**
4. **Phase 1: 添加基线质量闸门**

### 🟡 中优先级（完善功能）
5. Phase 4: 添加 main_axis_mode 字段
6. Phase 4: 添加 writing_rules
7. Phase 1: 实现补视角策略

### 🟢 低优先级（优化）
8. Phase 4: 添加 validation 步骤
9. 添加更多日志和调试信息

## 改进计划

我将按以下顺序修改代码：

1. **修改 Phase 3 的主轴选择逻辑** (phase3_evolution.py)
   - 为每个视角计算重构版本
   - 使用新的评分公式
   - 判定写作模式

2. **更新 Phase 4 生成逻辑** (phase4_guidance.py)
   - 添加 main_axis_mode
   - 根据模式生成不同的 guidance
   - 添加 writing_rules

3. **修正 Phase 2 的评分细节** (phase2_stress_test.py)
   - 确认公式权重
   - 确认阈值设置

4. **增强 Phase 1 的质量控制** (phase1_multiview_baseline.py)
   - 添加基线质量闸门
   - 实现补视角策略（可选）

