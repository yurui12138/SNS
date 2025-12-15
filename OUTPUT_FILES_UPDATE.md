# SNS 输出文件更新说明

## 问题描述

用户报告系统输出缺少关键的机器可读文件：

**之前的输出** ❌:
```
./output/
├── multiview_baseline.json
├── fit_vectors.json
├── igfinder2_report.md       ← 人类可读，但名称过时
└── igfinder2_results.json    ← 完整结果，但不是主要接口
```

**设计要求** ✅:
1. `audit_report.md` - 人类可读的审计报告
2. `guidance_pack.json` - **机器可读的结构化指导包**（缺失！）

---

## 修复方案

### 1. 新增 `guidance_pack.json`

**目的**: 为下游自动化综述生成系统提供机器可执行的指导。

**内容结构**:
```json
{
  // 核心元数据
  "topic": "research_topic",
  "generation_date": "ISO-8601",
  "schema_version": "2.0",
  
  // 写作策略（关键！）
  "writing_mode": "DELTA_FIRST | ANCHOR_PLUS_DELTA",
  "writing_rules": {
    "do": ["具体的写作建议"],
    "dont": ["应该避免的做法"]
  },
  
  // 分类体系（已应用演化）
  "taxonomy": {
    "main_axis": {
      "facet": "FACET_LABEL",
      "tree": { /* 完整树结构 */ },
      "weight": 0.35
    },
    "aux_axis": { /* 可选辅助轴 */ }
  },
  
  // 结构化大纲（带约束）
  "outline": [
    {
      "section": "章节名称",
      "subsections": [
        {
          "subsection": "小节名称",
          "required_nodes": ["/path/to/node"],
          "required_citations": ["Paper2024"],
          "must_answer": ["必须回答的问题"],
          "evidence_cards": [
            {
              "text": "证据文本",
              "citation": "Smith2024",
              "page": 5
            }
          ]
        }
      ]
    }
  ],
  
  // 演化上下文
  "evolution_summary": [
    {
      "operation": "ADD_NODE | SPLIT_NODE | RENAME_NODE",
      "view": "视角ID",
      "parent": "父节点路径",
      "new_node": "新节点名称",
      "justification": "为什么需要这个改变"
    }
  ],
  
  // 必答问题
  "must_answer_questions": ["问题1", "问题2"],
  
  // 透明度数据
  "reconstruction_scores": [
    {
      "view_id": "review_001/FACET",
      "fit_gain": 12.5,
      "stress_reduction": 0.45,
      "edit_cost": 2.0,
      "combined_score": 8.3
    }
  ]
}
```

### 2. 重命名 `igfinder2_report.md` → `audit_report.md`

**原因**:
- "igfinder2" 是旧名称（已重构为 SNS）
- "audit_report" 更准确描述其作用：审计现有分类体系的适配性

**内容增强**:
```markdown
# SNS Audit Report
## Self-Nonself Modeling Analysis

**Research Topic**: {topic}
**Analysis Date**: {date}

---

## Executive Summary

This audit report documents the SNS (Self-Nonself) modeling process...

## Phase 1: Self Construction (Multi-view Baseline)
[详细的多视角分类体系分析]

## Phase 2: Nonself Identification (Stress Test)
[论文适配性测试结果，带证据]

## Phase 3: Adaptation (Evolution Proposal)
[最小必要结构更新提案]

## Phase 4: Writing Guidance
[写作模式判定和规则]
```

### 3. 重命名 `igfinder2_results.json` → `sns_results.json`

**内容**: 保持不变，仍然是完整的结果归档，但名称更新为反映 SNS 品牌。

---

## 代码修改

### 修改文件: `knowledge_storm/sns/engine_v2.py`

#### 1. 更新 `_save_results()` 方法

```python
def _save_results(self):
    """Save final results to disk."""
    if not self.results:
        return
    
    # ✅ 新增：保存机器可读的指导包
    if self.results.delta_aware_guidance:
        guidance_pack_path = os.path.join(
            self.args.output_dir, 
            "guidance_pack.json"  # ← 新增！
        )
        logger.info(f"Saving guidance pack to {guidance_pack_path}")
        self._save_guidance_pack(guidance_pack_path)
    
    # ✅ 重命名：审计报告
    report_path = os.path.join(
        self.args.output_dir, 
        "audit_report.md"  # ← 从 igfinder2_report.md 改名
    )
    logger.info(f"Saving audit report to {report_path}")
    self._generate_markdown_report(report_path)
    
    # ✅ 重命名：完整结果
    results_path = os.path.join(
        self.args.output_dir, 
        "sns_results.json"  # ← 从 igfinder2_results.json 改名
    )
    logger.info(f"Saving complete results to {results_path}")
    with open(results_path, 'w') as f:
        json.dump(self.results.to_dict(), f, indent=2)
```

#### 2. 新增 `_save_guidance_pack()` 方法

```python
def _save_guidance_pack(self, output_path: str):
    """
    Save machine-readable guidance pack for downstream systems.
    
    The guidance pack is the PRIMARY OUTPUT for automated survey 
    generation systems.
    """
    guidance = self.results.delta_aware_guidance
    
    guidance_pack = {
        "topic": guidance.topic,
        "generation_date": guidance.generation_date.isoformat(),
        "schema_version": "2.0",
        
        # 写作策略（关键！）
        "writing_mode": guidance.main_axis_mode.value,
        "writing_rules": guidance.writing_rules.to_dict(),
        
        # 分类体系
        "taxonomy": {
            "main_axis": {
                "facet": guidance.main_axis.facet.value,
                "tree": guidance.main_axis.tree.to_dict(),
                "weight": guidance.main_axis.weight,
            },
            "aux_axis": guidance.aux_axis.to_dict() if guidance.aux_axis else None,
        },
        
        # 结构化大纲
        "outline": [s.to_dict() for s in guidance.outline],
        
        # 演化上下文
        "evolution_summary": [e.to_dict() for e in guidance.evolution_summary],
        
        # 必答问题
        "must_answer_questions": guidance.must_answer_questions,
        
        # 透明度
        "reconstruction_scores": [s.to_dict() for s in guidance.reconstruction_scores],
    }
    
    with open(output_path, 'w') as f:
        json.dump(guidance_pack, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Guidance pack saved: {output_path}")
    logger.info(f"   - Writing mode: {guidance.main_axis_mode.value}")
    logger.info(f"   - Sections: {len(guidance.outline)}")
    logger.info(f"   - Rules: {len(guidance.writing_rules.do)} do, "
                f"{len(guidance.writing_rules.dont)} dont")
```

#### 3. 更新 `_generate_markdown_report()` 标题

```python
def _generate_markdown_report(self, output_path: str):
    """Generate markdown audit report."""
    report = f"""# SNS Audit Report
## Self-Nonself Modeling Analysis

**Research Topic**: {self.results.topic}
**Analysis Date**: {self.results.generation_date.strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

This audit report documents the SNS modeling process...
"""
    # ... rest of report generation
```

---

## 输出文件对比

### 修改前 ❌

```
./output/
├── multiview_baseline.json       # 中间文件
├── fit_vectors.json              # 中间文件
├── stress_clusters.json          # 中间文件（如果保存）
├── evolution_proposal.json       # 中间文件（如果保存）
├── delta_guidance.json           # 中间文件（如果保存）
├── igfinder2_report.md           # 主要输出（人类可读）
└── igfinder2_results.json        # 完整归档
```

**问题**:
- ❌ 缺少机器可读的指导包
- ❌ 文件名过时（igfinder2 → sns）
- ❌ 没有明确区分"主要输出"和"中间文件"

### 修改后 ✅

```
./output/
# 主要输出（必需）
├── audit_report.md               # ✅ 人类可读的审计报告
└── guidance_pack.json            # ✅ 机器可读的指导包（新增！）

# 完整归档
└── sns_results.json              # ✅ 重命名，完整结果

# 中间文件（调试用，可选）
├── multiview_baseline.json
├── fit_vectors.json
├── stress_clusters.json
├── evolution_proposal.json
└── delta_guidance.json
```

**改进**:
- ✅ 两个主要输出文件清晰明确
- ✅ `guidance_pack.json` 包含所有机器可执行的约束
- ✅ 文件名反映 SNS 品牌
- ✅ 结构化、规范化的 JSON 格式

---

## 使用示例

### 下游系统集成

```python
import json

# 1. 加载 guidance_pack.json
with open("output/guidance_pack.json") as f:
    guidance = json.load(f)

# 2. 检查写作模式
if guidance["writing_mode"] == "DELTA_FIRST":
    # 优先组织新兴趋势和结构性变化
    strategy = DeltaFirstStrategy(
        rules=guidance["writing_rules"],
        outline=guidance["outline"]
    )
else:
    # 使用主轴作为基础，整合新论文
    strategy = AnchorPlusDeltaStrategy(
        taxonomy=guidance["taxonomy"]["main_axis"],
        evolution=guidance["evolution_summary"]
    )

# 3. 遍历大纲生成内容
for section in guidance["outline"]:
    for subsection in section["subsections"]:
        # 获取约束
        required_nodes = subsection["required_nodes"]
        required_citations = subsection["required_citations"]
        must_answer = subsection["must_answer"]
        evidence_cards = subsection["evidence_cards"]
        
        # 生成内容（基于约束）
        content = generate_subsection(
            title=subsection["subsection"],
            constraints={
                "nodes": required_nodes,
                "citations": required_citations,
                "questions": must_answer,
            },
            evidence=evidence_cards
        )

# 4. 应用写作规则
for rule in guidance["writing_rules"]["do"]:
    validate_rule(content, rule, expected=True)

for rule in guidance["writing_rules"]["dont"]:
    validate_rule(content, rule, expected=False)
```

### 人类审查流程

```bash
# 1. 研究人员阅读审计报告
cat output/audit_report.md

# 2. 理解系统的分析过程
# - Phase 1: 构建了哪些视角？
# - Phase 2: 哪些论文不适配？
# - Phase 3: 提出了什么结构更新？
# - Phase 4: 判定了什么写作模式？

# 3. 检查机器指导包（如果需要调试）
cat output/guidance_pack.json | jq '.writing_mode'
cat output/guidance_pack.json | jq '.evolution_summary'
```

---

## 验证清单

修改完成后，请验证：

- [ ] `guidance_pack.json` 文件生成
- [ ] `audit_report.md` 文件生成（旧名 `igfinder2_report.md` 已弃用）
- [ ] `sns_results.json` 文件生成（旧名 `igfinder2_results.json` 已弃用）
- [ ] `guidance_pack.json` 包含所有必需字段：
  - [ ] `writing_mode`
  - [ ] `writing_rules` (do/dont)
  - [ ] `taxonomy` (main_axis + aux_axis)
  - [ ] `outline` (sections + subsections + evidence_cards)
  - [ ] `evolution_summary`
  - [ ] `must_answer_questions`
  - [ ] `reconstruction_scores`
- [ ] JSON 格式有效（可解析）
- [ ] 日志输出正确显示新文件名

---

## 总结

### 问题
用户报告系统输出缺少机器可读的 `guidance_pack.json` 文件。

### 根本原因
实现时遗漏了设计文档中的关键要求：
- 设计要求：`audit_report.md` + `guidance_pack.json`
- 实际输出：`igfinder2_report.md` + ❌（缺失）

### 解决方案
1. ✅ 新增 `_save_guidance_pack()` 方法
2. ✅ 更新 `_save_results()` 调用新方法
3. ✅ 重命名输出文件（igfinder2 → sns, audit_report）
4. ✅ 更新 README.md 文档说明

### 影响
- ✅ 下游系统现在可以加载机器可读的指导包
- ✅ 输出文件名反映 SNS 品牌
- ✅ 明确区分"主要输出"和"中间文件"
- ✅ 完整实现论文设计方案

---

**修复完成**: 2025-12-15  
**修复人员**: Claude (genspark-ai-developer)
