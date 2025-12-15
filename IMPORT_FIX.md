# SNS Import 修复说明

## 问题描述

在重构 IG-Finder → SNS 时，`modules/__init__.py` 文件仍然引用旧的 v1.0 模块，导致 `ModuleNotFoundError`。

## 修复内容

### 1. 更新 `knowledge_storm/sns/modules/__init__.py`

**之前（错误）**:
```python
from .cognitive_self_construction import ...  # 不存在
from .innovative_nonself_identification import ...  # 不存在
from .mind_map_manager import ...  # 不存在
from .report_generation import ...  # 不存在
```

**之后（正确）**:
```python
from .phase1_multiview_baseline import Phase1Pipeline, ...
from .phase2_stress_test import Phase2Pipeline, ...
from .phase3_evolution import Phase3Pipeline, ...
from .phase4_guidance import Phase4Pipeline, ...
```

### 2. 更新 `knowledge_storm/sns/evaluation/__init__.py`

添加了缺失的 `print_metrics_report` 导出。

### 3. 更新文档字符串

将所有 "IG-Finder 2.0" 更新为 "SNS (Self-Nonself)"。

## 验证

所有 `__init__.py` 文件已通过 Python 语法检查：
```bash
python3 -m py_compile knowledge_storm/sns/__init__.py
python3 -m py_compile knowledge_storm/sns/modules/__init__.py
python3 -m py_compile knowledge_storm/sns/evaluation/__init__.py
```

## 使用说明

现在可以正常导入 SNS 模块：

```python
from knowledge_storm.sns import SNSRunner, SNSArguments
from knowledge_storm.sns.modules import (
    Phase1Pipeline, Phase2Pipeline, 
    Phase3Pipeline, Phase4Pipeline
)
```

## 运行示例

```bash
python run_sns_complete.py \
    --topic "deepfake" \
    --output-dir "./results_deepfake" \
    --openai-api-key "sk-..." \
    --openai-api-base "https://yunwu.ai/v1/"
```

## 依赖安装

确保安装所有依赖：
```bash
pip install -e .
# 或
pip install -r requirements.txt
```

必需依赖：
- dspy_ai
- sentence-transformers
- numpy
- litellm

可选依赖（用于完整功能）：
- hdbscan (Phase 3 聚类)
- transformers, torch (NLI 冲突检测)
- scikit-learn (评估指标)
