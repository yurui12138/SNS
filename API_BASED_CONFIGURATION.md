# SNS API-Based Configuration Guide

## 概述

SNS系统已更新为使用API调用而非本地模型，无需GPU资源。本指南说明如何配置和使用API-based的嵌入模型和NLI模型。

---

## 核心设计原则

### 1. **嵌入模型 (Embeddings)**
- **使用**: OpenAI `text-embedding-ada-002` 或 `text-embedding-3-small`
- **优势**: 
  - 无需本地GPU
  - 高质量语义理解
  - 支持大批量处理
- **Fallback**: TF-IDF (sklearn, 无API调用)

### 2. **NLI模型 (Natural Language Inference)**
- **使用**: LLM API (GPT-3.5-turbo, GPT-4, Claude等) 通过zero-shot prompting
- **优势**:
  - 无需本地GPU
  - 灵活的推理能力
  - 支持多语言
- **Fallback**: 规则based (关键词+反义词检测)

---

## 配置方式

### 方式1: 环境变量配置 (推荐)

```bash
# OpenAI API配置
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选,自定义endpoint

# 或使用Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2023-05-15"
```

### 方式2: 代码配置

#### Phase 2 Pipeline配置

```python
from knowledge.sns.engine_v2 import SNSRunner, SNSArguments, SNSLMConfigs

# 配置API密钥
embedding_api_key = "sk-..."  # OpenAI API key
nli_api_key = "sk-..."        # 可以使用同一个key

# 创建SNS配置
args = SNSArguments(
    topic="Your Research Topic",
    output_dir="./output",
    # ... 其他参数
)

# 配置LM模型
lm_configs = SNSLMConfigs(
    phase1_lm=your_lm,
    phase2_lm=your_lm,
    phase3_lm=your_lm,
    phase4_lm=your_lm,
    # Phase 2 嵌入配置
    phase2_embedding_type="openai",
    phase2_embedding_model="text-embedding-ada-002",
    phase2_embedding_api_key=embedding_api_key,
    # Phase 2 NLI配置  
    phase2_nli_type="llm",
    phase2_nli_model="gpt-3.5-turbo",
    phase2_nli_api_key=nli_api_key,
)

# 运行SNS
runner = SNSRunner(args, lm_configs)
results = runner.run()
```

---

## 详细配置选项

### 嵌入模型配置

#### OpenAI Embeddings

```python
{
    "embedding_model_type": "openai",
    "embedding_model_name": "text-embedding-ada-002",  # 或 "text-embedding-3-small"
    "embedding_api_key": "sk-...",
    "embedding_api_base": None,  # 可选,自定义endpoint
}
```

**可用模型**:
- `text-embedding-ada-002`: 1536维, 性价比高
- `text-embedding-3-small`: 512-1536维可调, 更新更快
- `text-embedding-3-large`: 256-3072维可调, 最高质量

#### Azure OpenAI Embeddings

```python
{
    "embedding_model_type": "azure",
    "embedding_model_name": "your-deployment-name",
    "embedding_api_key": "...",
    "embedding_api_base": "https://your-resource.openai.azure.com",
}
```

#### Fallback (TF-IDF)

```python
{
    "embedding_model_type": "fallback",
    # 无需API key
}
```

---

### NLI模型配置

#### LLM-based NLI

```python
{
    "nli_model_type": "llm",
    "nli_llm_model": "gpt-3.5-turbo",  # 或 "gpt-4", "claude-3-haiku"等
    "nli_api_key": "sk-...",
    "nli_api_base": None,  # 可选
}
```

**推荐模型**:
- `gpt-3.5-turbo`: 速度快, 成本低 ✅ 推荐
- `gpt-4`: 质量高, 成本较高
- `gpt-4-turbo`: 平衡选择
- `claude-3-haiku`: Claude系列, 速度快
- `claude-3-sonnet`: Claude系列, 质量高

#### Rule-based Fallback

```python
{
    "nli_model_type": "rule-based",
    # 无需API key
}
```

---

## 成本估算

### 嵌入模型 (Embeddings)

以`text-embedding-ada-002`为例:

| 任务 | Token数 | 成本 (USD) |
|------|---------|-----------|
| 1篇论文 (标题+摘要) | ~500 tokens | $0.0001 |
| 1个Taxonomy节点 | ~200 tokens | $0.00004 |
| Phase 2完整运行 (100篇论文, 50个节点) | ~60K tokens | $0.012 |

**月度预算建议**: $5-20 (取决于论文数量)

### NLI模型 (LLM-based)

以`gpt-3.5-turbo`为例:

| 任务 | Token数 | 成本 (USD) |
|------|---------|-----------|
| 1次NLI推理 | ~150 tokens | $0.0002 |
| Phase 2冲突检测 (100篇论文 × 5 candidates × 3 tests) | ~225K tokens | $0.30 |

**月度预算建议**: $10-50

### 总预算

- **小规模** (50篇论文/月): ~$5-15/月
- **中规模** (200篇论文/月): ~$20-60/月  
- **大规模** (500篇论文/月): ~$50-150/月

---

## 性能优化策略

### 1. 批量处理

```python
# Embeddings支持批量处理 (最多2048条)
embeddings = embedder.encode(texts, batch_size=100)
```

### 2. 缓存策略

```python
# 缓存论文embeddings
paper_embedding_cache = {}

def get_paper_embedding(paper_id, paper_text):
    if paper_id not in paper_embedding_cache:
        paper_embedding_cache[paper_id] = embedder.encode([paper_text])[0]
    return paper_embedding_cache[paper_id]
```

### 3. 智能Fallback

系统自动fallback策略:
1. **第一选择**: API模型 (OpenAI embeddings + LLM NLI)
2. **第二选择**: TF-IDF embeddings + Rule-based NLI
3. **错误恢复**: 单个API调用失败时返回零向量/中性标签

---

## 错误处理

### API限流 (Rate Limiting)

```python
# litellm自动处理重试
import litellm
litellm.num_retries = 3
litellm.retry_delay = 2  # 秒
```

### API超时

```python
# 设置超时
litellm.request_timeout = 30  # 秒
```

### API不可用时的Fallback

```python
try:
    embedder = create_embedding_model(model_type="openai", ...)
except Exception as e:
    logger.warning(f"API不可用: {e}, 使用TF-IDF fallback")
    embedder = create_embedding_model(model_type="fallback")
```

---

## 最佳实践

### 1. 开发环境

```python
# 使用fallback模型进行快速测试
args = SNSArguments(..., use_api_models=False)
```

### 2. 生产环境

```python
# 使用API模型获得最佳性能
args = SNSArguments(
    ...,
    use_api_models=True,
    embedding_model="text-embedding-ada-002",
    nli_model="gpt-3.5-turbo"
)
```

### 3. 混合策略

```python
# Embeddings使用API, NLI使用规则
lm_configs = SNSLMConfigs(
    phase2_embedding_type="openai",
    phase2_nli_type="rule-based",  # 节省成本
)
```

---

## 安全建议

### 1. API Key管理

```bash
# 使用.env文件 (不要提交到git)
echo "OPENAI_API_KEY=sk-..." > .env
echo ".env" >> .gitignore

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
```

### 2. Key Rotation

定期轮换API keys:
```python
import os
api_key = os.getenv("OPENAI_API_KEY_CURRENT")
```

### 3. 成本监控

```python
# 记录API调用
import litellm
litellm.success_callback = ["langfuse"]  # 或其他监控工具
```

---

## 常见问题

### Q1: 如何完全禁用API调用?

```python
lm_configs = SNSLMConfigs(
    phase2_embedding_type="fallback",
    phase2_nli_type="rule-based",
)
```

### Q2: 如何使用自定义API endpoint?

```python
lm_configs = SNSLMConfigs(
    phase2_embedding_api_base="https://your-proxy.com/v1",
    phase2_nli_api_base="https://your-proxy.com/v1",
)
```

### Q3: Azure OpenAI如何配置?

```python
lm_configs = SNSLMConfigs(
    phase2_embedding_type="azure",
    phase2_embedding_model="your-deployment-name",
    phase2_embedding_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    phase2_embedding_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
```

### Q4: 成本过高怎么办?

优化建议:
1. 使用`text-embedding-3-small` (比ada-002便宜5倍)
2. NLI改用`rule-based`
3. 减少top-k候选数量
4. 启用embedding缓存

---

## 测试

### 单元测试

```python
# 测试API连接
from knowledge.sns.embeddings import OpenAIEmbedding

embedder = OpenAIEmbedding(api_key="sk-...")
emb = embedder.encode(["test text"])
assert emb.shape[1] == 1536  # ada-002维度
```

### 集成测试

```bash
# 运行完整pipeline (小数据集)
python run_sns_example.py --topic "test" --max_papers 10
```

---

## 迁移指南

### 从本地模型迁移

**之前**:
```python
Phase2Pipeline(lm, embedding_model="specter2", nli_model_type="deberta")
```

**之后**:
```python
Phase2Pipeline(
    lm,
    embedding_model_type="openai",
    embedding_model_name="text-embedding-ada-002",
    embedding_api_key="sk-...",
    nli_model_type="llm",
    nli_llm_model="gpt-3.5-turbo",
    nli_api_key="sk-..."
)
```

---

## 总结

✅ **优势**:
- 无需GPU资源
- 部署简单
- 性能稳定
- 可扩展性强

⚠️ **注意事项**:
- 需要API keys
- 有API调用成本
- 依赖网络连接

📊 **推荐配置** (生产环境):
- Embeddings: `text-embedding-ada-002` (OpenAI)
- NLI: `gpt-3.5-turbo` (快速) 或 `rule-based` (省钱)
- Fallback: 启用 (TF-IDF + rule-based)

---

**更新日期**: 2025-12-15  
**版本**: v2.0 (API-based)
