# SNS方法实现改进建议

**日期**: 2025-12-15  
**项目**: SNS (Self-Nonself) for Automatic Survey Generation

---

## 执行摘要

本文档基于对SNS方法说明文档与当前代码实现的详细对比分析,提出针对性的改进建议。**当前实现完整度为80%**,存在4个Critical级别的缺失功能,但架构优秀且关键基础设施(Embeddings, NLI)已实现,预计**5-7个工作日可补齐所有Critical issues**。

### 关键发现

✅ **优势**:
- 数据结构体系完整 (100%对齐方法说明)
- Pipeline架构清晰 (Phase 1-4流程完整)
- 关键基础设施已实现 (embeddings.py, nli.py存在且完整)
- 设计决策正确 (Reconstruct-then-select, Writing Mode等)

⚠️ **需要改进**:
- Phase 2未使用真实Embeddings和NLI (使用placeholder)
- 补视角策略未实现 (只有warning)
- SPLIT_NODE和RENAME_NODE未实现 (只有TODO)
- Taxonomy_v2未应用Evolution (输出的是原始版本)

---

## 改进优先级

### 🔴 Priority 1 (Critical - Week 1)

#### 1.1 集成真实Embeddings和NLI到Phase 2

**当前问题**:
- `phase2_stress_test.py` 使用keyword overlap计算相似度
- `FitTester._calculate_conflict()` 使用简单keyword匹配

**改进方案**:
```python
# 在 EmbeddingBasedRetriever.__init__()
from ..embeddings import create_embedding_model
self.embedder = create_embedding_model(model_type="specter2", device="cpu")

# 在 FitTester.__init__()
from ..nli import create_nli_model
self.nli_model = create_nli_model(model_type="deberta", device="cpu")
```

**预期效果**:
- Coverage分数准确度提升 (基于SPECTER2)
- Conflict分数准确度提升 (基于DeBERTa-MNLI)
- FIT判定准确率提升

**工作量**: 4-6小时

---

#### 1.2 实现补视角策略

**当前问题**:
- `_check_baseline_quality()` 只warning,不补救
- 当综述质量不足时系统无法恢复

**改进方案**:
创建 `knowledge/sns/modules/compensatory_view.py`:
```python
class CompensatoryViewInducer:
    def should_induce(self, baseline) -> bool:
        # 检查unique facets < 2 或 dominant > 60%
    
    def induce_view(self, baseline, papers, topic) -> TaxonomyView:
        # 1. 对论文HDBSCAN聚类
        # 2. LLM生成簇标签
        # 3. 构建flat tree (root + leaves)
        # 4. 选择unique facet
        # 5. 创建TaxonomyView with weight=0.5
```

**集成到Phase1Pipeline**:
```python
def run(self, topic):
    # ... 现有代码 ...
    
    # 新增: 补视角检查与触发
    if self.compensatory_inducer.should_induce(baseline):
        compensatory_view = self.compensatory_inducer.induce_view(
            baseline, self.cached_papers, topic
        )
        if compensatory_view:
            baseline.views.append(compensatory_view)
            baseline.__post_init__()  # 重新归一化权重
```

**预期效果**:
- Baseline始终满足质量标准
- 系统鲁棒性提升

**工作量**: 1-2天

---

### 🟡 Priority 2 (High - Week 2-3)

#### 2.1 实现SPLIT_NODE操作

**当前问题**:
- 只有ADD_NODE,无法处理overcrowded节点

**改进方案**:
在 `phase3_evolution.py` 添加:
```python
def _propose_split_node(self, cluster, view, fit_vectors) -> Optional[SplitNodeOperation]:
    # 1. 识别overcrowded叶节点 (论文数 > 15)
    # 2. 对节点内论文sub-clustering
    # 3. LLM生成子节点定义
    # 4. 计算fit_gain
    # 5. 返回SplitNodeOperation (cost=2.0)
```

**预期效果**:
- 可以处理"节点过度拥挤"场景
- Evolution建议更全面

**工作量**: 1天

---

#### 2.2 实现RENAME_NODE操作

**当前问题**:
- 无法处理语义漂移节点

**改进方案**:
在 `phase3_evolution.py` 添加:
```python
def _propose_rename_node(self, cluster, view, fit_vectors) -> Optional[RenameNodeOperation]:
    # 1. 识别FORCE_FIT率高的叶节点 (drift > 30%)
    # 2. 分析lost_novelty提取新主题
    # 3. LLM生成新名称和定义
    # 4. 计算fit_gain
    # 5. 返回RenameNodeOperation (cost=0.5)
```

**预期效果**:
- 可以处理"节点语义漂移"场景
- Evolution建议完整

**工作量**: 1天

---

#### 2.3 应用Evolution到Taxonomy生成v2

**当前问题**:
- `guidance_pack.json`输出的taxonomy是原始版本
- 下游系统看不到结构更新

**改进方案**:
创建 `knowledge/sns/modules/taxonomy_evolution_applier.py`:
```python
def apply_evolution_to_taxonomy(view, operations) -> TaxonomyView:
    view_v2 = copy.deepcopy(view)
    
    for op in operations:
        if isinstance(op, AddNodeOperation):
            # 添加新节点到tree
            new_node = TaxonomyTreeNode(...)
            view_v2.tree.add_node(new_node)
            view_v2.node_definitions[new_path] = NodeDefinition(...)
        
        elif isinstance(op, SplitNodeOperation):
            # 将叶节点变为内部节点,添加子节点
        
        elif isinstance(op, RenameNodeOperation):
            # 更新节点名称和定义
    
    return view_v2
```

**集成到Phase4**:
```python
# 在axis selection之前应用evolution
baseline_v2 = apply_evolutions_to_all_views(baseline, evolution_proposal)
main_axis, mode = select_main_axis_with_mode(scores, baseline_v2)
```

**预期效果**:
- 下游系统看到演化后的taxonomy
- 输出完整性提升

**工作量**: 1-2天

---

### 🟢 Priority 3 (Medium - Week 4)

#### 3.1 增强Must-answer Questions

**当前问题**:
- 问题过于通用 ("What are the key approaches?")

**改进方案**:
```python
def _generate_must_answer_questions_enhanced(...):
    questions = []
    
    # 1. 基础结构问题
    questions.append(f"What are the main dimensions in {main_axis.facet_label}?")
    
    # 2. 每个evolution operation一个问题
    for op in evolution_proposal.operations:
        if isinstance(op, AddNodeOperation):
            questions.append(
                f"Why was '{op.new_node.name}' added? "
                f"What papers don't fit existing structure?"
            )
    
    # 3. 每个STRONG_SHIFT cluster一个问题
    for cluster in clusters:
        if cluster.cluster_type == ClusterType.STRONG_SHIFT:
            questions.append(
                f"How do cluster {cluster.cluster_id}'s {len(cluster.papers)} papers "
                f"challenge existing {view.facet_label} organization?"
            )
    
    # 4. 旧结构不足证据问题 (关键!)
    questions.append(
        "What evidence shows existing taxonomies are insufficient? "
        "Cite specific FORCE_FIT or UNFITTABLE cases."
    )
    
    return questions
```

**预期效果**:
- 问题更具针对性
- 引导下游系统回答关键演化问题

**工作量**: 4小时

---

#### 3.2 提升Evidence Cards质量

**当前问题**:
- Evidence cards只包含abstract前200字

**改进方案**:
```python
def _create_subsection(...):
    # ... 现有代码 ...
    
    # 新增: 从fit_reports提取精确evidence
    evidence_cards = []
    for paper in relevant_papers[:5]:
        fv = next((f for f in fit_vectors if f.paper_id == paper.url), None)
        if fv:
            report = next((r for r in fv.fit_reports if r.view_id == main_axis.view_id), None)
            if report and report.lost_novelty:
                # 使用lost_novelty作为evidence
                for ln in report.lost_novelty[:2]:
                    evidence_cards.append(EvidenceCard(
                        paper_id=paper.url,
                        title=paper.title,
                        claim=ln.bullet,
                        quote=ln.evidence[0].quote if ln.evidence else "",
                        page=ln.evidence[0].page if ln.evidence else 0
                    ))
    
    # Fallback to abstract if no fit_reports
    if not evidence_cards:
        # 现有逻辑
```

**预期效果**:
- Evidence cards包含精确的novelty quotes
- 可追溯性增强

**工作量**: 2小时

---

## 实施路线图

### Week 1: 核心功能补齐

| 任务 | 优先级 | 工作量 | 负责 |
|-----|--------|--------|------|
| 1.1 集成Embeddings+NLI | 🔴 Critical | 4-6h | Dev Team |
| 1.2 实现补视角策略 | 🔴 Critical | 1-2d | Dev Team |
| 测试 | 🔴 Critical | 4h | QA Team |

**里程碑**: Phase 2准确性提升,baseline质量保证

---

### Week 2: Evolution完整化

| 任务 | 优先级 | 工作量 | 负责 |
|-----|--------|--------|------|
| 2.1 实现SPLIT_NODE | 🟡 High | 1d | Dev Team |
| 2.2 实现RENAME_NODE | 🟡 High | 1d | Dev Team |
| 2.3 应用Evolution到Taxonomy | 🔴 Critical | 1-2d | Dev Team |
| 测试 | 🟡 High | 4h | QA Team |

**里程碑**: Evolution proposal完整,输出包含v2 taxonomy

---

### Week 3: 质量提升

| 任务 | 优先级 | 工作量 | 负责 |
|-----|--------|--------|------|
| 3.1 增强Must-answer Questions | 🟢 Medium | 4h | Dev Team |
| 3.2 提升Evidence Cards | 🟢 Medium | 2h | Dev Team |
| 文档更新 | 🟢 Medium | 4h | Dev Team |
| 端到端测试 | 🟡 High | 1d | QA Team |

**里程碑**: 输出质量达到生产标准

---

### Week 4: 优化与部署

| 任务 | 优先级 | 工作量 | 负责 |
|-----|--------|--------|------|
| 性能优化 (缓存, batch) | 🟢 Medium | 1d | Dev Team |
| 参数调优 | 🟢 Medium | 1d | Dev Team |
| Beta部署 | 🟡 High | 0.5d | DevOps |
| 用户反馈收集 | 🟡 High | - | PM |

**里程碑**: 系统ready for production

---

## 测试计划

### 单元测试

```python
# Test 1: Embedding integration
def test_embedding_retriever():
    retriever = EmbeddingBasedRetriever("specter2")
    sim = retriever._compute_similarity("deep learning", "neural networks")
    assert 0 <= sim <= 1
    assert sim > 0.5  # 应该相似

# Test 2: NLI integration
def test_nli_conflict():
    tester = FitTester(retriever, nli_model)
    claims = PaperClaims(...)  # supervised learning paper
    node_def = NodeDefinition(  # unsupervised node
        exclusion_criteria=["supervised learning methods"]
    )
    conflict = tester._calculate_conflict(claims, node_def)
    assert conflict > 0.5  # 应该有冲突

# Test 3: Compensatory view
def test_compensatory_inducer():
    inducer = CompensatoryViewInducer(embedder, lm)
    baseline = MultiViewBaseline(...)  # 只有1个facet
    assert inducer.should_induce(baseline) == True
    
    view = inducer.induce_view(baseline, papers, topic)
    assert view is not None
    assert view.facet_label != baseline.views[0].facet_label

# Test 4: Evolution applier
def test_evolution_applier():
    view = TaxonomyView(...)
    operations = [AddNodeOperation(...)]
    
    view_v2 = apply_evolution_to_taxonomy(view, operations)
    
    # 验证新节点存在
    assert "new_node_path" in view_v2.tree.nodes
    assert "new_node_path" in view_v2.node_definitions
```

### 集成测试

```python
# Test E2E: Complete pipeline
def test_sns_pipeline_e2e():
    runner = SNSRunner(args, lm_configs, rm)
    
    results = runner.run(
        do_phase1=True,
        do_phase2=True,
        do_phase3=True,
        do_phase4=True
    )
    
    # 验证输出完整性
    assert results.multiview_baseline is not None
    assert len(results.fit_vectors) > 0
    assert results.evolution_proposal is not None
    assert results.delta_aware_guidance is not None
    
    # 验证guidance_pack.json
    guidance_pack_path = os.path.join(args.output_dir, "guidance_pack.json")
    assert os.path.exists(guidance_pack_path)
    
    with open(guidance_pack_path) as f:
        pack = json.load(f)
        assert "writing_mode" in pack
        assert "taxonomy" in pack
        assert "outline" in pack
        assert "evolution_summary" in pack
    
    # 验证taxonomy包含evolution
    if results.evolution_proposal.operations:
        # 应该有新增的节点
        main_axis = results.delta_aware_guidance.main_axis
        for op in results.evolution_proposal.operations:
            if isinstance(op, AddNodeOperation):
                assert any(op.new_node.name in node.name 
                          for node in main_axis.tree.nodes.values())
```

---

## 性能优化建议

### 1. Embedding缓存

```python
class CachedEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = {}  # text -> embedding
    
    def encode(self, texts):
        to_encode = []
        cached_embeddings = []
        
        for text in texts:
            if text in self.cache:
                cached_embeddings.append(self.cache[text])
            else:
                to_encode.append(text)
        
        if to_encode:
            new_embeddings = self.embedder.encode(to_encode)
            for text, emb in zip(to_encode, new_embeddings):
                self.cache[text] = emb
        
        # 合并缓存和新计算的
        # ...
```

### 2. NLI批处理

```python
# 收集所有需要检测的(claim, exclusion)对
all_pairs = []
for claim in claims:
    for exclusion in node_def.exclusion_criteria:
        all_pairs.append((claim, exclusion))

# 批量推理
if len(all_pairs) > 0:
    premises = [p[1] for p in all_pairs]
    hypotheses = [p[0] for p in all_pairs]
    conflict_scores = nli_model.compute_contradiction_scores_batch(premises, hypotheses)
    max_conflict = max(conflict_scores)
```

### 3. 并行处理

```python
from concurrent.futures import ThreadPoolExecutor

def process_paper(paper):
    claims = self.claim_extractor.extract_claims(paper)
    fit_vector = self.stress_tester.test_paper(claims, baseline)
    return fit_vector

# 并行处理多篇论文
with ThreadPoolExecutor(max_workers=4) as executor:
    fit_vectors = list(executor.map(process_paper, papers))
```

---

## 部署清单

### 依赖更新

```bash
# requirements.txt 更新
sentence-transformers==2.2.2  # for SPECTER2
transformers==4.35.0          # for DeBERTa-MNLI
torch==2.1.0                  # for model inference
hdbscan==0.8.33              # for clustering
```

### 配置更新

```python
# SNSArguments 新增参数
@dataclass
class SNSArguments:
    # ... 现有参数 ...
    
    # 新增
    embedding_model: str = "specter2"  # or "scincl", "sbert", "fallback"
    nli_model: str = "deberta"         # or "roberta", "fallback"
    enable_compensatory_view: bool = True
    min_facet_diversity: int = 2
    max_dominant_facet_ratio: float = 0.6
```

### 环境变量

```bash
# 可选: 设置模型缓存路径
export TRANSFORMERS_CACHE=/path/to/model/cache
export SENTENCE_TRANSFORMERS_HOME=/path/to/model/cache
```

---

## 风险评估与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|-----|-----|---------|
| Embedding模型下载失败 | 中 | 高 | - Fallback to TF-IDF<br>- Pre-download models |
| NLI推理太慢 | 中 | 中 | - Batch processing<br>- Use smaller model (base vs large) |
| 补视角质量不稳定 | 低 | 中 | - LLM生成label加强prompt<br>- 人工review option |
| Evolution应用导致tree不一致 | 低 | 高 | - 充分测试<br>- Tree structure validation<br>- Rollback机制 |
| 性能下降 | 中 | 中 | - Embedding/NLI缓存<br>- 并行处理<br>- GPU加速option |

---

## 成功标准

### 功能完整性

- [ ] Phase 2使用真实SPECTER2和DeBERTa-MNLI
- [ ] Baseline质量不足时自动触发补视角
- [ ] Evolution proposal包含ADD/SPLIT/RENAME三种操作
- [ ] guidance_pack.json包含演化后的taxonomy_v2
- [ ] Must-answer questions针对具体evolution和stress points

### 质量指标

- [ ] FIT判定准确率 > 85% (人工抽样验证)
- [ ] Baseline unique facets ≥ 2 (100%满足)
- [ ] Evolution operations有证据支持 (100%有evidence_spans)
- [ ] guidance_pack.json通过schema验证 (100%通过)

### 性能指标

- [ ] End-to-end pipeline运行时间 < 30分钟 (100篇论文, 15篇综述)
- [ ] Embedding推理 < 10ms/paper (cached)
- [ ] NLI推理 < 50ms/对 (batched)

---

## 附录: 快速修复脚本

### 脚本1: 快速集成Embeddings到Phase2

```python
# scripts/quick_fix_phase2_embeddings.py

import sys
sys.path.insert(0, '/home/user/webapp')

from knowledge.sns.modules import phase2_stress_test
from knowledge.sns import embeddings

# Patch EmbeddingBasedRetriever
original_init = phase2_stress_test.EmbeddingBasedRetriever.__init__

def new_init(self, embedding_model_name="specter2"):
    self.model_name = embedding_model_name
    self.embedder = embeddings.create_embedding_model(
        model_type=embedding_model_name,
        device="cpu"
    )

def new_compute_similarity(self, text1, text2):
    emb1 = self.embedder.encode([text1])[0]
    emb2 = self.embedder.encode([text2])[0]
    return self.embedder.similarity(emb1, emb2)

# Apply patches
phase2_stress_test.EmbeddingBasedRetriever.__init__ = new_init
phase2_stress_test.EmbeddingBasedRetriever._compute_similarity = new_compute_similarity

print("✅ Phase 2 Embeddings patched successfully")
```

### 脚本2: 验证改进效果

```python
# scripts/validate_improvements.py

def validate_phase2_improvements():
    from knowledge.sns.modules.phase2_stress_test import EmbeddingBasedRetriever
    
    retriever = EmbeddingBasedRetriever("specter2")
    
    # Test 1: 使用真实embeddings
    sim = retriever._compute_similarity("deep learning", "neural networks")
    assert 0 <= sim <= 1, "Similarity out of range"
    assert sim > 0.5, "Related terms should have high similarity"
    
    print("✅ Embeddings working correctly")
    
def validate_evolution_applier():
    from knowledge.sns.modules.taxonomy_evolution_applier import apply_evolution_to_taxonomy
    
    # Create test data
    # ...
    
    # Apply evolution
    view_v2 = apply_evolution_to_taxonomy(view, operations)
    
    # Validate tree structure
    assert len(view_v2.tree.nodes) > len(view.tree.nodes)
    
    print("✅ Evolution applier working correctly")

if __name__ == "__main__":
    validate_phase2_improvements()
    validate_evolution_applier()
    
    print("\n🎉 All validations passed!")
```

---

**文档版本**: 1.0  
**作者**: Claude (AI Code Assistant)  
**最后更新**: 2025-12-15  
**状态**: Ready for implementation
