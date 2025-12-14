# IG-Finder: Innovation Gap Finder

**Identifying Verifiable Innovation Gaps in Scientific Research**

## 🌟 Overview

IG-Finder (Innovation Gap Finder) is an advanced AI framework designed to identify **verifiable innovation gaps** in scientific research by modeling cognitive baselines and detecting frontier research that deviates from established consensus. 

### The Problem
Existing automatic review generation systems suffer from **"lagging reviews"** - they fail to identify true innovations because they lack proper domain cognitive baseline modeling.

### The Solution
IG-Finder adapts the **immune system's self-nonself recognition mechanism** to scientific knowledge modeling:
- **"Cognitive Self"**: Consensus knowledge extracted from existing review papers
- **"Innovative Non-self"**: Emerging research clusters that logically deviate from the consensus

### Key Innovation
Instead of generating reviews directly, IG-Finder produces a **comprehensive innovation gap report** that can serve as enhanced input for downstream automatic review systems, dramatically improving their ability to recognize and articulate true innovation.

## 🏗️ Architecture

IG-Finder operates in two phases:

### Phase 1: Cognitive Self Construction
Build the domain's cognitive baseline by:
- Retrieving existing review papers
- Extracting consensus claims and domain development
- Structuring knowledge into a dynamic mind map

### Phase 2: Innovative Non-self Identification
Identify innovation gaps by:
- Retrieving frontier research papers
- Performing difference-aware analysis against the cognitive baseline
- Identifying emerging research clusters
- Marking evolution states on the mind map
- Generating comprehensive innovation gap reports

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Configure API Keys

Create a configuration file or set environment variables:

```bash
# For Tavily search (Recommended)
export TAVILY_API_KEY="your_tavily_api_key"

# For Yunwu AI (OpenAI-compatible proxy)
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="https://yunwu.ai/v1/"

# Alternative: Bing or You.com search
export BING_SEARCH_API_KEY="your_bing_key"
# or
export YDC_API_KEY="your_you_key"
```

### Run IG-Finder

**Option 1: Quick Start (Recommended for testing)**

```bash
python examples/ig_finder_examples/quick_start_yunwu.py \
    --topic "transformer models in natural language processing"
```

**Option 2: Full Configuration**

```bash
python examples/ig_finder_examples/run_ig_finder_tavily.py \
    --topic "your research topic" \
    --output-dir ./output \
    --top-k-reviews 20 \
    --top-k-research 30 \
    --min-cluster-size 3
```

## 📊 Output

IG-Finder generates comprehensive outputs:

1. **Cognitive Baseline** (`cognitive_baseline.json`)
   - Domain consensus knowledge
   - Key concepts and relationships
   - Research paradigms

2. **Innovation Analysis** (`phase2_results.json`)
   - Difference perception records
   - Deviation analysis
   - Cluster identification

3. **Innovation Gap Report** (`innovation_gap_report.md`)
   - Executive summary
   - Identified innovation clusters
   - Detailed analysis with evidence
   - Recommendations for downstream systems

4. **Dynamic Mind Map**
   - Hierarchical knowledge structure
   - Evolution state tracking
   - Innovation markers

## 🎯 Use Cases

- **Automatic Review Generation**: Provide enhanced input to overcome "lagging review" problems
- **Literature Survey**: Quickly identify research gaps in a domain
- **Research Planning**: Discover emerging trends and unexplored areas
- **Innovation Assessment**: Evaluate novelty of research directions
- **Academic Intelligence**: Track domain evolution and paradigm shifts

## 🔧 Configuration

### Search Engines
- **Tavily** (Recommended): Fast, academic-focused, stable API
- **Bing Search**: Web-based search with academic sources
- **You.com**: Alternative web search engine

### Language Models
- OpenAI GPT-4/GPT-3.5
- Azure OpenAI
- Compatible OpenAI proxies (e.g., Yunwu AI)
- Other OpenAI-compatible endpoints

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--topic` | Research topic to analyze | Required |
| `--output-dir` | Output directory | `./ig_finder_output` |
| `--retriever` | Search engine: `tavily`, `bing`, `you` | `tavily` |
| `--top-k-reviews` | Number of review papers to retrieve | 15 |
| `--top-k-research` | Number of research papers to retrieve | 25 |
| `--min-cluster-size` | Minimum papers per innovation cluster | 3 |
| `--deviation-threshold` | Innovation detection threshold | 0.7 |

## 🛠️ Advanced Usage

### Incremental Execution

Run phases separately for debugging or iterative refinement:

```bash
# Phase 1 only: Build cognitive baseline
python examples/ig_finder_examples/run_ig_finder_tavily.py \
    --topic "your topic" \
    --skip-phase2

# Phase 2 only: Identify innovations (requires Phase 1 output)
python examples/ig_finder_examples/run_ig_finder_tavily.py \
    --topic "your topic" \
    --skip-phase1
```

### Python API

```python
from knowledge_storm.ig_finder.engine import IGFinderRunner
from knowledge_storm.ig_finder.dataclass import IGFinderRunnerArguments
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import TavilySearchRM

# Configure language models
lm_configs = IGFinderLMConfigs()
openai_kwargs = {
    'api_key': 'your_api_key',
    'api_base': 'https://yunwu.ai/v1/',
    'temperature': 1.0,
}
lm = LitellmModel(model='gpt-4o', max_tokens=3000, **openai_kwargs)
lm_configs.set_all_models(lm)

# Configure retrieval
rm = TavilySearchRM(tavily_api_key='your_tavily_key', k=15)

# Create runner
args = IGFinderRunnerArguments(
    topic="your research topic",
    output_dir="./output",
)
runner = IGFinderRunner(args, lm_configs, rm)

# Execute
runner.run()
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

IG-Finder builds upon concepts from the STORM project:
- **STORM Paper**: [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
- **Co-STORM Paper**: [Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations](https://arxiv.org/abs/2408.15232)

Special thanks to the STORM team at Stanford OVAL for their foundational work on knowledge curation systems.



# IG-Finder: Innovation Gap Finder Framework

## 概述

IG-Finder (Innovation Gap Finder) 是一个基于免疫系统"自我-非我识别"机制的科学知识建模框架。该框架通过构建领域认知基线并识别与之偏离的创新研究簇,显式发现可验证的创新性认知缺口,为下游自动综述系统提供高质量的创新引导。

## 核心设计理念

### 1. 免疫学隐喻

**自我-非我识别机制**:
- **认知自我 (Cognitive Self)**: 从已有综述中提取的领域共识和认知边界
- **创新非我 (Innovative Non-self)**: 偏离共识但内部逻辑自洽的新兴研究簇
- **差异感知推理 (Difference-aware Reasoning)**: 多视角专家代理在前沿文献与共识之间的对比分析

### 2. 两阶段工作流

#### 阶段一: 认知自我构建 (Cognitive Self Construction)
1. **综述检索**: 基于主题检索高质量的已有综述文献
2. **共识提取**: 从综述中结构化提取领域发展脉络、研究范式、主流方法论
3. **动态建模**: 将提取的共识填充到动态思维导图,建立认知基线

#### 阶段二: 创新非我识别 (Innovative Non-self Identification)
1. **前沿检索**: 检索最新的研究型论文(非综述)
2. **差异分析**: 多视角专家代理对比前沿文献与认知基线
3. **演化标注**: 在思维导图上标注知识演化状态(延续/偏离/创新)
4. **簇识别**: 识别内部逻辑自洽的创新研究簇
5. **缺口报告**: 生成面向创新性认知缺口的结构化报告

## 技术架构

### 核心组件设计

```
IG-Finder Framework
│
├── 1. CognitiveSelfConstructionModule (认知自我构建模块)
│   ├── ReviewRetriever (综述检索器)
│   ├── ConsensusExtractor (共识提取器)
│   └── CognitiveBaselineBuilder (认知基线构建器)
│
├── 2. InnovativeNonSelfIdentificationModule (创新非我识别模块)
│   ├── FrontierPaperRetriever (前沿论文检索器)
│   ├── DifferenceAwareAnalyzer (差异感知分析器)
│   ├── ExpertPerspectiveGenerator (专家视角生成器)
│   └── InnovationClusterIdentifier (创新簇识别器)
│
├── 3. DynamicMindMapManager (动态思维导图管理器)
│   ├── KnowledgeEvolutionTracker (知识演化追踪器)
│   ├── ConceptRelationshipGraph (概念关系图)
│   └── EvolutionStateAnnotator (演化状态标注器)
│
├── 4. InnovationGapReportGenerator (创新缺口报告生成器)
│   ├── GapSynthesizer (缺口综合器)
│   ├── EvidenceOrganizer (证据组织器)
│   └── ReportFormatter (报告格式化器)
│
└── 5. IGFinderRunner (主执行引擎)
    ├── Configuration Management
    ├── Pipeline Orchestration
    └── State Persistence
```

### 数据结构设计

#### 1. CognitiveBaseline (认知基线)
```python
@dataclass
class CognitiveBaseline:
    """表示从已有综述中提取的领域认知基线"""
    topic: str
    review_papers: List[ReviewPaper]  # 源综述列表
    consensus_map: KnowledgeBase  # 共识思维导图
    research_paradigms: List[ResearchParadigm]  # 研究范式
    mainstream_methods: List[Method]  # 主流方法
    knowledge_boundaries: Dict[str, Boundary]  # 知识边界
    temporal_coverage: TimeRange  # 时间覆盖范围
```

#### 2. EvolutionState (演化状态)
```python
class EvolutionState(Enum):
    """知识节点的演化状态"""
    CONSENSUS = "consensus"  # 共识:已有综述中的确立知识
    CONTINUATION = "continuation"  # 延续:继续深化共识内容
    DEVIATION = "deviation"  # 偏离:与共识不同但未形成体系
    INNOVATION = "innovation"  # 创新:偏离共识且内部自洽的新簇
    POTENTIAL_GAP = "potential_gap"  # 潜在缺口:需要进一步验证
```

#### 3. InnovationCluster (创新簇)
```python
@dataclass
class InnovationCluster:
    """表示一个创新研究簇"""
    cluster_id: str
    name: str
    core_papers: List[ResearchPaper]  # 核心论文
    deviation_from_consensus: DeviationAnalysis  # 与共识的偏离分析
    internal_coherence_score: float  # 内部逻辑一致性得分
    innovation_dimensions: List[str]  # 创新维度(方法/数据/范式等)
    supporting_evidence: List[Evidence]  # 支撑证据
    knowledge_path: List[str]  # 在思维导图中的路径
```

#### 4. InnovationGapReport (创新缺口报告)
```python
@dataclass
class InnovationGapReport:
    """最终输出的创新缺口报告"""
    topic: str
    cognitive_baseline_summary: str  # 认知基线摘要
    identified_clusters: List[InnovationCluster]  # 识别的创新簇
    gap_analysis: Dict[str, GapAnalysis]  # 按维度的缺口分析
    evolution_narrative: str  # 知识演化叙述
    mind_map_visualization: Dict  # 思维导图可视化数据
    recommendation_for_review: str  # 给综述系统的建议
```

### 模块详细设计

#### Module 1: CognitiveSelfConstructionModule

**功能**: 从已有综述中构建领域认知基线

**子组件**:

1. **ReviewRetriever**
   - 检索策略: 优先选择高引用、近期发表的综述
   - 过滤条件: 排除研究型论文,只保留综述/Survey类文献
   - 输出: 排序的综述列表

2. **ConsensusExtractor**
   - 提取内容:
     - 领域发展历史和关键里程碑
     - 主流研究范式和方法论
     - 公认的挑战和未解决问题
     - 研究子领域的分类体系
   - 使用LLM进行结构化信息抽取

3. **CognitiveBaselineBuilder**
   - 将提取的共识组织到动态思维导图
   - 标记所有节点为CONSENSUS状态
   - 记录每个共识节点的来源综述

#### Module 2: InnovativeNonSelfIdentificationModule

**功能**: 识别偏离认知基线的创新研究簇

**子组件**:

1. **FrontierPaperRetriever**
   - 时间过滤: 检索认知基线时间范围之后的论文
   - 类型过滤: 排除综述,只保留研究型论文
   - 相关性排序: 确保与主题相关

2. **DifferenceAwareAnalyzer**
   - 多视角专家代理设计:
     - **方法论专家**: 关注研究方法的创新
     - **数据范式专家**: 关注数据和实验设计
     - **理论框架专家**: 关注概念和理论创新
     - **应用领域专家**: 关注应用场景扩展
   - 对比分析流程:
     ```
     For each paper:
       For each expert perspective:
         1. 提取论文核心主张
         2. 匹配认知基线中的相关节点
         3. 进行差异性分析
         4. 评估偏离程度和创新潜力
     ```

3. **ExpertPerspectiveGenerator**
   - 基于主题动态生成相关的专家视角
   - 参考STORM的perspective-guided机制

4. **InnovationClusterIdentifier**
   - 聚类算法: 基于语义相似度和共同偏离模式
   - 一致性验证: 检查簇内论文的逻辑自洽性
   - 标注策略:
     - 单篇偏离 → DEVIATION
     - 多篇形成簇且逻辑自洽 → INNOVATION
     - 与共识方向一致 → CONTINUATION

#### Module 3: DynamicMindMapManager

**功能**: 管理动态演化的思维导图

**特性**:
- 继承Co-STORM的KnowledgeBase结构
- 扩展功能:
  - 演化状态标注
  - 时间戳追踪
  - 多源信息关联(综述 vs 研究论文)
  - 偏离度量化

**核心方法**:
```python
class DynamicMindMapManager:
    def update_with_consensus(self, consensus_data):
        """用共识数据初始化思维导图"""
        
    def annotate_evolution_state(self, node, state, evidence):
        """标注节点的演化状态"""
        
    def track_deviation(self, node, baseline_node, deviation_metrics):
        """追踪偏离信息"""
        
    def identify_innovation_paths(self):
        """识别标记为INNOVATION的知识路径"""
        
    def export_visualization(self):
        """导出可视化数据"""
```

#### Module 4: InnovationGapReportGenerator

**功能**: 生成结构化的创新缺口报告

**报告结构**:
```
# Innovation Gap Report: [Topic]

## Executive Summary
- 认知基线概述
- 识别的创新簇数量
- 主要创新方向

## Part I: Cognitive Baseline
### 1.1 Field Development History
### 1.2 Established Research Paradigms
### 1.3 Mainstream Methodologies
### 1.4 Known Challenges

## Part II: Innovation Clusters
For each cluster:
  ### Cluster Name
  - Core Papers
  - Deviation Analysis
  - Innovation Dimensions
  - Internal Coherence Evidence

## Part III: Gap Analysis by Dimension
- Methodological Gaps
- Data Paradigm Gaps
- Theoretical Framework Gaps
- Application Domain Gaps

## Part IV: Knowledge Evolution Narrative
- 从共识到创新的演化路径
- 关键转折点分析
- 未来研究方向建议

## Part V: Mind Map Visualization
- 交互式思维导图数据
- 演化状态分布统计

## Part VI: Recommendations for Review Generation
- 建议的综述组织结构
- 需要重点阐述的创新点
- 引用证据的优先级
```

### 工作流程

```
Input: Topic (e.g., "自动综述生成")

┌─────────────────────────────────────────┐
│ Phase 1: Cognitive Self Construction   │
└─────────────────────────────────────────┘
  ↓
1.1 Retrieve Review Papers
  - Search Query: "[Topic] survey OR review"
  - Filter: publication_type=review
  - Sort: by citations and recency
  ↓
1.2 Extract Consensus from Reviews
  - For each review:
    - Extract: paradigms, methods, timeline, challenges
    - Organize: into hierarchical structure
  ↓
1.3 Build Cognitive Baseline
  - Initialize dynamic mind map
  - Populate with consensus nodes (state=CONSENSUS)
  - Record source reviews for each node

┌─────────────────────────────────────────┐
│ Phase 2: Innovative Non-self Identify  │
└─────────────────────────────────────────┘
  ↓
2.1 Retrieve Frontier Papers
  - Search Query: "[Topic]"
  - Filter: publication_type=research_article
  - Filter: date > cognitive_baseline.temporal_coverage.end
  ↓
2.2 Generate Expert Perspectives
  - Based on topic and cognitive baseline
  - Create specialized agents:
    - Methodology Expert
    - Data Paradigm Expert
    - Theory Expert
    - Application Expert
  ↓
2.3 Multi-perspective Difference Analysis
  - For each paper:
    - For each expert:
      - Extract paper's core claims
      - Match with baseline nodes
      - Analyze differences
      - Assess innovation potential
    - Aggregate expert opinions
  ↓
2.4 Update Mind Map with Evolution States
  - Add new nodes for frontier concepts
  - Annotate states: CONTINUATION/DEVIATION/INNOVATION
  - Record deviation metrics
  ↓
2.5 Identify Innovation Clusters
  - Cluster papers by:
    - Semantic similarity
    - Common deviation patterns
  - Validate internal coherence
  - Mark coherent clusters as INNOVATION
  ↓
2.6 Generate Innovation Gap Report
  - Synthesize cognitive baseline summary
  - Describe each innovation cluster
  - Provide gap analysis by dimensions
  - Construct evolution narrative
  - Export mind map visualization
  
Output: InnovationGapReport (替代简单的topic description)
```

## 与STORM/Co-STORM的关系

### 借鉴的组件

1. **从STORM借鉴**:
   - 整体Pipeline架构 (STORMWikiRunner → IGFinderRunner)
   - 多视角专家机制 (perspective-guided question asking)
   - 信息检索和引用管理
   - 模块化设计理念

2. **从Co-STORM借鉴**:
   - 动态思维导图 (KnowledgeBase)
   - 知识节点的层级组织
   - 协作式信息整合
   - 实时状态更新机制

### 关键创新点

1. **两阶段认知模型**: 区分"已知共识"和"创新偏离"
2. **演化状态标注**: 显式追踪知识演化信号
3. **差异感知推理**: 专家代理在共识与前沿之间进行对比
4. **创新簇识别**: 不仅发现单点创新,还识别系统性创新模式
5. **输出重定位**: 生成面向创新缺口的报告而非综述文章


# IG-Finder 实现总结

## 项目概述

已成功实现 **IG-Finder (Innovation Gap Finder)** 框架，这是一个基于免疫系统"自我-非我识别"机制的科学知识建模系统，用于识别研究领域中可验证的创新性认知缺口。

## 核心创新点

### 1. 理论创新：免疫学隐喻
- **认知自我 (Cognitive Self)**: 从已有综述中提取的领域共识，代表"已知的已知"
- **创新非我 (Innovative Non-self)**: 偏离共识但内部逻辑自洽的新兴研究簇，代表"系统性创新"
- **差异感知推理**: 多视角专家在前沿文献与共识之间的对比分析

### 2. 方法论创新：两阶段工作流

#### 阶段一：认知自我构建
```
输入: 研究主题
↓
检索综述论文 → 提取结构化共识 → 构建认知基线思维导图
↓
输出: 标记为CONSENSUS状态的动态知识库
```

#### 阶段二：创新非我识别
```
输入: 认知基线 + 研究主题
↓
检索前沿论文 → 多视角差异分析 → 识别创新簇 → 验证内部一致性
↓
输出: 标注演化状态的思维导图 + 创新簇列表
```

### 3. 输出重定位
不生成综述本身，而是生成**创新缺口报告**，作为下游自动综述系统的高质量输入，解决"滞后性综述"问题。

## 实现架构

### 核心模块 (7个主要组件)

#### 1. 数据结构层 (`dataclass.py` - 16.8KB)
```python
- CognitiveBaseline: 认知基线数据结构
- InnovationCluster: 创新簇表示
- InnovationGapReport: 最终报告
- EvolutionState: 知识演化状态枚举
- ExtendedKnowledgeNode: 扩展的知识节点
- 以及其他支持类型（ReviewPaper, ResearchPaper, DeviationAnalysis等）
```

#### 2. 认知自我构建模块 (`cognitive_self_construction.py` - 20.9KB)
```python
- ReviewRetriever: 检索高质量综述
  • 策略: 优先"survey/review"关键词
  • 过滤: 排除研究型论文
  • 排序: 按引用和相关性

- ConsensusExtractor: 提取共识知识
  • 使用 dspy.ChainOfThought 进行结构化抽取
  • 提取: 研究范式、主流方法、知识边界、概念层次
  • 输出: 结构化的共识数据

- CognitiveBaselineBuilder: 构建基线
  • 聚合多篇综述的共识
  • 构建层级化思维导图
  • 所有节点标记为 CONSENSUS 状态
```

#### 3. 创新非我识别模块 (`innovative_nonself_identification.py` - 25.6KB)
```python
- FrontierPaperRetriever: 检索前沿论文
  • 过滤: 排除综述，保留研究论文
  • 时间: 优先最新发表
  • 相关性: 确保与主题匹配

- ExpertPerspectiveGenerator: 生成专家视角
  • 方法论专家: 关注研究方法创新
  • 数据范式专家: 关注数据和实验设计
  • 理论框架专家: 关注概念和理论创新
  • 应用领域专家: 关注应用场景扩展

- DifferenceAwareAnalyzer: 差异感知分析
  • 多视角: 每篇论文从多个专家角度分析
  • 匹配: 与认知基线节点进行匹配
  • 评估: 偏离程度、创新潜力评分
  • 输出: DeviationAnalysis 对象

- InnovationClusterIdentifier: 创新簇识别
  • 聚类: 按偏离维度分组论文
  • 验证: 使用LLM验证内部逻辑一致性
  • 评分: 计算内部连贯性得分
  • 过滤: 只保留连贯的创新簇
```

#### 4. 动态思维导图管理器 (`mind_map_manager.py` - 9.0KB)
```python
- EvolutionStateAnnotator: 演化状态标注
  • 基于偏离分数自动标注状态
  • CONSENSUS: 共识节点
  • CONTINUATION: 延续方向
  • DEVIATION: 孤立偏离
  • INNOVATION: 簇化创新

- DynamicMindMapManager: 思维导图管理
  • update_with_phase2_results(): 更新思维导图
  • identify_innovation_paths(): 识别创新路径
  • get_evolution_state_distribution(): 统计状态分布
  • export_visualization_data(): 导出可视化数据
```

#### 5. 报告生成模块 (`report_generation.py` - 16.8KB)
```python
- InnovationGapReportGenerator: 报告生成器
  • _generate_baseline_summary(): 总结认知基线
  • _perform_gap_analysis(): 按维度分析缺口
  • _generate_evolution_narrative(): 生成演化叙述
  • _generate_recommendations(): 生成下游建议
  • format_report_as_markdown(): 格式化为Markdown
```

#### 6. 执行引擎 (`engine.py` - 17.3KB)
```python
- IGFinderLMConfigs: 语言模型配置
  • consensus_extraction_lm: 共识提取
  • deviation_analysis_lm: 偏离分析
  • cluster_validation_lm: 簇验证
  • report_generation_lm: 报告生成

- IGFinderArguments: 运行参数
  • topic, output_dir
  • top_k_reviews, top_k_research_papers
  • min_cluster_size, deviation_threshold

- IGFinderRunner: 主执行引擎
  • run_phase1_cognitive_self_construction()
  • run_phase2_innovative_nonself_identification()
  • generate_innovation_gap_report()
  • run(): 完整流程
```

#### 7. 示例和文档
```
- examples/ig_finder_examples/run_ig_finder_gpt.py (6.2KB)
  • 命令行参数解析
  • 完整使用示例
  • 结果展示

- examples/ig_finder_examples/README.md (9.5KB)
  • 安装指南
  • 使用教程
  • 参数说明
  • 故障排查
  • 与STORM集成示例

- IG_FINDER_DESIGN.md (10.3KB)
  • 完整设计文档
  • 架构说明
  • 数据结构规范
  • 工作流程详解
```

## 技术特点

### 1. 继承STORM生态系统
- **接口兼容**: 继承 `interface.py` 的抽象类
- **检索器复用**: 使用STORM的 `Retriever` 接口
- **LM系统**: 复用STORM的多LM系统范式
- **知识库**: 扩展Co-STORM的 `KnowledgeBase` 和 `KnowledgeNode`

### 2. 模块化设计
- **高内聚低耦合**: 每个模块职责单一
- **可测试性**: 清晰的接口便于单元测试
- **可扩展性**: 易于添加新的专家视角或聚类算法
- **可配置性**: 丰富的参数控制行为

### 3. DSPy集成
- 使用 `dspy.Signature` 定义LLM任务
- 使用 `dspy.ChainOfThought` 进行复杂推理
- 利用 `dspy.context` 管理LM切换

### 4. 持久化和增量执行
- 自动保存中间结果（JSON格式）
- 支持跳过已完成阶段
- 便于调试和迭代优化

## 使用示例

### 命令行使用
```bash
python examples/ig_finder_examples/run_ig_finder_gpt.py \
    --topic "automatic literature review generation" \
    --output-dir ./output \
    --retriever bing \
    --top-k-reviews 10 \
    --top-k-research 30
```

### Python API使用
```python
from knowledge_storm.ig_finder import (
    IGFinderRunner,
    IGFinderLMConfigs,
    IGFinderArguments,
)
from knowledge_storm.rm import BingSearch

# 配置
lm_configs = IGFinderLMConfigs()
lm_configs.init(lm_type="openai")

rm = BingSearch(bing_search_api_key=os.getenv('BING_API_KEY'))

args = IGFinderArguments(
    topic="automatic literature review generation",
    output_dir="./output",
)

# 执行
runner = IGFinderRunner(args, lm_configs, rm)
report = runner.run()

# 结果分析
for cluster in report.identified_clusters:
    print(f"{cluster.name}: {len(cluster.core_papers)} papers")
    print(f"  Innovation: {', '.join(cluster.innovation_dimensions)}")
```

### 与STORM集成
```python
# 1. 使用IG-Finder识别创新缺口
ig_report = ig_runner.run()

# 2. 构建增强的主题描述
enhanced_topic = f"{topic}\n\nInnovation Focus:\n"
for cluster in ig_report.identified_clusters:
    enhanced_topic += f"- {cluster.name}: {cluster.cluster_summary}\n"

# 3. 传递给STORM生成创新型综述
storm_runner.run(topic=enhanced_topic, ...)
```

## 输出结果

### 创新缺口报告结构
```markdown
# Innovation Gap Report: [Topic]

## Executive Summary
- 认知基线概述
- 识别的创新簇数量
- 主要创新方向

## Part I: Cognitive Baseline
- 领域发展历史
- 已确立的研究范式
- 主流方法论
- 已知挑战

## Part II: Innovation Clusters
For each cluster:
  - 核心论文列表
  - 偏离分析
  - 创新维度
  - 内部连贯性证据
  - 潜在影响

## Part III: Gap Analysis by Dimension
- 方法论缺口
- 数据范式缺口
- 理论框架缺口
- 应用领域缺口

## Part IV: Knowledge Evolution Narrative
- 从共识到创新的演化路径
- 关键转折点分析

## Part V: Mind Map Visualization
- 演化状态分布统计
- 交互式思维导图数据

## Part VI: Recommendations for Review Generation
- 建议的综述组织结构
- 需要重点阐述的创新点
- 引用证据的优先级
```

### 文件输出
```
output/
├── cognitive_baseline.json          # 认知基线（可复用）
├── phase2_results.json              # 阶段2结果
├── innovation_gap_report.json       # JSON格式报告
└── innovation_gap_report.md         # Markdown格式报告（人类可读）
```

## 代码统计

### 代码量
```
文件                                      行数    字节
========================================================
dataclass.py                            ~600    16.8KB
engine.py                               ~520    17.3KB
cognitive_self_construction.py          ~620    20.9KB
innovative_nonself_identification.py    ~760    25.6KB
mind_map_manager.py                     ~270    9.0KB
report_generation.py                    ~500    16.8KB
__init__.py (modules)                   ~40     1.1KB
__init__.py (ig_finder)                 ~50     1.0KB
--------------------------------------------------------
核心代码总计                            ~3360   ~108KB

run_ig_finder_gpt.py                    ~200    6.2KB
README.md (examples)                    ~380    9.5KB
IG_FINDER_DESIGN.md                     ~420    10.3KB
--------------------------------------------------------
文档和示例总计                          ~1000   ~26KB

总计                                    ~4360   ~134KB
```

### 复杂度指标
- **模块数**: 7个主要模块
- **类数**: ~20个核心类
- **函数数**: ~60+个方法
- **dspy.Signature数**: 6个LLM任务定义



## 设计决策和权衡

### 1. 为什么选择两阶段设计？
- **认知清晰**: 明确区分"已知"和"创新"
- **可解释性**: 便于追溯创新识别的依据
- **可调试性**: 每个阶段可独立验证
- **可复用性**: 认知基线可跨查询复用

### 2. 为什么使用多专家视角？
- **全面性**: 不同角度发现不同类型创新
- **鲁棒性**: 减少单一视角的偏见
- **细粒度**: 能够识别特定维度的创新
- **借鉴STORM**: 延续STORM的perspective-guided设计

### 3. 为什么需要内部一致性验证？
- **质量控制**: 避免将噪声误认为创新
- **可信度**: 确保识别的创新有足够支撑
- **聚焦**: 关注系统性创新而非孤立案例

### 4. 为什么不直接生成综述？
- **定位差异**: IG-Finder是"发现"工具，STORM是"生成"工具
- **模块化**: 分离关注点，提升可组合性
- **灵活性**: 报告可用于多种下游任务
- **研究价值**: 创新缺口本身就是有价值的输出

## 与现有系统对比

| 特性 | STORM | Co-STORM | IG-Finder |
|------|-------|----------|-----------|
| **主要目标** | 生成维基风格文章 | 人机协作知识整理 | 识别创新缺口 |
| **输入** | 主题描述 | 主题描述 | 主题描述 |
| **核心机制** | 多视角问答 | 协作对话 | 自我-非我识别 |
| **知识组织** | 静态大纲 | 动态思维导图 | 演化状态思维导图 |
| **输出** | 带引用的文章 | 带引用的文章 | 创新缺口报告 |
| **创新识别** | ❌ | ❌ | ✅ |
| **认知基线建模** | ❌ | ❌ | ✅ |
| **演化追踪** | ❌ | 部分支持 | ✅ |

## 未来改进方向

### 短期（1-3个月）
- [ ] 添加单元测试和集成测试
- [ ] 支持更多检索后端（Semantic Scholar, arXiv API）
- [ ] 优化LLM提示词以提高提取质量
- [ ] 添加进度条和详细日志
- [ ] 实现结果缓存机制

### 中期（3-6个月）
- [ ] 集成更先进的聚类算法（基于语义嵌入）
- [ ] 支持引用网络分析
- [ ] 添加时序分析（跨多个时间窗口）
- [ ] 开发Web UI用于可视化探索
- [ ] 实现增量更新机制（基于新论文持续更新）

### 长期（6-12个月）
- [ ] 多语言支持（中文、德语等科研语言）
- [ ] 跨领域创新识别（跨学科知识迁移）
- [ ] 自动化评估系统（与人工标注对比）
- [ ] 与学术数据库深度集成
- [ ] 发表研究论文验证方法有效性

## 潜在应用场景

### 1. 学术研究
- **文献综述准备**: 快速了解领域创新前沿
- **研究机会识别**: 发现尚未充分探索的方向
- **论文定位**: 帮助研究者理解自己工作的创新性

### 2. 科研管理
- **基金评审**: 识别真正创新的研究提案
- **战略规划**: 为机构确定研究优先级
- **人才评估**: 评估研究者的创新贡献

### 3. 教育培训
- **课程设计**: 帮助教师了解领域最新发展
- **学生指导**: 为研究生选题提供参考
- **知识更新**: 追踪快速发展领域的变化

### 4. 产业应用
- **技术监控**: 追踪竞争对手的创新动向
- **投资决策**: 识别有潜力的新兴技术
- **产品规划**: 发现未被满足的市场需求

## 技术债务和已知限制

### 当前限制
1. **依赖LLM质量**: 提取和分析质量取决于LLM能力
2. **检索覆盖度**: 受限于检索系统的索引范围
3. **计算成本**: 大量LLM调用导致成本较高
4. **时间延迟**: 完整流程可能需要数分钟到数十分钟
5. **领域泛化**: 在某些高度专业化领域可能效果较差

### 技术债务
1. **错误处理**: 需要更细粒度的异常处理
2. **性能优化**: 可并行化的部分未充分优化
3. **测试覆盖**: 缺少自动化测试
4. **文档完善**: API文档可以更详细

## 验证和评估

### 定性验证
- ✅ 框架能够成功执行完整流程
- ✅ 生成的报告结构清晰完整
- ✅ 识别的创新簇具有一定合理性
- ✅ 代码遵循STORM项目规范

### 待完成的定量评估
- [ ] 与人工标注的创新点进行对比
- [ ] 计算识别的准确率和召回率
- [ ] 测试不同参数设置的影响
- [ ] 在多个研究领域验证泛化能力

## 结论

IG-Finder框架已成功实现，具有以下优势：

### 理论贡献
- 将免疫学隐喻引入科学知识建模
- 提出认知基线与创新识别的两阶段方法
- 定义了知识演化状态的分类体系

### 工程实现
- 完整的模块化实现（~134KB代码和文档）
- 良好的代码结构和可扩展性
- 丰富的文档和使用示例
- 与STORM生态系统无缝集成

### 实用价值
- 为下游综述生成系统提供高质量输入
- 帮助研究者快速把握领域创新动态
- 支持多种学术和产业应用场景

### 下一步
- 等待PR审核和反馈
- 根据反馈进行优化改进
- 在真实研究场景中测试验证
- 收集用户反馈持续迭代


## 🙏 致谢

本项目站在巨人的肩膀上：

- **STORM团队** (Yijia Shao et al.): 提供了优秀的基础框架
- **Co-STORM团队** (Yucheng Jiang et al.): 动态思维导图的灵感来源
- **DSPy**: 简化了LLM应用开发
- **OpenAI/Azure/Together**: 提供强大的语言模型


# IG-Finder 使用指南

## 什么是 IG-Finder？

IG-Finder (Innovation Gap Finder，创新缺口发现器) 是一个自动识别科研领域创新缺口的框架。它通过两个阶段的分析：

1. **认知自我构建**：分析已有综述，建立领域共识基线
2. **创新非我识别**：分析最新研究论文，识别偏离共识但内部逻辑自洽的创新研究簇

最终生成一份详细的**创新缺口报告**，可作为自动综述生成系统的高质量输入。

## 核心理念

借鉴免疫系统的"自我-非我识别"机制：
- **认知自我 (Self)** = 已有综述中的共识知识
- **创新非我 (Non-self)** = 偏离共识的新兴研究簇
- **目标** = 识别真正的创新，而非重复已知内容

## 快速开始

### 1. 安装

```bash
cd /home/user/webapp
pip install -e .
```

### 2. 配置 API 密钥

```bash
# OpenAI API (必需)
export OPENAI_API_KEY="your_openai_api_key"

# 搜索引擎 API (选择其一)
export BING_SEARCH_API_KEY="your_bing_api_key"
# 或
export YDC_API_KEY="your_you_api_key"
```

### 3. 运行示例

```bash
python examples/ig_finder_examples/run_ig_finder_gpt.py \
    --topic "自动综述生成" \
    --output-dir ./output \
    --retriever bing
```

## 详细使用

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--topic` | 研究主题（必需） | - |
| `--output-dir` | 输出目录 | `./ig_finder_output` |
| `--retriever` | 搜索引擎（bing/you） | `bing` |
| `--top-k-reviews` | 检索综述数量 | 10 |
| `--top-k-research` | 检索研究论文数量 | 30 |
| `--min-cluster-size` | 最小簇大小 | 2 |
| `--deviation-threshold` | 偏离阈值(0-1) | 0.5 |

### Python API 使用

```python
from knowledge_storm.ig_finder import (
    IGFinderRunner,
    IGFinderLMConfigs,
    IGFinderArguments,
)
from knowledge_storm.rm import BingSearch
import os

# 配置语言模型
lm_configs = IGFinderLMConfigs()
lm_configs.init(lm_type="openai")

# 配置检索器
rm = BingSearch(
    bing_search_api_key=os.getenv('BING_SEARCH_API_KEY'),
    k=10
)

# 配置参数
args = IGFinderArguments(
    topic="自动综述生成",
    output_dir="./output",
    top_k_reviews=10,
    top_k_research_papers=30,
)

# 创建并运行
runner = IGFinderRunner(args, lm_configs, rm)
report = runner.run()

# 查看结果
print(f"识别到 {len(report.identified_clusters)} 个创新簇")
for cluster in report.identified_clusters:
    print(f"- {cluster.name}: {len(cluster.core_papers)} 篇论文")
```

## 输出结果

运行完成后，输出目录包含：

```
output/
├── cognitive_baseline.json          # 认知基线（可复用）
├── phase2_results.json              # 第二阶段结果
├── innovation_gap_report.json       # 完整报告(JSON)
└── innovation_gap_report.md         # 完整报告(Markdown)
```

### 报告内容

创新缺口报告包括：

1. **概述摘要**：识别的创新簇数量和主要方向
2. **认知基线**：领域发展历史、研究范式、主流方法
3. **创新簇详情**：
   - 核心论文列表
   - 创新维度（方法论/数据/理论/应用）
   - 偏离分析
   - 潜在影响
4. **缺口分析**：按不同维度分析创新机会
5. **演化叙述**：从共识到创新的知识演化路径
6. **推荐建议**：给下游综述生成系统的指导

## 适用场景

### 学术研究
- 📚 文献综述准备：快速了解领域前沿
- 🔍 研究选题：发现尚未充分探索的方向
- 📝 论文写作：理解自己工作的创新定位

### 科研管理
- 💰 基金评审：识别真正创新的项目
- 🎯 战略规划：确定研究优先级
- 👥 人才评估：评估研究者的创新贡献

## 推荐研究主题

### AI/机器学习
- "自动综述生成"
- "多智能体强化学习"
- "神经架构搜索"
- "少样本学习"
- "大语言模型推理"

### 交叉领域
- "AI药物发现"
- "计算神经科学"
- "人机协作"
- "可解释AI"

## 参数调优建议

### 1. 主题描述
- ✅ **推荐**：具体明确，如"Transformer在时间序列预测中的应用"
- ❌ **避免**：过于宽泛，如"机器学习"

### 2. 检索数量
- **综述少**：增加 `--top-k-reviews` 到 15-20
- **论文少**：增加 `--top-k-research` 到 40-50
- **成本控制**：减少检索数量，但可能影响全面性

### 3. 阈值设置
- **保守策略**：`--deviation-threshold 0.7`（只识别显著创新）
- **激进策略**：`--deviation-threshold 0.3`（识别更多潜在创新）
- **平衡策略**：`--deviation-threshold 0.5`（默认）

### 4. 簇大小
- **严格要求**：`--min-cluster-size 3`（需要3+篇论文支撑）
- **宽松要求**：`--min-cluster-size 1`（接受单篇创新论文）
- **推荐设置**：`--min-cluster-size 2`（默认）

## 与 STORM 集成

IG-Finder 可以为 STORM 提供更好的输入：

```python
# 第一步：识别创新缺口
from knowledge_storm.ig_finder import IGFinderRunner
ig_runner = IGFinderRunner(ig_args, lm_configs, rm)
gap_report = ig_runner.run()

# 第二步：构建增强的主题描述
enhanced_topic = f"{topic}\n\n重点关注以下创新方向：\n"
for cluster in gap_report.identified_clusters:
    enhanced_topic += f"\n## {cluster.name}\n"
    enhanced_topic += f"{cluster.cluster_summary}\n"
    enhanced_topic += f"关键论文：\n"
    for paper in cluster.core_papers[:3]:
        enhanced_topic += f"- {paper.title} ({paper.year})\n"

# 第三步：使用 STORM 生成创新型综述
from knowledge_storm import STORMWikiRunner
storm_runner = STORMWikiRunner(storm_args, lm_configs, rm)
storm_runner.run(
    topic=enhanced_topic,
    do_research=True,
    do_generate_article=True,
)
```

## 常见问题

### Q1: 没有找到综述怎么办？
- 尝试用英文描述主题
- 添加领域关键词（如"survey", "review"）
- 扩大搜索范围（增加 `top-k-reviews`）

### Q2: 没有识别到创新簇？
- 降低 `deviation-threshold`
- 减小 `min-cluster-size`
- 检查是否是成熟稳定的领域（可能确实缺少创新）

### Q3: API 调用太多成本高？
- 减少检索数量
- 使用分阶段执行（`--skip-phase1` 或 `--skip-phase2`）
- 复用认知基线（同一领域多次查询）

### Q4: 内存不足？
- 减少 `top-k-research`
- 分批处理
- 使用更大内存的机器

## 技术支持

- **设计文档**：查看 `IG_FINDER_DESIGN.md` 了解详细架构
- **实现总结**：查看 `IG_FINDER_IMPLEMENTATION_SUMMARY.md` 了解技术细节
- **示例文档**：查看 `examples/ig_finder_examples/README.md` 了解更多用法
- **GitHub**: https://github.com/yurui12138/storm
- **Pull Request**: https://github.com/yurui12138/storm/pull/1

## 贡献指南

欢迎贡献改进！可以：
- 报告 bug 或提出功能请求
- 改进文档和示例
- 优化算法和提示词
- 添加新的检索后端
- 实现可视化界面

## 许可证

继承 STORM 项目的许可证。

## 致谢

IG-Finder 基于以下优秀工作：
- **STORM** (Shao et al., NAACL 2024): 提供了基础框架
- **Co-STORM** (Jiang et al., EMNLP 2024): 提供了动态思维导图的灵感
- **DSPy**: 提供了优雅的 LLM 编程范式

---
**开始使用 IG-Finder，发现科研创新的新机会！** 🚀

---
