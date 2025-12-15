"""
Quick Fixes for SNS Critical Issues

This module provides improved configurations and patched functions
to address the critical issues found in the deepfake test run.
"""
from knowledge_storm.sns import SNSArguments


def get_improved_sns_args(topic: str, output_dir: str) -> SNSArguments:
    """
    返回改进后的 SNS 配置参数。
    
    主要改进：
    1. 启用真实的 embedding 模型（SPECTER2）
    2. 增加 review 和 research paper 数量
    3. 降低聚类阈值
    
    Args:
        topic: 研究主题
        output_dir: 输出目录
        
    Returns:
        改进后的 SNSArguments
    """
    return SNSArguments(
        topic=topic,
        output_dir=output_dir,
        
        # ✅ 修复1: 增加样本数量（从 5/10 增加到 15/30）
        top_k_reviews=15,           # 更多review papers
        top_k_research_papers=30,   # 更多research papers
        
        # ✅ 修复2: 降低聚类阈值（从 3 降低到 2）
        min_cluster_size=2,         # 更容易形成聚类
        
        # ✅ 修复3: 启用真实的 embedding 模型
        embedding_model="allenai/specter2",  # 不再使用 dummy
        
        # 保持其他默认值
        save_intermediate_results=True,
        lambda_regularization=0.8,
    )


def get_relaxed_fit_thresholds():
    """
    返回放宽后的 FitScore 阈值。
    
    需要在 phase2_stress_test.py 中应用这些阈值。
    """
    return {
        # 原始阈值（太严格）
        "original": {
            "UNFITTABLE_coverage": 0.45,
            "UNFITTABLE_conflict": 0.55,
            "FORCE_FIT_residual": 0.45,
        },
        
        # 建议阈值（更合理）
        "recommended": {
            "UNFITTABLE_coverage": 0.25,   # 从 0.45 降低到 0.25
            "UNFITTABLE_conflict": 0.70,   # 从 0.55 提高到 0.70
            "FORCE_FIT_residual": 0.60,    # 从 0.45 提高到 0.60
        }
    }


def get_improved_fit_score_weights():
    """
    返回改进后的 FitScore 计算权重。
    
    需要在 phase2_stress_test.py 中应用。
    """
    return {
        # 原始权重
        "original": {
            "coverage": 1.0,
            "conflict": 0.8,
            "residual": 0.4,
        },
        
        # 建议权重（降低 residual 的惩罚）
        "recommended": {
            "coverage": 1.0,
            "conflict": 0.6,   # 从 0.8 降低到 0.6
            "residual": 0.2,   # 从 0.4 降低到 0.2
        }
    }


def validate_sns_results(fit_vectors, stress_clusters, baseline):
    """
    验证 SNS 运行结果的质量并输出警告。
    
    Args:
        fit_vectors: FitVector 列表
        stress_clusters: StressCluster 列表
        baseline: MultiViewBaseline
        
    Returns:
        dict: 质量指标和警告信息
    """
    import logging
    logger = logging.getLogger(__name__)
    
    warnings = []
    metrics = {}
    
    # 检查1: Fit rate
    total_tests = sum(len(fv.fit_reports) for fv in fit_vectors)
    fit_count = sum(1 for fv in fit_vectors 
                    for fr in fv.fit_reports 
                    if str(fr.label) == "FIT")
    force_fit_count = sum(1 for fv in fit_vectors 
                          for fr in fv.fit_reports 
                          if str(fr.label) == "FORCE_FIT")
    unfittable_count = sum(1 for fv in fit_vectors 
                           for fr in fv.fit_reports 
                           if str(fr.label) == "UNFITTABLE")
    
    fit_rate = fit_count / total_tests if total_tests > 0 else 0
    metrics['fit_rate'] = fit_rate
    metrics['force_fit_rate'] = force_fit_count / total_tests if total_tests > 0 else 0
    metrics['unfittable_rate'] = unfittable_count / total_tests if total_tests > 0 else 0
    
    if fit_rate < 0.1:
        warnings.append({
            "severity": "HIGH",
            "category": "Low Fit Rate",
            "message": f"Very low fit rate: {fit_rate:.1%}",
            "possible_causes": [
                "Embedding model quality (check embedding_model parameter)",
                "NodeDefinition quality (review extraction might have failed)",
                "Baseline diversity (may need more review papers)",
            ],
            "recommendations": [
                "Use 'allenai/specter2' instead of 'dummy' embedding",
                "Increase top_k_reviews to 15-20",
                "Check review paper extraction logs for errors",
            ]
        })
    
    # 检查2: Cluster count
    cluster_count = len(stress_clusters)
    metrics['cluster_count'] = cluster_count
    
    if cluster_count == 0:
        warnings.append({
            "severity": "HIGH",
            "category": "No Stress Clusters",
            "message": "No stress clusters formed",
            "possible_causes": [
                "Too few stressed papers (need at least 5-10)",
                "min_cluster_size too large",
                "Papers too dissimilar (no clear failure patterns)",
            ],
            "recommendations": [
                "Increase top_k_research_papers to 30-50",
                "Decrease min_cluster_size to 2",
                "Check if HDBSCAN is properly installed",
            ]
        })
    
    # 检查3: Baseline quality
    unique_facets = len(set(v.facet_label.value for v in baseline.views))
    metrics['unique_facets'] = unique_facets
    metrics['total_views'] = len(baseline.views)
    
    if unique_facets < 2:
        warnings.append({
            "severity": "MEDIUM",
            "category": "Low Baseline Diversity",
            "message": f"Only {unique_facets} unique facets in baseline",
            "possible_causes": [
                "Too few review papers retrieved",
                "Review papers focus on similar aspects",
                "LLM extraction failed for some reviews",
            ],
            "recommendations": [
                "Increase top_k_reviews to 15-20",
                "Use broader search queries",
                "Check LLM extraction logs",
            ]
        })
    
    # 检查4: Average coverage
    avg_coverage = sum(fr.scores.coverage 
                       for fv in fit_vectors 
                       for fr in fv.fit_reports) / total_tests if total_tests > 0 else 0
    metrics['avg_coverage'] = avg_coverage
    
    if avg_coverage < 0.2:
        warnings.append({
            "severity": "HIGH",
            "category": "Low Coverage Scores",
            "message": f"Average coverage: {avg_coverage:.3f} (too low)",
            "possible_causes": [
                "Poor embedding quality",
                "NodeDefinitions lack keywords",
                "Semantic mismatch between papers and taxonomy",
            ],
            "recommendations": [
                "Enable SPECTER2 embedding model",
                "Improve NodeDefinition extraction prompt",
                "Add more keywords to node definitions",
            ]
        })
    
    # 输出总结
    logger.info("="*80)
    logger.info("SNS Results Validation Summary")
    logger.info("="*80)
    logger.info(f"Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    if warnings:
        logger.warning(f"\n⚠️  Found {len(warnings)} issues:")
        for i, warning in enumerate(warnings, 1):
            logger.warning(f"\n{i}. [{warning['severity']}] {warning['category']}")
            logger.warning(f"   {warning['message']}")
            logger.warning(f"   Possible causes:")
            for cause in warning['possible_causes']:
                logger.warning(f"     - {cause}")
            logger.warning(f"   Recommendations:")
            for rec in warning['recommendations']:
                logger.warning(f"     ✓ {rec}")
    else:
        logger.info("\n✅ No critical issues found!")
    
    logger.info("="*80)
    
    return {
        'metrics': metrics,
        'warnings': warnings
    }


# 示例用法
if __name__ == "__main__":
    # 获取改进的配置
    args = get_improved_sns_args("deepfake", "./output_improved")
    print("Improved SNS Arguments:")
    print(f"  top_k_reviews: {args.top_k_reviews}")
    print(f"  top_k_research_papers: {args.top_k_research_papers}")
    print(f"  min_cluster_size: {args.min_cluster_size}")
    print(f"  embedding_model: {args.embedding_model}")
    
    # 显示建议的阈值
    print("\nRecommended FitScore Thresholds:")
    thresholds = get_relaxed_fit_thresholds()
    for key, value in thresholds['recommended'].items():
        original = thresholds['original'][key]
        print(f"  {key}: {original} → {value}")
    
    # 显示建议的权重
    print("\nRecommended FitScore Weights:")
    weights = get_improved_fit_score_weights()
    for key, value in weights['recommended'].items():
        original = weights['original'][key]
        print(f"  {key}: {original} → {value}")
