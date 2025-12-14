"""
示例脚本：使用 arXiv 作为文献检索后端运行 IG-Finder。

本脚本支持：
- 使用 arXiv API 进行论文检索
- 自定义 OpenAI 兼容代理（例如 yunwu.ai）

用法：
    python examples/ig_finder_examples/run_ig_finder.py --topic "automatic literature review generation"
"""

import os
import sys
import argparse
import logging
import warnings

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from knowledge_storm.ig_finder import (
    IGFinderRunner,
    IGFinderLMConfigs,
    IGFinderArguments,
)
from knowledge_storm.rm import ArxivSearchRM
from knowledge_storm.lm import LitellmModel
from knowledge_storm.interface import Retriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")


def main():
    parser = argparse.ArgumentParser(description='Run IG-Finder')
    parser.add_argument(
        '--topic',
        type=str,
        required=True,
        help='Research topic to analyze (e.g., "automatic literature review generation")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ig_finder_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--openai-api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--openai-api-base',
        type=str,
        default=None,
        help='OpenAI API base URL (or set OPENAI_API_BASE environment variable)'
    )
    parser.add_argument(
        '--top-k-reviews',
        type=int,
        default=3,
        help='Number of review papers to retrieve'
    )
    parser.add_argument(
        '--top-k-research',
        type=int,
        default=5,
        help='Number of research papers to retrieve'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum number of papers to form a cluster'
    )
    parser.add_argument(
        '--deviation-threshold',
        type=float,
        default=0.5,
        help='Minimum deviation score (0-1) to consider'
    )
    parser.add_argument(
        '--skip-phase1',
        action='store_true',
        help='Skip Phase 1 (load from saved results)'
    )
    parser.add_argument(
        '--skip-phase2',
        action='store_true',
        help='Skip Phase 2 (load from saved results)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model name to use (default: gpt-4o)'
    )
    
    args = parser.parse_args()
    
    def _sanitize_base(url: str) -> str:
        import re
        s = url or ""
        s = s.strip()
        return re.sub(r"[^a-zA-Z0-9:/._-]", "", s)

    # openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    # openai_api_base = args.openai_api_base or os.getenv('OPENAI_API_BASE')
    openai_api_key = 'sk-w15EcYnx4q4Q2xJAJik8bjZ5gW5rXz2oyzoBPopd4xnJ519H'
    openai_api_base = 'https://api.yunwu.ai/v1'
    openai_api_base = _sanitize_base(openai_api_base) if openai_api_base else None

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set. Please provide --openai-api-key or set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Log configuration
    logger.info("="*80)
    logger.info("IG-Finder Configuration:")
    logger.info(f"  Topic: {args.topic}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  API Base: {openai_api_base if openai_api_base else 'Default OpenAI'}")
    logger.info(f"  Search Engine: arXiv")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info("="*80)
    
    # Setup LM configs with custom API base
    logger.info("Initializing language model configurations...")
    lm_configs = IGFinderLMConfigs()
    
    # Prepare OpenAI kwargs
    openai_kwargs = {
        'api_key': openai_api_key,
        'temperature': 1.0,
        'top_p': 0.9,
    }
    debug_dir = os.path.join(args.output_dir, 'raw_lm')
    openai_kwargs['debug_responses_dir'] = debug_dir
    
    # Add custom API base if provided
    if openai_api_base:
        openai_kwargs['api_base'] = openai_api_base
        logger.info(f"Using custom OpenAI API base: {openai_api_base}")
    
    # Create language models for different tasks
    # Use GPT-4o for complex reasoning tasks
    consensus_lm = LitellmModel(model=args.model, max_tokens=3000, **openai_kwargs)
    deviation_lm = LitellmModel(model=args.model, max_tokens=2000, **openai_kwargs)
    cluster_lm = LitellmModel(model=args.model, max_tokens=1500, **openai_kwargs)
    report_lm = LitellmModel(model=args.model, max_tokens=4000, **openai_kwargs)
    
    # Set the language models
    lm_configs.set_consensus_extraction_lm(consensus_lm)
    lm_configs.set_deviation_analysis_lm(deviation_lm)
    lm_configs.set_cluster_validation_lm(cluster_lm)
    lm_configs.set_report_generation_lm(report_lm)
    
    logger.info(f"Language models initialized with {args.model}")
    
    logger.info("Initializing arXiv search engine...")
    rm = Retriever(
        rm=ArxivSearchRM(
            k=10,
        )
    )
    logger.info("arXiv search engine ready")
    
    # Setup arguments
    ig_args = IGFinderArguments(
        topic=args.topic,
        output_dir=args.output_dir,
        top_k_reviews=args.top_k_reviews,
        top_k_research_papers=args.top_k_research,
        min_cluster_size=args.min_cluster_size,
        deviation_threshold=args.deviation_threshold,
        save_intermediate_results=True,
    )
    
    # Create runner
    logger.info("Creating IG-Finder runner...")
    runner = IGFinderRunner(
        args=ig_args,
        lm_configs=lm_configs,
        rm=rm,
    )
    
    # Run pipeline
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting IG-Finder for topic: {args.topic}")
    logger.info(f"{'='*80}\n")
    
    try:
        report = runner.run(
            do_phase1=not args.skip_phase1,
            do_phase2=not args.skip_phase2,
            do_generate_report=True,
        )
        
        # Print summary
        runner.summary()
        
        # Print key results
        print("\n" + "="*80)
        print("KEY RESULTS")
        print("="*80)
        print(f"\nIdentified {len(report.identified_clusters)} innovation clusters:")
        for i, cluster in enumerate(report.identified_clusters, 1):
            print(f"\n{i}. {cluster.name}")
            print(f"   - Papers: {len(cluster.core_papers)}")
            print(f"   - Dimensions: {', '.join(cluster.innovation_dimensions)}")
            print(f"   - Coherence: {cluster.internal_coherence_score:.2f}")
            print(f"   - Summary: {cluster.cluster_summary[:200]}...")
        
        print("\n" + "="*80)
        print("INNOVATION GAP ANALYSIS")
        print("="*80)
        for dimension, gap in report.gap_analysis_by_dimension.items():
            print(f"\n{dimension}:")
            print(f"   - Evidence Strength: {gap.evidence_strength:.2f}")
            print(f"   - {gap.gap_description[:200]}...")
        
        print("\n" + "="*80)
        print(f"Full report saved to: {args.output_dir}/innovation_gap_report.md")
        print("="*80 + "\n")
        
        logger.info("IG-Finder execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
