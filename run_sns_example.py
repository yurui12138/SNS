"""
Example script for running IG-Finder 2.0

This demonstrates the new multi-view atlas stress test approach.
"""
import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from knowledge.sns import (
    SNSRunner,
    SNSArguments,
    SNSLMConfigs,
)
from knowledge.rm import ArxivSearchRM
from knowledge.lm import LitellmModel
from knowledge.interface import Retriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run IG-Finder 2.0')
    parser.add_argument(
        '--topic',
        type=str,
        required=True,
        help='Research topic to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./igfinder2_output',
        help='Output directory'
    )
    parser.add_argument(
        '--top-k-reviews',
        type=int,
        default=5,
        help='Number of review papers to retrieve'
    )
    parser.add_argument(
        '--top-k-research',
        type=int,
        default=10,
        help='Number of research papers to retrieve'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='API base URL'
    )
    
    args = parser.parse_args()
    
    # Setup API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY') or 'sk-w15EcYnx4q4Q2xJAJik8bjZ5gW5rXz2oyzoBPopd4xnJ519H'
    api_base = args.api_base or os.getenv('OPENAI_API_BASE') or 'https://api.yunwu.ai/v1'
    
    logger.info("="*80)
    logger.info("IG-FINDER 2.0: Multi-view Atlas Stress Test")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"Model: {args.model}")
    logger.info(f"API Base: {api_base}")
    logger.info("="*80)
    
    # Setup language models
    logger.info("Initializing language models...")
    lm_configs = SNSLMConfigs()
    
    openai_kwargs = {
        'api_key': api_key,
        'api_base': api_base,
        'temperature': 0.0,  # Temperature=0 for reproducibility
        'top_p': 0.9,
    }
    
    # Create language models for different tasks
    consensus_lm = LitellmModel(model=args.model, max_tokens=3000, **openai_kwargs)
    deviation_lm = LitellmModel(model=args.model, max_tokens=2000, **openai_kwargs)
    cluster_lm = LitellmModel(model=args.model, max_tokens=1500, **openai_kwargs)
    report_lm = LitellmModel(model=args.model, max_tokens=4000, **openai_kwargs)
    
    lm_configs.set_consensus_extraction_lm(consensus_lm)
    lm_configs.set_deviation_analysis_lm(deviation_lm)
    lm_configs.set_cluster_validation_lm(cluster_lm)
    lm_configs.set_report_generation_lm(report_lm)
    
    logger.info("Language models initialized")
    
    # Setup retriever
    logger.info("Initializing arXiv retriever...")
    rm = Retriever(rm=ArxivSearchRM(k=10))
    
    # Create IG-Finder 2.0 arguments
    igfinder2_args = SNSArguments(
        topic=args.topic,
        output_dir=args.output_dir,
        top_k_reviews=args.top_k_reviews,
        top_k_research_papers=args.top_k_research,
        min_cluster_size=2,
        save_intermediate_results=True,
        embedding_model="dummy",  # Using simple embedding for demo
        lambda_regularization=0.8,
    )
    
    # Create runner
    logger.info("Creating IG-Finder 2.0 runner...")
    runner = SNSRunner(
        args=igfinder2_args,
        lm_configs=lm_configs,
        rm=rm,
    )
    
    # Run pipeline
    logger.info("\n" + "="*80)
    logger.info("Starting IG-Finder 2.0 Pipeline")
    logger.info("="*80 + "\n")
    
    try:
        results = runner.run(
            do_phase1=True,
            do_phase2=True,
            do_phase3=False,  # Not yet implemented
            do_phase4=False,  # Not yet implemented
        )
        
        # Print summary
        runner.summary()
        
        # Print key findings
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        print(f"\n✓ Extracted {len(results.multiview_baseline.views)} taxonomy views from reviews")
        
        for view in results.multiview_baseline.views:
            print(f"  - {view.view_id}: {view.facet_label.value} (weight={view.weight:.3f})")
            print(f"    Source: {view.review_title[:80]}...")
            print(f"    Leaf nodes: {len(view.tree.get_leaf_nodes())}")
        
        print(f"\n✓ Tested {len(results.fit_vectors)} papers against multi-view baseline")
        
        # Find high-stress papers
        high_stress = [fv for fv in results.fit_vectors if fv.stress_score > 0.5]
        print(f"\n✓ Identified {len(high_stress)} high-stress papers (stress > 0.5)")
        
        if high_stress:
            print("\n  Top 5 high-stress papers:")
            for fv in sorted(high_stress, key=lambda x: x.stress_score, reverse=True)[:5]:
                print(f"    - Stress: {fv.stress_score:.3f}, Unfittable: {fv.unfittable_score:.3f}")
                print(f"      Paper: {fv.paper_id[:80]}...")
        
        print(f"\n✓ Main organizational axis: {results.delta_aware_guidance.main_axis.facet_label.value}")
        if results.delta_aware_guidance.aux_axis:
            print(f"✓ Auxiliary axis: {results.delta_aware_guidance.aux_axis.facet_label.value}")
        
        print("\n" + "="*80)
        print(f"✓ Complete results saved to: {args.output_dir}")
        print("  - multiview_baseline.json: Multi-view taxonomy atlas")
        print("  - fit_vectors.json: Stress test results for all papers")
        print("  - igfinder2_results.json: Complete structured results")
        print("  - igfinder2_report.md: Human-readable report")
        print("="*80 + "\n")
        
        logger.info("IG-Finder 2.0 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
