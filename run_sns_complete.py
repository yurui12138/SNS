#!/usr/bin/env python3
"""
SNS: Self-Nonself Modeling - Complete End-to-End Example

This script demonstrates the full SNS pipeline:
1. Phase 1: Multi-view Baseline (Self Construction)
2. Phase 2: Multi-view Stress Test (Nonself Identification)
3. Phase 3: Stress Clustering & Evolution (Adaptation)
4. Phase 4: Delta-aware Guidance Generation
5. Evaluation Framework

Usage:
    python run_sns_complete.py --topic "transformer neural networks" --output-dir ./output

Requirements:
    - OpenAI API key (or compatible endpoint)
    - arXiv API access
    - Optional: GPU for embedding/NLI models
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import ArxivSearchRM as ArxivRM
from knowledge_storm.sns.engine_v2 import SNSRunner, SNSArguments
from knowledge_storm.interface import LMConfigs, Retriever
from knowledge_storm.sns.evaluation import compute_all_metrics, print_metrics_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run SNS (Self-Nonself) complete pipeline')
    
    # Basic arguments
    parser.add_argument('--topic', type=str, required=True,
                       help='Research topic to analyze')
    parser.add_argument('--output-dir', type=str, default='./sns_output',
                       help='Output directory for results')
    
    # API configuration
    parser.add_argument('--openai-api-key', type=str,
                       default=os.environ.get('OPENAI_API_KEY'),
                       help='OpenAI API key')
    parser.add_argument('--openai-api-base', type=str,
                       default=os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                       help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='LLM model to use')
    
    # Pipeline parameters
    parser.add_argument('--top-k-reviews', type=int, default=15,
                       help='Number of review papers to retrieve')
    parser.add_argument('--top-k-research', type=int, default=30,
                       help='Number of research papers to analyze')
    parser.add_argument('--min-cluster-size', type=int, default=3,
                       help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--lambda-reg', type=float, default=0.8,
                       help='Lambda regularization for evolution objective')
    
    # Embedding/NLI configuration
    parser.add_argument('--embedding-model', type=str, default='dummy',
                       choices=['dummy', 'specter2', 'scincl', 'sbert'],
                       help='Embedding model for semantic similarity')
    parser.add_argument('--nli-model', type=str, default='dummy',
                       choices=['dummy', 'deberta', 'roberta'],
                       help='NLI model for conflict detection')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for neural models')
    
    # Phase control
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1 (load from cache)')
    parser.add_argument('--skip-phase2', action='store_true',
                       help='Skip Phase 2 (load from cache)')
    parser.add_argument('--skip-phase3', action='store_true',
                       help='Skip Phase 3 (load from cache)')
    parser.add_argument('--skip-phase4', action='store_true',
                       help='Skip Phase 4 (load from cache)')
    
    # Evaluation
    parser.add_argument('--run-evaluation', action='store_true',
                       help='Run evaluation framework after pipeline')
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.openai_api_key:
        logger.error("OpenAI API key is required. Set via --openai-api-key or OPENAI_API_KEY env var")
        return
    
    # Configure LLMs
    logger.info("Configuring language models...")
    lm_configs = LMConfigs()
    
    # Use same model for all tasks (can be customized)
    for attr_name in ['consensus_extraction_lm', 'deviation_analysis_lm', 
                      'cluster_validation_lm', 'report_generation_lm']:
        setattr(lm_configs, attr_name, LitellmModel(
            model=args.model,
            max_tokens=3000,
            temperature=0,
            api_key=args.openai_api_key,
            api_base=args.openai_api_base
        ))
    
    # Configure retrieval
    logger.info("Configuring retrieval module...")
    rm = Retriever(rm=ArxivRM())
    logger.info(f"Retriever type: {type(rm)}")
    logger.info(f"Inner RM type: {type(rm.rm)}")
    
    # Create runner arguments
    runner_args = SNSArguments(
        topic=args.topic,
        output_dir=args.output_dir,
        top_k_reviews=args.top_k_reviews,
        top_k_research_papers=args.top_k_research,
        min_cluster_size=args.min_cluster_size,
        save_intermediate_results=True,
        embedding_model=args.embedding_model,
        lambda_regularization=args.lambda_reg
    )
    
    # Create runner
    logger.info("Initializing IG-Finder 2.0 Runner...")
    runner = SNSRunner(
        args=runner_args,
        lm_configs=lm_configs,
        rm=rm
    )
    
    # Run pipeline
    logger.info("="*80)
    logger.info("Starting IG-Finder 2.0 Complete Pipeline")
    logger.info("="*80)
    
    try:
        results = runner.run(
            do_phase1=not args.skip_phase1,
            do_phase2=not args.skip_phase2,
            do_phase3=not args.skip_phase3,
            do_phase4=not args.skip_phase4
        )
        
        # Print summary
        runner.summary()
        
        # Run evaluation if requested
        if args.run_evaluation:
            logger.info("\n" + "="*80)
            logger.info("Running Evaluation Framework")
            logger.info("="*80)
            
            metrics = compute_all_metrics(
                fit_vectors=results.fit_vectors,
                evolution_proposal=results.evolution_proposal,
                original_tree=results.multiview_baseline.views[0].tree if results.multiview_baseline.views else None,
                evolved_tree=None  # Would need to apply operations to get evolved tree
            )
            
            # Print metrics report
            print_metrics_report(metrics)
            
            # Save metrics
            import json
            metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
            with open(metrics_path, 'w') as f:
                # Convert non-serializable values
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                        serializable_metrics[k] = v
                    else:
                        serializable_metrics[k] = str(v)
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        logger.info("\n✅ IG-Finder 2.0 pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {args.output_dir}")
        logger.info("\nGenerated files:")
        logger.info(f"  - multiview_baseline.json     (Phase 1 output)")
        logger.info(f"  - fit_vectors.json            (Phase 2 output)")
        logger.info(f"  - stress_clusters.json        (Phase 3 output)")
        logger.info(f"  - evolution_proposal.json     (Phase 3 output)")
        logger.info(f"  - delta_guidance.json         (Phase 4 output)")
        logger.info(f"  - igfinder2_results.json      (Complete results)")
        logger.info(f"  - igfinder2_report.md         (Human-readable report)")
        if args.run_evaluation:
            logger.info(f"  - evaluation_metrics.json     (Evaluation metrics)")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
