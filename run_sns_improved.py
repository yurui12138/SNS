"""
Improved SNS Example Script

This script demonstrates running SNS with improved configurations
to address the critical issues found in testing.
"""
import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from knowledge_storm.sns import (
    SNSRunner,
    SNSLMConfigs,
)
from knowledge_storm.rm import ArxivSearchRM
from knowledge_storm.lm import LitellmModel
from knowledge_storm.interface import Retriever

# Import quick fixes
from quick_fixes import (
    get_improved_sns_args,
    validate_sns_results,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run SNS with Improved Configuration'
    )
    parser.add_argument(
        '--topic',
        type=str,
        required=True,
        help='Research topic to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sns_output_improved',
        help='Output directory'
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
    parser.add_argument(
        '--use-dummy-embedding',
        action='store_true',
        help='Use dummy embedding (for testing only, not recommended)'
    )
    
    args = parser.parse_args()
    
    # Setup API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    api_base = args.api_base or os.getenv('OPENAI_API_BASE')
    
    if not api_key:
        logger.error("No API key provided. Use --api-key or set OPENAI_API_KEY")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("SNS: Self-Nonself Modeling (IMPROVED VERSION)")
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
        'temperature': 0.0,
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
    
    logger.info("✅ Language models initialized")
    
    # Setup retriever
    logger.info("Initializing arXiv retriever...")
    rm = Retriever(rm=ArxivSearchRM(k=20))  # Increased from 10 to 20
    
    # ✅ Get improved SNS arguments
    logger.info("\n" + "="*80)
    logger.info("Using IMPROVED configuration:")
    logger.info("="*80)
    
    sns_args = get_improved_sns_args(args.topic, args.output_dir)
    
    # Override embedding if user wants dummy (not recommended)
    if args.use_dummy_embedding:
        logger.warning("⚠️  Using dummy embedding (NOT RECOMMENDED for production)")
        sns_args.embedding_model = "dummy"
    
    logger.info(f"  top_k_reviews: {sns_args.top_k_reviews} (increased from 5)")
    logger.info(f"  top_k_research_papers: {sns_args.top_k_research_papers} (increased from 10)")
    logger.info(f"  min_cluster_size: {sns_args.min_cluster_size} (decreased from 3)")
    logger.info(f"  embedding_model: {sns_args.embedding_model} (changed from 'dummy')")
    logger.info("="*80 + "\n")
    
    # Check if SPECTER2 is available
    if sns_args.embedding_model == "allenai/specter2":
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("✅ sentence-transformers is available")
        except ImportError:
            logger.error("❌ sentence-transformers not installed!")
            logger.error("   Please install: pip install sentence-transformers")
            logger.error("   Falling back to dummy embedding...")
            sns_args.embedding_model = "dummy"
    
    # Create runner
    logger.info("Creating SNS runner...")
    runner = SNSRunner(
        args=sns_args,
        lm_configs=lm_configs,
        rm=rm,
    )
    
    # Run pipeline
    logger.info("\n" + "="*80)
    logger.info("Starting SNS Pipeline")
    logger.info("="*80 + "\n")
    
    try:
        results = runner.run(
            do_phase1=True,
            do_phase2=True,
            do_phase3=True,   # Enable Phase 3
            do_phase4=True,   # Enable Phase 4
        )
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EXECUTION SUMMARY")
        logger.info("="*80)
        runner.summary()
        
        # ✅ Validate results and show warnings
        logger.info("\n" + "="*80)
        logger.info("QUALITY VALIDATION")
        logger.info("="*80)
        
        validation = validate_sns_results(
            results.fit_vectors,
            results.stress_clusters,
            results.multiview_baseline
        )
        
        # Print key findings
        logger.info("\n" + "="*80)
        logger.info("KEY FINDINGS")
        logger.info("="*80)
        
        logger.info(f"\n📊 Multi-view Baseline:")
        logger.info(f"  Extracted {len(results.multiview_baseline.views)} taxonomy views")
        
        unique_facets = len(set(v.facet_label.value for v in results.multiview_baseline.views))
        logger.info(f"  Unique facets: {unique_facets}")
        
        for view in results.multiview_baseline.views:
            logger.info(f"\n  View {view.view_id}:")
            logger.info(f"    Facet: {view.facet_label.value}")
            logger.info(f"    Weight: {view.weight:.3f}")
            logger.info(f"    Leaf nodes: {len(view.tree.get_leaf_nodes())}")
            logger.info(f"    Source: {view.review_title[:80]}...")
        
        logger.info(f"\n📝 Stress Test Results:")
        logger.info(f"  Tested {len(results.fit_vectors)} papers")
        
        # Calculate fit statistics
        total_tests = sum(len(fv.fit_reports) for fv in results.fit_vectors)
        fit_count = sum(1 for fv in results.fit_vectors 
                        for fr in fv.fit_reports 
                        if str(fr.label) == "FIT")
        force_fit_count = sum(1 for fv in results.fit_vectors 
                              for fr in fv.fit_reports 
                              if str(fr.label) == "FORCE_FIT")
        unfittable_count = sum(1 for fv in results.fit_vectors 
                               for fr in fv.fit_reports 
                               if str(fr.label) == "UNFITTABLE")
        
        logger.info(f"  FIT: {fit_count} ({fit_count/total_tests*100:.1f}%)")
        logger.info(f"  FORCE_FIT: {force_fit_count} ({force_fit_count/total_tests*100:.1f}%)")
        logger.info(f"  UNFITTABLE: {unfittable_count} ({unfittable_count/total_tests*100:.1f}%)")
        
        # Find high-stress papers
        high_stress = [fv for fv in results.fit_vectors if fv.stress_score > 0.5]
        logger.info(f"\n🔴 High-stress papers: {len(high_stress)} (stress > 0.5)")
        
        if high_stress:
            logger.info("\n  Top 3 high-stress papers:")
            for fv in sorted(high_stress, key=lambda x: x.stress_score, reverse=True)[:3]:
                logger.info(f"    - Stress: {fv.stress_score:.3f}, Unfittable: {fv.unfittable_score:.3f}")
                logger.info(f"      Paper: {fv.paper_id[:80]}...")
        
        logger.info(f"\n🔬 Evolution Analysis:")
        logger.info(f"  Stress clusters: {len(results.stress_clusters)}")
        logger.info(f"  Evolution operations: {len(results.evolution_proposal.operations)}")
        
        if results.evolution_proposal.operations:
            logger.info(f"\n  Proposed operations:")
            for op in results.evolution_proposal.operations[:5]:  # Show top 5
                logger.info(f"    - {op.operation_type}: {op.parent_path} → {op.new_node_proposal.name if hasattr(op, 'new_node_proposal') else 'N/A'}")
        
        logger.info(f"\n📖 Delta-aware Guidance:")
        if results.delta_aware_guidance:
            logger.info(f"  Main axis: {results.delta_aware_guidance.main_axis.facet_label.value}")
            if results.delta_aware_guidance.aux_axis:
                logger.info(f"  Auxiliary axis: {results.delta_aware_guidance.aux_axis.facet_label.value}")
            logger.info(f"  Writing mode: {results.delta_aware_guidance.main_axis_mode.value}")
            logger.info(f"  Outline sections: {len(results.delta_aware_guidance.outline)}")
            logger.info(f"  Writing rules: {len(results.delta_aware_guidance.writing_rules.do)} do, {len(results.delta_aware_guidance.writing_rules.dont)} dont")
        
        # Output files summary
        logger.info("\n" + "="*80)
        logger.info("OUTPUT FILES")
        logger.info("="*80)
        logger.info(f"📁 Output directory: {args.output_dir}")
        logger.info("\nPrimary outputs:")
        logger.info("  ✅ audit_report.md - Human-readable audit report")
        logger.info("  ✅ guidance_pack.json - Machine-readable guidance for downstream systems")
        logger.info("\nIntermediate files:")
        logger.info("  - multiview_baseline.json - Multi-view taxonomy atlas")
        logger.info("  - fit_vectors.json - Stress test results")
        logger.info("  - stress_clusters.json - Clustered stressed papers")
        logger.info("  - evolution_proposal.json - Proposed structure updates")
        logger.info("  - sns_results.json - Complete results bundle")
        logger.info("="*80 + "\n")
        
        logger.info("✅ SNS completed successfully!")
        
        # Final recommendations
        if validation['warnings']:
            logger.info("\n" + "="*80)
            logger.info("⚠️  RECOMMENDATIONS FOR NEXT RUN")
            logger.info("="*80)
            
            high_warnings = [w for w in validation['warnings'] if w['severity'] == 'HIGH']
            if high_warnings:
                logger.info("\nHigh priority:")
                for warning in high_warnings:
                    logger.info(f"  - {warning['category']}: {warning['message']}")
                    if warning['recommendations']:
                        logger.info(f"    Recommendation: {warning['recommendations'][0]}")
            
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
