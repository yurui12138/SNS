"""
IG-Finder 2.0 Engine

Main execution engine orchestrating all phases.
"""
import logging
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from ..interface import LMConfigs, Retriever
from ..lm import LitellmModel
from .dataclass import ResearchPaper, ReviewPaper
from .dataclass_v2 import (
    MultiViewBaseline,
    FitVector,
    IGFinder2Results,
    DeltaAwareGuidance,
    StressCluster,
    EvolutionProposal,
)
from .modules.phase1_multiview_baseline import Phase1Pipeline
from .modules.phase2_stress_test import Phase2Pipeline
from .modules.phase3_evolution import Phase3Pipeline
from .modules.phase4_guidance import Phase4Pipeline

logger = logging.getLogger(__name__)


@dataclass
class IGFinder2Arguments:
    """Arguments for IG-Finder 2.0 runner."""
    topic: str
    output_dir: str
    top_k_reviews: int = 15
    top_k_research_papers: int = 30
    min_cluster_size: int = 3
    save_intermediate_results: bool = True
    embedding_model: str = "dummy"  # In production: "allenai/specter2"
    lambda_regularization: float = 0.8


class IGFinder2Runner:
    """
    Main runner for IG-Finder 2.0.
    
    Orchestrates all phases:
    1. Multi-view Baseline Construction
    2. Multi-view Stress Test
    3. Stress Clustering & Minimal Evolution (simplified in v1)
    4. Delta-aware Guidance Generation (simplified in v1)
    """
    
    def __init__(
        self,
        args: IGFinder2Arguments,
        lm_configs: LMConfigs,
        rm: Retriever,
    ):
        self.args = args
        self.lm_configs = lm_configs
        self.rm = rm
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize pipelines
        self.phase1 = Phase1Pipeline(
            retriever=rm,
            lm=lm_configs.consensus_extraction_lm,
            top_k_reviews=args.top_k_reviews
        )
        
        self.phase2 = Phase2Pipeline(
            lm=lm_configs.deviation_analysis_lm,
            embedding_model=args.embedding_model
        )
        
        self.phase3 = Phase3Pipeline(
            lm=lm_configs.cluster_validation_lm,
            min_cluster_size=args.min_cluster_size,
            lambda_reg=args.lambda_regularization
        )
        
        self.phase4 = Phase4Pipeline()
        
        # State
        self.baseline: Optional[MultiViewBaseline] = None
        self.fit_vectors: List[FitVector] = []
        self.research_papers: List = []
        self.stress_clusters: List[StressCluster] = []
        self.evolution_proposal: Optional[EvolutionProposal] = None
        self.results: Optional[IGFinder2Results] = None
    
    def run(
        self,
        do_phase1: bool = True,
        do_phase2: bool = True,
        do_phase3: bool = True,
        do_phase4: bool = True,
    ) -> IGFinder2Results:
        """
        Run the complete IG-Finder 2.0 pipeline.
        
        Args:
            do_phase1: Run Phase 1 (multi-view baseline)
            do_phase2: Run Phase 2 (stress test)
            do_phase3: Run Phase 3 (clustering & evolution)
            do_phase4: Run Phase 4 (guidance generation)
            
        Returns:
            IGFinder2Results object
        """
        logger.info("="*80)
        logger.info("IG-FINDER 2.0: Multi-view Atlas Stress Test")
        logger.info(f"Topic: {self.args.topic}")
        logger.info("="*80)
        
        # Phase 1: Multi-view Baseline
        if do_phase1:
            baseline_path = os.path.join(self.args.output_dir, "multiview_baseline.json")
            if os.path.exists(baseline_path) and not do_phase1:
                logger.info("Loading existing multi-view baseline...")
                with open(baseline_path, 'r') as f:
                    self.baseline = MultiViewBaseline.from_dict(json.load(f))
            else:
                self.baseline = self.phase1.run(self.args.topic)
                
                if self.args.save_intermediate_results:
                    logger.info(f"Saving baseline to {baseline_path}")
                    with open(baseline_path, 'w') as f:
                        json.dump(self.baseline.to_dict(), f, indent=2)
        
        if not self.baseline:
            raise ValueError("No baseline available. Run Phase 1 first.")
        
        # Phase 2: Stress Test
        if do_phase2:
            fit_vectors_path = os.path.join(self.args.output_dir, "fit_vectors.json")
            if os.path.exists(fit_vectors_path) and not do_phase2:
                logger.info("Loading existing fit vectors...")
                with open(fit_vectors_path, 'r') as f:
                    data = json.load(f)
                    self.fit_vectors = [FitVector.from_dict(fv) for fv in data]
            else:
                # Retrieve research papers
                self.research_papers = self._retrieve_research_papers()
                
                # Run stress test
                self.fit_vectors = self.phase2.run(self.research_papers, self.baseline)
                
                if self.args.save_intermediate_results:
                    logger.info(f"Saving fit vectors to {fit_vectors_path}")
                    with open(fit_vectors_path, 'w') as f:
                        json.dump([fv.to_dict() for fv in self.fit_vectors], f, indent=2)
        
        # Phase 3: Stress Clustering & Minimal Evolution
        if do_phase3:
            clusters_path = os.path.join(self.args.output_dir, "stress_clusters.json")
            proposal_path = os.path.join(self.args.output_dir, "evolution_proposal.json")
            
            if os.path.exists(clusters_path) and os.path.exists(proposal_path) and not do_phase3:
                logger.info("Loading existing stress clusters and evolution proposal...")
                with open(clusters_path, 'r') as f:
                    data = json.load(f)
                    self.stress_clusters = [StressCluster.from_dict(c) for c in data]
                with open(proposal_path, 'r') as f:
                    self.evolution_proposal = EvolutionProposal.from_dict(json.load(f))
            else:
                self.stress_clusters, self.evolution_proposal = self.phase3.run(
                    self.fit_vectors,
                    self.research_papers,
                    self.baseline
                )
                
                if self.args.save_intermediate_results:
                    logger.info(f"Saving stress clusters to {clusters_path}")
                    with open(clusters_path, 'w') as f:
                        json.dump([c.to_dict() for c in self.stress_clusters], f, indent=2)
                    logger.info(f"Saving evolution proposal to {proposal_path}")
                    with open(proposal_path, 'w') as f:
                        json.dump(self.evolution_proposal.to_dict(), f, indent=2)
        else:
            self.stress_clusters = []
            self.evolution_proposal = EvolutionProposal(
                operations=[],
                total_fit_gain=0.0,
                total_edit_cost=0.0,
                objective_value=0.0
            )
        
        # Phase 4: Delta-aware Guidance Generation
        if do_phase4:
            guidance_path = os.path.join(self.args.output_dir, "delta_guidance.json")
            
            if os.path.exists(guidance_path) and not do_phase4:
                logger.info("Loading existing delta-aware guidance...")
                with open(guidance_path, 'r') as f:
                    delta_guidance = DeltaAwareGuidance.from_dict(json.load(f))
            else:
                delta_guidance = self.phase4.run(
                    topic=self.args.topic,
                    baseline=self.baseline,
                    fit_vectors=self.fit_vectors,
                    papers=self.research_papers,
                    clusters=self.stress_clusters,
                    evolution_proposal=self.evolution_proposal
                )
                
                if self.args.save_intermediate_results:
                    logger.info(f"Saving delta-aware guidance to {guidance_path}")
                    with open(guidance_path, 'w') as f:
                        json.dump(delta_guidance.to_dict(), f, indent=2)
        else:
            delta_guidance = self._generate_simplified_guidance()
        
        # Assemble results
        self.results = IGFinder2Results(
            topic=self.args.topic,
            multiview_baseline=self.baseline,
            fit_vectors=self.fit_vectors,
            stress_clusters=self.stress_clusters,
            evolution_proposal=self.evolution_proposal,
            delta_aware_guidance=delta_guidance,
            statistics=self._compute_statistics(),
            generation_date=datetime.now()
        )
        
        # Save final results
        self._save_results()
        
        logger.info("="*80)
        logger.info("IG-FINDER 2.0 COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return self.results
    
    def _retrieve_research_papers(self) -> List:
        """Retrieve research papers (non-review)."""
        from ...interface import Information
        
        logger.info("Retrieving research papers...")
        
        queries = [
            f"{self.args.topic}",
            f"{self.args.topic} method",
            f"{self.args.topic} approach",
            f"recent advances in {self.args.topic}",
        ]
        
        all_papers = []
        for query in queries:
            try:
                results = self.rm.retrieve(query=query, exclude_urls=[])
                all_papers.extend(results)
            except Exception as e:
                logger.warning(f"Error retrieving papers with query '{query}': {e}")
        
        # Remove duplicates
        seen_urls = set()
        unique_papers = []
        for paper in all_papers:
            if paper.url not in seen_urls:
                seen_urls.add(paper.url)
                unique_papers.append(paper)
        
        # Filter out reviews
        research_papers = []
        review_keywords = ['survey', 'review', 'overview']
        for paper in unique_papers:
            title_lower = paper.title.lower()
            if not any(kw in title_lower for kw in review_keywords):
                research_papers.append(paper)
        
        logger.info(f"Retrieved {len(research_papers)} research papers")
        
        return research_papers[:self.args.top_k_research_papers]
    
    def _generate_simplified_guidance(self) -> DeltaAwareGuidance:
        """Generate simplified delta-aware guidance."""
        from .dataclass_v2 import Section, Subsection, EvidenceCard, EvolutionSummaryItem
        
        # Simplified: Just use first view as main axis
        if not self.baseline or not self.baseline.views:
            raise ValueError("No baseline views available")
        
        main_axis = self.baseline.views[0]
        aux_axis = self.baseline.views[1] if len(self.baseline.views) > 1 else None
        
        # Create simple outline
        outline = [
            Section(
                section=f"{main_axis.facet_label.value} (Main Organization)",
                subsections=[
                    Subsection(
                        subsection="Overview",
                        required_nodes=[],
                        required_citations=[],
                        must_answer=["What are the main organizational dimensions?"],
                        evidence_cards=[]
                    )
                ]
            )
        ]
        
        guidance = DeltaAwareGuidance(
            topic=self.args.topic,
            main_axis=main_axis,
            aux_axis=aux_axis,
            outline=outline,
            evolution_summary=[],
            must_answer_questions=[
                "What are the key organizational dimensions in existing reviews?",
                "Which papers show structural stress?",
                "What minimal changes are needed to accommodate new research?"
            ]
        )
        
        return guidance
    
    def _compute_statistics(self) -> dict:
        """Compute statistics about the results."""
        if not self.baseline or not self.fit_vectors:
            return {}
        
        total_papers = len(self.fit_vectors)
        total_views = len(self.baseline.views)
        
        # Count labels
        fit_count = sum(
            1 for fv in self.fit_vectors
            for report in fv.fit_reports
            if report.label.value == "FIT"
        )
        force_fit_count = sum(
            1 for fv in self.fit_vectors
            for report in fv.fit_reports
            if report.label.value == "FORCE_FIT"
        )
        unfittable_count = sum(
            1 for fv in self.fit_vectors
            for report in fv.fit_reports
            if report.label.value == "UNFITTABLE"
        )
        
        total_tests = total_papers * total_views
        
        return {
            "total_papers": total_papers,
            "total_views": total_views,
            "total_tests": total_tests,
            "fit_count": fit_count,
            "force_fit_count": force_fit_count,
            "unfittable_count": unfittable_count,
            "fit_rate": fit_count / total_tests if total_tests > 0 else 0.0,
            "force_fit_rate": force_fit_count / total_tests if total_tests > 0 else 0.0,
            "unfittable_rate": unfittable_count / total_tests if total_tests > 0 else 0.0,
            "avg_stress_score": sum(fv.stress_score for fv in self.fit_vectors) / total_papers if total_papers > 0 else 0.0,
            "avg_unfittable_score": sum(fv.unfittable_score for fv in self.fit_vectors) / total_papers if total_papers > 0 else 0.0,
        }
    
    def _save_results(self):
        """Save final results to disk."""
        if not self.results:
            return
        
        # Save complete results as JSON
        results_path = os.path.join(self.args.output_dir, "igfinder2_results.json")
        logger.info(f"Saving complete results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)
        
        # Save human-readable report
        report_path = os.path.join(self.args.output_dir, "igfinder2_report.md")
        logger.info(f"Saving human-readable report to {report_path}")
        self._generate_markdown_report(report_path)
    
    def _generate_markdown_report(self, output_path: str):
        """Generate markdown report."""
        if not self.results:
            return
        
        stats = self.results.statistics
        
        report = f"""# IG-Finder 2.0 Report

## Topic
{self.results.topic}

## Generation Date
{self.results.generation_date.strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

- **Total Papers Analyzed**: {stats.get('total_papers', 0)}
- **Total Taxonomy Views**: {stats.get('total_views', 0)}
- **Total Fit Tests**: {stats.get('total_tests', 0)}

### Fit Test Results

- **FIT**: {stats.get('fit_count', 0)} ({stats.get('fit_rate', 0.0)*100:.1f}%)
- **FORCE_FIT**: {stats.get('force_fit_count', 0)} ({stats.get('force_fit_rate', 0.0)*100:.1f}%)
- **UNFITTABLE**: {stats.get('unfittable_count', 0)} ({stats.get('unfittable_rate', 0.0)*100:.1f}%)

### Stress Scores

- **Average Stress Score**: {stats.get('avg_stress_score', 0.0):.3f}
- **Average Unfittable Score**: {stats.get('avg_unfittable_score', 0.0):.3f}

## Multi-view Baseline

"""
        
        for view in self.results.multiview_baseline.views:
            report += f"""### View {view.view_id}: {view.facet_label.value}
- **Source**: {view.review_title}
- **Weight**: {view.weight:.3f}
- **Rationale**: {view.facet_rationale}
- **Leaf Nodes**: {len(view.tree.get_leaf_nodes())}

"""
        
        report += """## High-Stress Papers

Papers with stress_score > 0.5:

"""
        
        for fv in sorted(self.fit_vectors, key=lambda x: x.stress_score, reverse=True)[:10]:
            if fv.stress_score > 0.5:
                report += f"""- Paper ID: {fv.paper_id}
  - Stress Score: {fv.stress_score:.3f}
  - Unfittable Score: {fv.unfittable_score:.3f}
  
"""
        
        report += """## Delta-aware Guidance

### Main Axis
"""
        guidance = self.results.delta_aware_guidance
        report += f"- **Facet**: {guidance.main_axis.facet_label.value}\n"
        report += f"- **Source**: {guidance.main_axis.review_title}\n\n"
        
        if guidance.aux_axis:
            report += """### Auxiliary Axis
"""
            report += f"- **Facet**: {guidance.aux_axis.facet_label.value}\n"
            report += f"- **Source**: {guidance.aux_axis.review_title}\n\n"
        
        report += """### Key Questions to Address

"""
        for question in guidance.must_answer_questions:
            report += f"- {question}\n"
        
        report += "\n---\n\n"
        report += "*Generated by IG-Finder 2.0*\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
    
    def summary(self):
        """Print summary of results."""
        if not self.results:
            logger.warning("No results available")
            return
        
        stats = self.results.statistics
        
        print("\n" + "="*80)
        print("IG-FINDER 2.0 SUMMARY")
        print("="*80)
        print(f"\nTopic: {self.results.topic}")
        print(f"\nAnalyzed {stats.get('total_papers', 0)} papers across {stats.get('total_views', 0)} taxonomy views")
        print(f"\nFit Test Results:")
        print(f"  FIT:         {stats.get('fit_count', 0):4d} ({stats.get('fit_rate', 0.0)*100:5.1f}%)")
        print(f"  FORCE_FIT:   {stats.get('force_fit_count', 0):4d} ({stats.get('force_fit_rate', 0.0)*100:5.1f}%)")
        print(f"  UNFITTABLE:  {stats.get('unfittable_count', 0):4d} ({stats.get('unfittable_rate', 0.0)*100:5.1f}%)")
        print(f"\nAverage Stress Score: {stats.get('avg_stress_score', 0.0):.3f}")
        print(f"Average Unfittable Score: {stats.get('avg_unfittable_score', 0.0):.3f}")
        print(f"\nResults saved to: {self.args.output_dir}")
        print("="*80 + "\n")
