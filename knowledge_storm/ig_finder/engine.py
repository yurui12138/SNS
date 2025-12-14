"""
IG-Finder Engine

Main execution engine for the Innovation Gap Finder framework.
Orchestrates the two-phase workflow: Cognitive Self Construction and
Innovative Non-self Identification.
"""

import dspy
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

from ..interface import LMConfigs, Retriever
from ..lm import LitellmModel
from ..dataclass import KnowledgeBase
from .dataclass import CognitiveBaseline, InnovationGapReport
from .modules import (
    CognitiveSelfConstructionModule,
    InnovativeNonSelfIdentificationModule,
    DynamicMindMapManager,
    InnovationGapReportGenerator,
)

logger = logging.getLogger(__name__)


class IGFinderLMConfigs(LMConfigs):
    """
    Language model configurations for IG-Finder framework.
    
    Similar to STORMWikiLMConfigs but adapted for IG-Finder's specific needs.
    """
    
    def __init__(self):
        super().__init__()
        self.consensus_extraction_lm = None  # For extracting consensus from reviews
        self.deviation_analysis_lm = None    # For analyzing deviations
        self.cluster_validation_lm = None    # For validating innovation clusters
        self.report_generation_lm = None     # For generating final report
    
    def init(
        self,
        lm_type: Literal["openai", "azure", "together"],
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Initialize with default configurations based on provider."""
        if lm_type == "openai":
            openai_kwargs = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
                "api_base": None,
            }
            # Use GPT-4 for complex reasoning tasks
            self.consensus_extraction_lm = LitellmModel(
                model="gpt-4o", max_tokens=3000, **openai_kwargs
            )
            self.deviation_analysis_lm = LitellmModel(
                model="gpt-4o", max_tokens=2000, **openai_kwargs
            )
            self.cluster_validation_lm = LitellmModel(
                model="gpt-4o", max_tokens=1500, **openai_kwargs
            )
            self.report_generation_lm = LitellmModel(
                model="gpt-4o", max_tokens=4000, **openai_kwargs
            )
        elif lm_type == "azure":
            azure_kwargs = {
                "api_key": os.getenv("AZURE_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": os.getenv("AZURE_API_VERSION"),
            }
            self.consensus_extraction_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=3000, **azure_kwargs, model_type="chat"
            )
            self.deviation_analysis_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=2000, **azure_kwargs, model_type="chat"
            )
            self.cluster_validation_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=1500, **azure_kwargs, model_type="chat"
            )
            self.report_generation_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=4000, **azure_kwargs, model_type="chat"
            )
        elif lm_type == "together":
            together_kwargs = {
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "temperature": temperature,
                "top_p": top_p,
            }
            model_name = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            self.consensus_extraction_lm = LitellmModel(
                model=model_name, max_tokens=3000, model_type="chat", **together_kwargs
            )
            self.deviation_analysis_lm = LitellmModel(
                model=model_name, max_tokens=2000, model_type="chat", **together_kwargs
            )
            self.cluster_validation_lm = LitellmModel(
                model=model_name, max_tokens=1500, model_type="chat", **together_kwargs
            )
            self.report_generation_lm = LitellmModel(
                model=model_name, max_tokens=4000, model_type="chat", **together_kwargs
            )
        else:
            raise Exception(
                f"Unsupported LM type: {lm_type}. Choose from 'openai', 'azure', or 'together'."
            )
    
    def set_consensus_extraction_lm(self, model: dspy.LM):
        self.consensus_extraction_lm = model
    
    def set_deviation_analysis_lm(self, model: dspy.LM):
        self.deviation_analysis_lm = model
    
    def set_cluster_validation_lm(self, model: dspy.LM):
        self.cluster_validation_lm = model
    
    def set_report_generation_lm(self, model: dspy.LM):
        self.report_generation_lm = model


@dataclass
class IGFinderArguments:
    """Arguments for configuring IG-Finder pipeline."""
    
    topic: str = field(
        metadata={"help": "Research topic to analyze"}
    )
    output_dir: str = field(
        default="./ig_finder_output",
        metadata={"help": "Output directory for results"}
    )
    top_k_reviews: int = field(
        default=10,
        metadata={"help": "Number of review papers to retrieve for cognitive baseline"}
    )
    top_k_research_papers: int = field(
        default=30,
        metadata={"help": "Number of research papers to retrieve for innovation identification"}
    )
    min_cluster_size: int = field(
        default=2,
        metadata={"help": "Minimum number of papers to form an innovation cluster"}
    )
    deviation_threshold: float = field(
        default=0.5,
        metadata={"help": "Minimum deviation score (0-1) to consider as significant"}
    )
    save_intermediate_results: bool = field(
        default=True,
        metadata={"help": "Whether to save intermediate results"}
    )


class IGFinderRunner:
    """
    Main execution engine for IG-Finder framework.
    
    Usage:
        >>> runner = IGFinderRunner(args, lm_configs, rm)
        >>> runner.run()
        >>> report = runner.get_report()
    """
    
    def __init__(
        self,
        args: IGFinderArguments,
        lm_configs: IGFinderLMConfigs,
        rm: Retriever,
    ):
        self.args = args
        self.lm_configs = lm_configs
        self.rm = rm
        
        # Initialize modules
        self.phase1_module = CognitiveSelfConstructionModule(
            retriever=rm,
            consensus_extraction_lm=lm_configs.consensus_extraction_lm,
            top_k_reviews=args.top_k_reviews,
        )
        
        self.phase2_module = InnovativeNonSelfIdentificationModule(
            retriever=rm,
            analysis_lm=lm_configs.deviation_analysis_lm,
            top_k_papers=args.top_k_research_papers,
            min_cluster_size=args.min_cluster_size,
            deviation_threshold=args.deviation_threshold,
        )
        
        self.mind_map_manager = DynamicMindMapManager()
        
        self.report_generator = InnovationGapReportGenerator(
            lm=lm_configs.report_generation_lm
        )
        
        # State variables
        self.cognitive_baseline: Optional[CognitiveBaseline] = None
        self.innovation_clusters = None
        self.papers_with_deviations = None
        self.final_report: Optional[InnovationGapReport] = None
        
        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"IGFinderRunner initialized for topic: {args.topic}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_phase1_cognitive_self_construction(self) -> CognitiveBaseline:
        """
        Execute Phase 1: Cognitive Self Construction.
        
        Returns:
            CognitiveBaseline object
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: COGNITIVE SELF CONSTRUCTION")
        logger.info("="*80 + "\n")
        
        cognitive_baseline = self.phase1_module.construct_cognitive_self(self.args.topic)
        self.cognitive_baseline = cognitive_baseline
        
        # Save intermediate results
        if self.args.save_intermediate_results:
            self._save_cognitive_baseline(cognitive_baseline)
        
        return cognitive_baseline
    
    def run_phase2_innovative_nonself_identification(
        self,
        cognitive_baseline: Optional[CognitiveBaseline] = None,
    ):
        """
        Execute Phase 2: Innovative Non-self Identification.
        
        Args:
            cognitive_baseline: Optional baseline (uses self.cognitive_baseline if not provided)
        """
        if cognitive_baseline is None:
            if self.cognitive_baseline is None:
                raise ValueError("Must run Phase 1 first or provide cognitive_baseline")
            cognitive_baseline = self.cognitive_baseline
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: INNOVATIVE NON-SELF IDENTIFICATION")
        logger.info("="*80 + "\n")
        
        papers_with_deviations, innovation_clusters = self.phase2_module.identify_innovative_nonself(
            self.args.topic,
            cognitive_baseline,
        )
        
        self.papers_with_deviations = papers_with_deviations
        self.innovation_clusters = innovation_clusters
        
        # Update mind map with evolution states
        logger.info("\nUpdating mind map with evolution states...")
        updated_mind_map = self.mind_map_manager.update_with_phase2_results(
            cognitive_baseline,
            papers_with_deviations,
            innovation_clusters,
        )
        
        # Save intermediate results
        if self.args.save_intermediate_results:
            self._save_phase2_results(innovation_clusters, papers_with_deviations)
        
        return innovation_clusters
    
    def generate_innovation_gap_report(self) -> InnovationGapReport:
        """
        Generate the final innovation gap report.
        
        Returns:
            InnovationGapReport object
        """
        if self.cognitive_baseline is None or self.innovation_clusters is None:
            raise ValueError("Must run both phases before generating report")
        
        logger.info("\n" + "="*80)
        logger.info("GENERATING INNOVATION GAP REPORT")
        logger.info("="*80 + "\n")
        
        # Get visualization data from mind map
        mind_map_viz_data = self.mind_map_manager.export_visualization_data(
            self.cognitive_baseline.consensus_map
        )
        
        # Generate report
        report = self.report_generator.generate_report(
            topic=self.args.topic,
            cognitive_baseline=self.cognitive_baseline,
            innovation_clusters=self.innovation_clusters,
            papers_with_deviations=self.papers_with_deviations,
            mind_map_visualization_data=mind_map_viz_data,
        )
        
        self.final_report = report
        
        # Save report
        self._save_report(report)
        
        return report
    
    def run(
        self,
        do_phase1: bool = True,
        do_phase2: bool = True,
        do_generate_report: bool = True,
    ) -> InnovationGapReport:
        """
        Execute the complete IG-Finder pipeline.
        
        Args:
            do_phase1: Whether to run Phase 1 (cognitive self construction)
            do_phase2: Whether to run Phase 2 (innovative non-self identification)
            do_generate_report: Whether to generate final report
            
        Returns:
            InnovationGapReport object
        """
        logger.info("\n" + "="*80)
        logger.info(f"IG-FINDER: Innovation Gap Finder for '{self.args.topic}'")
        logger.info("="*80 + "\n")
        
        # Phase 1
        if do_phase1:
            self.run_phase1_cognitive_self_construction()
        else:
            logger.info("Skipping Phase 1 (loading from saved state if available)")
            self._load_cognitive_baseline()
        
        # Phase 2
        if do_phase2:
            self.run_phase2_innovative_nonself_identification()
        else:
            logger.info("Skipping Phase 2 (loading from saved state if available)")
            self._load_phase2_results()
        
        # Generate report
        if do_generate_report:
            report = self.generate_innovation_gap_report()
        else:
            logger.info("Skipping report generation")
            report = self.final_report
        
        logger.info("\n" + "="*80)
        logger.info("IG-FINDER PIPELINE COMPLETE")
        logger.info("="*80 + "\n")
        
        return report
    
    def get_report(self) -> Optional[InnovationGapReport]:
        """Get the generated report."""
        return self.final_report
    
    def _save_cognitive_baseline(self, baseline: CognitiveBaseline):
        """Save cognitive baseline to file."""
        output_file = self.output_dir / "cognitive_baseline.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(baseline.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved cognitive baseline to {output_file}")
    
    def _save_phase2_results(self, clusters, papers_with_deviations):
        """Save Phase 2 results to file."""
        output_file = self.output_dir / "phase2_results.json"
        data = {
            "innovation_clusters": [c.to_dict() for c in clusters],
            "num_papers_analyzed": len(papers_with_deviations),
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Phase 2 results to {output_file}")
    
    def _save_report(self, report: InnovationGapReport):
        """Save final report to file."""
        # Save JSON version
        json_file = self.output_dir / "innovation_gap_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved report (JSON) to {json_file}")
        
        # Save Markdown version
        md_file = self.output_dir / "innovation_gap_report.md"
        md_content = self.report_generator.format_report_as_markdown(report)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logger.info(f"Saved report (Markdown) to {md_file}")
    
    def _load_cognitive_baseline(self):
        """Load cognitive baseline from file."""
        input_file = self.output_dir / "cognitive_baseline.json"
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.cognitive_baseline = CognitiveBaseline.from_dict(data)
            logger.info(f"Loaded cognitive baseline from {input_file}")
        else:
            logger.warning(f"Cognitive baseline file not found: {input_file}")
    
    def _load_phase2_results(self):
        """Load Phase 2 results from file."""
        input_file = self.output_dir / "phase2_results.json"
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            from .dataclass import InnovationCluster
            self.innovation_clusters = [
                InnovationCluster.from_dict(c) for c in data["innovation_clusters"]
            ]
            logger.info(f"Loaded Phase 2 results from {input_file}")
        else:
            logger.warning(f"Phase 2 results file not found: {input_file}")
    
    def summary(self):
        """Print summary of execution."""
        print("\n" + "="*80)
        print("IG-FINDER EXECUTION SUMMARY")
        print("="*80)
        
        if self.cognitive_baseline:
            print(f"\nPhase 1: Cognitive Self Construction")
            print(f"  - Topic: {self.args.topic}")
            print(f"  - Review papers analyzed: {len(self.cognitive_baseline.review_papers)}")
            print(f"  - Research paradigms identified: {len(self.cognitive_baseline.research_paradigms)}")
            print(f"  - Mainstream methods: {len(self.cognitive_baseline.mainstream_methods)}")
            print(f"  - Knowledge boundaries: {len(self.cognitive_baseline.knowledge_boundaries)}")
        
        if self.innovation_clusters:
            print(f"\nPhase 2: Innovative Non-self Identification")
            print(f"  - Research papers analyzed: {len(self.papers_with_deviations) if self.papers_with_deviations else 0}")
            print(f"  - Innovation clusters identified: {len(self.innovation_clusters)}")
            print(f"  - Total papers in clusters: {sum(len(c.core_papers) for c in self.innovation_clusters)}")
        
        if self.final_report:
            print(f"\nFinal Report")
            print(f"  - Generation date: {self.final_report.generation_date}")
            print(f"  - Gap analysis dimensions: {len(self.final_report.gap_analysis_by_dimension)}")
            print(f"  - Innovation paths: {len(self.final_report.mind_map_visualization_data.get('innovation_paths', []))}")
        
        print(f"\nOutput directory: {self.output_dir}")
        print("="*80 + "\n")
