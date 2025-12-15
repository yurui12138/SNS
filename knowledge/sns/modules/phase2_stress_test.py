"""
Phase 2: Multi-view Stress Test

This module implements stress testing of papers against the multi-view baseline.
Includes paper claim extraction, candidate retrieval, and fit scoring.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import dspy

from ...interface import Information
from ..dataclass import ResearchPaper
from ..embeddings import create_embedding_model, compute_hybrid_similarity
from ..nli import create_nli_model, compute_max_conflict_score
from ..dataclass_v2 import (
    MultiViewBaseline,
    TaxonomyView,
    TaxonomyTreeNode,
    NodeDefinition,
    PaperClaims,
    PaperClaim,
    EvidenceSpan,
    FitLabel,
    FitScores,
    FitReport,
    FitVector,
    LostNovelty,
    ConflictEvidence,
)
from ..schemas_v2 import create_paper_claim_extractor
from ..parsing import safe_json_loads

logger = logging.getLogger(__name__)


class PaperClaimExtractor:
    """
    Extracts structured claims from research papers.
    
    Uses LLM with fixed JSON schema (temperature=0).
    Enforces exactly 3 novelty bullets.
    """
    
    def __init__(self, lm):
        self.lm = lm
        self.extractor = create_paper_claim_extractor(lm)
    
    def extract_claims(self, paper: Information) -> Optional[PaperClaims]:
        """
        Extract structured claims from a paper.
        
        Args:
            paper: Information object containing paper data
            
        Returns:
            PaperClaims object or None if extraction fails
        """
        logger.info(f"Extracting claims from paper: {paper.title}")
        
        try:
            # Prepare input
            paper_text = " ".join(paper.snippets)[:15000]
            
            # Call LLM (wrap in dspy context)
            with dspy.context(lm=self.lm):
                result = self.extractor(
                    paper_title=paper.title,
                    paper_abstract=paper.description,
                    paper_text=paper_text
                )
            
            # Parse JSON
            claims_data = safe_json_loads(result.claims_json)
            
            if not claims_data:
                logger.warning(f"Failed to parse claims JSON for paper: {paper.title}")
                return None
            
            # Build PaperClaims
            paper_claims = self._build_paper_claims(claims_data, paper)
            
            # Validate: must have exactly 3 novelty bullets
            if len(paper_claims.novelty_bullets) != 3:
                logger.warning(f"Paper has {len(paper_claims.novelty_bullets)} novelty bullets, expected 3")
                # Pad or trim to exactly 3
                while len(paper_claims.novelty_bullets) < 3:
                    paper_claims.novelty_bullets.append(PaperClaim(
                        text="Additional contribution (placeholder)",
                        evidence=[]
                    ))
                paper_claims.novelty_bullets = paper_claims.novelty_bullets[:3]
            
            return paper_claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}", exc_info=True)
            return None
    
    def _build_paper_claims(self, data: Dict, paper: Information) -> PaperClaims:
        """Build PaperClaims from parsed JSON."""
        
        claims = data.get("claims", {})
        
        def parse_claim(claim_data):
            if isinstance(claim_data, dict):
                return PaperClaim(
                    text=claim_data.get("text", ""),
                    evidence=[self._parse_evidence(e) for e in claim_data.get("evidence", [])]
                )
            return None
        
        def parse_claim_list(claim_list):
            return [parse_claim(c) for c in claim_list if parse_claim(c)]
        
        return PaperClaims(
            paper_id=paper.url,
            problem=parse_claim(claims.get("problem")),
            core_idea=parse_claim_list(claims.get("core_idea", [])),
            mechanism=parse_claim_list(claims.get("mechanism", [])),
            training=parse_claim_list(claims.get("training", [])),
            evaluation=parse_claim_list(claims.get("evaluation", [])),
            novelty_bullets=parse_claim_list(claims.get("novelty_bullets", [])),
            keywords=data.get("keywords", []),
            tasks_datasets=data.get("tasks_datasets", []),
            methods_components=data.get("methods_components", [])
        )
    
    def _parse_evidence(self, e: Dict) -> EvidenceSpan:
        """Parse evidence span."""
        return EvidenceSpan(
            claim="",
            page=e.get("page", 0),
            section=e.get("section", ""),
            char_start=0,
            char_end=0,
            quote=e.get("quote", "")
        )


class EmbeddingBasedRetriever:
    """
    Retrieves top-K candidate leaf nodes using embedding similarity.
    
    Uses API-based embedding models (OpenAI, Azure, or TF-IDF fallback).
    No GPU required - all computation on API side.
    """
    
    def __init__(
        self,
        embedding_model_type: str = "openai",
        embedding_model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize retriever with API-based embeddings.
        
        Args:
            embedding_model_type: "openai", "azure", or "fallback"
            embedding_model_name: Specific model name or deployment
            api_key: API key (or set via environment variable)
            api_base: Custom API base URL
        """
        self.model_type = embedding_model_type
        self.model_name = embedding_model_name
        logger.info(f"Initializing EmbeddingBasedRetriever with {embedding_model_type}/{embedding_model_name}")
        
        # Load API-based embedding model
        try:
            self.embedder = create_embedding_model(
                model_type=embedding_model_type,
                model_name=embedding_model_name,
                api_key=api_key,
                api_base=api_base
            )
            logger.info(f"Successfully initialized {embedding_model_type} embedding model")
        except Exception as e:
            logger.warning(f"Failed to initialize {embedding_model_type}, using TF-IDF fallback: {e}")
            self.embedder = create_embedding_model(model_type="fallback")
    
    def retrieve_candidates(
        self, 
        paper_claims: PaperClaims,
        view: TaxonomyView,
        k: int = 5
    ) -> List[Tuple[TaxonomyTreeNode, float]]:
        """
        Retrieve top-K candidate leaf nodes for a paper.
        
        Args:
            paper_claims: Claims extracted from paper
            view: Taxonomy view to search in
            k: Number of candidates to return
            
        Returns:
            List of (node, similarity_score) tuples
        """
        # Construct paper representation
        paper_text = self._paper_to_text(paper_claims)
        
        # Get all leaf nodes
        leaf_nodes = view.tree.get_leaf_nodes()
        
        # Compute similarities (placeholder implementation)
        candidates = []
        for leaf in leaf_nodes:
            leaf_text = self._leaf_to_text(leaf, view)
            similarity = self._compute_similarity(paper_text, leaf_text)
            candidates.append((leaf, similarity))
        
        # Sort by similarity and return top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    def _paper_to_text(self, claims: PaperClaims) -> str:
        """Convert paper claims to text for embedding."""
        parts = []
        parts.extend([c.text for c in claims.core_idea])
        parts.extend([c.text for c in claims.novelty_bullets])
        parts.extend(claims.keywords)
        return " ".join(parts)
    
    def _leaf_to_text(self, leaf: TaxonomyTreeNode, view: TaxonomyView) -> str:
        """Convert leaf node to text for embedding."""
        parts = [leaf.name]
        
        if leaf.path in view.node_definitions:
            node_def = view.node_definitions[leaf.path]
            parts.append(node_def.definition)
            parts.extend(node_def.canonical_keywords)
        
        return " ".join(parts)
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using real embeddings.
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            # Use real embedding model
            emb1 = self.embedder.encode([text1])[0]
            emb2 = self.embedder.encode([text2])[0]
            similarity = self.embedder.similarity(emb1, emb2)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Embedding computation failed: {e}, using fallback")
            # Fallback to keyword overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0


class FitTester:
    """
    Performs fit test on a paper against a leaf node.
    
    Implements Coverage / Conflict / Residual scoring from design doc.
    """
    
    def __init__(
        self,
        retriever: EmbeddingBasedRetriever,
        nli_model=None,
        nli_model_type: str = "llm",
        nli_llm_model: str = "gpt-3.5-turbo",
        nli_api_key: Optional[str] = None,
        nli_api_base: Optional[str] = None
    ):
        """
        Initialize FitTester with API-based NLI.
        
        Args:
            retriever: Embedding-based retriever
            nli_model: Pre-initialized NLI model (optional)
            nli_model_type: "llm" or "rule-based"
            nli_llm_model: LLM model name if using LLM
            nli_api_key: API key for NLI LLM
            nli_api_base: Custom API base for NLI LLM
        """
        self.retriever = retriever
        
        # Initialize NLI model for conflict detection
        if nli_model is None:
            logger.info(f"Initializing NLI model for conflict detection: {nli_model_type}")
            try:
                self.nli_model = create_nli_model(
                    model_type=nli_model_type,
                    llm_model=nli_llm_model,
                    api_key=nli_api_key,
                    api_base=nli_api_base
                )
                logger.info(f"Successfully initialized {nli_model_type} NLI model")
            except Exception as e:
                logger.warning(f"Failed to initialize NLI model: {e}, using rule-based fallback")
                self.nli_model = create_nli_model(model_type="rule-based")
        else:
            self.nli_model = nli_model
    
    def test_fit(
        self,
        paper_claims: PaperClaims,
        leaf: TaxonomyTreeNode,
        node_def: NodeDefinition,
        view: TaxonomyView
    ) -> Tuple[FitLabel, FitScores, List[LostNovelty], List[ConflictEvidence]]:
        """
        Test how well a paper fits into a leaf node.
        
        Args:
            paper_claims: Claims from paper
            leaf: Candidate leaf node
            node_def: Definition of the leaf node
            view: Parent taxonomy view
            
        Returns:
            Tuple of (label, scores, lost_novelty, conflict_evidence)
        """
        # Calculate three components
        coverage = self._calculate_coverage(paper_claims, leaf, node_def)
        conflict = self._calculate_conflict(paper_claims, node_def)
        residual = self._calculate_residual(paper_claims, leaf, node_def)
        
        # Combined fit score
        fit_score = coverage - 0.8 * conflict - 0.4 * residual
        
        scores = FitScores(
            coverage=coverage,
            conflict=conflict,
            residual=residual,
            fit_score=fit_score
        )
        
        # Determine label based on thresholds
        label = self._determine_label(coverage, conflict, residual)
        
        # Extract evidence
        lost_novelty = self._extract_lost_novelty(paper_claims, leaf, node_def, residual)
        conflict_evidence = self._extract_conflict_evidence(paper_claims, node_def, conflict)
        
        return label, scores, lost_novelty, conflict_evidence
    
    def _calculate_coverage(
        self, 
        claims: PaperClaims, 
        leaf: TaxonomyTreeNode,
        node_def: NodeDefinition
    ) -> float:
        """
        Calculate coverage score.
        
        Formula: Coverage = 0.7 * cov_sem + 0.3 * cov_lex
        """
        # Semantic coverage (placeholder)
        paper_text = " ".join([c.text for c in claims.core_idea + claims.novelty_bullets])
        leaf_text = node_def.definition + " " + " ".join(node_def.canonical_keywords)
        cov_sem = self.retriever._compute_similarity(paper_text, leaf_text)
        
        # Lexical coverage (Jaccard)
        paper_keywords = set(claims.keywords)
        leaf_keywords = set(node_def.canonical_keywords)
        
        if paper_keywords or leaf_keywords:
            intersection = len(paper_keywords & leaf_keywords)
            union = len(paper_keywords | leaf_keywords)
            cov_lex = intersection / union if union > 0 else 0.0
        else:
            cov_lex = 0.0
        
        coverage = 0.7 * cov_sem + 0.3 * cov_lex
        return coverage
    
    def _calculate_conflict(self, claims: PaperClaims, node_def: NodeDefinition) -> float:
        """
        Calculate conflict score using real NLI model.
        
        Formula: Conflict = max_{h in Exclusion+Boundary} P_NLI(contradiction | claim, h)
        """
        if not node_def.exclusion_criteria and not node_def.boundary_statements:
            return 0.0
        
        # Combine all claims
        all_claims_text = " ".join([
            c.text for claim_list in [claims.core_idea, claims.mechanism, claims.novelty_bullets]
            for c in claim_list
        ])
        
        if not all_claims_text.strip():
            return 0.0
        
        try:
            # Use real NLI model for conflict detection
            max_conflict = compute_max_conflict_score(
                claim_text=all_claims_text,
                node_definition_text=node_def.definition,
                exclusion_criteria=node_def.exclusion_criteria + node_def.boundary_statements,
                nli_model=self.nli_model
            )
            return max_conflict
        except Exception as e:
            logger.warning(f"NLI conflict detection failed: {e}, using fallback")
            # Fallback to keyword-based detection
            max_conflict = 0.0
            all_claims = [c.text for claim_list in [claims.core_idea, claims.mechanism, claims.novelty_bullets]
                         for c in claim_list]
            all_exclusions = node_def.exclusion_criteria + node_def.boundary_statements
            
            for claim in all_claims:
                for exclusion in all_exclusions:
                    conflict_score = self._keyword_conflict_score(claim, exclusion)
                    max_conflict = max(max_conflict, conflict_score)
            
            return max_conflict
    
    def _keyword_conflict_score(self, claim: str, exclusion: str) -> float:
        """
        Placeholder conflict detection based on keywords.
        
        In production: Replace with NLI model prediction.
        """
        claim_words = set(claim.lower().split())
        exclusion_words = set(exclusion.lower().split())
        
        overlap = len(claim_words & exclusion_words)
        total = len(exclusion_words)
        
        return min(1.0, overlap / max(total, 1))
    
    def _calculate_residual(
        self,
        claims: PaperClaims,
        leaf: TaxonomyTreeNode,
        node_def: NodeDefinition
    ) -> float:
        """
        Calculate residual score (contribution loss).
        
        Formula: Residual = 1 - max_{b in NoveltyBullets} cos(emb(b), leaf_vector)
        """
        leaf_text = node_def.definition + " " + " ".join(node_def.canonical_keywords)
        
        max_similarity = 0.0
        for novelty in claims.novelty_bullets:
            similarity = self.retriever._compute_similarity(novelty.text, leaf_text)
            max_similarity = max(max_similarity, similarity)
        
        residual = 1.0 - max_similarity
        return residual
    
    def _determine_label(self, coverage: float, conflict: float, residual: float) -> FitLabel:
        """
        Determine fit label based on thresholds.
        
        Rules from design doc:
        - If coverage < 0.45 or conflict > 0.55: UNFITTABLE
        - Else if residual > 0.45: FORCE_FIT
        - Else: FIT
        """
        if coverage < 0.45 or conflict > 0.55:
            return FitLabel.UNFITTABLE
        elif residual > 0.45:
            return FitLabel.FORCE_FIT
        else:
            return FitLabel.FIT
    
    def _extract_lost_novelty(
        self,
        claims: PaperClaims,
        leaf: TaxonomyTreeNode,
        node_def: NodeDefinition,
        residual: float
    ) -> List[LostNovelty]:
        """Extract novelty contributions that are lost."""
        lost = []
        
        leaf_text = node_def.definition + " " + " ".join(node_def.canonical_keywords)
        
        for novelty in claims.novelty_bullets:
            similarity = self.retriever._compute_similarity(novelty.text, leaf_text)
            if similarity < 0.55:  # Low similarity = lost
                lost.append(LostNovelty(
                    bullet=novelty.text,
                    evidence=novelty.evidence,
                    similarity_to_leaf=similarity
                ))
        
        return lost
    
    def _extract_conflict_evidence(
        self,
        claims: PaperClaims,
        node_def: NodeDefinition,
        conflict: float
    ) -> List[ConflictEvidence]:
        """Extract evidence of conflicts."""
        conflicts = []
        
        all_claims = []
        for claim_list in [claims.core_idea, claims.mechanism]:
            all_claims.extend([c.text for c in claim_list])
        
        all_boundaries = node_def.exclusion_criteria + node_def.boundary_statements
        
        for boundary in all_boundaries:
            for claim in all_claims:
                conflict_score = self._keyword_conflict_score(claim, boundary)
                if conflict_score > 0.5:  # High conflict
                    conflicts.append(ConflictEvidence(
                        boundary=boundary,
                        nli_contradiction=conflict_score,
                        paper_claim=claim
                    ))
        
        return conflicts


class MultiViewStressTester:
    """
    Performs stress test across all views in the baseline.
    """
    
    def __init__(self, retriever: EmbeddingBasedRetriever, nli_model=None):
        self.retriever = retriever
        self.fit_tester = FitTester(retriever, nli_model)
    
    def test_paper(
        self,
        paper_claims: PaperClaims,
        baseline: MultiViewBaseline
    ) -> FitVector:
        """
        Test a paper against all views in the baseline.
        
        Args:
            paper_claims: Claims extracted from paper
            baseline: Multi-view baseline
            
        Returns:
            FitVector containing fit reports for all views
        """
        fit_reports = []
        
        for view in baseline.views:
            report = self._test_paper_against_view(paper_claims, view)
            fit_reports.append(report)
        
        # Calculate weighted scores
        stress_score = sum(
            view.weight * (1 if report.label != FitLabel.FIT else 0)
            for view, report in zip(baseline.views, fit_reports)
        )
        
        unfittable_score = sum(
            view.weight * (1 if report.label == FitLabel.UNFITTABLE else 0)
            for view, report in zip(baseline.views, fit_reports)
        )
        
        return FitVector(
            paper_id=paper_claims.paper_id,
            fit_reports=fit_reports,
            stress_score=stress_score,
            unfittable_score=unfittable_score
        )
    
    def _test_paper_against_view(
        self,
        paper_claims: PaperClaims,
        view: TaxonomyView
    ) -> FitReport:
        """Test paper against a single view."""
        
        # Get candidate leaves
        candidates = self.retriever.retrieve_candidates(paper_claims, view, k=5)
        
        if not candidates:
            # No candidates found
            return FitReport(
                paper_id=paper_claims.paper_id,
                view_id=view.view_id,
                facet_label=view.facet_label,
                best_leaf_path=None,
                label=FitLabel.UNFITTABLE,
                scores=FitScores(0.0, 1.0, 1.0, -1.0),
                lost_novelty=[],
                conflict_evidence=[]
            )
        
        # Test each candidate and pick best
        best_leaf = None
        best_score = -float('inf')
        best_result = None
        
        for leaf, _ in candidates:
            if leaf.path not in view.node_definitions:
                continue
            
            node_def = view.node_definitions[leaf.path]
            label, scores, lost, conflicts = self.fit_tester.test_fit(
                paper_claims, leaf, node_def, view
            )
            
            if scores.fit_score > best_score:
                best_score = scores.fit_score
                best_leaf = leaf
                best_result = (label, scores, lost, conflicts)
        
        if best_result is None:
            # No valid candidates
            return FitReport(
                paper_id=paper_claims.paper_id,
                view_id=view.view_id,
                facet_label=view.facet_label,
                best_leaf_path=None,
                label=FitLabel.UNFITTABLE,
                scores=FitScores(0.0, 1.0, 1.0, -1.0),
                lost_novelty=[],
                conflict_evidence=[]
            )
        
        label, scores, lost, conflicts = best_result
        
        return FitReport(
            paper_id=paper_claims.paper_id,
            view_id=view.view_id,
            facet_label=view.facet_label,
            best_leaf_path=best_leaf.path if best_leaf else None,
            label=label,
            scores=scores,
            lost_novelty=lost,
            conflict_evidence=conflicts
        )


# ============================================================================
# Main Phase 2 Pipeline
# ============================================================================

class Phase2Pipeline:
    """
    Complete Phase 2 pipeline: Multi-view Stress Test.
    
    Uses API-based embeddings and NLI (no GPU required).
    """
    
    def __init__(
        self,
        lm,
        # Embedding configuration (API-based)
        embedding_model_type: str = "openai",
        embedding_model_name: str = "text-embedding-ada-002",
        embedding_api_key: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        # NLI configuration (LLM-based or rule-based)
        nli_model_type: str = "llm",
        nli_llm_model: str = "gpt-3.5-turbo",
        nli_api_key: Optional[str] = None,
        nli_api_base: Optional[str] = None
    ):
        """
        Initialize Phase 2 pipeline with API-based models.
        
        Args:
            lm: DSPy language model for claim extraction
            embedding_model_type: "openai", "azure", or "fallback"
            embedding_model_name: Model name or deployment
            embedding_api_key: API key for embeddings
            embedding_api_base: Custom API base for embeddings
            nli_model_type: "llm" or "rule-based"
            nli_llm_model: LLM model name for NLI
            nli_api_key: API key for NLI
            nli_api_base: Custom API base for NLI
        """
        self.claim_extractor = PaperClaimExtractor(lm)
        
        # Initialize API-based retriever
        self.retriever = EmbeddingBasedRetriever(
            embedding_model_type=embedding_model_type,
            embedding_model_name=embedding_model_name,
            api_key=embedding_api_key,
            api_base=embedding_api_base
        )
        
        # Initialize API-based NLI model
        logger.info(f"Initializing NLI model: {nli_model_type}")
        try:
            nli_model = create_nli_model(
                model_type=nli_model_type,
                llm_model=nli_llm_model,
                api_key=nli_api_key,
                api_base=nli_api_base
            )
            logger.info(f"Successfully initialized {nli_model_type} NLI model")
        except Exception as e:
            logger.warning(f"Failed to initialize NLI model, using rule-based fallback: {e}")
            nli_model = create_nli_model(model_type="rule-based")
        
        self.stress_tester = MultiViewStressTester(self.retriever, nli_model)
    
    def run(
        self,
        papers: List[Information],
        baseline: MultiViewBaseline
    ) -> List[FitVector]:
        """
        Run complete Phase 2 pipeline.
        
        Args:
            papers: List of research papers
            baseline: Multi-view baseline from Phase 1
            
        Returns:
            List of FitVector objects
        """
        logger.info("="*80)
        logger.info("PHASE 2: Multi-view Stress Test")
        logger.info("="*80)
        
        fit_vectors = []
        
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.title}")
            
            # Extract claims
            claims = self.claim_extractor.extract_claims(paper)
            if not claims:
                logger.warning(f"Failed to extract claims for paper: {paper.title}")
                continue
            
            # Stress test
            fit_vector = self.stress_tester.test_paper(claims, baseline)
            fit_vectors.append(fit_vector)
            
            logger.info(f"  Stress score: {fit_vector.stress_score:.3f}")
            logger.info(f"  Unfittable score: {fit_vector.unfittable_score:.3f}")
        
        logger.info(f"Phase 2 completed: Tested {len(fit_vectors)} papers")
        logger.info("="*80)
        
        return fit_vectors
