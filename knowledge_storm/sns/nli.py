"""
Natural Language Inference (NLI) module for conflict detection.

Uses DeBERTa-MNLI or similar models to detect contradictions between
taxonomy definitions and paper claims.
"""
import logging
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")

logger = logging.getLogger(__name__)


class NLILabel(Enum):
    """NLI prediction labels."""
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class NLIModel:
    """
    Natural Language Inference model for detecting semantic conflicts.
    
    Uses DeBERTa-v3-large-mnli or similar MNLI-trained models.
    Predicts entailment, neutral, or contradiction between two texts.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large-mnli",
        device: str = "cpu",
        batch_size: int = 8
    ):
        """
        Initialize NLI model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for inference
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for NLI. "
                            "Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading NLI model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            # Label mapping (MNLI standard)
            self.label_map = {
                0: NLILabel.CONTRADICTION,
                1: NLILabel.NEUTRAL,
                2: NLILabel.ENTAILMENT
            }
            
            logger.info("NLI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise
    
    def predict(self, premise: str, hypothesis: str) -> Tuple[NLILabel, float]:
        """
        Predict NLI relation between premise and hypothesis.
        
        Args:
            premise: Premise text (e.g., taxonomy node definition)
            hypothesis: Hypothesis text (e.g., paper claim)
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_idx].item()
        
        label = self.label_map[pred_idx]
        
        return label, confidence
    
    def predict_batch(
        self,
        premises: List[str],
        hypotheses: List[str]
    ) -> List[Tuple[NLILabel, float]]:
        """
        Batch prediction for efficiency.
        
        Args:
            premises: List of premise texts
            hypotheses: List of hypothesis texts
            
        Returns:
            List of (label, confidence) tuples
        """
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have same length")
        
        results = []
        
        for i in range(0, len(premises), self.batch_size):
            batch_premises = premises[i:i + self.batch_size]
            batch_hypotheses = hypotheses[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                pred_indices = torch.argmax(probs, dim=-1).cpu().numpy()
                confidences = probs.cpu().numpy()
            
            # Extract results
            for j, pred_idx in enumerate(pred_indices):
                label = self.label_map[pred_idx]
                confidence = confidences[j, pred_idx]
                results.append((label, float(confidence)))
        
        return results
    
    def compute_contradiction_score(self, premise: str, hypothesis: str) -> float:
        """
        Compute contradiction score between premise and hypothesis.
        
        Returns probability of contradiction in [0, 1].
        Used for Conflict score in Phase 2.
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            Contradiction probability
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get contradiction probability (index 0)
            contradiction_prob = probs[0, 0].item()
        
        return contradiction_prob
    
    def compute_contradiction_scores_batch(
        self,
        premises: List[str],
        hypotheses: List[str]
    ) -> List[float]:
        """
        Batch computation of contradiction scores.
        
        Args:
            premises: List of premise texts
            hypotheses: List of hypothesis texts
            
        Returns:
            List of contradiction probabilities
        """
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have same length")
        
        scores = []
        
        for i in range(0, len(premises), self.batch_size):
            batch_premises = premises[i:i + self.batch_size]
            batch_hypotheses = hypotheses[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get contradiction probabilities (index 0)
                contradiction_probs = probs[:, 0].cpu().numpy()
            
            scores.extend(contradiction_probs.tolist())
        
        return scores


class FallbackNLI:
    """
    Fallback NLI using simple keyword-based heuristics.
    
    Used when transformers library is not available.
    Not as accurate as neural models, but provides basic functionality.
    """
    
    def __init__(self):
        logger.warning("Using fallback NLI (no neural models available)")
        
        # Negation indicators
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody',
            'nothing', 'nowhere', 'without', 'lack', 'lacks', 'lacking'
        }
        
        # Contradiction indicators
        self.contradiction_pairs = [
            ('increase', 'decrease'),
            ('improve', 'worsen'),
            ('effective', 'ineffective'),
            ('success', 'failure'),
            ('positive', 'negative'),
            ('high', 'low'),
            ('large', 'small'),
            ('more', 'less'),
        ]
    
    def predict(self, premise: str, hypothesis: str) -> Tuple[NLILabel, float]:
        """Simple rule-based prediction."""
        
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        
        # Check for negation
        premise_negated = bool(premise_words & self.negation_words)
        hypothesis_negated = bool(hypothesis_words & self.negation_words)
        
        # Check for contradiction pairs
        has_contradiction = False
        for word1, word2 in self.contradiction_pairs:
            if (word1 in premise_words and word2 in hypothesis_words) or \
               (word2 in premise_words and word1 in hypothesis_words):
                has_contradiction = True
                break
        
        # Negation XOR logic
        if premise_negated != hypothesis_negated or has_contradiction:
            return NLILabel.CONTRADICTION, 0.6
        
        # Check overlap
        overlap = len(premise_words & hypothesis_words)
        if overlap > 5:
            return NLILabel.ENTAILMENT, 0.5
        
        return NLILabel.NEUTRAL, 0.4
    
    def compute_contradiction_score(self, premise: str, hypothesis: str) -> float:
        """Compute simple contradiction score."""
        label, confidence = self.predict(premise, hypothesis)
        
        if label == NLILabel.CONTRADICTION:
            return confidence
        else:
            return 0.0


def create_nli_model(
    model_type: str = "deberta",
    model_name: Optional[str] = None,
    device: str = "cpu"
):
    """
    Factory function to create NLI model.
    
    Args:
        model_type: Type of model ('deberta', 'roberta', 'fallback')
        model_name: Optional custom model name
        device: Device to run on
        
    Returns:
        NLI model instance
    """
    if not TRANSFORMERS_AVAILABLE and model_type != "fallback":
        logger.warning("transformers not available, using fallback NLI")
        return FallbackNLI()
    
    if model_type == "deberta":
        model_name = model_name or "microsoft/deberta-v3-large-mnli"
        return NLIModel(model_name, device)
    
    elif model_type == "roberta":
        model_name = model_name or "roberta-large-mnli"
        return NLIModel(model_name, device)
    
    elif model_type == "fallback":
        return FallbackNLI()
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# Helper Functions for Phase 2
# ============================================================================

def compute_max_conflict_score(
    claim_text: str,
    node_definition_text: str,
    exclusion_criteria: List[str],
    nli_model
) -> float:
    """
    Compute maximum conflict score between a claim and a node.
    
    Formula from design doc:
    Conflict = max NLI_contradiction(claim, excl_i)
    
    Args:
        claim_text: Paper claim text
        node_definition_text: Node definition
        exclusion_criteria: List of exclusion criteria
        nli_model: NLI model instance
        
    Returns:
        Maximum contradiction score
    """
    if not exclusion_criteria:
        return 0.0
    
    # Check against definition
    def_conflict = nli_model.compute_contradiction_score(
        node_definition_text,
        claim_text
    )
    
    # Check against each exclusion criterion
    if hasattr(nli_model, 'compute_contradiction_scores_batch'):
        # Batch computation
        premises = exclusion_criteria
        hypotheses = [claim_text] * len(exclusion_criteria)
        excl_conflicts = nli_model.compute_contradiction_scores_batch(premises, hypotheses)
        max_excl_conflict = max(excl_conflicts) if excl_conflicts else 0.0
    else:
        # Sequential computation
        max_excl_conflict = 0.0
        for criterion in exclusion_criteria:
            conflict = nli_model.compute_contradiction_score(criterion, claim_text)
            max_excl_conflict = max(max_excl_conflict, conflict)
    
    # Return maximum across all checks
    return max(def_conflict, max_excl_conflict)
