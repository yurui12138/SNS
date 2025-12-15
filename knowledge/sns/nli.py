"""
Natural Language Inference (NLI) for conflict detection - LLM API-based version.

This module uses LLM API calls instead of local NLI models to avoid GPU requirements.
Uses zero-shot prompting with GPT-3.5/GPT-4 or other LLMs for entailment detection.

Key Design:
1. LLM-based NLI: Uses structured prompting for entailment/contradiction detection
2. Rule-based fallback: Keyword-based heuristics (no GPU required)
3. Batch processing support for efficiency
"""
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("litellm not available. Install with: pip install litellm")

logger = logging.getLogger(__name__)


class NLILabel(Enum):
    """NLI prediction labels."""
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass
class NLIResult:
    """Result of NLI prediction."""
    label: NLILabel
    confidence: float
    contradiction_score: float  # Probability of contradiction (0-1)
    
    def __post_init__(self):
        # Ensure contradiction_score is set
        if self.label == NLILabel.CONTRADICTION:
            self.contradiction_score = max(self.contradiction_score, self.confidence)


class LLMNLIModel:
    """
    LLM-based NLI model using structured prompting.
    
    Uses GPT-3.5-turbo, GPT-4, or other LLMs to perform entailment detection.
    No GPU required - all computation on API side.
    """
    
    # NLI prompt template
    NLI_PROMPT = """You are an expert in natural language inference. Given a PREMISE and a HYPOTHESIS, determine their logical relationship.

PREMISE: {premise}

HYPOTHESIS: {hypothesis}

Task: Determine if the HYPOTHESIS is:
- ENTAILMENT: The hypothesis logically follows from the premise (is supported by it)
- CONTRADICTION: The hypothesis contradicts or conflicts with the premise
- NEUTRAL: The hypothesis is neither entailed nor contradicted by the premise

Respond with ONLY ONE WORD: ENTAILMENT, CONTRADICTION, or NEUTRAL.

Your answer:"""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 10
    ):
        """
        Initialize LLM-based NLI model.
        
        Args:
            model: LLM model name (gpt-3.5-turbo, gpt-4, claude-3-haiku, etc.)
            api_key: API key (or set via environment variable)
            api_base: Custom API base URL
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Max tokens in response
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LLM NLI. "
                            "Install with: pip install litellm")
        
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized LLM NLI model: {model}")
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Predict NLI relationship between premise and hypothesis.
        
        Args:
            premise: The premise statement
            hypothesis: The hypothesis statement
            
        Returns:
            NLIResult with label and confidence
        """
        # Format prompt
        prompt = self.NLI_PROMPT.format(premise=premise, hypothesis=hypothesis)
        
        try:
            # Call LLM API
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract prediction
            prediction = response['choices'][0]['message']['content'].strip().upper()
            
            # Parse label
            if "ENTAILMENT" in prediction:
                label = NLILabel.ENTAILMENT
                confidence = 0.9
                contradiction_score = 0.0
            elif "CONTRADICTION" in prediction:
                label = NLILabel.CONTRADICTION
                confidence = 0.9
                contradiction_score = 0.9
            elif "NEUTRAL" in prediction:
                label = NLILabel.NEUTRAL
                confidence = 0.8
                contradiction_score = 0.0
            else:
                # Fallback: try to infer from response
                logger.warning(f"Unexpected NLI response: {prediction}, defaulting to NEUTRAL")
                label = NLILabel.NEUTRAL
                confidence = 0.5
                contradiction_score = 0.0
            
            return NLIResult(
                label=label,
                confidence=confidence,
                contradiction_score=contradiction_score
            )
            
        except Exception as e:
            logger.error(f"Error in LLM NLI prediction: {e}")
            # Return neutral with low confidence as fallback
            return NLIResult(
                label=NLILabel.NEUTRAL,
                confidence=0.3,
                contradiction_score=0.0
            )
    
    def predict_batch(
        self,
        premise_hypothesis_pairs: List[Tuple[str, str]]
    ) -> List[NLIResult]:
        """
        Predict NLI for a batch of premise-hypothesis pairs.
        
        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples
            
        Returns:
            List of NLIResult objects
        """
        results = []
        
        for premise, hypothesis in premise_hypothesis_pairs:
            result = self.predict(premise, hypothesis)
            results.append(result)
        
        return results


class RuleBasedNLIModel:
    """
    Rule-based NLI fallback using keyword matching.
    
    Uses simple heuristics to detect contradictions.
    No GPU or API required.
    """
    
    # Negation words
    NEGATION_WORDS = {
        'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing',
        'nobody', 'nowhere', "n't", 'cannot', 'cant', 'without',
        'lack', 'lacks', 'absent', 'exclude', 'excluding'
    }
    
    # Antonym pairs (simple examples)
    ANTONYMS = {
        'good': 'bad', 'hot': 'cold', 'big': 'small', 'fast': 'slow',
        'high': 'low', 'strong': 'weak', 'increase': 'decrease',
        'improve': 'worsen', 'success': 'failure', 'correct': 'incorrect',
        'true': 'false', 'accept': 'reject', 'include': 'exclude',
        'always': 'never', 'all': 'none', 'everything': 'nothing'
    }
    
    def __init__(self):
        """Initialize rule-based NLI."""
        # Build bidirectional antonym map
        self.antonym_map = {}
        for word1, word2 in self.ANTONYMS.items():
            self.antonym_map[word1] = word2
            self.antonym_map[word2] = word1
        
        logger.info("Initialized rule-based NLI fallback")
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Predict NLI using rule-based heuristics.
        
        Args:
            premise: The premise statement
            hypothesis: The hypothesis statement
            
        Returns:
            NLIResult with label and confidence
        """
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        premise_tokens = set(premise_lower.split())
        hypothesis_tokens = set(hypothesis_lower.split())
        
        # Rule 1: Check for negation patterns
        premise_has_negation = bool(premise_tokens & self.NEGATION_WORDS)
        hypothesis_has_negation = bool(hypothesis_tokens & self.NEGATION_WORDS)
        
        # If one has negation and the other doesn't, might be contradiction
        if premise_has_negation != hypothesis_has_negation:
            # Check for overlapping content
            overlap = premise_tokens & hypothesis_tokens
            if len(overlap) > 3:  # Significant overlap
                return NLIResult(
                    label=NLILabel.CONTRADICTION,
                    confidence=0.6,
                    contradiction_score=0.6
                )
        
        # Rule 2: Check for antonyms
        for token in hypothesis_tokens:
            if token in self.antonym_map:
                antonym = self.antonym_map[token]
                if antonym in premise_tokens:
                    return NLIResult(
                        label=NLILabel.CONTRADICTION,
                        confidence=0.7,
                        contradiction_score=0.7
                    )
        
        # Rule 3: High overlap suggests entailment
        if len(premise_tokens) > 0 and len(hypothesis_tokens) > 0:
            overlap_ratio = len(premise_tokens & hypothesis_tokens) / len(hypothesis_tokens)
            if overlap_ratio > 0.7:
                return NLIResult(
                    label=NLILabel.ENTAILMENT,
                    confidence=0.5,
                    contradiction_score=0.0
                )
        
        # Default: neutral
        return NLIResult(
            label=NLILabel.NEUTRAL,
            confidence=0.4,
            contradiction_score=0.0
        )
    
    def predict_batch(
        self,
        premise_hypothesis_pairs: List[Tuple[str, str]]
    ) -> List[NLIResult]:
        """Predict NLI for a batch."""
        return [self.predict(p, h) for p, h in premise_hypothesis_pairs]


def compute_contradiction_score(
    claim: str,
    definition: str,
    nli_model,
    exclusion_criteria: Optional[List[str]] = None
) -> float:
    """
    Compute contradiction score between a claim and definition.
    
    This is the key function for Phase 2 conflict detection.
    
    Args:
        claim: Paper claim/novelty statement
        definition: Node definition to test against
        nli_model: NLI model instance (LLM or rule-based)
        exclusion_criteria: Optional exclusion criteria to test
        
    Returns:
        Contradiction score (0-1), higher = more contradictory
    """
    # Test claim against definition
    result = nli_model.predict(premise=definition, hypothesis=claim)
    contradiction_score = result.contradiction_score
    
    # Also test against exclusion criteria if provided
    if exclusion_criteria:
        exclusion_scores = []
        for exclusion in exclusion_criteria:
            exc_result = nli_model.predict(premise=exclusion, hypothesis=claim)
            # If claim matches exclusion (entailment), that's a conflict
            if exc_result.label == NLILabel.ENTAILMENT:
                exclusion_scores.append(exc_result.confidence)
        
        if exclusion_scores:
            max_exclusion_score = max(exclusion_scores)
            # Combine with main contradiction score
            contradiction_score = max(contradiction_score, max_exclusion_score)
    
    return contradiction_score


def compute_max_conflict_score(
    claim: str,
    node_definition_text: str,
    exclusion_criteria: List[str],
    nli_model
) -> float:
    """
    Compute maximum conflict score between claim and node.
    
    Tests claim against:
    1. Node definition (looking for contradiction)
    2. Exclusion criteria (looking for matches)
    
    Args:
        claim: Paper claim
        node_definition_text: Node definition text
        exclusion_criteria: List of exclusion statements
        nli_model: NLI model instance
        
    Returns:
        Maximum conflict score (0-1)
    """
    return compute_contradiction_score(
        claim=claim,
        definition=node_definition_text,
        nli_model=nli_model,
        exclusion_criteria=exclusion_criteria
    )


def create_nli_model(
    model_type: str = "llm",
    llm_model: str = "gpt-3.5-turbo",
    **kwargs
) -> Union[LLMNLIModel, RuleBasedNLIModel]:
    """
    Factory function to create NLI models.
    
    Args:
        model_type: Type of model ("llm" or "rule-based")
        llm_model: LLM model name if using LLM
        **kwargs: Additional arguments for model initialization
        
    Returns:
        NLI model instance
    """
    if model_type == "llm":
        return LLMNLIModel(model=llm_model, **kwargs)
    elif model_type == "rule-based":
        return RuleBasedNLIModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'llm' or 'rule-based'.")
