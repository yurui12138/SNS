"""
Embedding and similarity computation infrastructure.

Provides unified interface for different embedding models:
- SPECTER2 for scientific papers
- SciNCL for cross-encoder reranking
- Sentence-BERT for general text
"""
import logging
import numpy as np
from typing import List, Optional, Union, Dict
from abc import ABC, abstractmethod
import os

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("litellm not available. Install with: pip install litellm")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        pass
    
    @abstractmethod
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings."""
        pass
    
    def batch_similarity(self, query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarities between query and candidate embeddings."""
        # Cosine similarity by default
        return np.dot(query_embeddings, candidate_embeddings.T) / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) *
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True).T
        )


class SPECTER2Embedding(EmbeddingModel):
    """
    SPECTER2 embedding model for scientific papers.
    
    Uses allenai/specter2_base for encoding scientific abstracts/titles.
    Optimized for semantic similarity in research papers.
    """
    
    def __init__(self, model_name: str = "allenai/specter2_base", device: str = "cpu"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for SPECTER2. "
                            "Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading SPECTER2 model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info("SPECTER2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SPECTER2 model: {e}")
            logger.info("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings (titles, abstracts, or combined)
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))


class SciNCLEmbedding(EmbeddingModel):
    """
    SciNCL embedding model for scientific literature.
    
    Alternative to SPECTER2, optimized for citation recommendation
    and semantic search in academic papers.
    """
    
    def __init__(self, model_name: str = "malteos/scincl", device: str = "cpu"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for SciNCL. "
                            "Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading SciNCL model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info("SciNCL model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SciNCL model: {e}")
            logger.info("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))


class SentenceBERTEmbedding(EmbeddingModel):
    """
    General-purpose Sentence-BERT embedding.
    
    Fast and efficient for general semantic similarity tasks.
    Good fallback when domain-specific models are not available.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. "
                            "Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Sentence-BERT model loaded successfully")
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))


class FallbackEmbedding(EmbeddingModel):
    """
    Fallback embedding using simple TF-IDF.
    
    Used when no neural embedding models are available.
    """
    
    def __init__(self):
        logger.warning("Using fallback TF-IDF embedding (no neural models available)")
        self.vocab: Optional[Dict[str, int]] = None
        self.idf: Optional[np.ndarray] = None
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary and IDF from texts."""
        from collections import Counter
        import math
        
        # Build vocabulary
        word_counts = Counter()
        doc_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
            doc_counts.update(set(words))
        
        # Create vocab
        vocab_list = [word for word, _ in word_counts.most_common(5000)]
        self.vocab = {word: i for i, word in enumerate(vocab_list)}
        
        # Calculate IDF
        n_docs = len(texts)
        self.idf = np.array([
            math.log(n_docs / (doc_counts.get(word, 0) + 1))
            for word in vocab_list
        ])
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts using TF-IDF."""
        if not texts:
            return np.array([])
        
        # Build vocab if first time
        if self.vocab is None:
            self._build_vocab(texts)
        
        # Create TF-IDF vectors
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(len(self.vocab))
            
            for word in words:
                if word in self.vocab:
                    idx = self.vocab[word]
                    vec[idx] += 1
            
            # Apply IDF
            vec = vec * self.idf
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            vectors.append(vec)
        
        return np.array(vectors)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-10
        ))


def create_embedding_model(
    model_type: str = "specter2",
    model_name: Optional[str] = None,
    device: str = "cpu"
) -> EmbeddingModel:
    """
    Factory function to create embedding model.
    
    Args:
        model_type: Type of model ('specter2', 'scincl', 'sbert', 'fallback')
        model_name: Optional custom model name
        device: Device to run model on ('cpu' or 'cuda')
        
    Returns:
        EmbeddingModel instance
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE and model_type != "fallback":
        logger.warning("sentence-transformers not available, using fallback")
        model_type = "fallback"
    
    if model_type == "specter2":
        model_name = model_name or "allenai/specter2_base"
        return SPECTER2Embedding(model_name, device)
    
    elif model_type == "scincl":
        model_name = model_name or "malteos/scincl"
        return SciNCLEmbedding(model_name, device)
    
    elif model_type == "sbert":
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceBERTEmbedding(model_name, device)
    
    elif model_type == "fallback":
        return FallbackEmbedding()
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# Hybrid Similarity Computation
# ============================================================================

def compute_hybrid_similarity(
    text1: str,
    text2: str,
    embedding_model: EmbeddingModel,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3
) -> float:
    """
    Compute hybrid similarity combining semantic and lexical matching.
    
    Formula from design doc: Coverage = 0.7 * semantic + 0.3 * lexical
    
    Args:
        text1: First text
        text2: Second text
        embedding_model: Embedding model for semantic similarity
        semantic_weight: Weight for semantic similarity (default 0.7)
        lexical_weight: Weight for lexical similarity (default 0.3)
        
    Returns:
        Hybrid similarity score in [0, 1]
    """
    # Semantic similarity
    emb1 = embedding_model.encode([text1])[0]
    emb2 = embedding_model.encode([text2])[0]
    semantic_sim = embedding_model.similarity(emb1, emb2)
    
    # Lexical similarity (Jaccard)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        lexical_sim = 0.0
    else:
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        lexical_sim = intersection / union if union > 0 else 0.0
    
    # Combined score
    hybrid_score = semantic_weight * semantic_sim + lexical_weight * lexical_sim
    
    return float(np.clip(hybrid_score, 0.0, 1.0))


def compute_top_k_matches(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int = 5
) -> List[int]:
    """
    Find top-K most similar candidates to query.
    
    Args:
        query_embedding: Query embedding (1D array)
        candidate_embeddings: Candidate embeddings (2D array, shape [n_candidates, dim])
        k: Number of top matches to return
        
    Returns:
        List of indices of top-K candidates
    """
    # Compute similarities
    similarities = np.dot(candidate_embeddings, query_embedding) / (
        np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top-K
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    return top_k_indices.tolist()
