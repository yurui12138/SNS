"""
Embedding and similarity computation infrastructure - API-based version.

This module uses API calls instead of local models to avoid GPU requirements.
Supports OpenAI, Azure, and other embedding API providers.

Key Design:
1. OpenAIEmbedding: Uses text-embedding-ada-002 or text-embedding-3-small
2. FallbackEmbedding: TF-IDF based (no GPU required)
3. Hybrid similarity: Combines semantic + lexical matching
"""
import logging
import numpy as np
from typing import List, Optional, Union, Dict
from abc import ABC, abstractmethod
from collections import Counter
import math

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("litellm not available. Install with: pip install litellm")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Install with: pip install scikit-learn")

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
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if len(candidate_embeddings.shape) == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Normalize
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        candidate_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        
        return np.dot(query_norm, candidate_norm.T)


class OpenAIEmbedding(EmbeddingModel):
    """
    OpenAI API-based embedding model.
    
    Uses text-embedding-ada-002 or text-embedding-3-small/large.
    No GPU required - all computation on API side.
    """
    
    def __init__(
        self, 
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model: Model name (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            api_base: Custom API base URL (for Azure or proxy)
            dimensions: Output dimensions (only for text-embedding-3-* models)
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for OpenAI embeddings. "
                            "Install with: pip install litellm")
        
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.dimensions = dimensions
        
        # Validate model and dimensions
        if dimensions and not model.startswith("text-embedding-3"):
            logger.warning(f"dimensions parameter only supported for text-embedding-3-* models, ignoring")
            self.dimensions = None
        
        logger.info(f"Initialized OpenAI embedding model: {model}")
    
    def encode(self, texts: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        """
        Encode texts to embeddings via OpenAI API.
        
        Args:
            texts: List of text strings
            batch_size: Batch size (OpenAI supports up to 2048 inputs per request)
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter empty texts
        valid_texts = [t if t.strip() else " " for t in texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i+batch_size]
            
            try:
                # Use litellm for API call
                response = litellm.embedding(
                    model=self.model,
                    input=batch,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    dimensions=self.dimensions
                )
                
                # Extract embeddings
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size + 1}: {e}")
                # Return zero embeddings for failed batch
                dim = 1536 if self.model == "text-embedding-ada-002" else (self.dimensions or 1536)
                all_embeddings.extend([[0.0] * dim] * len(batch))
        
        return np.array(all_embeddings)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        ))


class AzureOpenAIEmbedding(OpenAIEmbedding):
    """
    Azure OpenAI API-based embedding model.
    
    Wrapper for Azure-specific configuration.
    """
    
    def __init__(
        self,
        deployment_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: str = "2023-05-15"
    ):
        """
        Initialize Azure OpenAI embedding model.
        
        Args:
            deployment_name: Azure deployment name
            api_key: Azure API key (or set AZURE_OPENAI_API_KEY env var)
            api_base: Azure API base URL (or set AZURE_OPENAI_ENDPOINT env var)
            api_version: API version
        """
        # Azure uses deployment name as model
        super().__init__(
            model=f"azure/{deployment_name}",
            api_key=api_key,
            api_base=api_base
        )
        self.api_version = api_version
        logger.info(f"Initialized Azure OpenAI embedding: {deployment_name}")


class FallbackEmbedding(EmbeddingModel):
    """
    TF-IDF based fallback embedding (no GPU required).
    
    Uses sklearn TfidfVectorizer for text vectorization.
    Suitable for when API is unavailable or for testing.
    """
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize TF-IDF embedding.
        
        Args:
            max_features: Maximum number of features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for FallbackEmbedding. "
                            "Install with: pip install scikit-learn")
        
        self.max_features = max_features
        self.vectorizer = None
        logger.info(f"Initialized TF-IDF fallback embedding (max_features={max_features})")
    
    def encode(self, texts: List[str], fit: bool = False, **kwargs) -> np.ndarray:
        """
        Encode texts to TF-IDF vectors.
        
        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizer (first time only)
            
        Returns:
            numpy array of shape (len(texts), max_features)
        """
        if not texts:
            return np.array([])
        
        # Initialize vectorizer if needed
        if self.vectorizer is None or fit:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            embeddings = self.vectorizer.fit_transform(texts).toarray()
        else:
            embeddings = self.vectorizer.transform(texts).toarray()
        
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        ))


def compute_hybrid_similarity(
    text1: str,
    text2: str,
    embedding1: Optional[np.ndarray] = None,
    embedding2: Optional[np.ndarray] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3
) -> float:
    """
    Compute hybrid similarity combining semantic and lexical matching.
    
    Args:
        text1, text2: Input texts
        embedding1, embedding2: Pre-computed embeddings (optional)
        embedding_model: Model for computing embeddings if not provided
        semantic_weight: Weight for semantic similarity (0-1)
        lexical_weight: Weight for lexical similarity (0-1)
        
    Returns:
        Combined similarity score (0-1)
    """
    # Semantic similarity
    if embedding1 is not None and embedding2 is not None:
        semantic_sim = float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        ))
    elif embedding_model is not None:
        embs = embedding_model.encode([text1, text2])
        semantic_sim = float(np.dot(embs[0], embs[1]) / (
            np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
        ))
    else:
        semantic_sim = 0.0
    
    # Lexical similarity (Jaccard)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        lexical_sim = 0.0
    else:
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        lexical_sim = intersection / union if union > 0 else 0.0
    
    # Combine
    combined = semantic_weight * semantic_sim + lexical_weight * lexical_sim
    
    return combined


def compute_top_k_matches(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_texts: List[str],
    query_text: str = "",
    top_k: int = 5,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3
) -> List[tuple]:
    """
    Find top-k most similar candidates using hybrid similarity.
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: Candidate embedding matrix
        candidate_texts: Candidate text strings
        query_text: Query text (for lexical matching)
        top_k: Number of results to return
        semantic_weight: Weight for semantic similarity
        lexical_weight: Weight for lexical similarity
        
    Returns:
        List of (index, similarity_score) tuples, sorted by score
    """
    # Compute semantic similarities
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    candidate_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    semantic_sims = np.dot(candidate_norm, query_norm)
    
    # Compute hybrid scores
    hybrid_scores = []
    for i, (candidate_text, semantic_sim) in enumerate(zip(candidate_texts, semantic_sims)):
        if query_text and lexical_weight > 0:
            # Add lexical similarity
            tokens_q = set(query_text.lower().split())
            tokens_c = set(candidate_text.lower().split())
            
            if tokens_q and tokens_c:
                intersection = len(tokens_q & tokens_c)
                union = len(tokens_q | tokens_c)
                lexical_sim = intersection / union if union > 0 else 0.0
            else:
                lexical_sim = 0.0
            
            hybrid_score = semantic_weight * semantic_sim + lexical_weight * lexical_sim
        else:
            hybrid_score = semantic_sim
        
        hybrid_scores.append((i, float(hybrid_score)))
    
    # Sort by score
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    return hybrid_scores[:top_k]


def create_embedding_model(
    model_type: str = "openai",
    model_name: str = "text-embedding-ada-002",
    **kwargs
) -> EmbeddingModel:
    """
    Factory function to create embedding models.
    
    Args:
        model_type: Type of model ("openai", "azure", "fallback")
        model_name: Specific model name
        **kwargs: Additional arguments for model initialization
        
    Returns:
        EmbeddingModel instance
    """
    if model_type == "openai":
        return OpenAIEmbedding(model=model_name, **kwargs)
    elif model_type == "azure":
        return AzureOpenAIEmbedding(deployment_name=model_name, **kwargs)
    elif model_type == "fallback":
        return FallbackEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
