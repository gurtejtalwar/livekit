import os
import asyncio
from collections import OrderedDict
from typing import Optional, Tuple
from dotenv import load_dotenv

import numpy as np
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv(override=True)
class LRUCache:
    """Simple thread-safe LRU for async apps (no awaits needed)."""
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        # mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: str):
        self.cache[key] = value
        self.cache.move_to_end(key)

        # evict least recently used
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# create global cache instance
mongo_lru_cache = LRUCache(max_size=512)

# Update the cache class to support async operations
class SemanticLRUCache:
    def __init__(self, embed_model, capacity=200, similarity_threshold=0.95):
        """
        Semantic cache using cosine similarity for matching.
        
        Args:
            embed_model: OpenAI embedding model instance
            capacity: Maximum number of cached items
            similarity_threshold: Minimum cosine similarity (0-1) to consider a cache hit
        """
        self.embed_model = embed_model
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.cache = OrderedDict()  # query_text -> (embedding, value)
        self._lock = asyncio.Lock()  # Thread-safe operations
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(emb1, dtype=np.float32)
        emb2 = np.array(emb2, dtype=np.float32)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, question: str) -> Optional[Tuple[str, any, float]]:
        """
        Find cached value for semantically similar question (synchronous).
        
        Returns:
            Tuple of (matched_question, cached_value, similarity_score) if hit, None if miss
        """
        if not self.cache:
            return None
        
        # Get embedding for the question
        query_embedding = self.embed_model.get_query_embedding(question)
        query_emb = np.array(query_embedding, dtype=np.float32)
        
        # Find most similar cached question
        best_similarity = -1
        best_key = None
        
        for cached_question, (cached_emb, _) in self.cache.items():
            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = cached_question
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            # Move to end (most recently used)
            self.cache.move_to_end(best_key)
            _, value = self.cache[best_key]
            return (best_key, value, best_similarity)
        
        return None
    
    async def set_async(self, question: str, value: any):
        """
        Store question and value in cache asynchronously.
        Uses lock to prevent race conditions.
        """
        async with self._lock:
            # Get embedding for the question
            embedding = self.embed_model.get_query_embedding(question)
            emb_array = np.array(embedding, dtype=np.float32)
            
            # Remove old entry if question exists
            if question in self.cache:
                del self.cache[question]
            
            # Add new entry
            self.cache[question] = (emb_array, value)
            self.cache.move_to_end(question)
            
            # Evict oldest if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def set(self, question: str, value: any):
        """Synchronous version of set (kept for compatibility)."""
        embedding = self.embed_model.get_query_embedding(question)
        emb_array = np.array(embedding, dtype=np.float32)
        
        if question in self.cache:
            del self.cache[question]
        
        self.cache[question] = (emb_array, value)
        self.cache.move_to_end(question)
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
embed_model = OpenAIEmbedding(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-small"
    )
# Initialize cache with your embed_model
semantic_context_cache = SemanticLRUCache(
    embed_model=embed_model,  # Your OpenAI embedding model
    capacity=200,
    similarity_threshold=0.95
)