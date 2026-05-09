"""
Hybrid Search Optimization for Multi-Bit Adaptive Quantization

Combines semantic (vector) search with keyword-based search for improved retrieval accuracy.
Uses reciprocal rank fusion to combine results from both methods.

Key Benefits:
- Better recall than pure vector search
- Captures exact keyword matches that embeddings might miss
- More robust to query variations
- Improved precision for specific terminology
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extract keywords from text using various methods."""
    
    def __init__(self):
        # Common English stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'it', 'its', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'not', 'no',
            'so', 'if', 'as', 'from', 'into', 'through', 'during', 'before', 'after'
        }
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.
        
        Uses a combination of:
        1. TF (Term Frequency) - words appearing frequently
        2. Word length - longer words tend to be more meaningful
        3. Position - words appearing early are often important
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        # Tokenize and clean
        tokens = self._tokenize(text)
        
        if not tokens:
            return []
        
        # Calculate term frequency
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # Score keywords based on multiple factors
        scored_keywords = []
        for word, freq in tf.items():
            if word in self.stop_words or len(word) < 3:
                continue
            
            # Scoring formula:
            # - Term frequency (higher is better)
            # - Word length bonus (longer words more meaningful)
            # - Capitalization bonus (proper nouns often important)
            score = freq * (1 + 0.1 * len(word))
            
            if word[0].isupper():
                score *= 1.2  # Bonus for capitalized words
            
            scored_keywords.append((word, score))
        
        # Sort by score and return top keywords
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored_keywords[:max_keywords]]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that handles punctuation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes within words
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove standalone apostrophes
        tokens = [t.strip("'") for t in tokens]
        
        # Filter empty strings
        return [t for t in tokens if t]


class BM25Scorer:
    """
    BM25 (Best Matching 25) scoring for keyword-based retrieval.
    
    Industry-standard algorithm for ranking documents by relevance to a query.
    Better than simple TF-IDF for short documents.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.5-1.0)
        """
        self.k1 = k1
        self.b = b
        self.document_freq = defaultdict(int)  # How many docs contain each term
        self.document_lengths = {}  # Length of each document
        self.avg_doc_length = 0.0
        self.num_documents = 0
    
    def index_documents(self, documents: Dict[str, str]):
        """
        Index a collection of documents for BM25 scoring.
        
        Args:
            documents: Dictionary mapping doc_id to document text
        """
        extractor = KeywordExtractor()
        self.num_documents = len(documents)
        total_length = 0
        
        # Clear previous indices
        self.document_freq.clear()
        self.document_lengths.clear()
        
        # Process each document
        for doc_id, text in documents.items():
            # Extract keywords/tokens
            tokens = extractor._tokenize(text)
            doc_length = len(tokens)
            self.document_lengths[doc_id] = doc_length
            total_length += doc_length
            
            # Count document frequency (unique terms per doc)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.document_freq[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = total_length / max(self.num_documents, 1)
    
    def score_query(self, query: str, doc_id: str, doc_text: str) -> float:
        """
        Calculate BM25 score for a query against a document.
        
        Args:
            query: Search query
            doc_id: Document ID
            doc_text: Document text
            
        Returns:
            BM25 relevance score (higher = more relevant)
        """
        extractor = KeywordExtractor()
        query_terms = extractor._tokenize(query)
        doc_tokens = extractor._tokenize(doc_text)
        
        if not query_terms or not doc_tokens:
            return 0.0
        
        # Calculate term frequencies in document
        doc_tf = defaultdict(int)
        for token in doc_tokens:
            doc_tf[token] += 1
        
        doc_length = self.document_lengths.get(doc_id, len(doc_tokens))
        score = 0.0
        
        for term in query_terms:
            # Get document frequency
            df = self.document_freq.get(term, 0)
            
            if df == 0:
                continue
            
            # IDF (Inverse Document Frequency) component
            idf = np.log((self.num_documents - df + 0.5) / (df + 0.5) + 1.0)
            
            # TF (Term Frequency) component with saturation
            tf = doc_tf.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            # BM25 score for this term
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    def score_all_documents(self, query: str, documents: Dict[str, str]) -> Dict[str, float]:
        """
        Score all documents against a query.
        
        Args:
            query: Search query
            documents: Dictionary mapping doc_id to document text
            
        Returns:
            Dictionary mapping doc_id to BM25 score
        """
        scores = {}
        for doc_id, doc_text in documents.items():
            score = self.score_query(query, doc_id, doc_text)
            if score > 0:
                scores[doc_id] = score
        
        return scores


class ReciprocalRankFusion:
    """
    Combine ranked lists from different retrieval methods using Reciprocal Rank Fusion (RRF).
    
    RRF is a robust method for combining rankings that doesn't require score normalization.
    Formula: RRF_score(d) = Σ (1 / (k + rank_i(d)))
    where k is a constant (typically 60) and rank_i is the rank in list i.
    """
    
    def __init__(self, k: float = 60.0):
        """
        Initialize RRF combiner.
        
        Args:
            k: Constant for RRF calculation (typically 60)
        """
        self.k = k
    
    def fuse_rankings(
        self,
        rankings: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked lists into a single combined ranking.
        
        Args:
            rankings: List of ranked lists, each containing (doc_id, score) tuples
            weights: Optional weights for each ranking source
            
        Returns:
            Combined ranked list of (doc_id, rrf_score) tuples
        """
        if not rankings:
            return []
        
        if weights is None:
            weights = [1.0] * len(rankings)
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        for rank_list, weight in zip(rankings, weights):
            for rank, (doc_id, _) in enumerate(rank_list, start=1):
                rrf_scores[doc_id] += weight / (self.k + rank)
        
        # Sort by RRF score (descending)
        combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return combined


class HybridSearchOptimizer:
    """
    Hybrid search system combining semantic (vector) and keyword (BM25) search.
    
    Architecture:
    1. Vector Search: Semantic similarity using embeddings
    2. Keyword Search: Exact term matching using BM25
    3. Result Fusion: Combine using Reciprocal Rank Fusion
    
    Benefits:
    - Better recall: Captures both semantic and lexical matches
    - Improved precision: Reduces false positives
    - Robust: Works well even when one method fails
    """
    
    def __init__(
        self,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: float = 60.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Initialize hybrid search optimizer.
        
        Args:
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            rrf_k: RRF constant
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.bm25 = BM25Scorer(k1=bm25_k1, b=bm25_b)
        self.extractor = KeywordExtractor()
        
        # Cache for indexed documents
        self.indexed_documents = {}
        
        logger.info(
            f"HybridSearchOptimizer initialized: "
            f"vector_weight={vector_weight}, keyword_weight={keyword_weight}"
        )
    
    def index_memories(self, memories: Dict[str, str]):
        """
        Index memories for keyword-based search.
        
        Args:
            memories: Dictionary mapping memory_id to memory content/text
        """
        self.indexed_documents = memories
        self.bm25.index_documents(memories)
        logger.info(f"Indexed {len(memories)} memories for keyword search")
    
    def hybrid_search(
        self,
        query: str,
        vector_results: List[Tuple[str, float]],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining vector and keyword results.
        
        Args:
            query: Search query
            vector_results: Results from vector search [(doc_id, distance), ...]
                           Note: Lower distance = more similar
            k: Number of final results to return
            
        Returns:
            Combined ranked results [(doc_id, combined_score), ...]
        """
        # Convert vector distances to scores (invert so higher = better)
        # Normalize to 0-1 range
        if vector_results:
            min_dist = min(dist for _, dist in vector_results)
            max_dist = max(dist for _, dist in vector_results)
            dist_range = max_dist - min_dist if max_dist != min_dist else 1.0
            
            vector_scores = [
                (doc_id, 1.0 - ((dist - min_dist) / dist_range))
                for doc_id, dist in vector_results
            ]
        else:
            vector_scores = []
        
        # Perform keyword search on indexed documents
        keyword_scores_dict = self.bm25.score_all_documents(
            query, self.indexed_documents
        )
        
        # Convert to ranked list
        keyword_scores = sorted(
            keyword_scores_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top-k for each method before fusion
        top_k_vector = vector_scores[:k * 2]  # Get extra for fusion
        top_k_keyword = keyword_scores[:k * 2]
        
        # Fuse rankings using RRF
        fused = self.rrf.fuse_rankings(
            rankings=[top_k_vector, top_k_keyword],
            weights=[self.vector_weight, self.keyword_weight]
        )
        
        # Return top-k results
        return fused[:k]
    
    def get_search_analytics(self, query: str, vector_results: List[Tuple[str, float]]) -> Dict:
        """
        Get analytics about the hybrid search performance.
        
        Args:
            query: Search query
            vector_results: Original vector search results
            
        Returns:
            Dictionary with search analytics
        """
        # Extract query keywords
        query_keywords = self.extractor.extract_keywords(query, max_keywords=5)
        
        # Analyze overlap between vector and keyword results
        vector_ids = set(doc_id for doc_id, _ in vector_results)
        
        keyword_scores_dict = self.bm25.score_all_documents(
            query, self.indexed_documents
        )
        keyword_ids = set(keyword_scores_dict.keys())
        
        overlap = vector_ids.intersection(keyword_ids)
        
        return {
            'query': query,
            'query_keywords': query_keywords,
            'vector_results_count': len(vector_results),
            'keyword_results_count': len(keyword_ids),
            'overlap_count': len(overlap),
            'overlap_percentage': (len(overlap) / max(len(vector_ids), 1)) * 100,
            'search_method': 'hybrid'
        }


def create_hybrid_search_wrapper(memory_system):
    """
    Create a wrapper function to add hybrid search to an existing memory system.
    
    This is a convenience function for easy integration.
    
    Args:
        memory_system: AgenticMemorySystem instance
        
    Returns:
        HybridSearchOptimizer instance configured with the memory system
    """
    optimizer = HybridSearchOptimizer(
        vector_weight=0.6,
        keyword_weight=0.4
    )
    
    # Index all existing memories
    memories_dict = {
        note_id: note.content
        for note_id, note in memory_system.memories.items()
    }
    optimizer.index_memories(memories_dict)
    
    return optimizer
