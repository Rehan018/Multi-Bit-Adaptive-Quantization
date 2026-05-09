"""
Memory Evolution Enhancement using TurboQuant Distortion Analysis

This module enhances the memory evolution system by using quantization distortion
as a mathematical trigger for memory evolution decisions, reducing LLM calls
and improving efficiency.

Based on TurboQuant paper (arXiv:2504.19874v1) principles.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agentic_memory.turboquant import TurboQuantCompressor
import logging

logger = logging.getLogger(__name__)


class TurboQuantEvolutionAnalyzer:
    """
    Analyze memory evolution potential using quantization distortion.
    
    Key Concept:
    - Low distortion → memory is redundant (similar to existing memories)
    - High distortion → memory is unique (new information)
    
    This allows us to make evolution decisions based on mathematical
    properties rather than always calling LLM.
    """
    
    def __init__(
        self,
        bit_width: int = 4,
        evolution_threshold: float = 0.0001,
        dimension: int = 384
    ):
        """
        Initialize TurboQuant Evolution Analyzer.
        
        Args:
            bit_width: Bits per coordinate for quantization
            evolution_threshold: MSE threshold below which memory is considered redundant
            dimension: Embedding dimension
        """
        self.bit_width = bit_width
        self.evolution_threshold = evolution_threshold
        self.dimension = dimension
        
        # Initialize compressor for distortion calculation
        self.compressor = TurboQuantCompressor(
            bit_width=bit_width,
            use_qjl=False,  # Don't need QJL for evolution analysis
            dimension=dimension
        )
        
        logger.info(
            f"TurboQuantEvolutionAnalyzer initialized: "
            f"bit_width={bit_width}, threshold={evolution_threshold}, "
            f"dimension={dimension}"
        )
    
    def calculate_memory_distortion(
        self,
        new_embedding: np.ndarray,
        related_embeddings: List[np.ndarray]
    ) -> Dict:
        """
        Calculate distortion between new memory and related memories.
        
        This measures how "redundant" or "unique" the new memory is.
        
        Key Insight:
        - Compress new embedding and related embeddings
        - If new embedding can be reconstructed from related ones with low error → redundant
        - If reconstruction error is high → unique (contains new information)
        
        Args:
            new_embedding: Embedding of new memory
            related_embeddings: Embeddings of related memories
            
        Returns:
            Dictionary with distortion metrics:
            - min_distortion: Minimum distortion to any related memory
            - avg_distortion: Average distortion to all related memories
            - max_distortion: Maximum distortion to any related memory
            - should_evolve: Boolean indicating if evolution is needed
            - redundancy_score: Score from 0 (unique) to 1 (highly redundant)
        """
        if not related_embeddings:
            return {
                'min_distortion': float('inf'),
                'avg_distortion': float('inf'),
                'max_distortion': float('inf'),
                'should_evolve': False,
                'redundancy_score': 0.0,
                'reason': 'No related memories found'
            }
        
        # Calculate similarity-based distortion to each related memory
        # Lower distance = higher redundancy
        new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-10)
        
        distortions = []
        for related_emb in related_embeddings:
            related_norm = related_emb / (np.linalg.norm(related_emb) + 1e-10)
            
            # Cosine distance (1 - cosine_similarity)
            cosine_sim = np.dot(new_norm, related_norm)
            cosine_dist = 1.0 - cosine_sim
            
            # Use TurboQuant to compress and measure reconstruction
            # This adds quantization-aware distortion
            compressed = self.compressor.compress(related_emb)
            reconstructed = self.compressor.decompress(compressed)
            recon_norm = reconstructed / (np.linalg.norm(reconstructed) + 1e-10)
            
            # Combined distortion: base distance + quantization error
            quant_error = np.mean((related_emb - reconstructed) ** 2)
            total_distortion = cosine_dist + quant_error
            
            distortions.append(total_distortion)
        
        # Compute statistics
        min_distortion = float(np.min(distortions))
        avg_distortion = float(np.mean(distortions))
        max_distortion = float(np.max(distortions))
        
        # Determine if evolution is needed
        # Low distortion = redundant (should evolve)
        should_evolve = min_distortion < self.evolution_threshold
        
        # Calculate redundancy score (0 = unique, 1 = highly redundant)
        # Lower distortion = higher redundancy
        # Normalize: if min_distortion == 0 → redundancy = 1.0
        #            if min_distortion >= threshold*10 → redundancy = 0.0
        redundancy_score = max(0.0, 1.0 - (min_distortion / (self.evolution_threshold * 10)))
        redundancy_score = min(1.0, redundancy_score)
        
        return {
            'min_distortion': min_distortion,
            'avg_distortion': avg_distortion,
            'max_distortion': max_distortion,
            'should_evolve': should_evolve,
            'redundancy_score': redundancy_score,
            'reason': self._get_evolution_reason(
                should_evolve, min_distortion, redundancy_score
            )
        }
    
    def _get_evolution_reason(
        self,
        should_evolve: bool,
        min_distortion: float,
        redundancy_score: float
    ) -> str:
        """Generate human-readable reason for evolution decision."""
        
        if should_evolve:
            if redundancy_score > 0.9:
                return "Highly redundant - very similar to existing memories"
            elif redundancy_score > 0.7:
                return "Moderately redundant - significant overlap with existing memories"
            else:
                return "Slightly redundant - some similarity to existing memories"
        else:
            if redundancy_score < 0.1:
                return "Highly unique - contains new information"
            elif redundancy_score < 0.3:
                return "Mostly unique - minimal overlap with existing memories"
            else:
                return "Distinct but related - evolution not required yet"
    
    def batch_analyze_evolution(
        self,
        new_embeddings: List[np.ndarray],
        existing_embeddings: List[np.ndarray],
        top_k_related: int = 5
    ) -> List[Dict]:
        """
        Batch analyze multiple new memories for evolution potential.
        
        Args:
            new_embeddings: List of new memory embeddings
            existing_embeddings: List of all existing memory embeddings
            top_k_related: Number of most related memories to consider
            
        Returns:
            List of distortion analysis results for each new memory
        """
        results = []
        
        for i, new_emb in enumerate(new_embeddings):
            # Find top-k most related existing memories
            related = self._find_most_related(
                new_emb, existing_embeddings, top_k_related
            )
            
            # Analyze distortion
            analysis = self.calculate_memory_distortion(new_emb, related)
            analysis['memory_index'] = i
            
            results.append(analysis)
        
        return results
    
    def _find_most_related(
        self,
        query_embedding: np.ndarray,
        existing_embeddings: List[np.ndarray],
        top_k: int
    ) -> List[np.ndarray]:
        """Find top-k most related existing embeddings."""
        if not existing_embeddings:
            return []
        
        # Calculate cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = []
        
        for emb in existing_embeddings:
            emb_norm = emb / np.linalg.norm(emb)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append(similarity)
        
        # Get top-k indices
        similarities = np.array(similarities)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top-k embeddings
        return [existing_embeddings[i] for i in top_k_indices]
    
    def update_threshold_based_on_data(
        self,
        sample_embeddings: List[np.ndarray],
        target_evolution_rate: float = 0.2
    ):
        """
        Automatically calibrate evolution threshold based on sample data.
        
        Args:
            sample_embeddings: Sample of existing memory embeddings
            target_evolution_rate: Desired fraction of memories that trigger evolution
        """
        if len(sample_embeddings) < 10:
            logger.warning("Not enough samples for threshold calibration")
            return
        
        # Calculate pairwise distortions
        distortions = []
        sample_size = min(100, len(sample_embeddings))  # Limit for performance
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                compressed = self.compressor.compress(sample_embeddings[i])
                reconstructed = self.compressor.decompress(compressed)
                mse = np.mean((sample_embeddings[i] - reconstructed) ** 2)
                distortions.append(mse)
        
        # Set threshold to target evolution rate percentile
        if distortions:
            self.evolution_threshold = float(np.percentile(
                distortions, target_evolution_rate * 100
            ))
            logger.info(
                f"Auto-calibrated threshold: {self.evolution_threshold} "
                f"(target evolution rate: {target_evolution_rate})"
            )


class EnhancedMemoryEvolution:
    """
    Enhanced memory evolution system that combines TurboQuant distortion
    analysis with traditional LLM-based evolution for optimal results.
    
    This system:
    1. Uses TurboQuant distortion for quick filtering (saves LLM calls)
    2. Only calls LLM when distortion indicates potential evolution needed
    3. Provides detailed evolution analytics
    """
    
    def __init__(
        self,
        bit_width: int = 4,
        evolution_threshold: float = 0.0001,
        dimension: int = 384,
        enable_distortion_filter: bool = True
    ):
        """
        Initialize Enhanced Memory Evolution system.
        
        Args:
            bit_width: Bits for quantization
            evolution_threshold: MSE threshold for evolution trigger
            dimension: Embedding dimension
            enable_distortion_filter: Whether to use distortion as pre-filter
        """
        self.enable_distortion_filter = enable_distortion_filter
        self.analytics = TurboQuantEvolutionAnalyzer(
            bit_width=bit_width,
            evolution_threshold=evolution_threshold,
            dimension=dimension
        )
        
        # Statistics tracking
        self.stats = {
            'total_analyses': 0,
            'llm_calls_saved': 0,
            'evolution_triggers': 0,
            'unique_memories': 0
        }
    
    def should_evolve_memory(
        self,
        new_embedding: np.ndarray,
        related_embeddings: List[np.ndarray],
        use_distortion_filter: bool = None
    ) -> Dict:
        """
        Determine if a new memory should trigger evolution.
        
        This is the enhanced version that uses TurboQuant distortion
        to make faster decisions and save LLM calls.
        
        Args:
            new_embedding: Embedding of new memory
            related_embeddings: Embeddings of related memories
            use_distortion_filter: Override default filter setting
            
        Returns:
            Dictionary with:
            - should_evolve: Boolean
            - reason: Explanation
            - distortion_metrics: Quantitative measures
            - llm_call_needed: Whether LLM should still be called
        """
        if use_distortion_filter is None:
            use_distortion_filter = self.enable_distortion_filter
        
        self.stats['total_analyses'] += 1
        
        if use_distortion_filter:
            # Use TurboQuant distortion analysis (fast, no LLM call)
            distortion_analysis = self.analytics.calculate_memory_distortion(
                new_embedding, related_embeddings
            )
            
            should_evolve = distortion_analysis['should_evolve']
            
            if should_evolve:
                self.stats['evolution_triggers'] += 1
                llm_call_needed = True  # Still call LLM for detailed evolution
            else:
                self.stats['unique_memories'] += 1
                self.stats['llm_calls_saved'] += 1
                llm_call_needed = False  # Skip LLM call, memory is unique
            
            result = {
                'should_evolve': should_evolve,
                'llm_call_needed': llm_call_needed,
                'distortion_metrics': distortion_analysis,
                'reason': distortion_analysis['reason'],
                'method': 'turboquant_distortion',
                'llm_calls_saved': self.stats['llm_calls_saved']
            }
        else:
            # Traditional approach: always call LLM
            result = {
                'should_evolve': True,
                'llm_call_needed': True,
                'distortion_metrics': None,
                'reason': 'Traditional LLM-based approach',
                'method': 'llm_only'
            }
        
        return result
    
    def get_evolution_statistics(self) -> Dict:
        """Get evolution system statistics."""
        total = self.stats['total_analyses']
        
        if total == 0:
            return {
                'total_analyses': 0,
                'llm_calls_saved': 0,
                'llm_call_savings_percent': 0,
                'evolution_triggers': 0,
                'unique_memories': 0,
                'evolution_rate': 0
            }
        
        return {
            'total_analyses': total,
            'llm_calls_saved': self.stats['llm_calls_saved'],
            'llm_call_savings_percent': (self.stats['llm_calls_saved'] / total) * 100,
            'evolution_triggers': self.stats['evolution_triggers'],
            'unique_memories': self.stats['unique_memories'],
            'evolution_rate': (self.stats['evolution_triggers'] / total) * 100
        }
