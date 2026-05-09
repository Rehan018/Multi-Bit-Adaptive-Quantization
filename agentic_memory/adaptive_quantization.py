"""
Multi-Bit Adaptive Quantization

Dynamically selects optimal bit-width (2/3/4-bit) for each memory based on:
- Retrieval frequency (popular memories get higher quality)
- Memory age (recent memories get higher quality)
- User-defined importance scores
- Content complexity

Benefits:
- Optimal storage-quality tradeoff
- Important memories preserved with high fidelity
- Less important memories compressed more aggressively
- Adaptive to usage patterns over time
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime, timedelta

from .turboquant import TurboQuantCompressor

logger = logging.getLogger(__name__)


class ImportanceLevel(Enum):
    """Memory importance levels for adaptive quantization."""
    CRITICAL = 4    # 4-bit compression (highest quality)
    HIGH = 3        # 3-bit compression
    MEDIUM = 3      # 3-bit compression
    LOW = 2         # 2-bit compression (maximum compression)


class AdaptiveQuantizationPolicy:
    """
    Determines optimal bit-width for memory compression based on multiple factors.
    
    Policy Components:
    1. Retrieval Frequency: Frequently accessed memories → higher bit-width
    2. Memory Age: Recent memories → higher bit-width
    3. Importance Score: User-defined importance → higher bit-width
    4. Content Complexity: Complex/technical content → higher bit-width
    """
    
    def __init__(
        self,
        frequency_weight: float = 0.4,
        age_weight: float = 0.3,
        importance_weight: float = 0.3,
        recent_threshold_days: int = 7,
        high_frequency_threshold: int = 10
    ):
        """
        Initialize adaptive quantization policy.
        
        Args:
            frequency_weight: Weight for retrieval frequency (0-1)
            age_weight: Weight for memory age (0-1)
            importance_weight: Weight for importance score (0-1)
            recent_threshold_days: Days considered "recent"
            high_frequency_threshold: Retrieval count for "high frequency"
        """
        self.frequency_weight = frequency_weight
        self.age_weight = age_weight
        self.importance_weight = importance_weight
        self.recent_threshold_days = recent_threshold_days
        self.high_frequency_threshold = high_frequency_threshold
        
        # Validate weights sum to 1.0
        total_weight = frequency_weight + age_weight + importance_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            self.frequency_weight /= total_weight
            self.age_weight /= total_weight
            self.importance_weight /= total_weight
        
        logger.info(
            f"AdaptiveQuantizationPolicy initialized: "
            f"freq_w={frequency_weight}, age_w={age_weight}, "
            f"imp_w={importance_weight}"
        )
    
    def calculate_importance_score(
        self,
        retrieval_count: int = 0,
        days_since_creation: int = 0,
        user_importance: float = 0.5,
        content_length: int = 0
    ) -> float:
        """
        Calculate composite importance score (0-1).
        
        Args:
            retrieval_count: Number of times memory was retrieved
            days_since_creation: Days since memory was created
            user_importance: User-defined importance (0-1)
            content_length: Length of memory content
            
        Returns:
            Importance score (0-1, higher = more important)
        """
        # Frequency score (0-1)
        freq_score = min(retrieval_count / self.high_frequency_threshold, 1.0)
        
        # Age score (0-1, recent = high score)
        if days_since_creation <= self.recent_threshold_days:
            age_score = 1.0
        else:
            # Decay after recent threshold
            age_score = max(0.0, 1.0 - (days_since_creation - self.recent_threshold_days) / 90)
        
        # Importance score (already 0-1)
        imp_score = user_importance
        
        # Composite score
        composite = (
            self.frequency_weight * freq_score +
            self.age_weight * age_score +
            self.importance_weight * imp_score
        )
        
        return min(max(composite, 0.0), 1.0)
    
    def determine_bit_width(self, importance_score: float) -> int:
        """
        Determine optimal bit-width based on importance score.
        
        Args:
            importance_score: Score from calculate_importance_score()
            
        Returns:
            Bit-width (2, 3, or 4)
        """
        if importance_score >= 0.7:
            return 4  # Critical/High importance
        elif importance_score >= 0.4:
            return 3  # Medium importance
        else:
            return 2  # Low importance
    
    def get_importance_level(self, importance_score: float) -> ImportanceLevel:
        """Get importance level enum from score."""
        bit_width = self.determine_bit_width(importance_score)
        return ImportanceLevel(bit_width)


class MultiBitAdaptiveCompressor:
    """
    Compressor that uses different bit-widths for different memories.
    
    Maintains separate compressors for each bit-width and selects
    the appropriate one based on memory importance.
    """
    
    def __init__(
        self,
        dimension: int = 384,
        use_qjl: bool = True,
        policy: Optional[AdaptiveQuantizationPolicy] = None
    ):
        """
        Initialize multi-bit adaptive compressor.
        
        Args:
            dimension: Embedding dimension
            use_qjl: Whether to use QJL for all compressors
            policy: Quantization policy (creates default if None)
        """
        self.dimension = dimension
        self.use_qjl = use_qjl
        self.policy = policy or AdaptiveQuantizationPolicy()
        
        # Create compressors for each bit-width
        self.compressors = {
            2: TurboQuantCompressor(bit_width=2, use_qjl=use_qjl, dimension=dimension),
            3: TurboQuantCompressor(bit_width=3, use_qjl=use_qjl, dimension=dimension),
            4: TurboQuantCompressor(bit_width=4, use_qjl=use_qjl, dimension=dimension)
        }
        
        # Statistics tracking
        self.stats = {
            'total_compressed': 0,
            'by_bit_width': {2: 0, 3: 0, 4: 0},
            'total_original_bytes': 0,
            'total_compressed_bytes': 0
        }
        
        logger.info(
            f"MultiBitAdaptiveCompressor initialized: "
            f"dimension={dimension}, QJL={use_qjl}"
        )
    
    def compress_adaptive(
        self,
        embedding: np.ndarray,
        metadata: Dict
    ) -> Tuple[Dict, int]:
        """
        Compress embedding with adaptive bit-width selection.
        
        Args:
            embedding: Embedding vector to compress
            metadata: Memory metadata containing:
                - retrieval_count: Number of retrievals
                - timestamp: Creation timestamp
                - importance: User-defined importance (0-1)
                - content_length: Length of content
                
        Returns:
            Tuple of (compressed_data, bit_width_used)
        """
        # Extract metadata for importance calculation
        retrieval_count = metadata.get('retrieval_count', 0)
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        user_importance = metadata.get('importance', 0.5)
        content = metadata.get('content', '')
        
        # Calculate days since creation
        try:
            created_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days_since = (datetime.now(created_date.tzinfo) - created_date).days
        except Exception:
            days_since = 0
        
        # Calculate importance score
        importance_score = self.policy.calculate_importance_score(
            retrieval_count=retrieval_count,
            days_since_creation=days_since,
            user_importance=user_importance,
            content_length=len(content)
        )
        
        # Determine optimal bit-width
        bit_width = self.policy.determine_bit_width(importance_score)
        
        # Compress with selected bit-width
        compressor = self.compressors[bit_width]
        compressed = compressor.compress(embedding)
        
        # Add metadata about compression
        compressed['bit_width'] = bit_width
        compressed['importance_score'] = importance_score
        
        # Update statistics
        self.stats['total_compressed'] += 1
        self.stats['by_bit_width'][bit_width] += 1
        self.stats['total_original_bytes'] += embedding.nbytes
        compressed_size = compressed['indices'].nbytes
        if 'qjl_bits' in compressed:
            compressed_size += compressed['qjl_bits'].nbytes
        self.stats['total_compressed_bytes'] += compressed_size
        
        return compressed, bit_width
    
    def decompress(self, compressed: Dict) -> np.ndarray:
        """
        Decompress embedding using the bit-width stored in metadata.
        
        Args:
            compressed: Compressed data with 'bit_width' field
            
        Returns:
            Reconstructed embedding
        """
        bit_width = compressed.get('bit_width', 4)  # Default to 4-bit
        
        if bit_width not in self.compressors:
            raise ValueError(f"Invalid bit-width: {bit_width}")
        
        compressor = self.compressors[bit_width]
        return compressor.decompress(compressed)
    
    def get_statistics(self) -> Dict:
        """Get compression statistics by bit-width."""
        if self.stats['total_compressed'] == 0:
            return {
                'total_compressed': 0,
                'distribution': {2: 0, 3: 0, 4: 0},
                'percentages': {2: 0, 3: 0, 4: 0},
                'average_bit_width': 0,
                'compression_ratio': 0
            }
        
        total = self.stats['total_compressed']
        distribution = self.stats['by_bit_width']
        
        percentages = {
            bw: (count / total * 100) for bw, count in distribution.items()
        }
        
        avg_bit_width = sum(bw * count for bw, count in distribution.items()) / total
        
        compression_ratio = (
            self.stats['total_original_bytes'] / 
            max(self.stats['total_compressed_bytes'], 1)
        )
        
        return {
            'total_compressed': total,
            'distribution': distribution,
            'percentages': percentages,
            'average_bit_width': avg_bit_width,
            'compression_ratio': compression_ratio,
            'original_size_mb': self.stats['total_original_bytes'] / (1024 * 1024),
            'compressed_size_mb': self.stats['total_compressed_bytes'] / (1024 * 1024),
            'space_saved_mb': (
                self.stats['total_original_bytes'] - 
                self.stats['total_compressed_bytes']
            ) / (1024 * 1024)
        }
    
    def update_policy(self, **kwargs):
        """Update quantization policy parameters."""
        for key, value in kwargs.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                logger.info(f"Updated policy parameter: {key} = {value}")


def create_adaptive_compressor(
    dimension: int = 384,
    use_qjl: bool = True,
    **policy_kwargs
) -> MultiBitAdaptiveCompressor:
    """
    Convenience function to create adaptive compressor with custom policy.
    
    Args:
        dimension: Embedding dimension
        use_qjl: Enable QJL
        **policy_kwargs: Parameters for AdaptiveQuantizationPolicy
        
    Returns:
        Configured MultiBitAdaptiveCompressor
    """
    policy = AdaptiveQuantizationPolicy(**policy_kwargs)
    return MultiBitAdaptiveCompressor(
        dimension=dimension,
        use_qjl=use_qjl,
        policy=policy
    )
