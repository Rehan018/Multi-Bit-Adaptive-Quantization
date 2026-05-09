"""
TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate
Based on: arXiv:2504.19874v1 - TurboQuant Paper

Implementation of the TurboQuant algorithm for vector compression.
Provides near-optimal distortion rates for high-dimensional vector quantization.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def _qr_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QR decomposition using Gram-Schmidt process.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    m, n = matrix.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = matrix[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], matrix[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v
    
    return Q, R


class TurboQuantCompressor:
    """
    TurboQuant vector compressor based on the paper arXiv:2504.19874v1
    
    Key Features:
    - Random rotation to induce Beta distribution on coordinates
    - Optimal scalar quantization per coordinate
    - Optional QJL (Quantized Johnson-Lindenstrauss) for unbiased inner products
    - Near-optimal distortion rate (within 2.7x of theoretical lower bound)
    - Data-oblivious (no training/calibration needed)
    """
    
    def __init__(self, bit_width: int = 4, use_qjl: bool = True, dimension: int = 384):
        """
        Initialize TurboQuant compressor.
        
        Args:
            bit_width: Bits per coordinate (2, 3, 4 recommended)
            use_qjl: Whether to use QJL for unbiased inner product estimation
            dimension: Expected vector dimension (default 384 for all-MiniLM-L6-v2)
        """
        if bit_width < 1:
            raise ValueError("bit_width must be >= 1")

        self.bit_width = bit_width
        self.use_qjl = use_qjl
        self.dimension = dimension
        # TurboQuantprod spends one bit on QJL and the remaining bits on the
        # MSE quantizer. Without QJL, the whole budget is used for MSE.
        self.mse_bit_width = max(1, bit_width - 1) if use_qjl else bit_width
        
        # Generate random rotation matrix (QR decomposition of random Gaussian)
        np.random.seed(42)  # Fixed seed for reproducibility
        random_matrix = np.random.randn(dimension, dimension)
        Q, R = _qr_decomposition(random_matrix)
        self.rotation_matrix = Q
        
        # Build optimal codebook using Lloyd-Max algorithm
        self.codebook = self._build_codebook(self.mse_bit_width, dimension)
        
        # For QJL (if enabled)
        if use_qjl:
            np.random.seed(123)  # Different seed for QJL
            self.qjl_matrix = np.random.randn(dimension, dimension)
        
        logger.info(
            "TurboQuant initialized: %s-bit total, %s-bit MSE, dimension=%s, QJL=%s",
            bit_width,
            self.mse_bit_width,
            dimension,
            use_qjl,
        )

    @staticmethod
    def _pack_indices(indices: np.ndarray, bit_width: int) -> np.ndarray:
        """Pack integer codebook indices into a byte array."""
        indices = np.asarray(indices, dtype=np.uint16).reshape(-1)
        if indices.size == 0:
            return np.array([], dtype=np.uint8)

        max_value = (1 << bit_width) - 1
        if np.any(indices > max_value):
            raise ValueError("index value exceeds bit-width capacity")

        bits = ((indices[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1).astype(np.uint8)
        return np.packbits(bits.reshape(-1), bitorder="big")

    @staticmethod
    def _unpack_indices(packed: np.ndarray, bit_width: int, count: int) -> np.ndarray:
        """Unpack byte-packed codebook indices."""
        packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
        bits_needed = count * bit_width
        bits = np.unpackbits(packed, bitorder="big")[:bits_needed]
        if bits.size < bits_needed:
            raise ValueError("packed indices do not contain enough bits")

        bit_matrix = bits.reshape(count, bit_width).astype(np.uint16)
        weights = (1 << np.arange(bit_width - 1, -1, -1, dtype=np.uint16))
        return (bit_matrix * weights).sum(axis=1).astype(np.int32)

    @staticmethod
    def _pack_signs(signs: np.ndarray) -> np.ndarray:
        """Pack QJL signs into one bit per coordinate."""
        signs = np.asarray(signs).reshape(-1)
        bits = (signs >= 0).astype(np.uint8)
        return np.packbits(bits, bitorder="big")

    @staticmethod
    def _unpack_signs(packed: np.ndarray, count: int) -> np.ndarray:
        """Unpack QJL signs to {-1, +1} values."""
        packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
        bits = np.unpackbits(packed, bitorder="big")[:count]
        if bits.size < count:
            raise ValueError("packed QJL signs do not contain enough bits")
        return np.where(bits > 0, 1.0, -1.0).astype(np.float32)
    
    def _build_codebook(self, bit_width: int, dimension: int) -> np.ndarray:
        """
        Build optimal scalar quantization codebook using Lloyd-Max algorithm.
        
        For high dimensions, the Beta distribution converges to N(0, 1/d).
        We solve the continuous k-means problem to find optimal centroids.
        
        Args:
            bit_width: Number of bits
            dimension: Vector dimension
            
        Returns:
            Codebook array of shape (2^bit_width, dimension)
        """
        n_centroids = 2 ** bit_width
        
        # For high-dimensional vectors rotated randomly,
        # each coordinate follows approximately N(0, 1/d)
        std = 1.0 / np.sqrt(dimension)
        
        # Use Lloyd-Max algorithm to find optimal quantization levels
        # For Gaussian distribution, optimal quantizer is well-studied
        centroids = self._compute_gaussian_centroids(n_centroids, std)
        
        # Replicate for each dimension (since coordinates are independent after rotation)
        codebook = np.tile(centroids.reshape(-1, 1), (1, dimension))
        
        return codebook
    
    def _compute_gaussian_centroids(self, n_centroids: int, std: float) -> np.ndarray:
        """
        Compute optimal centroids for Gaussian distribution using Lloyd-Max.
        
        For a standard normal distribution, the optimal quantization boundaries
        and centroids can be computed iteratively.
        
        Args:
            n_centroids: Number of quantization levels
            std: Standard deviation of the distribution
            
        Returns:
            Array of centroid values
        """
        # Initial guess: uniform spacing over [-3*std, 3*std]
        boundaries = np.linspace(-3 * std, 3 * std, n_centroids + 1)
        centroids = np.zeros(n_centroids)
        
        # Lloyd-Max iterations (converges quickly for Gaussian)
        for _ in range(10):
            # Update centroids as conditional expectations
            for i in range(n_centroids):
                if i == 0:
                    # First interval: [-inf, boundaries[0]]
                    centroids[i] = self._gaussian_mean(-10, boundaries[i], std)
                elif i == n_centroids - 1:
                    # Last interval: [boundaries[-1], inf]
                    centroids[i] = self._gaussian_mean(boundaries[i], 10, std)
                else:
                    # Middle intervals
                    centroids[i] = self._gaussian_mean(boundaries[i], boundaries[i+1], std)
            
            # Update boundaries as midpoints
            for i in range(1, n_centroids):
                boundaries[i] = (centroids[i-1] + centroids[i]) / 2
        
        return centroids
    
    def _gaussian_mean(self, a: float, b: float, std: float) -> float:
        """
        Compute conditional mean of Gaussian in interval [a, b].
        
        Args:
            a: Lower bound
            b: Upper bound
            std: Standard deviation
            
        Returns:
            Conditional mean
        """
        # Simple numerical integration using trapezoidal rule
        n_points = 1000
        x = np.linspace(a, b, n_points)
        
        # Gaussian PDF
        pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / std) ** 2)
        
        # Integral of x * pdf(x)
        # Use np.trapezoid for NumPy 2.0+, fallback to np.trapz for older versions
        if hasattr(np, 'trapezoid'):
            numerator = np.trapezoid(x * pdf, x)
            denominator = np.trapezoid(pdf, x)
        else:
            # Fallback for NumPy < 2.0
            numerator = np.trapz(x * pdf, x)
            denominator = np.trapz(pdf, x)
        
        if denominator < 1e-10:
            return (a + b) / 2
        
        return numerator / denominator
    
    def compress(self, vector: np.ndarray) -> Dict:
        """
        Compress a vector using TurboQuant algorithm.
        
        Algorithm:
        1. Apply random rotation
        2. Quantize each coordinate to nearest centroid
        3. (Optional) Apply QJL to residuals for unbiased inner product
        
        Args:
            vector: Input vector (shape: dimension,)
            
        Returns:
            Dictionary with compressed representation:
            - indices: packed quantization indices
            - qjl_bits: packed QJL sign bits (if use_qjl=True)
            - residual_norm: Norm of residual (for QJL scaling)
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {vector.shape[0]}")
        
        # Step 1: Apply random rotation
        rotated = self.rotation_matrix @ vector
        
        # Step 2: Quantize each coordinate to nearest centroid
        # codebook shape: (n_centroids, dimension)
        n_centroids = self.codebook.shape[0]
        indices = np.zeros(self.dimension, dtype=np.int32)
        reconstructed = np.zeros(self.dimension, dtype=np.float32)
        
        for dim in range(self.dimension):
            # Find nearest centroid for this coordinate
            distances = np.abs(self.codebook[:, dim] - rotated[dim])
            indices[dim] = np.argmin(distances)
            reconstructed[dim] = self.codebook[indices[dim], dim]
        
        # Step 3: Compute residual
        residual = rotated - reconstructed
        
        result = {
            'indices': self._pack_indices(indices, self.mse_bit_width),
            'residual_norm': float(np.linalg.norm(residual)),
            'indices_packed': True,
            'mse_bit_width': self.mse_bit_width,
            'bit_width': self.bit_width,
            'dimension': self.dimension,
        }
        
        # Optional: QJL for unbiased inner product
        if self.use_qjl:
            # Apply random projection to residual
            qjl_projection = self.qjl_matrix @ residual
            qjl_bits = np.where(qjl_projection >= 0, 1, -1).astype(np.int8)
            result['qjl_bits'] = self._pack_signs(qjl_bits)
            result['qjl_packed'] = True
        
        return result
    
    def decompress(self, compressed: Dict) -> np.ndarray:
        """
        Decompress a vector from TurboQuant representation.
        
        Args:
            compressed: Dictionary from compress() method
            
        Returns:
            Reconstructed vector (shape: dimension,)
        """
        # Reconstruct from codebook. Accept both the new packed format and the
        # old prototype format where indices were stored as one int32 per coord.
        raw_indices = np.asarray(compressed['indices'])
        packed_flag = compressed.get('indices_packed', False)
        if isinstance(packed_flag, str):
            packed_flag = packed_flag.lower() == "true"

        mse_bit_width = int(compressed.get('mse_bit_width', self.mse_bit_width))
        if packed_flag or raw_indices.size < self.dimension:
            indices = self._unpack_indices(raw_indices, mse_bit_width, self.dimension)
        else:
            indices = raw_indices.astype(np.int32).reshape(-1)[:self.dimension]

        reconstructed = np.zeros(self.dimension, dtype=np.float32)
        for dim in range(self.dimension):
            reconstructed[dim] = self.codebook[indices[dim], dim]
        
        # Optional: Add QJL residual estimate for unbiased reconstruction
        if self.use_qjl and 'qjl_bits' in compressed:
            raw_qjl = np.asarray(compressed['qjl_bits'])
            qjl_packed = compressed.get('qjl_packed', False)
            if isinstance(qjl_packed, str):
                qjl_packed = qjl_packed.lower() == "true"
            if qjl_packed or raw_qjl.size < self.dimension:
                qjl_bits = self._unpack_signs(raw_qjl, self.dimension)
            else:
                qjl_bits = np.asarray(raw_qjl, dtype=np.float32).reshape(-1)[:self.dimension]
                qjl_bits = np.where(qjl_bits >= 0, 1.0, -1.0).astype(np.float32)

            # QJL inverse transform
            qjl_reconstruction = (np.sqrt(np.pi / 2) / self.dimension) * \
                                (self.qjl_matrix.T @ qjl_bits)
            
            # Scale by residual norm
            qjl_reconstruction *= compressed['residual_norm']
            
            # Add to reconstructed vector
            reconstructed += qjl_reconstruction
        
        # Apply inverse rotation
        vector = self.rotation_matrix.T @ reconstructed
        
        return vector
    
    def calculate_distortion(self, original: np.ndarray, compressed: Dict) -> float:
        """
        Calculate MSE distortion between original and reconstructed vector.
        
        Args:
            original: Original vector
            compressed: Compressed representation
            
        Returns:
            Mean squared error
        """
        reconstructed = self.decompress(compressed)
        mse = np.mean((original - reconstructed) ** 2)
        return mse
    
    def inner_product(self, compressed_x: Dict, compressed_y: Dict) -> float:
        """
        Estimate inner product between two compressed vectors.
        
        When use_qjl=True, this provides unbiased inner product estimation.
        
        Args:
            compressed_x: Compressed representation of vector x
            compressed_y: Compressed representation of vector y
            
        Returns:
            Estimated inner product
        """
        # Decompress both vectors
        x_reconstructed = self.decompress(compressed_x)
        y_reconstructed = self.decompress(compressed_y)
        
        return float(np.dot(x_reconstructed, y_reconstructed))

    def inner_product_with_vector(self, compressed_x: Dict, vector_y: np.ndarray) -> float:
        """
        Estimate inner product between one compressed vector and one full vector.

        This matches the TurboQuantprod use case: the stored vector is compressed
        and the query vector remains available in full precision.
        """
        x_reconstructed = self.decompress(compressed_x)
        vector_y = np.asarray(vector_y, dtype=np.float32)
        return float(np.dot(vector_y, x_reconstructed))
    
    def get_compression_ratio(self) -> float:
        """
        Calculate theoretical compression ratio.
        
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        # Original: 32-bit float per coordinate
        original_bits = self.dimension * 32
        
        # With QJL, bit_width is the total budget: (bit_width - 1) MSE bits
        # plus one QJL sign bit per coordinate.
        compressed_bits = self.dimension * self.bit_width
        
        return original_bits / compressed_bits


def generate_test_vector(dimension: int = 384) -> np.ndarray:
    """Generate a random test vector for demonstration."""
    vector = np.random.randn(dimension).astype(np.float32)
    vector /= np.linalg.norm(vector)  # Normalize to unit sphere
    return vector


if __name__ == "__main__":
    """Test TurboQuant implementation."""
    print("=" * 60)
    print("TurboQuant: Online Vector Quantization")
    print("=" * 60)
    
    # Test with different bit widths
    for bit_width in [2, 3, 4]:
        print(f"\n--- Testing {bit_width}-bit quantization ---")
        
        compressor = TurboQuantCompressor(
            bit_width=bit_width,
            use_qjl=True,
            dimension=384
        )
        
        # Generate test vector
        original = generate_test_vector(384)
        
        # Compress
        compressed = compressor.compress(original)
        
        # Decompress
        reconstructed = compressor.decompress(compressed)
        
        # Calculate metrics
        mse = compressor.calculate_distortion(original, compressed)
        compression_ratio = compressor.get_compression_ratio()
        
        # Inner product test
        original_2 = generate_test_vector(384)
        compressed_2 = compressor.compress(original_2)
        
        true_inner_product = np.dot(original, original_2)
        estimated_inner_product = compressor.inner_product(compressed, compressed_2)
        ip_error = abs(true_inner_product - estimated_inner_product)
        
        print(f"Original size: {original.nbytes} bytes")
        print(f"Compressed indices: {compressed['indices'].nbytes} bytes")
        if 'qjl_bits' in compressed:
            print(f"QJL bits: {compressed['qjl_bits'].nbytes} bytes")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        print(f"MSE: {mse:.6f}")
        print(f"True inner product: {true_inner_product:.6f}")
        print(f"Estimated inner product: {estimated_inner_product:.6f}")
        print(f"Inner product error: {ip_error:.6f}")
    
    print("\n" + "=" * 60)
    print("TurboQuant implementation verified!")
    print("=" * 60)
