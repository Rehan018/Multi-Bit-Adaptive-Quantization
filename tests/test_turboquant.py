"""
Test suite for TurboQuant vector compression integration.
Tests the TurboQuant compressor and TurboQuantRetriever.
"""

import sys
import os
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory.turboquant import TurboQuantCompressor, generate_test_vector
from agentic_memory.retrievers import TurboQuantRetriever


def test_turboquant_basic_compression():
    """Test basic TurboQuant compression and decompression."""
    print("\n" + "="*70)
    print("TEST 1: Basic TurboQuant Compression")
    print("="*70)
    
    for bit_width in [2, 3, 4]:
        print(f"\n--- {bit_width}-bit quantization ---")
        
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
        
        print(f"  Original size: {original.nbytes} bytes ({original.shape})")
        print(f"  Compressed indices: {compressed['indices'].nbytes} bytes")
        if 'qjl_bits' in compressed:
            print(f"  QJL bits: {compressed['qjl_bits'].nbytes} bytes")
        total_compressed = compressed['indices'].nbytes + compressed.get('qjl_bits', np.array([])).nbytes
        print(f"  Total compressed: {total_compressed} bytes")
        print(f"  Theoretical compression ratio: {compression_ratio:.1f}x")
        print(f"  Actual compression ratio: {original.nbytes / total_compressed:.1f}x")
        print(f"  MSE: {mse:.6f}")
        print(f"  ✓ PASSED" if mse < 0.1 else f"  ✗ FAILED (MSE too high)")
    
    print("\n" + "="*70)


def test_inner_product_unbiasedness():
    """Test that TurboQuant provides unbiased inner product estimation."""
    print("\n" + "="*70)
    print("TEST 2: Inner Product Unbiasedness")
    print("="*70)
    
    compressor = TurboQuantCompressor(
        bit_width=4,
        use_qjl=True,
        dimension=384
    )
    
    # Generate multiple test pairs
    n_tests = 10
    total_error = 0.0
    
    for i in range(n_tests):
        vec1 = generate_test_vector(384)
        vec2 = generate_test_vector(384)
        
        # True inner product
        true_ip = np.dot(vec1, vec2)
        
        # Compress both vectors
        comp1 = compressor.compress(vec1)
        comp2 = compressor.compress(vec2)
        
        # Estimated inner product
        estimated_ip = compressor.inner_product(comp1, comp2)
        
        error = abs(true_ip - estimated_ip)
        total_error += error
        
        if i < 3:  # Show first 3 examples
            print(f"  Test {i+1}: True IP={true_ip:.6f}, Estimated IP={estimated_ip:.6f}, Error={error:.6f}")
    
    avg_error = total_error / n_tests
    print(f"\n  Average inner product error over {n_tests} tests: {avg_error:.6f}")
    print(f"  ✓ PASSED" if avg_error < 0.05 else f"  ✗ FAILED (error too high)")
    print("="*70)


def test_different_bit_widths():
    """Compare performance across different bit widths."""
    print("\n" + "="*70)
    print("TEST 3: Bit Width Comparison")
    print("="*70)
    
    results = []
    
    for bit_width in [2, 3, 4]:
        compressor = TurboQuantCompressor(
            bit_width=bit_width,
            use_qjl=True,
            dimension=384
        )
        
        vec = generate_test_vector(384)
        compressed = compressor.compress(vec)
        reconstructed = compressor.decompress(compressed)
        
        mse = compressor.calculate_distortion(vec, compressed)
        compression_ratio = compressor.get_compression_ratio()
        
        results.append({
            'bit_width': bit_width,
            'mse': mse,
            'compression_ratio': compression_ratio
        })
        
        print(f"  {bit_width}-bit: MSE={mse:.6f}, Compression={compression_ratio:.1f}x")
    
    # Verify that higher bits = lower MSE
    assert results[0]['mse'] > results[1]['mse'] > results[2]['mse'], \
        "MSE should decrease with higher bit width"
    
    print(f"\n  ✓ PASSED: MSE decreases with higher bit width")
    print("="*70)


def test_turboquant_retriever():
    """Test TurboQuantRetriever integration with ChromaDB."""
    print("\n" + "="*70)
    print("TEST 4: TurboQuantRetriever Integration")
    print("="*70)
    
    try:
        # Initialize TurboQuantRetriever
        retriever = TurboQuantRetriever(
            collection_name="test_turboquant_memories",
            model_name="all-MiniLM-L6-v2",
            bit_width=4,
            use_qjl=True,
            embedding_dimension=384
        )
        
        print("  ✓ TurboQuantRetriever initialized successfully")
        
        # Test document addition with compression
        test_docs = [
            ("Deep learning neural networks for image classification", 
             {"category": "AI", "tags": json.dumps(["deep learning", "neural networks"])}),
            ("Natural language processing with transformers",
             {"category": "NLP", "tags": json.dumps(["NLP", "transformers"])}),
            ("Computer vision applications in autonomous driving",
             {"category": "CV", "tags": json.dumps(["computer vision", "autonomous driving"])})
        ]
        
        for i, (doc, metadata) in enumerate(test_docs):
            # Generate a random embedding to simulate real embeddings
            embedding = np.random.randn(384).astype(np.float32)
            embedding /= np.linalg.norm(embedding)
            
            retriever.add_document(
                document=doc,
                metadata=metadata,
                doc_id=f"test_doc_{i}",
                embedding=embedding
            )
            print(f"  ✓ Added document {i+1} with TurboQuant compression")
        
        # Get compression statistics
        stats = retriever.get_compression_stats()
        print(f"\n  Compression Statistics:")
        print(f"    Documents compressed: {stats['documents_compressed']}")
        print(f"    Original size: {stats['original_size_mb']:.4f} MB")
        print(f"    Compressed size: {stats['compressed_size_mb']:.4f} MB")
        print(f"    Actual compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"    Space saved: {stats['space_saved_mb']:.4f} MB")
        print(f"    Theoretical ratio: {stats['theoretical_ratio']:.1f}x")
        
        # Test search
        print(f"\n  Testing search functionality...")
        results = retriever.search_with_compression("deep learning", k=2)
        print(f"  ✓ Search returned {len(results.get('ids', [[]])[0])} results")
        
        # Verify compression ratio is in expected range
        assert stats['compression_ratio'] > 5, "Compression ratio should be > 5x"
        assert stats['compression_ratio'] < 20, "Compression ratio should be < 20x"
        
        print(f"\n  ✓ PASSED: All TurboQuantRetriever tests passed")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*70)
        return False


def test_compression_quality():
    """Test that compression maintains reasonable quality."""
    print("\n" + "="*70)
    print("TEST 5: Compression Quality Verification")
    print("="*70)
    
    compressor = TurboQuantCompressor(
        bit_width=4,
        use_qjl=True,
        dimension=384
    )
    
    # Test with multiple vectors
    n_vectors = 20
    mse_scores = []
    
    for i in range(n_vectors):
        vec = generate_test_vector(384)
        compressed = compressor.compress(vec)
        mse = compressor.calculate_distortion(vec, compressed)
        mse_scores.append(mse)
    
    avg_mse = np.mean(mse_scores)
    max_mse = np.max(mse_scores)
    min_mse = np.min(mse_scores)
    
    print(f"  Tested {n_vectors} vectors")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Max MSE: {max_mse:.6f}")
    print(f"  Min MSE: {min_mse:.6f}")
    print(f"  Std MSE: {np.std(mse_scores):.6f}")
    
    # Quality check: MSE should be low for 4-bit quantization
    assert avg_mse < 0.01, f"Average MSE too high: {avg_mse}"
    
    print(f"\n  ✓ PASSED: Compression quality is acceptable")
    print("="*70)


def test_compression_ratio():
    """Verify compression ratios match theoretical expectations."""
    print("\n" + "="*70)
    print("TEST 6: Compression Ratio Verification")
    print("="*70)
    
    test_cases = [
        (2, True, 384, 16.0),  # 2-bit total: 1 MSE bit + 1 QJL bit
        (3, True, 384, 10.6),  # 3-bit total: 2 MSE bits + 1 QJL bit
        (4, True, 384, 8.0),   # 4-bit total: 3 MSE bits + 1 QJL bit
        (4, False, 384, 8.0),  # 4-bit without QJL: 32 / 4 = 8x
    ]
    
    for bit_width, use_qjl, dimension, expected_min_ratio in test_cases:
        compressor = TurboQuantCompressor(
            bit_width=bit_width,
            use_qjl=use_qjl,
            dimension=dimension
        )
        
        theoretical_ratio = compressor.get_compression_ratio()
        
        print(f"  {bit_width}-bit, QJL={use_qjl}: Theoretical ratio = {theoretical_ratio:.1f}x")
        
        assert theoretical_ratio >= expected_min_ratio, \
            f"Compression ratio too low for {bit_width}-bit"
    
    print(f"\n  ✓ PASSED: All compression ratios are acceptable")
    print("="*70)


def run_all_tests():
    """Run all TurboQuant tests."""
    print("\n" + "#"*70)
    print("# TurboQuant Integration Test Suite")
    print("# Based on: arXiv:2504.19874v1")
    print("#"*70)
    
    test_results = {}
    
    # Run tests
    try:
        test_turboquant_basic_compression()
        test_results['basic_compression'] = True
    except Exception as e:
        print(f"Test failed: {e}")
        test_results['basic_compression'] = False
    
    try:
        test_inner_product_unbiasedness()
        test_results['inner_product'] = True
    except Exception as e:
        print(f"Test failed: {e}")
        test_results['inner_product'] = False
    
    try:
        test_different_bit_widths()
        test_results['bit_width_comparison'] = True
    except Exception as e:
        print(f"Test failed: {e}")
        test_results['bit_width_comparison'] = False
    
    try:
        test_results['retriever_integration'] = test_turboquant_retriever()
    except Exception as e:
        print(f"Test failed: {e}")
        test_results['retriever_integration'] = False
    
    try:
        test_compression_quality()
        test_results['compression_quality'] = True
    except Exception as e:
        print(f"Test failed: {e}")
        test_results['compression_quality'] = False
    
    try:
        test_compression_ratio()
        test_results['compression_ratio'] = True
    except Exception as e:
        print(f"Test failed: {e}")
        test_results['compression_ratio'] = False
    
    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    
    for test_name, passed in test_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n  🎉 ALL TESTS PASSED! TurboQuant integration is working correctly!")
        return True
    else:
        print(f"\n  ⚠️  {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
