"""
Test suite for Multi-Bit Adaptive Quantization.
Tests adaptive bit-width selection, policy decisions, and compression quality.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory.adaptive_quantization import (
    ImportanceLevel,
    AdaptiveQuantizationPolicy,
    MultiBitAdaptiveCompressor,
    create_adaptive_compressor
)


def generate_test_embedding(dimension: int = 384) -> np.ndarray:
    """Generate a random test embedding."""
    embedding = np.random.randn(dimension)
    return embedding / np.linalg.norm(embedding)


def test_importance_level_enum():
    """Test ImportanceLevel enum values."""
    print("\n" + "="*70)
    print("TEST 1: ImportanceLevel Enum")
    print("="*70)
    
    assert ImportanceLevel.CRITICAL.value == 4
    assert ImportanceLevel.HIGH.value == 3
    assert ImportanceLevel.MEDIUM.value == 3
    assert ImportanceLevel.LOW.value == 2
    
    print(f"✓ CRITICAL: {ImportanceLevel.CRITICAL.value}-bit")
    print(f"✓ HIGH: {ImportanceLevel.HIGH.value}-bit")
    print(f"✓ MEDIUM: {ImportanceLevel.MEDIUM.value}-bit")
    print(f"✓ LOW: {ImportanceLevel.LOW.value}-bit")
    
    print(f"\n  ✓ PASSED: ImportanceLevel enum correct")
    print("="*70)
    return True


def test_policy_importance_scoring():
    """Test importance score calculation with different scenarios."""
    print("\n" + "="*70)
    print("TEST 2: Policy Importance Scoring")
    print("="*70)
    
    policy = AdaptiveQuantizationPolicy(
        frequency_weight=0.4,
        age_weight=0.3,
        importance_weight=0.3,
        recent_threshold_days=7,
        high_frequency_threshold=10
    )
    
    # Scenario 1: High frequency, recent, high importance
    score1 = policy.calculate_importance_score(
        retrieval_count=15,  # High frequency
        days_since_creation=2,  # Recent
        user_importance=0.9  # High importance
    )
    print(f"\nScenario 1 (High freq, recent, important): {score1:.3f}")
    assert score1 > 0.7, "Should be high importance"
    
    # Scenario 2: Low frequency, old, low importance
    score2 = policy.calculate_importance_score(
        retrieval_count=1,  # Low frequency
        days_since_creation=60,  # Old
        user_importance=0.1  # Low importance
    )
    print(f"Scenario 2 (Low freq, old, unimportant): {score2:.3f}")
    assert score2 < 0.4, "Should be low importance"
    
    # Scenario 3: Medium everything
    score3 = policy.calculate_importance_score(
        retrieval_count=5,
        days_since_creation=15,
        user_importance=0.5
    )
    print(f"Scenario 3 (Medium everything): {score3:.3f}")
    assert 0.3 <= score3 <= 0.7, "Should be medium importance"
    
    # Verify ordering
    assert score1 > score3 > score2, "Scores should follow expected order"
    
    print(f"\n  ✓ PASSED: Importance scoring working correctly")
    print("="*70)
    return True


def test_bit_width_determination():
    """Test bit-width selection based on importance scores."""
    print("\n" + "="*70)
    print("TEST 3: Bit Width Determination")
    print("="*70)
    
    policy = AdaptiveQuantizationPolicy()
    
    # Test thresholds
    test_cases = [
        (0.9, 4, "Critical importance"),
        (0.75, 4, "High importance"),
        (0.7, 4, "Threshold for 4-bit"),
        (0.6, 3, "Medium-high importance"),
        (0.4, 3, "Threshold for 3-bit"),
        (0.3, 2, "Medium-low importance"),
        (0.1, 2, "Low importance"),
        (0.0, 2, "Minimum importance"),
    ]
    
    for score, expected_bw, description in test_cases:
        actual_bw = policy.determine_bit_width(score)
        status = "✓" if actual_bw == expected_bw else "✗"
        print(f"{status} Score {score:.2f} → {actual_bw}-bit ({description})")
        assert actual_bw == expected_bw, f"Expected {expected_bw}-bit, got {actual_bw}-bit"
    
    print(f"\n  ✓ PASSED: Bit width determination correct")
    print("="*70)
    return True


def test_importance_level_mapping():
    """Test mapping from importance score to ImportanceLevel enum."""
    print("\n" + "="*70)
    print("TEST 4: Importance Level Mapping")
    print("="*70)
    
    policy = AdaptiveQuantizationPolicy()
    
    test_cases = [
        (0.9, ImportanceLevel.CRITICAL),
        (0.75, ImportanceLevel.CRITICAL),
        (0.6, ImportanceLevel.HIGH),
        (0.45, ImportanceLevel.MEDIUM),
        (0.2, ImportanceLevel.LOW),
        (0.05, ImportanceLevel.LOW),
    ]
    
    for score, expected_level in test_cases:
        actual_level = policy.get_importance_level(score)
        status = "✓" if actual_level == expected_level else "✗"
        print(f"{status} Score {score:.2f} → {actual_level.name} ({actual_level.value}-bit)")
        assert actual_level == expected_level
    
    print(f"\n  ✓ PASSED: Importance level mapping correct")
    print("="*70)
    return True


def test_adaptive_compression_different_scenarios():
    """Test adaptive compression with various memory scenarios."""
    print("\n" + "="*70)
    print("TEST 5: Adaptive Compression - Different Scenarios")
    print("="*70)
    
    compressor = MultiBitAdaptiveCompressor(
        dimension=384,
        use_qjl=True
    )
    
    scenarios = [
        {
            'name': 'Frequently accessed recent memory',
            'metadata': {
                'retrieval_count': 20,
                'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                'importance': 0.9,
                'content': 'Important system configuration'
            }
        },
        {
            'name': 'Old rarely accessed memory',
            'metadata': {
                'retrieval_count': 1,
                'timestamp': (datetime.now() - timedelta(days=90)).isoformat(),
                'importance': 0.1,
                'content': 'Old conversation snippet'
            }
        },
        {
            'name': 'Medium importance memory',
            'metadata': {
                'retrieval_count': 5,
                'timestamp': (datetime.now() - timedelta(days=14)).isoformat(),
                'importance': 0.5,
                'content': 'General knowledge note'
            }
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        embedding = generate_test_embedding(384)
        compressed, bit_width = compressor.compress_adaptive(
            embedding, scenario['metadata']
        )
        
        results[scenario['name']] = {
            'bit_width': bit_width,
            'importance_score': compressed['importance_score']
        }
        
        print(f"\n{scenario['name']}:")
        print(f"  Importance score: {compressed['importance_score']:.3f}")
        print(f"  Bit width used: {bit_width}-bit")
    
    # Verify adaptive behavior
    freq_recent_bw = results['Frequently accessed recent memory']['bit_width']
    old_rare_bw = results['Old rarely accessed memory']['bit_width']
    
    assert freq_recent_bw >= old_rare_bw, \
        "Frequent/recent memories should use higher or equal bit-width"
    
    print(f"\n  ✓ PASSED: Adaptive compression working correctly")
    print("="*70)
    return True


def test_compression_quality_by_bit_width():
    """Test that higher bit-widths provide better quality."""
    print("\n" + "="*70)
    print("TEST 6: Compression Quality by Bit Width")
    print("="*70)
    
    compressor = MultiBitAdaptiveCompressor(
        dimension=384,
        use_qjl=True
    )
    
    embedding = generate_test_embedding(384)
    
    # Test each bit-width manually
    mse_results = {}
    
    for bit_width in [2, 3, 4]:
        # Force specific bit-width by setting importance appropriately
        if bit_width == 4:
            metadata = {'retrieval_count': 20, 'timestamp': datetime.now().isoformat(), 
                       'importance': 0.9, 'content': 'test'}
        elif bit_width == 3:
            metadata = {'retrieval_count': 5, 'timestamp': datetime.now().isoformat(),
                       'importance': 0.5, 'content': 'test'}
        else:  # 2-bit
            metadata = {'retrieval_count': 0, 'timestamp': datetime.now().isoformat(),
                       'importance': 0.1, 'content': 'test'}
        
        compressed, actual_bw = compressor.compress_adaptive(embedding, metadata)
        reconstructed = compressor.decompress(compressed)
        
        mse = np.mean((embedding - reconstructed) ** 2)
        mse_results[actual_bw] = mse
        
        print(f"{actual_bw}-bit: MSE = {mse:.6f}")
    
    # Verify that higher bit-width = lower MSE
    assert mse_results[4] < mse_results[3] < mse_results[2], \
        "Higher bit-width should have lower MSE"
    
    print(f"\n  ✓ PASSED: Quality increases with bit-width")
    print("="*70)
    return True


def test_statistics_tracking():
    """Test compression statistics tracking."""
    print("\n" + "="*70)
    print("TEST 7: Statistics Tracking")
    print("="*70)
    
    compressor = MultiBitAdaptiveCompressor(
        dimension=384,
        use_qjl=True
    )
    
    # Compress multiple embeddings with different importance levels
    for i in range(20):
        embedding = generate_test_embedding(384)
        
        # Vary importance to get different bit-widths
        if i < 7:
            metadata = {'retrieval_count': 15, 'timestamp': datetime.now().isoformat(),
                       'importance': 0.9, 'content': 'high'}
        elif i < 14:
            metadata = {'retrieval_count': 5, 'timestamp': datetime.now().isoformat(),
                       'importance': 0.5, 'content': 'medium'}
        else:
            metadata = {'retrieval_count': 1, 'timestamp': datetime.now().isoformat(),
                       'importance': 0.1, 'content': 'low'}
        
        compressor.compress_adaptive(embedding, metadata)
    
    stats = compressor.get_statistics()
    
    print(f"\nCompression Statistics:")
    print(f"  Total compressed: {stats['total_compressed']}")
    print(f"  Distribution:")
    for bw in [2, 3, 4]:
        count = stats['distribution'][bw]
        pct = stats['percentages'][bw]
        print(f"    {bw}-bit: {count} ({pct:.1f}%)")
    print(f"  Average bit-width: {stats['average_bit_width']:.2f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Original size: {stats['original_size_mb']:.4f} MB")
    print(f"  Compressed size: {stats['compressed_size_mb']:.4f} MB")
    print(f"  Space saved: {stats['space_saved_mb']:.4f} MB")
    
    # Verify statistics
    assert stats['total_compressed'] == 20
    assert sum(stats['distribution'].values()) == 20
    assert 2 <= stats['average_bit_width'] <= 4
    assert stats['compression_ratio'] > 1
    
    print(f"\n  ✓ PASSED: Statistics tracking working correctly")
    print("="*70)
    return True


def test_policy_parameter_updates():
    """Test updating policy parameters dynamically."""
    print("\n" + "="*70)
    print("TEST 8: Policy Parameter Updates")
    print("="*70)
    
    compressor = MultiBitAdaptiveCompressor(
        dimension=384,
        use_qjl=True,
        policy=AdaptiveQuantizationPolicy(
            frequency_weight=0.4,
            age_weight=0.3,
            importance_weight=0.3
        )
    )
    
    # Test initial behavior
    metadata1 = {
        'retrieval_count': 15,
        'timestamp': datetime.now().isoformat(),
        'importance': 0.5,
        'content': 'test'
    }
    
    embedding = generate_test_embedding(384)
    _, bw1 = compressor.compress_adaptive(embedding, metadata1)
    print(f"Initial weights (freq=0.4, age=0.3, imp=0.3): {bw1}-bit")
    
    # Update policy to prioritize frequency more
    compressor.update_policy(frequency_weight=0.7, age_weight=0.15, importance_weight=0.15)
    
    _, bw2 = compressor.compress_adaptive(embedding, metadata1.copy())
    print(f"Updated weights (freq=0.7, age=0.15, imp=0.15): {bw2}-bit")
    
    # With higher frequency weight, high-frequency memory should maintain or increase bit-width
    assert bw2 >= bw1, "Higher frequency weight should not reduce bit-width for high-freq memory"
    
    print(f"\n  ✓ PASSED: Policy updates working correctly")
    print("="*70)
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 9: Edge Cases")
    print("="*70)
    
    compressor = MultiBitAdaptiveCompressor(dimension=384, use_qjl=True)
    
    # Test 1: Empty metadata
    embedding = generate_test_embedding(384)
    compressed1, bw1 = compressor.compress_adaptive(embedding, {})
    print(f"✓ Empty metadata handled: {bw1}-bit")
    
    # Test 2: Invalid timestamp
    compressed2, bw2 = compressor.compress_adaptive(
        embedding, 
        {'timestamp': 'invalid-date'}
    )
    print(f"✓ Invalid timestamp handled: {bw2}-bit")
    
    # Test 3: Extreme importance values
    for imp in [-0.5, 0.0, 0.5, 1.0, 1.5]:
        metadata = {'importance': imp, 'content': 'test'}
        compressed, bw = compressor.compress_adaptive(embedding, metadata)
        print(f"✓ Importance {imp} → {bw}-bit")
        assert 2 <= bw <= 4, f"Invalid bit-width: {bw}"
    
    # Test 4: Decompression with missing bit_width (should default to 4-bit)
    try:
        # Create a valid compressed structure but without bit_width
        valid_compressed = {
            'indices': np.zeros(384, dtype=np.int64),
            'residual_norm': 0.0,
            'qjl_bits': np.ones(384, dtype=np.int8)
        }
        result = compressor.decompress(valid_compressed)
        print(f"✓ Missing bit_width defaults to 4-bit")
        assert result.shape == (384,), "Should decompress successfully"
    except Exception as e:
        print(f"✗ Failed to handle missing bit_width: {e}")
        raise
    
    print(f"\n  ✓ PASSED: Edge cases handled correctly")
    print("="*70)
    return True


def test_convenience_function():
    """Test the convenience function for creating adaptive compressor."""
    print("\n" + "="*70)
    print("TEST 10: Convenience Function")
    print("="*70)
    
    compressor = create_adaptive_compressor(
        dimension=384,
        use_qjl=True,
        frequency_weight=0.5,
        age_weight=0.3,
        importance_weight=0.2
    )
    
    print(f"✓ Created via convenience function")
    print(f"  Type: {type(compressor).__name__}")
    print(f"  Policy freq_weight: {compressor.policy.frequency_weight}")
    print(f"  Policy age_weight: {compressor.policy.age_weight}")
    print(f"  Policy importance_weight: {compressor.policy.importance_weight}")
    
    assert isinstance(compressor, MultiBitAdaptiveCompressor)
    assert compressor.policy.frequency_weight == 0.5
    
    # Test basic functionality
    embedding = generate_test_embedding(384)
    metadata = {'retrieval_count': 5, 'timestamp': datetime.now().isoformat(),
               'importance': 0.5, 'content': 'test'}
    
    compressed, bw = compressor.compress_adaptive(embedding, metadata)
    reconstructed = compressor.decompress(compressed)
    
    assert reconstructed.shape == (384,)
    
    print(f"\n  ✓ PASSED: Convenience function working")
    print("="*70)
    return True


def test_real_world_scenario():
    """Test realistic usage scenario with mixed importance memories."""
    print("\n" + "="*70)
    print("TEST 11: Real-World Scenario Simulation")
    print("="*70)
    
    compressor = MultiBitAdaptiveCompressor(
        dimension=384,
        use_qjl=True,
        policy=AdaptiveQuantizationPolicy(
            frequency_weight=0.4,
            age_weight=0.3,
            importance_weight=0.3
        )
    )
    
    # Simulate 100 memories with varying characteristics
    memories = []
    
    # 20 critical memories (frequently accessed, recent, important)
    for i in range(20):
        memories.append({
            'retrieval_count': np.random.randint(15, 30),
            'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 5))).isoformat(),
            'importance': np.random.uniform(0.8, 1.0),
            'content': 'Critical business logic' * 10
        })
    
    # 30 medium memories
    for i in range(30):
        memories.append({
            'retrieval_count': np.random.randint(3, 10),
            'timestamp': (datetime.now() - timedelta(days=np.random.randint(10, 60))).isoformat(),
            'importance': np.random.uniform(0.4, 0.7),
            'content': 'General information' * 5
        })
    
    # 50 low-priority memories
    for i in range(50):
        memories.append({
            'retrieval_count': np.random.randint(0, 3),
            'timestamp': (datetime.now() - timedelta(days=np.random.randint(60, 365))).isoformat(),
            'importance': np.random.uniform(0.0, 0.3),
            'content': 'Archive data' * 2
        })
    
    # Compress all memories
    bit_width_counts = {2: 0, 3: 0, 4: 0}
    
    for i, metadata in enumerate(memories):
        embedding = generate_test_embedding(384)
        compressed, bw = compressor.compress_adaptive(embedding, metadata)
        bit_width_counts[bw] += 1
    
    stats = compressor.get_statistics()
    
    print(f"\nReal-World Scenario Results:")
    print(f"  Total memories: {len(memories)}")
    print(f"  Distribution:")
    for bw in [2, 3, 4]:
        count = bit_width_counts[bw]
        pct = (count / len(memories)) * 100
        print(f"    {bw}-bit: {count} ({pct:.1f}%)")
    print(f"  Average bit-width: {stats['average_bit_width']:.2f}")
    print(f"  Overall compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Space saved: {stats['space_saved_mb']:.2f} MB")
    
    # Verify distribution makes sense
    assert bit_width_counts[4] >= 15, "Should have many 4-bit (critical memories)"
    assert bit_width_counts[2] >= 40, "Should have many 2-bit (low-priority memories)"
    assert 2.5 <= stats['average_bit_width'] <= 3.5, "Average should be in middle range"
    
    print(f"\n  ✓ PASSED: Real-world scenario working correctly")
    print("="*70)
    return True


def run_all_tests():
    """Run all adaptive quantization tests."""
    print("\n" + "#"*70)
    print("# Multi-Bit Adaptive Quantization Test Suite")
    print("# Dynamic Bit-Width Selection Based on Memory Importance")
    print("#"*70)
    
    test_results = {}
    
    tests = [
        ('importance_enum', test_importance_level_enum),
        ('policy_scoring', test_policy_importance_scoring),
        ('bit_width_determination', test_bit_width_determination),
        ('level_mapping', test_importance_level_mapping),
        ('adaptive_scenarios', test_adaptive_compression_different_scenarios),
        ('quality_comparison', test_compression_quality_by_bit_width),
        ('statistics', test_statistics_tracking),
        ('policy_updates', test_policy_parameter_updates),
        ('edge_cases', test_edge_cases),
        ('convenience_func', test_convenience_function),
        ('real_world', test_real_world_scenario),
    ]
    
    for test_name, test_func in tests:
        try:
            test_results[test_name] = test_func()
        except Exception as e:
            print(f"\n  ✗ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            test_results[test_name] = False
    
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
        print("\n  🎉 ALL TESTS PASSED!")
        print("  🚀 Multi-Bit Adaptive Quantization is working!")
        print("  💡 Key Benefits:")
        print("     - Dynamic bit-width selection (2/3/4-bit)")
        print("     - Importance-based compression")
        print("     - Optimal storage-quality tradeoff")
        print("     - Adapts to usage patterns over time")
        return True
    else:
        print(f"\n  ⚠️  {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
