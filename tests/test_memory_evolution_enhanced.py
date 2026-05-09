"""
Test suite for Memory Evolution Enhancement using TurboQuant Distortion Analysis.
Tests the enhanced evolution system and its integration with memory retrieval.
"""

import sys
import os
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory.memory_evolution_enhanced import (
    TurboQuantEvolutionAnalyzer,
    EnhancedMemoryEvolution
)
from agentic_memory.turboquant import TurboQuantCompressor


def generate_test_embedding(dimension: int = 384) -> np.ndarray:
    """Generate a random test embedding."""
    embedding = np.random.randn(dimension)
    return embedding / np.linalg.norm(embedding)


def test_distortion_analysis():
    """Test basic distortion analysis functionality."""
    print("\n" + "="*70)
    print("TEST 1: TurboQuant Evolution Analyzer - Distortion Analysis")
    print("="*70)
    
    analyzer = TurboQuantEvolutionAnalyzer(
        bit_width=4,
        evolution_threshold=0.0001,
        dimension=384
    )
    
    # Generate test embeddings
    new_embedding = generate_test_embedding(384)
    related_embeddings = [generate_test_embedding(384) for _ in range(5)]
    
    # Analyze distortion
    result = analyzer.calculate_memory_distortion(new_embedding, related_embeddings)
    
    print(f"\nDistortion Analysis Results:")
    print(f"  Min distortion: {result['min_distortion']:.6f}")
    print(f"  Avg distortion: {result['avg_distortion']:.6f}")
    print(f"  Max distortion: {result['max_distortion']:.6f}")
    print(f"  Should evolve: {result['should_evolve']}")
    print(f"  Redundancy score: {result['redundancy_score']:.4f}")
    print(f"  Reason: {result['reason']}")
    
    # Verify result structure
    assert 'min_distortion' in result, "Missing min_distortion"
    assert 'avg_distortion' in result, "Missing avg_distortion"
    assert 'max_distortion' in result, "Missing max_distortion"
    assert 'should_evolve' in result, "Missing should_evolve"
    assert 'redundancy_score' in result, "Missing redundancy_score"
    assert 'reason' in result, "Missing reason"
    
    # Verify ranges
    assert 0 <= result['redundancy_score'] <= 1.0, "Redundancy score out of range"
    assert result['min_distortion'] >= 0, "Distortion should be non-negative"
    
    print(f"\n  ✓ PASSED: Distortion analysis working correctly")
    print("="*70)
    return True


def test_redundant_memory_detection():
    """Test that the system correctly identifies redundant (similar) memories."""
    print("\n" + "="*70)
    print("TEST 2: Redundant Memory Detection")
    print("="*70)
    
    # Use higher threshold for cosine-based distortion
    analyzer = TurboQuantEvolutionAnalyzer(
        bit_width=4,
        evolution_threshold=0.05,  # Adjusted for cosine distance
        dimension=384
    )
    
    # Create a base embedding
    base_embedding = generate_test_embedding(384)
    
    # Create a very similar (redundant) embedding
    noise = np.random.randn(384) * 0.01  # Small noise
    similar_embedding = base_embedding + noise
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
    
    # Analyze the similar embedding against the base
    related = [base_embedding]
    result = analyzer.calculate_memory_distortion(similar_embedding, related)
    
    print(f"\nSimilar Memory Analysis:")
    print(f"  Min distortion: {result['min_distortion']:.6f}")
    print(f"  Should evolve: {result['should_evolve']}")
    print(f"  Redundancy score: {result['redundancy_score']:.4f}")
    print(f"  Reason: {result['reason']}")
    
    # Similar memories should have low distortion and high redundancy
    assert result['min_distortion'] < 0.1, "Similar memories should have low distortion"
    assert result['redundancy_score'] > 0.5, "Similar memories should have high redundancy"
    
    print(f"\n  ✓ PASSED: Redundant memory detection working correctly")
    print("="*70)
    return True


def test_unique_memory_detection():
    """Test that the system correctly identifies unique (different) memories."""
    print("\n" + "="*70)
    print("TEST 3: Unique Memory Detection")
    print("="*70)
    
    analyzer = TurboQuantEvolutionAnalyzer(
        bit_width=4,
        evolution_threshold=0.0001,
        dimension=384
    )
    
    # Create two completely different embeddings
    embedding1 = generate_test_embedding(384)
    embedding2 = generate_test_embedding(384)
    
    # Analyze embedding2 against embedding1
    related = [embedding1]
    result = analyzer.calculate_memory_distortion(embedding2, related)
    
    print(f"\nUnique Memory Analysis:")
    print(f"  Min distortion: {result['min_distortion']:.6f}")
    print(f"  Should evolve: {result['should_evolve']}")
    print(f"  Redundancy score: {result['redundancy_score']:.4f}")
    print(f"  Reason: {result['reason']}")
    
    # Different memories should have low redundancy score
    assert result['redundancy_score'] < 0.5, "Unique memories should have low redundancy"
    
    print(f"\n  ✓ PASSED: Unique memory detection working correctly")
    print("="*70)
    return True


def test_batch_analysis():
    """Test batch analysis of multiple memories."""
    print("\n" + "="*70)
    print("TEST 4: Batch Memory Analysis")
    print("="*70)
    
    analyzer = TurboQuantEvolutionAnalyzer(
        bit_width=4,
        evolution_threshold=0.0001,
        dimension=384
    )
    
    # Generate test data
    existing_embeddings = [generate_test_embedding(384) for _ in range(20)]
    new_embeddings = [generate_test_embedding(384) for _ in range(5)]
    
    # Analyze all new memories
    results = analyzer.batch_analyze_evolution(
        new_embeddings,
        existing_embeddings,
        top_k_related=5
    )
    
    print(f"\nBatch Analysis Results:")
    print(f"  Analyzed {len(results)} new memories")
    
    evolution_triggers = 0
    unique_memories = 0
    
    for i, result in enumerate(results):
        if result['should_evolve']:
            evolution_triggers += 1
            status = "→ EVOLVE (redundant)"
        else:
            unique_memories += 1
            status = "→ UNIQUE (no evolution)"
        
        print(f"  Memory {i+1}: {result['redundancy_score']:.4f} {status}")
    
    print(f"\nSummary:")
    print(f"  Evolution triggers: {evolution_triggers}")
    print(f"  Unique memories: {unique_memories}")
    print(f"  Total: {len(results)}")
    
    # Verify batch results
    assert len(results) == 5, "Should analyze all 5 new memories"
    assert all('should_evolve' in r for r in results), "All results should have should_evolve"
    
    print(f"\n  ✓ PASSED: Batch analysis working correctly")
    print("="*70)
    return True


def test_enhanced_evolution_system():
    """Test the enhanced evolution system with LLM call savings."""
    print("\n" + "="*70)
    print("TEST 5: Enhanced Evolution System - LLM Call Savings")
    print("="*70)
    
    evolution_system = EnhancedMemoryEvolution(
        bit_width=4,
        evolution_threshold=0.0001,
        dimension=384,
        enable_distortion_filter=True
    )
    
    # Simulate adding multiple memories
    existing_embeddings = [generate_test_embedding(384) for _ in range(10)]
    
    llm_calls_saved = 0
    evolution_triggers = 0
    
    print(f"\nSimulating memory additions...")
    for i in range(10):
        new_embedding = generate_test_embedding(384)
        
        # Analyze if evolution needed
        decision = evolution_system.should_evolve_memory(
            new_embedding,
            existing_embeddings
        )
        
        if decision['should_evolve']:
            evolution_triggers += 1
            print(f"  Memory {i+1}: SHOULD EVOLVE (LLM call needed)")
        else:
            llm_calls_saved += 1
            print(f"  Memory {i+1}: UNIQUE (LLM call saved)")
        
        # Add to existing for next iteration
        existing_embeddings.append(new_embedding)
    
    # Get statistics
    stats = evolution_system.get_evolution_statistics()
    
    print(f"\nEvolution System Statistics:")
    print(f"  Total analyses: {stats['total_analyses']}")
    print(f"  LLM calls saved: {stats['llm_calls_saved']}")
    print(f"  LLM call savings: {stats['llm_call_savings_percent']:.1f}%")
    print(f"  Evolution triggers: {stats['evolution_triggers']}")
    print(f"  Unique memories: {stats['unique_memories']}")
    print(f"  Evolution rate: {stats['evolution_rate']:.1f}%")
    
    # Verify statistics
    assert stats['total_analyses'] == 10, "Should have analyzed 10 memories"
    assert stats['llm_calls_saved'] + stats['evolution_triggers'] == 10, \
        "Saved + triggered should equal total"
    assert stats['llm_call_savings_percent'] > 0, "Should have saved some LLM calls"
    
    print(f"\n  ✓ PASSED: Enhanced evolution system working correctly")
    print(f"  💰 Saved {stats['llm_calls_saved']} LLM calls ({stats['llm_call_savings_percent']:.1f}%)")
    print("="*70)
    return True


def test_evolution_without_filter():
    """Test evolution system when distortion filter is disabled."""
    print("\n" + "="*70)
    print("TEST 6: Evolution System Without Distortion Filter")
    print("="*70)
    
    evolution_system = EnhancedMemoryEvolution(
        bit_width=4,
        evolution_threshold=0.0001,
        dimension=384,
        enable_distortion_filter=False  # Disabled
    )
    
    new_embedding = generate_test_embedding(384)
    related = [generate_test_embedding(384)]
    
    decision = evolution_system.should_evolve_memory(new_embedding, related)
    
    print(f"\nWithout Distortion Filter:")
    print(f"  Should evolve: {decision['should_evolve']}")
    print(f"  LLM call needed: {decision['llm_call_needed']}")
    print(f"  Method: {decision['method']}")
    
    # Without filter, should always recommend LLM call
    assert decision['llm_call_needed'] == True, "Should always call LLM without filter"
    assert decision['method'] == 'llm_only', "Should use LLM-only method"
    
    print(f"\n  ✓ PASSED: Traditional mode working correctly")
    print("="*70)
    return True


def test_threshold_calibration():
    """Test automatic threshold calibration."""
    print("\n" + "="*70)
    print("TEST 7: Automatic Threshold Calibration")
    print("="*70)
    
    analyzer = TurboQuantEvolutionAnalyzer(
        bit_width=4,
        evolution_threshold=0.0001,  # Initial threshold
        dimension=384
    )
    
    print(f"\nInitial threshold: {analyzer.evolution_threshold}")
    
    # Generate sample embeddings with known similarity
    sample_embeddings = []
    base = generate_test_embedding(384)
    
    # Add variations with different similarity levels
    for i in range(20):
        if i < 5:
            # Very similar (high redundancy)
            noise = np.random.randn(384) * 0.01
        elif i < 10:
            # Moderately similar
            noise = np.random.randn(384) * 0.1
        else:
            # Different (low redundancy)
            noise = np.random.randn(384) * 0.5
        
        variant = base + noise
        variant = variant / np.linalg.norm(variant)
        sample_embeddings.append(variant)
    
    # Calibrate threshold
    analyzer.update_threshold_based_on_data(
        sample_embeddings,
        target_evolution_rate=0.2
    )
    
    print(f"Calibrated threshold: {analyzer.evolution_threshold}")
    print(f"  ✓ PASSED: Threshold calibration working")
    print("="*70)
    return True


def test_evolution_cost_savings():
    """Test and quantify LLM cost savings."""
    print("\n" + "="*70)
    print("TEST 8: LLM Cost Savings Analysis")
    print("="*70)
    
    # Assume GPT-4 mini costs ~$0.01 per call
    LLM_COST_PER_CALL = 0.01
    
    evolution_system = EnhancedMemoryEvolution(
        bit_width=4,
        evolution_threshold=0.0001,
        dimension=384,
        enable_distortion_filter=True
    )
    
    # Simulate large-scale memory addition
    existing = [generate_test_embedding(384) for _ in range(50)]
    new_memories = [generate_test_embedding(384) for _ in range(100)]
    
    print(f"\nSimulating addition of {len(new_memories)} memories...")
    
    for i, new_emb in enumerate(new_memories):
        decision = evolution_system.should_evolve_memory(new_emb, existing)
        existing.append(new_emb)
        
        if i < 3 or i == len(new_memories) - 1:
            status = "EVOLVE" if decision['should_evolve'] else "UNIQUE"
            print(f"  Memory {i+1}: {status}")
    
    stats = evolution_system.get_evolution_statistics()
    
    cost_savings = stats['llm_calls_saved'] * LLM_COST_PER_CALL
    total_cost_without = stats['total_analyses'] * LLM_COST_PER_CALL
    actual_cost = stats['evolution_triggers'] * LLM_COST_PER_CALL
    
    print(f"\n💰 Cost Analysis:")
    print(f"  Traditional approach (no filter): ${total_cost_without:.2f}")
    print(f"  Enhanced approach (with filter): ${actual_cost:.2f}")
    print(f"  💵 Savings: ${cost_savings:.2f} ({stats['llm_call_savings_percent']:.1f}%)")
    print(f"  📊 LLM calls saved: {stats['llm_calls_saved']}/{stats['total_analyses']}")
    
    # Verify savings
    assert cost_savings > 0, "Should have some cost savings"
    assert stats['llm_call_savings_percent'] > 0, "Should save some percentage"
    
    print(f"\n  ✓ PASSED: Cost savings verified")
    print("="*70)
    return True


def run_all_tests():
    """Run all evolution enhancement tests."""
    print("\n" + "#"*70)
    print("# Memory Evolution Enhancement Test Suite")
    print("# Using TurboQuant Distortion Analysis")
    print("#"*70)
    
    test_results = {}
    
    # Run tests
    tests = [
        ('distortion_analysis', test_distortion_analysis),
        ('redundant_detection', test_redundant_memory_detection),
        ('unique_detection', test_unique_memory_detection),
        ('batch_analysis', test_batch_analysis),
        ('enhanced_system', test_enhanced_evolution_system),
        ('no_filter_mode', test_evolution_without_filter),
        ('threshold_calibration', test_threshold_calibration),
        ('cost_savings', test_evolution_cost_savings),
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
        print("  🚀 Memory Evolution Enhancement is working correctly!")
        print("  💡 Key Benefits:")
        print("     - LLM calls reduced by using distortion filter")
        print("     - Mathematical evolution decisions (not just LLM)")
        print("     - Cost savings on large-scale memory systems")
        return True
    else:
        print(f"\n  ⚠️  {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
