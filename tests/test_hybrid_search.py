"""
Test suite for Hybrid Search Optimization.
Tests keyword extraction, BM25 scoring, RRF fusion, and hybrid search integration.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory.hybrid_search import (
    KeywordExtractor,
    BM25Scorer,
    ReciprocalRankFusion,
    HybridSearchOptimizer
)


def test_keyword_extraction():
    """Test keyword extraction from text."""
    print("\n" + "="*70)
    print("TEST 1: Keyword Extraction")
    print("="*70)
    
    extractor = KeywordExtractor()
    
    # Test 1: Simple sentence
    text1 = "Machine learning algorithms are transforming artificial intelligence research"
    keywords1 = extractor.extract_keywords(text1, max_keywords=5)
    
    print(f"\nText: '{text1}'")
    print(f"Keywords: {keywords1}")
    
    assert len(keywords1) > 0, "Should extract at least one keyword"
    assert 'machine' in keywords1 or 'learning' in keywords1, "Should extract important words"
    
    # Test 2: Technical content
    text2 = "Deep neural networks with convolutional layers process image data efficiently"
    keywords2 = extractor.extract_keywords(text2, max_keywords=5)
    
    print(f"\nText: '{text2}'")
    print(f"Keywords: {keywords2}")
    
    assert len(keywords2) > 0, "Should extract keywords from technical text"
    
    # Test 3: Empty text
    keywords3 = extractor.extract_keywords("", max_keywords=5)
    assert len(keywords3) == 0, "Should return empty list for empty text"
    
    # Test 4: Stop words filtering
    text4 = "The cat is on the mat and the dog is in the house"
    keywords4 = extractor.extract_keywords(text4, max_keywords=5)
    
    print(f"\nText: '{text4}'")
    print(f"Keywords: {keywords4}")
    
    # Should filter out stop words like 'the', 'is', 'on', 'and', 'in'
    assert 'the' not in keywords4, "Should filter out stop word 'the'"
    assert 'is' not in keywords4, "Should filter out stop word 'is'"
    
    print(f"\n  ✓ PASSED: Keyword extraction working correctly")
    print("="*70)
    return True


def test_bm25_scoring():
    """Test BM25 scoring algorithm."""
    print("\n" + "="*70)
    print("TEST 2: BM25 Scoring")
    print("="*70)
    
    scorer = BM25Scorer(k1=1.5, b=0.75)
    
    # Create test documents
    documents = {
        'doc1': "Python programming language for data science",
        'doc2': "Java programming for enterprise applications",
        'doc3': "Machine learning with Python and TensorFlow",
        'doc4': "Web development using JavaScript and React"
    }
    
    # Index documents
    scorer.index_documents(documents)
    
    print(f"\nIndexed {len(documents)} documents")
    
    # Test query scoring
    query = "Python programming"
    
    scores = {}
    for doc_id, doc_text in documents.items():
        score = scorer.score_query(query, doc_id, doc_text)
        scores[doc_id] = score
        print(f"  {doc_id}: {score:.4f}")
    
    # doc1 and doc3 should have higher scores (contain "Python")
    assert scores['doc1'] > 0, "doc1 should match query"
    assert scores['doc3'] > 0, "doc3 should match query"
    
    # doc1 should score higher than doc2 (both have "programming" but only doc1 has "Python")
    assert scores['doc1'] > scores['doc2'], "doc1 should score higher than doc2"
    
    print(f"\n  ✓ PASSED: BM25 scoring working correctly")
    print("="*70)
    return True


def test_bm25_ranking():
    """Test that BM25 produces correct rankings."""
    print("\n" + "="*70)
    print("TEST 3: BM25 Ranking Quality")
    print("="*70)
    
    scorer = BM25Scorer()
    
    # Create documents with varying relevance
    documents = {
        'highly_relevant': "Artificial intelligence and machine learning algorithms",
        'moderately_relevant': "AI systems use intelligent algorithms",
        'slightly_relevant': "Computer science involves algorithms",
        'not_relevant': "Cooking recipes for Italian pasta"
    }
    
    scorer.index_documents(documents)
    
    # Query about AI/ML
    query = "artificial intelligence machine learning"
    
    scores = scorer.score_all_documents(query, documents)
    
    print(f"\nQuery: '{query}'")
    print("\nScores:")
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_id}: {score:.4f}")
    
    # Verify ranking order
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    assert ranked_docs[0][0] == 'highly_relevant', "Most relevant doc should rank first"
    
    # Check that not_relevant has the lowest score (or is not in results if score=0)
    if 'not_relevant' in scores:
        assert scores['not_relevant'] < scores['slightly_relevant'], \
            "not_relevant should score lower than slightly_relevant"
    
    print(f"\n  ✓ PASSED: BM25 ranking quality verified")
    print("="*70)
    return True


def test_reciprocal_rank_fusion():
    """Test RRF combination of multiple rankings."""
    print("\n" + "="*70)
    print("TEST 4: Reciprocal Rank Fusion")
    print("="*70)
    
    rrf = ReciprocalRankFusion(k=60.0)
    
    # Create two different rankings
    ranking1 = [
        ('doc_a', 0.9),
        ('doc_b', 0.8),
        ('doc_c', 0.7),
        ('doc_d', 0.6)
    ]
    
    ranking2 = [
        ('doc_b', 0.95),
        ('doc_c', 0.85),
        ('doc_e', 0.75),
        ('doc_f', 0.65)
    ]
    
    # Fuse rankings
    fused = rrf.fuse_rankings([ranking1, ranking2])
    
    print(f"\nRanking 1: {[doc for doc, _ in ranking1]}")
    print(f"Ranking 2: {[doc for doc, _ in ranking2]}")
    print(f"\nFused ranking:")
    for rank, (doc_id, score) in enumerate(fused, start=1):
        print(f"  {rank}. {doc_id} (RRF score: {score:.4f})")
    
    # doc_b appears in both rankings, should rank high
    assert fused[0][0] == 'doc_b' or fused[1][0] == 'doc_b', \
        "doc_b (in both lists) should rank highly"
    
    # doc_c also appears in both, should be in top results
    top_3_docs = [doc for doc, _ in fused[:3]]
    assert 'doc_c' in top_3_docs, "doc_c should be in top 3"
    
    print(f"\n  ✓ PASSED: RRF fusion working correctly")
    print("="*70)
    return True


def test_hybrid_search_integration():
    """Test complete hybrid search system."""
    print("\n" + "="*70)
    print("TEST 5: Hybrid Search Integration")
    print("="*70)
    
    optimizer = HybridSearchOptimizer(
        vector_weight=0.6,
        keyword_weight=0.4
    )
    
    # Create test memories
    memories = {
        'mem1': "Deep learning neural networks for image classification",
        'mem2': "Natural language processing with transformers and attention",
        'mem3': "Computer vision applications in autonomous vehicles",
        'mem4': "Reinforcement learning for game playing agents",
        'mem5': "Database management systems and SQL queries"
    }
    
    # Index memories
    optimizer.index_memories(memories)
    print(f"✓ Indexed {len(memories)} memories")
    
    # Simulate vector search results (doc_id, distance)
    # Lower distance = more similar
    vector_results = [
        ('mem1', 0.2),   # Most similar semantically
        ('mem2', 0.3),
        ('mem3', 0.4),
        ('mem4', 0.5),
        ('mem5', 0.8)    # Least similar
    ]
    
    # Perform hybrid search
    query = "deep learning neural networks"
    hybrid_results = optimizer.hybrid_search(query, vector_results, k=3)
    
    print(f"\nQuery: '{query}'")
    print(f"\nHybrid search results:")
    for rank, (doc_id, score) in enumerate(hybrid_results, start=1):
        print(f"  {rank}. {doc_id} (score: {score:.4f})")
        print(f"     Content: {memories[doc_id][:60]}...")
    
    # mem1 should rank highly (matches both semantically and lexically)
    top_doc = hybrid_results[0][0]
    assert top_doc == 'mem1', f"mem1 should rank first, got {top_doc}"
    
    print(f"\n  ✓ PASSED: Hybrid search integration working correctly")
    print("="*70)
    return True


def test_hybrid_vs_pure_vector():
    """Compare hybrid search with pure vector search."""
    print("\n" + "="*70)
    print("TEST 6: Hybrid vs Pure Vector Search Comparison")
    print("="*70)
    
    optimizer = HybridSearchOptimizer(
        vector_weight=0.6,
        keyword_weight=0.4
    )
    
    # Create memories where keyword matching matters
    memories = {
        'mem1': "Python code examples for beginners",
        'mem2': "Advanced Python programming techniques",
        'mem3': "JavaScript tutorial for web developers",
        'mem4': "Python data analysis with pandas library",
        'mem5': "C++ performance optimization guide"
    }
    
    optimizer.index_memories(memories)
    
    # Query with specific technical term
    query = "Python pandas"
    
    # Simulate vector search (might miss exact term matches)
    # In reality, embeddings might not capture "pandas" well
    vector_results = [
        ('mem3', 0.3),  # JavaScript (semantically somewhat related - programming)
        ('mem5', 0.4),  # C++ (also programming)
        ('mem1', 0.5),  # Python beginner
        ('mem2', 0.6),  # Python advanced
        ('mem4', 0.7)   # Python pandas (but vector search ranks it lower!)
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nPure Vector Search Results:")
    for rank, (doc_id, dist) in enumerate(vector_results[:3], start=1):
        print(f"  {rank}. {doc_id} (distance: {dist:.2f})")
        print(f"     Content: {memories[doc_id]}")
    
    # Hybrid search
    hybrid_results = optimizer.hybrid_search(query, vector_results, k=3)
    
    print(f"\nHybrid Search Results:")
    for rank, (doc_id, score) in enumerate(hybrid_results, start=1):
        print(f"  {rank}. {doc_id} (score: {score:.4f})")
        print(f"     Content: {memories[doc_id]}")
    
    # Hybrid should boost mem4 (contains both "Python" and "pandas")
    hybrid_doc_ids = [doc for doc, _ in hybrid_results]
    
    # mem4 should be in top 3 due to keyword matching
    assert 'mem4' in hybrid_doc_ids, "mem4 should appear in hybrid results (has 'Python pandas')"
    
    # mem4 should rank higher in hybrid than in pure vector
    vector_rank = next(i for i, (doc, _) in enumerate(vector_results) if doc == 'mem4')
    hybrid_rank = next(i for i, (doc, _) in enumerate(hybrid_results) if doc == 'mem4')
    
    print(f"\nmem4 ranking improvement:")
    print(f"  Vector search: rank {vector_rank + 1}")
    print(f"  Hybrid search: rank {hybrid_rank + 1}")
    print(f"  Improvement: {vector_rank - hybrid_rank} positions")
    
    assert hybrid_rank < vector_rank, "mem4 should rank higher in hybrid search"
    
    print(f"\n  ✓ PASSED: Hybrid search improves over pure vector search")
    print("="*70)
    return True


def test_search_analytics():
    """Test search analytics functionality."""
    print("\n" + "="*70)
    print("TEST 7: Search Analytics")
    print("="*70)
    
    optimizer = HybridSearchOptimizer()
    
    memories = {
        'mem1': "Machine learning algorithms",
        'mem2': "Deep neural networks",
        'mem3': "Natural language processing"
    }
    
    optimizer.index_memories(memories)
    
    query = "machine learning neural"
    vector_results = [('mem1', 0.2), ('mem2', 0.3), ('mem3', 0.5)]
    
    analytics = optimizer.get_search_analytics(query, vector_results)
    
    print(f"\nQuery: '{query}'")
    print(f"\nAnalytics:")
    print(f"  Query keywords: {analytics['query_keywords']}")
    print(f"  Vector results: {analytics['vector_results_count']}")
    print(f"  Keyword results: {analytics['keyword_results_count']}")
    print(f"  Overlap: {analytics['overlap_count']} ({analytics['overlap_percentage']:.1f}%)")
    print(f"  Search method: {analytics['search_method']}")
    
    # Verify analytics structure
    assert 'query_keywords' in analytics, "Should have query_keywords"
    assert 'overlap_count' in analytics, "Should have overlap_count"
    assert 'search_method' in analytics, "Should have search_method"
    
    # Should extract relevant keywords
    assert len(analytics['query_keywords']) > 0, "Should extract keywords"
    
    print(f"\n  ✓ PASSED: Search analytics working correctly")
    print("="*70)
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 8: Edge Cases and Error Handling")
    print("="*70)
    
    optimizer = HybridSearchOptimizer()
    
    # Test 1: Empty vector results
    hybrid_results = optimizer.hybrid_search("test query", [], k=5)
    print(f"Empty vector results: {len(hybrid_results)} results")
    # Should still work (just returns keyword results)
    
    # Test 2: No indexed documents
    optimizer2 = HybridSearchOptimizer()
    # Don't index any documents
    hybrid_results2 = optimizer2.hybrid_search("test", [('doc1', 0.5)], k=5)
    print(f"No indexed docs: {len(hybrid_results2)} results")
    # Should handle gracefully
    
    # Test 3: Very short query
    memories = {'doc1': "Some content here"}
    optimizer.index_memories(memories)
    hybrid_results3 = optimizer.hybrid_search("a", [('doc1', 0.5)], k=5)
    print(f"Short query: {len(hybrid_results3)} results")
    
    # Test 4: Identical documents
    memories_identical = {
        'doc1': "Same content",
        'doc2': "Same content",
        'doc3': "Same content"
    }
    optimizer.index_memories(memories_identical)
    hybrid_results4 = optimizer.hybrid_search("same", [('doc1', 0.1), ('doc2', 0.2)], k=5)
    print(f"Identical docs: {len(hybrid_results4)} results")
    
    print(f"\n  ✓ PASSED: Edge cases handled correctly")
    print("="*70)
    return True


def test_performance_characteristics():
    """Test performance characteristics of hybrid search."""
    print("\n" + "="*70)
    print("TEST 9: Performance Characteristics")
    print("="*70)
    
    import time
    
    optimizer = HybridSearchOptimizer()
    
    # Create large document collection
    num_docs = 100
    memories = {
        f'doc{i}': f"Document number {i} with some unique content about topic {i % 10}"
        for i in range(num_docs)
    }
    
    # Measure indexing time
    start_time = time.time()
    optimizer.index_memories(memories)
    indexing_time = time.time() - start_time
    
    print(f"Indexed {num_docs} documents in {indexing_time*1000:.2f}ms")
    
    # Measure search time
    vector_results = [(f'doc{i}', 0.1 * i) for i in range(20)]
    
    start_time = time.time()
    for _ in range(10):  # Run 10 searches
        results = optimizer.hybrid_search("topic 5", vector_results, k=5)
    search_time = (time.time() - start_time) / 10  # Average per search
    
    print(f"Average search time: {search_time*1000:.2f}ms")
    print(f"Results per search: {len(results)}")
    
    # Performance should be reasonable
    assert indexing_time < 1.0, f"Indexing too slow: {indexing_time}s"
    assert search_time < 0.1, f"Search too slow: {search_time}s"
    
    print(f"\n  ✓ PASSED: Performance within acceptable range")
    print("="*70)
    return True


def run_all_tests():
    """Run all hybrid search tests."""
    print("\n" + "#"*70)
    print("# Hybrid Search Optimization Test Suite")
    print("# Combining Semantic + Keyword Search")
    print("#"*70)
    
    test_results = {}
    
    # Run tests
    tests = [
        ('keyword_extraction', test_keyword_extraction),
        ('bm25_scoring', test_bm25_scoring),
        ('bm25_ranking', test_bm25_ranking),
        ('rrf_fusion', test_reciprocal_rank_fusion),
        ('hybrid_integration', test_hybrid_search_integration),
        ('hybrid_vs_vector', test_hybrid_vs_pure_vector),
        ('search_analytics', test_search_analytics),
        ('edge_cases', test_edge_cases),
        ('performance', test_performance_characteristics),
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
        print("  🚀 Hybrid Search Optimization is working correctly!")
        print("  💡 Key Benefits:")
        print("     - Better recall with semantic + keyword matching")
        print("     - Improved precision for technical terms")
        print("     - Robust to query variations")
        print("     - Industry-standard BM25 + RRF algorithms")
        return True
    else:
        print(f"\n  ⚠️  {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
