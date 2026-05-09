# Hybrid Search

Hybrid search combines semantic vector results with keyword search. It is useful because embeddings are good at meaning, but they are not always good at exact names, error messages, identifiers, or rare terms.

The implementation lives in:

```text
agentic_memory/hybrid_search.py
```

## What It Combines

Hybrid search uses three pieces:

1. Vector search results from an existing retriever.
2. BM25 keyword scoring over indexed memory text.
3. Reciprocal Rank Fusion (RRF) to merge the rankings.

The main class is:

```python
HybridSearchOptimizer
```

## Why It Helps

Pure vector search can miss documents that contain the exact phrase you care about. For example:

- an exception string
- an API name
- a command-line flag
- a product name
- a person's name
- a code symbol

BM25 catches exact lexical matches. Vector search catches semantic matches. RRF gives a stable combined ranking without needing the two score scales to match.

## Basic Usage

```python
from agentic_memory.hybrid_search import HybridSearchOptimizer

optimizer = HybridSearchOptimizer(
    vector_weight=0.6,
    keyword_weight=0.4,
)

memories = {
    "m1": "Python pandas dataframe merge error",
    "m2": "Vector databases store embedding representations",
    "m3": "Troubleshooting CUDA out of memory errors",
}

optimizer.index_memories(memories)

vector_results = [
    ("m2", 0.25),
    ("m1", 0.40),
    ("m3", 0.60),
]

results = optimizer.hybrid_search(
    query="pandas merge error",
    vector_results=vector_results,
    k=3,
)
```

`vector_results` should use lower distance as better, matching ChromaDB style.

## How Ranking Works

Vector distances are normalized into scores. BM25 produces keyword scores. Each ranked list is then fused with RRF:

```text
score(doc) = sum(weight / (rrf_k + rank))
```

The default `rrf_k` is `60.0`, which keeps the fusion stable and avoids overreacting to small rank changes.

## Tuning

Use these defaults first:

```python
HybridSearchOptimizer(
    vector_weight=0.6,
    keyword_weight=0.4,
    rrf_k=60.0,
    bm25_k1=1.5,
    bm25_b=0.75,
)
```

Adjust weights based on your domain:

- Technical docs: increase `keyword_weight`.
- Conversational memory: increase `vector_weight`.
- Code or logs: often use a stronger keyword weight.
- Open-ended semantic recall: keep vector weight higher.

## Keeping The Index Fresh

Call `index_memories()` whenever the memory set changes materially.

For small memory banks this is cheap. For very large or constantly changing collections, you may want incremental indexing later. The current implementation rebuilds the BM25 index from the supplied dictionary.

## Analytics

Use:

```python
analytics = optimizer.get_search_analytics(query, vector_results)
```

This reports:

- extracted query keywords
- vector result count
- keyword result count
- overlap count
- overlap percentage

Low overlap is not always bad. It often means the two methods are finding different useful evidence.

## When To Use It

Use hybrid search when:

- Queries contain exact terms.
- You search technical notes, docs, logs, or code.
- You see vector search returning plausible but wrong results.
- Names and identifiers matter.

## When To Avoid It

Avoid it when:

- The memory bank is tiny.
- Queries are purely semantic.
- Memory text changes constantly and reindexing is too expensive.
- You are not searching text.

## Verification

Run:

```bash
.venv/bin/pytest -q tests/test_hybrid_search.py
```

The tests cover keyword extraction, BM25 scoring, RRF fusion, hybrid ranking, analytics, edge cases, and basic performance.
