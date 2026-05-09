# Memory Evolution Enhancement

This document explains the lightweight memory evolution filter in Multi-Bit Adaptive Quantization. It uses embedding similarity and TurboQuant reconstruction error to decide whether a memory looks redundant enough to trigger deeper evolution logic.

The implementation lives in:

```text
agentic_memory/memory_evolution_enhanced.py
```

## Why It Exists

LLM-based memory evolution can be useful, but calling an LLM for every new memory is expensive and slow.

The enhancement adds a cheap first pass:

```text
new memory arrives
find related memories
measure redundancy with vector math
only call deeper evolution when needed
```

The goal is not to replace semantic reasoning. The goal is to avoid unnecessary LLM calls when the memory is clearly unique.

## Main Classes

### `TurboQuantEvolutionAnalyzer`

Computes distortion-style metrics between a new embedding and related embeddings.

It returns:

```python
{
    "min_distortion": float,
    "avg_distortion": float,
    "max_distortion": float,
    "should_evolve": bool,
    "redundancy_score": float,
    "reason": str,
}
```

Low distortion means the new memory is close to existing memories and may be redundant. High distortion means it likely contains new information.

### `EnhancedMemoryEvolution`

Wraps the analyzer and tracks operational stats:

- total analyses
- LLM calls saved
- evolution triggers
- unique memories
- evolution rate

## Basic Usage

```python
import numpy as np
from agentic_memory.memory_evolution_enhanced import EnhancedMemoryEvolution

evolution = EnhancedMemoryEvolution(
    bit_width=4,
    evolution_threshold=0.0001,
    dimension=384,
    enable_distortion_filter=True,
)

new_embedding = np.random.randn(384).astype(np.float32)
new_embedding /= np.linalg.norm(new_embedding)

related_embeddings = [
    np.random.randn(384).astype(np.float32)
    for _ in range(5)
]
related_embeddings = [
    emb / np.linalg.norm(emb)
    for emb in related_embeddings
]

decision = evolution.should_evolve_memory(
    new_embedding,
    related_embeddings,
)

print(decision)
print(evolution.get_evolution_statistics())
```

## How To Interpret The Decision

The decision has two separate meanings:

- `should_evolve`: the memory appears redundant enough to consider evolution.
- `llm_call_needed`: a deeper LLM-based step should run.

If the memory is unique, the system can store it directly and skip the LLM call.

## Thresholds

The default threshold is intentionally conservative. You should tune it with your own embeddings.

You can auto-calibrate from samples:

```python
evolution.analytics.update_threshold_based_on_data(
    sample_embeddings,
    target_evolution_rate=0.2,
)
```

This sets the threshold so roughly the target fraction of sample comparisons would trigger evolution.

## When To Use It

Use this filter when:

- Memory creation is frequent.
- LLM calls are a noticeable cost.
- Many memories are similar or repetitive.
- You already have embeddings available.

## When To Avoid It

Avoid it when:

- Every memory needs careful semantic merging.
- You do not have embeddings.
- False negatives are more expensive than LLM calls.
- Your memory bank is too small for redundancy filtering to matter.

## Practical Notes

This is a filter, not a final source of truth. It is useful because it is cheap. It should be paired with LLM evolution when the filter says a memory looks redundant or connected enough to inspect.

Random unrelated vectors will usually look unique. Similar vectors with small noise should trigger evolution if the threshold is reasonable.

## Verification

Run:

```bash
.venv/bin/pytest -q tests/test_memory_evolution_enhanced.py
```

The tests cover distortion analysis, redundant-memory detection, unique-memory detection, batch analysis, threshold calibration, and estimated LLM cost savings.
