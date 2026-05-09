# Adaptive Quantization

Adaptive quantization chooses a TurboQuant bit-width per memory instead of using the same compression setting for everything.

The implementation lives in:

```text
agentic_memory/adaptive_quantization.py
```

## Why It Exists

Not every memory deserves the same storage budget.

A memory that is retrieved often, recently created, or explicitly marked as important should keep more detail. Older or rarely used memories can usually tolerate stronger compression.

Adaptive quantization gives the memory store a simple policy layer:

```text
important memory   -> 4-bit TurboQuant
medium memory      -> 3-bit TurboQuant
low-priority memory -> 2-bit TurboQuant
```

## Main Classes

### `AdaptiveQuantizationPolicy`

Computes an importance score from:

- retrieval count
- age
- user-provided importance

The default weights are:

```python
frequency_weight = 0.4
age_weight = 0.3
importance_weight = 0.3
```

The score is then mapped to a bit-width:

| Importance score | Bit-width |
| ---: | ---: |
| `>= 0.7` | 4-bit |
| `>= 0.4` | 3-bit |
| `< 0.4` | 2-bit |

### `MultiBitAdaptiveCompressor`

Owns three TurboQuant compressors:

- 2-bit
- 3-bit
- 4-bit

It chooses the right one for each memory and stores the selected bit-width in the compressed payload.

## Basic Usage

```python
import numpy as np
from agentic_memory.adaptive_quantization import create_adaptive_compressor

compressor = create_adaptive_compressor(
    dimension=384,
    use_qjl=True,
)

embedding = np.random.randn(384).astype(np.float32)
embedding /= np.linalg.norm(embedding)

metadata = {
    "retrieval_count": 25,
    "timestamp": "2026-05-09T12:00:00",
    "importance": 0.8,
    "content": "Important project decision",
}

compressed, bit_width = compressor.compress_adaptive(embedding, metadata)
restored = compressor.decompress(compressed)

print(bit_width)
print(compressor.get_statistics())
```

## Metadata Inputs

The policy reads these fields:

```python
{
    "retrieval_count": 0,
    "timestamp": "...",
    "importance": 0.5,
    "content": "...",
}
```

Missing fields are handled with defaults. Invalid timestamps are treated as recent enough to avoid crashing the compression path.

## Statistics

`get_statistics()` returns:

- total compressed count
- distribution by bit-width
- average bit-width
- original size
- compressed size
- estimated space saved

This is useful for checking whether the policy is doing what you expect. If everything lands in one bucket, tune the thresholds or weights.

## When To Use It

Use adaptive quantization when:

- The memory bank is large.
- Memories have different importance levels.
- You track retrieval counts.
- You want a better storage-quality tradeoff than one fixed bit-width.
- You periodically re-evaluate old memories.

## When To Avoid It

Avoid it when:

- All memories are equally important.
- The system has very few memories.
- You do not track useful metadata.
- You need the simplest possible compression path.

## Practical Guidance

For most systems:

- Start with the default policy.
- Use 4-bit for critical memories.
- Let old low-use memories fall to 2-bit.
- Monitor the average bit-width.

If average bit-width stays above 3.5, the policy is too conservative. If retrieval quality drops, it is too aggressive.

## Verification

Run:

```bash
.venv/bin/pytest -q tests/test_adaptive_quantization.py
```

The tests cover policy scoring, bit-width selection, compression quality, statistics, edge cases, and a small real-world simulation.
