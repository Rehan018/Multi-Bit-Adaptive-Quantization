# TurboQuant in Multi-Bit Adaptive Quantization

This document explains the TurboQuant integration in this repository: what it does today, how to use it, and what to watch out for.

TurboQuant is used here to compress dense embedding vectors before storing them as metadata. The implementation is based on the TurboQuant paper, "Online Vector Quantization with Near-optimal Distortion Rate" (`arXiv:2504.19874v1`), with a practical Python/Numpy implementation for this repository.

## What It Does

The core class is `TurboQuantCompressor` in `agentic_memory/turboquant.py`.

It compresses a vector in three steps:

1. Rotate the vector with a fixed random orthogonal matrix.
2. Quantize each rotated coordinate with a scalar codebook.
3. If QJL is enabled, store a 1-bit sign sketch of the residual.

With QJL enabled, the bit budget follows the paper's two-stage design:

```text
total bit_width = MSE quantizer bits + QJL residual bit

2-bit mode = 1 MSE bit + 1 QJL bit
3-bit mode = 2 MSE bits + 1 QJL bit
4-bit mode = 3 MSE bits + 1 QJL bit
```

The packed representation stores:

```python
{
    "indices": packed_codebook_indices,
    "qjl_bits": packed_qjl_signs,
    "residual_norm": float,
    "indices_packed": True,
    "qjl_packed": True,
    "mse_bit_width": int,
    "bit_width": int,
    "dimension": int,
}
```

The old prototype stored one `int32` per coordinate, which was not real compression. That is fixed now. Indices and QJL signs are packed into bytes.

## Compression Ratios

For 384-dimensional `float32` embeddings:

| Mode | MSE bits | QJL bit | Stored bytes | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 2-bit + QJL | 1 | 1 | 96 bytes | 16.0x |
| 3-bit + QJL | 2 | 1 | 144 bytes | 10.7x |
| 4-bit + QJL | 3 | 1 | 192 bytes | 8.0x |
| 4-bit without QJL | 4 | 0 | 192 bytes | 8.0x |

The ratio is measured against a 384-dimensional `float32` vector:

```text
384 dims * 4 bytes = 1536 bytes
1536 / 192 = 8x
```

If your input vectors are `float64`, the apparent ratio will look larger because the original vector is twice as large. For real embedding storage, assume `float32`.

## Code Example

```python
import numpy as np
from agentic_memory.turboquant import TurboQuantCompressor

compressor = TurboQuantCompressor(
    bit_width=4,
    use_qjl=True,
    dimension=384,
)

embedding = np.random.randn(384).astype(np.float32)
embedding /= np.linalg.norm(embedding)

compressed = compressor.compress(embedding)
restored = compressor.decompress(compressed)

mse = compressor.calculate_distortion(embedding, compressed)
print(mse)
```

## Using It With Retrieval

`TurboQuantRetriever` lives in `agentic_memory/retrievers.py`.

```python
import numpy as np
from agentic_memory.retrievers import TurboQuantRetriever

retriever = TurboQuantRetriever(
    collection_name="compressed_memories",
    bit_width=4,
    use_qjl=True,
    embedding_dimension=384,
)

embedding = np.random.randn(384).astype(np.float32)
embedding /= np.linalg.norm(embedding)

retriever.add_document(
    document="A note about vector compression",
    metadata={"category": "engineering"},
    doc_id="note-1",
    embedding=embedding,
)

results = retriever.search_with_compression("vector compression", k=5)
```

`search_with_compression()` ranks compressed metadata when compressed vectors are present. If the collection has no compressed metadata, it falls back to normal ChromaDB search.

## Using It With `AgenticMemorySystem`

`AgenticMemorySystem` now accepts a custom retriever:

```python
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import TurboQuantRetriever

retriever = TurboQuantRetriever(bit_width=4, use_qjl=True)

memory = AgenticMemorySystem(
    retriever=retriever,
    llm_backend="openai",
    llm_model="gpt-4o-mini",
)
```

If no OpenAI key or Ollama backend is available, core memory CRUD/search still works. LLM-based memory evolution is skipped gracefully.

## When To Use It

Use TurboQuant when:

- You store many dense embeddings.
- Backup/export size matters.
- You can tolerate approximate reconstruction.
- You want a data-oblivious compressor with no training step.
- You want compressed metadata that can survive persistence and migration.

For most memory-retrieval use cases, start with `bit_width=4, use_qjl=True`.

## When Not To Use It

Avoid TurboQuant when:

- You need exact vector values later.
- Your memory bank is tiny and compression adds no practical value.
- Your embeddings are sparse or not roughly dense Euclidean vectors.
- You need a production GPU kernel today.
- You want ChromaDB's own internal vector index to physically store only compressed vectors.

That last point is important: this integration stores compressed vectors as metadata and can rank with them through `search_with_compression()`. ChromaDB itself is not replaced by a custom compressed-vector index.

## Current Limits

This implementation is intentionally practical, but it is not a full production vector database engine.

- It uses NumPy on CPU, not a GPU kernel.
- The codebook uses a Gaussian approximation for high-dimensional rotated coordinates.
- Compressed search currently scans compressed metadata in Python.
- The implementation is best suited for moderate memory banks, experiments, and compressed persistence.

The paper's accelerator-friendly claim refers to the algorithmic structure. This repository does not yet include CUDA, Triton, or vectorized production kernels.

## Verification

Run:

```bash
.venv/bin/pytest -q
```

The TurboQuant tests verify:

- Packed indices and QJL signs.
- Compression ratio greater than 5x for the retriever.
- MSE decreases as bit-width increases.
- Persistent compressed storage can round-trip data.

## Practical Recommendation

Use this as the compression layer for stored embeddings and persistent snapshots. For high-scale search over millions of vectors, the next step would be a real compressed index that avoids Python-side scanning.
