# Persistent Compressed Memory

This document explains the persistent compressed retriever in Multi-Bit Adaptive Quantization. It combines ChromaDB persistence with TurboQuant-compressed embedding metadata.

The implementation lives in:

```text
agentic_memory/persistent_compressed_retriever.py
```

## Why It Exists

Plain in-memory retrieval is fine for quick experiments, but it has two obvious limits:

- The memory store disappears when the process exits.
- Embeddings become expensive to keep, back up, or move around as the memory bank grows.

`PersistentCompressedRetriever` addresses both:

- ChromaDB data is stored on disk.
- If an embedding is supplied when adding a document, a packed TurboQuant version is stored in metadata.
- Compression stats are saved next to the ChromaDB directory.

## What Gets Stored

When compression is enabled and an embedding is provided, metadata gets fields like:

```python
{
    "compressed_indices": "[...]",
    "qjl_bits": "[...]",
    "residual_norm": "0.123",
    "indices_packed": "True",
    "qjl_packed": "True",
    "mse_bit_width": "3",
    "bit_width": "4",
    "compression_enabled": "true",
}
```

For a 384-dimensional `float32` vector in 4-bit QJL mode, the packed compressed representation is about 192 bytes instead of 1536 bytes, or roughly 8x smaller.

## Basic Usage

```python
import numpy as np
from agentic_memory.persistent_compressed_retriever import PersistentCompressedRetriever

retriever = PersistentCompressedRetriever(
    directory="./memory_db",
    collection_name="memories",
    bit_width=4,
    use_qjl=True,
    auto_compress=True,
    extend=True,
)

embedding = np.random.randn(384).astype(np.float32)
embedding /= np.linalg.norm(embedding)

retriever.add_document(
    document="The user prefers concise technical explanations.",
    metadata={"source": "conversation"},
    doc_id="memory-1",
    embedding=embedding,
)

results = retriever.search("technical explanations", k=5)
stats = retriever.get_compression_stats()
```

## Compression Stats

`get_compression_stats()` reports:

- total document count
- compressed vs uncompressed count
- original embedding bytes
- compressed metadata bytes
- actual compression ratio
- theoretical compressor ratio

The actual ratio is based on the packed arrays stored by this integration. It does not include every byte of ChromaDB's own storage files.

## Persistence Across Sessions

Use `extend=True` when opening an existing collection:

```python
retriever = PersistentCompressedRetriever(
    directory="./memory_db",
    collection_name="memories",
    extend=True,
)
```

Without `extend=True`, the class raises an error if the collection already exists. That is intentional; it prevents accidentally writing to the wrong existing collection.

## Migrating Existing Data

If a collection already has uncompressed embeddings, use:

```python
stats = retriever.migrate_to_compressed(batch_size=100)
```

Migration reads existing embeddings from ChromaDB, compresses them, and updates metadata. Documents without embeddings are skipped.

## When To Use It

Use persistent compressed memory when:

- Memories must survive restarts.
- You want smaller backups or exports.
- Multiple runs should reuse the same memory store.
- You are experimenting with long-lived agent memory.
- You want compression stats over time.

## When To Avoid It

Avoid it when:

- You only need a one-off demo.
- You do not have embeddings to compress.
- You need exact vector recovery.
- You need a high-throughput write path with hundreds of writes per second.
- You want ChromaDB's internal vector index itself to be compressed.

This retriever stores compressed vectors as metadata. ChromaDB still manages its normal collection storage.

## Recommended Settings

Start here:

```python
PersistentCompressedRetriever(
    bit_width=4,
    use_qjl=True,
    auto_compress=True,
)
```

Use lower bit-widths only when storage is more important than reconstruction quality.

## Troubleshooting

### Collection Already Exists

Use `extend=True` if you meant to open an existing collection.

### Compression Ratio Looks Wrong

Check that:

- You passed an embedding to `add_document()`.
- The embedding is `float32`.
- `auto_compress=True`.
- Metadata contains `compressed_indices`.

### Search Works But Embeddings Are Missing

ChromaDB does not always include embeddings in query responses unless requested. Compressed metadata is still available through document metadata.

## Verification

Run:

```bash
.venv/bin/pytest -q tests/test_persistent_compressed.py
```

The tests cover initialization, add/search, persistence across sessions, migration, and compression stats.
