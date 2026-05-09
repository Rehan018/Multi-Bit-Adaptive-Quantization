# Multi-Bit Adaptive Quantization

A Python library for building compressed, structured memory systems for LLM agents. It gives agents a memory layer that stores rich notes instead of flat vectors, compresses embeddings 8–16× using a research-grade algorithm, and retrieves them intelligently through hybrid semantic and keyword search.

The Python package is `agentic_memory` (kept for backward compatibility). The current focus is the compression and retrieval layer built on top of the original agentic memory work.

**Related research:**
- [Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110) — upstream memory system design
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — compression algorithm implemented here
- Original memory system repo: [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)

---

## Overview

Most vector stores treat memory as a blob of text attached to a float array. That works fine for single-turn retrieval, but it breaks down when you want memory to be useful over time — tracking how often something was accessed, what it's related to, or how it has changed.

This project wraps ChromaDB in a structured memory layer. Every memory is a `MemoryNote` with explicit fields for keywords, tags, context, links to related memories, access timestamps, and evolution history. On top of that structured layer, the project adds:

- **TurboQuant compression** — 8–16× size reduction using random rotation and scalar quantization
- **Adaptive bit-width selection** — automatically assigns 2, 3, or 4 bits per coordinate based on how important a memory is
- **Hybrid search** — combines semantic vector search with BM25 keyword ranking via Reciprocal Rank Fusion
- **Memory evolution** — an LLM analyzes new memories against their neighbors and updates links, tags, and context
- **Persistent storage** — optional disk-backed ChromaDB that survives process restarts

---

## Objective

Build a memory system for LLM agents that is:

1. **Structurally rich** — memories carry metadata that agents and developers can use, not just raw content
2. **Storage-efficient at scale** — embedding compression makes large memory banks practical
3. **Retrieval-accurate** — hybrid search catches both semantic matches and exact keyword matches that embeddings miss
4. **Self-organizing** — an LLM periodically updates links and context as new memories arrive

---

## Problem Statement

Three concrete problems motivated this project:

**Storage cost.** A 384-dim float32 embedding takes 1,536 bytes. A million agent memories take 1.5 GB in vectors alone, before content or metadata. TurboQuant compresses each embedding to 96–192 bytes with provably near-optimal distortion.

**Not all memories matter equally.** Compressing every memory at the same bit-width wastes quality on stale or rarely-retrieved notes while potentially under-serving critical ones. The adaptive quantization layer scores each memory by retrieval frequency, age, and explicit importance, then picks the right bit-width for it.

**Exact terms fall through semantic search.** Embeddings are good at meaning but bad at specific identifiers, technical terms, and short phrases. The hybrid search layer runs BM25 alongside the vector search and fuses both rankings, so neither type of match dominates.

---

## Why This Project Matters

If you are building an agent that accumulates memory over time — a personal assistant, a long-running research agent, a customer support bot — you will eventually hit these walls: storage cost, retrieval accuracy, and memory staleness. This project is an exploration of how to address all three in a single coherent system, grounded in an actual quantization research paper rather than ad hoc heuristics.

The TurboQuant implementation specifically is interesting because it is data-oblivious (no training corpus needed), mathematically grounded (near-optimal distortion rate within 2.7× of the theoretical lower bound per the paper), and implemented entirely in NumPy — no ML framework dependency.

---

## Key Features

- **Structured memory notes** — content, keywords, tags, context, category, links, timestamps, retrieval count, evolution history
- **TurboQuant compression** — 2/3/4-bit quantization with optional QJL for unbiased inner product estimation
- **Adaptive compression policy** — per-memory bit-width based on importance score (frequency + age + user weight)
- **Hybrid BM25 + vector search** — Reciprocal Rank Fusion combining semantic and keyword rankings
- **Memory evolution** — LLM-driven link strengthening and context updating between related memories
- **Distortion-based evolution filter** — uses quantization distortion to pre-filter LLM calls, saving cost
- **Persistent storage** — disk-backed ChromaDB with optional automatic compression
- **Multi-agent memory sharing** — `CopiedChromaRetriever` creates isolated copies of a shared starting collection
- **LLM-agnostic** — works with OpenAI or local Ollama; core CRUD and search work with no LLM at all

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector store | ChromaDB |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| Compression | TurboQuant (NumPy, custom implementation) |
| Keyword search | rank-bm25, custom BM25Scorer |
| LLM backend | litellm (OpenAI API or Ollama) |
| Numerics | NumPy, scikit-learn |
| Packaging | setuptools, pyproject.toml |
| Tests | pytest |

---

## System Architecture

The system has four layers that work together:

```
User / Agent
     │
     ▼
AgenticMemorySystem          ← main entry point (memory_system.py)
     │
     ├── MemoryNote           ← structured data object per memory
     │
     ├── LLMController        ← metadata extraction + evolution decisions
     │       ├── OpenAIController
     │       └── OllamaController
     │
     ├── Retriever            ← pluggable storage backends
     │       ├── ChromaRetriever            (in-memory, default)
     │       ├── PersistentChromaRetriever  (disk-backed)
     │       ├── TurboQuantRetriever        (compressed metadata)
     │       └── CopiedChromaRetriever      (isolated clone)
     │
     ├── TurboQuantCompressor ← rotation → quantization → optional QJL
     │
     ├── HybridSearchOptimizer← BM25 + vector → RRF fusion
     │
     ├── AdaptiveQuantizationPolicy ← importance scoring → bit-width
     │
     └── TurboQuantEvolutionAnalyzer ← distortion-based evolution filter
```

### Memory lifecycle

```
add_note(content, tags, ...)
  │
  ├── analyze_content() → LLM extracts keywords, context, tags
  │
  ├── process_memory() → find k nearest neighbors
  │       └── LLM decides: strengthen links? update neighbor context?
  │
  ├── retriever.add_document() → embed + store in ChromaDB
  │       └── (if TurboQuantRetriever) compress embedding → store packed bytes in metadata
  │
  └── evo_cnt check → consolidate_memories() every N evolutions
```

```
search_agentic(query, k)
  │
  ├── ChromaDB semantic search → top-k results
  │
  └── Expand with linked memories (neighbors from evolution)
```

### TurboQuant compression pipeline

```
Input vector (384 × float32 = 1,536 bytes)
  │
  ├── Random orthogonal rotation (pre-computed, fixed seed)
  │
  ├── Per-coordinate scalar quantization
  │       └── Lloyd-Max codebook (optimal for Gaussian/Beta distribution)
  │
  ├── (optional) QJL — random projection of residual → 1 sign bit per coord
  │       └── Enables unbiased inner product estimation without full reconstruction
  │
  └── Pack indices into bytes → 96 / 144 / 192 bytes (2/3/4-bit)
```

---

## Folder Structure

```
A-mem/
├── agentic_memory/
│   ├── __init__.py
│   ├── memory_system.py               # AgenticMemorySystem, MemoryNote
│   ├── retrievers.py                  # ChromaRetriever variants + TurboQuantRetriever
│   ├── turboquant.py                  # TurboQuant compression algorithm
│   ├── llm_controller.py              # OpenAI / Ollama wrappers
│   ├── hybrid_search.py               # BM25Scorer, RRF, HybridSearchOptimizer
│   ├── adaptive_quantization.py       # AdaptiveQuantizationPolicy, MultiBitAdaptiveCompressor
│   ├── memory_evolution_enhanced.py   # TurboQuantEvolutionAnalyzer, EnhancedMemoryEvolution
│   └── persistent_compressed_retriever.py  # PersistentCompressedRetriever
├── tests/
│   ├── test_turboquant.py
│   ├── test_memory_system.py
│   ├── test_hybrid_search.py
│   ├── test_adaptive_quantization.py
│   ├── test_persistent_compressed.py
│   ├── test_memory_evolution_enhanced.py
│   ├── test_retriever.py
│   └── conftest.py
├── docs/
│   ├── turboquant-guide.md
│   ├── persistent-compressed-memory.md
│   ├── adaptive-quantization.md
│   ├── hybrid-search.md
│   └── memory-evolution.md
├── examples/
│   └── sovereign_memory.py            # End-to-end example with Ollama
├── Figure/
│   ├── framework.jpg                  # System architecture diagram
│   ├── intro-a.jpg                    # Traditional memory system
│   ├── intro-b.jpg                    # Agentic memory system
│   └── mermaid-diagrams.md            # Mermaid source for diagrams
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Setup Instructions

Python 3.8 or higher is required.

```bash
# Clone and enter the repo
git clone https://github.com/agiresearch/multi-bit-adaptive-quantization
cd multi-bit-adaptive-quantization

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

The first import of `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) automatically.

For NLTK tokenizer data (used by hybrid search):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Only if using OpenAI backend | Passed to the OpenAI client for LLM calls |

For Ollama, no environment variable is needed — just have the Ollama daemon running locally and the model pulled:
```bash
ollama pull llama3
```

If neither backend is configured, the system still runs. LLM-based metadata extraction and memory evolution are silently skipped. Basic CRUD and semantic search continue to work.

---

## How to Run

### No-LLM mode (local only)

```python
from agentic_memory.memory_system import AgenticMemorySystem

memory = AgenticMemorySystem()  # no llm_backend needed

memory_id = memory.add_note(
    content="The user prefers dark mode and keyboard shortcuts.",
    tags=["preference", "ui"],
    category="User Profile",
)

note = memory.read(memory_id)
print(note.content)

results = memory.search_agentic("user interface preferences", k=5)
for item in results:
    print(item["content"])

memory.update(memory_id, tags=["preference", "ui", "verified"])
memory.delete(memory_id)
```

### With OpenAI (LLM metadata extraction + evolution)

```python
import os
from agentic_memory.memory_system import AgenticMemorySystem

memory = AgenticMemorySystem(
    model_name="all-MiniLM-L6-v2",
    llm_backend="openai",
    llm_model="gpt-4o-mini",
)

memory_id = memory.add_note(
    content="Attention mechanisms scale quadratically with sequence length."
)

note = memory.read(memory_id)
print(note.keywords)   # extracted by LLM
print(note.context)    # extracted by LLM
print(note.tags)       # extracted by LLM
```

### With Ollama (fully local)

```python
from agentic_memory.memory_system import AgenticMemorySystem

memory = AgenticMemorySystem(
    llm_backend="ollama",
    llm_model="llama3",
)

memory_id = memory.add_note(content="User values data privacy above convenience.")
```

### End-to-end example

```bash
python examples/sovereign_memory.py
```

This demonstrates initializing with Ollama, storing a memory, and retrieving it by semantic query.

---

## Compression

### TurboQuant retriever

```python
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import TurboQuantRetriever
import numpy as np

retriever = TurboQuantRetriever(
    collection_name="compressed_memories",
    bit_width=4,        # 2, 3, or 4
    use_qjl=True,       # adds 1 QJL sign bit per coordinate
    embedding_dimension=384,
)

memory = AgenticMemorySystem(retriever=retriever)

# When passing a pre-computed embedding, it gets compressed
embedding = np.random.randn(384).astype(np.float32)
embedding /= np.linalg.norm(embedding)

retriever.add_document(
    document="Some memory content",
    metadata={"category": "test"},
    doc_id="mem_001",
    embedding=embedding,
)

# Compressed similarity search
results = retriever.search_with_compression("query text", k=5)
print(retriever.get_compression_stats())
```

### Adaptive bit-width compression

```python
from agentic_memory.adaptive_quantization import MultiBitAdaptiveCompressor
import numpy as np

compressor = MultiBitAdaptiveCompressor(dimension=384, use_qjl=True)

embedding = np.random.randn(384).astype(np.float32)
metadata = {
    "retrieval_count": 42,     # frequently accessed → 4-bit
    "timestamp": "2026-01-01T00:00:00",
    "importance": 0.9,
    "content": "critical user preference",
}

compressed, bit_width = compressor.compress_adaptive(embedding, metadata)
print(f"Stored at {bit_width}-bit quality")
print(compressor.get_statistics())
```

### Persistent compressed retriever

```python
from agentic_memory.persistent_compressed_retriever import PersistentCompressedRetriever
from agentic_memory.memory_system import AgenticMemorySystem

retriever = PersistentCompressedRetriever(
    directory="~/.agent_memories",
    collection_name="long_term",
    bit_width=4,
    use_qjl=True,
    extend=True,   # add to existing collection if it exists
)

memory = AgenticMemorySystem(retriever=retriever)
```

### Compression reference table

For 384-dimensional `float32` embeddings (1,536 bytes uncompressed):

| Mode | Stored bytes | Compression |
|------|-------------:|------------:|
| 2-bit + QJL | 96 bytes | 16.0× |
| 3-bit + QJL | 144 bytes | 10.7× |
| 4-bit + QJL | 192 bytes | 8.0× |

---

## Hybrid Search

```python
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.hybrid_search import create_hybrid_search_wrapper

memory = AgenticMemorySystem()

memory.add_note(content="Asyncio uses an event loop for concurrency in Python.")
memory.add_note(content="JavaScript Promises handle async via callbacks.")
memory.add_note(content="Go goroutines are lightweight threads managed by the runtime.")

# Build BM25 index over current memories
optimizer = create_hybrid_search_wrapper(memory)

# Get vector results first, then fuse with keyword results
vector_results = memory.retriever.collection.query(
    query_texts=["event loop async"],
    n_results=5
)
vector_pairs = list(zip(
    vector_results["ids"][0],
    vector_results["distances"][0]
))

# Fuse vector + BM25 rankings
fused = optimizer.hybrid_search(
    query="event loop async",
    vector_results=vector_pairs,
    k=3,
)
print(fused)   # [(doc_id, rrf_score), ...]
```

---

## Model Details

**Embedding model:** `all-MiniLM-L6-v2` from SentenceTransformers
- 384-dimensional output
- Good balance of speed and quality for semantic similarity tasks
- Downloaded automatically on first use

**TurboQuant compressor:**
- Implemented from scratch in NumPy based on arXiv:2504.19874
- Random orthogonal rotation pre-computed at init with fixed seed (reproducible)
- Codebook built via Lloyd-Max iterations over Gaussian distribution
- `bit_width` parameter controls MSE + QJL budget: at 4-bit with QJL, 3 bits go to the MSE quantizer and 1 bit to QJL
- No training data required — data-oblivious design

**LLM:** User-supplied. The system issues two types of prompts:
1. **Content analysis** — extracts keywords, context, tags from memory content (one call per `add_note` if LLM is configured)
2. **Evolution decision** — asks whether a new memory should strengthen links or update neighbors (one call per note if neighbors exist)

Using `gpt-4o-mini` keeps the cost low. Ollama makes both fully local.

---

## Testing

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run all tests
pytest -q

# Run specific test files
pytest -q tests/test_turboquant.py
pytest -q tests/test_memory_system.py
pytest -q tests/test_hybrid_search.py
pytest -q tests/test_adaptive_quantization.py
pytest -q tests/test_persistent_compressed.py
pytest -q tests/test_memory_evolution_enhanced.py
```

The TurboQuant tests (`test_turboquant.py`) do not require an LLM — they test the compression algorithm directly against NumPy assertions. The memory system tests require no LLM either, since `AgenticMemorySystem` degrades gracefully when no key is available.

Tests that specifically test LLM evolution behavior will need a valid `OPENAI_API_KEY` or a running Ollama server.

---

## Feature Documentation

Detailed guides for each component are in the `docs/` directory:

- [TurboQuant guide](docs/turboquant-guide.md) — algorithm walkthrough, bit-width tradeoffs
- [Persistent compressed memory](docs/persistent-compressed-memory.md) — setup, migration
- [Adaptive quantization](docs/adaptive-quantization.md) — importance scoring, policy tuning
- [Hybrid search](docs/hybrid-search.md) — BM25 parameters, RRF weights
- [Memory evolution](docs/memory-evolution.md) — evolution threshold, LLM prompts

---

## Design Decisions and Trade-offs

**Why ChromaDB instead of a dedicated ANN index?**
ChromaDB handles embedding, storage, and similarity search in one library with minimal setup. For the scale this project targets (thousands to low millions of memories), it is practical. If you needed 100M+ vectors with sub-millisecond latency, you would replace the retriever with Faiss or Pinecone. The retriever interface is pluggable specifically to allow that.

**Why store compressed vectors in ChromaDB metadata instead of replacing the index?**
ChromaDB manages its own vector index internally. Injecting TurboQuant directly into ChromaDB's ANN index would require forking ChromaDB. Instead, compressed vectors are stored as JSON in metadata. The `search_with_compression` path scans those compressed representations in Python for ranking. This means compressed search is O(n) and best treated as experimental — ChromaDB's native search is still the production path.

**Why adaptive bit-width instead of one fixed compression level?**
A uniform 2-bit compression is great for storage but hurts retrieval quality on frequently-accessed memories. Uniform 4-bit is safe but wastes storage on memories that never get retrieved again. The adaptive policy is a middle ground — it costs one importance calculation per memory, but lets the bit-width track actual usage rather than being a static deployment decision.

**Why use distortion to pre-filter LLM evolution calls?**
Every call to the LLM evolution prompt costs latency and money. The distortion filter checks whether a new memory's embedding is geometrically close (low quantization error) to existing memories. If it is redundant, there is little reason to ask an LLM to evolve it. This reduces LLM calls on duplicate or near-duplicate memories, which is common in real agent workflows.

---

## Limitations

- **No GPU support.** TurboQuant is NumPy CPU code. The Gram-Schmidt QR decomposition and per-coordinate codebook lookup run on CPU only. For very large batches this is slow.
- **Compressed search is linear.** `search_with_compression` scans all compressed metadata in Python. It is not suitable as a primary retrieval path for large collections.
- **Consolidation loses compression stats.** When `consolidate_memories()` runs (triggered every `evo_threshold` evolutions), it rebuilds the retriever as a plain `ChromaRetriever`, which does not carry forward any compression state from a `TurboQuantRetriever`. This is a known limitation of the current architecture.
- **LLM prompts are one-size-fits-all.** The evolution and analysis prompts are fixed templates. They work reasonably well but are not tuned for specific domains.
- **No authentication or access control.** This is a local library, not a service. There is no concept of users or permissions.
- **No benchmark results in this repo.** The compression ratios are verified by tests. End-to-end retrieval quality improvements from hybrid search or adaptive quantization are not benchmarked against a labeled dataset here.

---

## Future Improvements

- **GPU-accelerated TurboQuant** — the rotation and quantization steps are embarrassingly parallelizable; a Triton kernel would make batch compression fast
- **Pluggable ANN index** — wire TurboQuant into a proper HNSW or IVF index (Faiss, Hnswlib) to make compressed search sublinear
- **Streaming evolution** — currently evolution runs synchronously on `add_note`; background async processing would remove the latency spike
- **Retriever-agnostic consolidation** — `consolidate_memories()` should rebuild whatever retriever type is currently active, preserving compression settings
- **Domain-tuned LLM prompts** — configurable prompt templates per deployment domain (code, medical, legal)
- **Benchmark suite** — retrieval quality benchmarks on BEIR or similar to quantify the effect of adaptive quantization and hybrid search

---

## Author / Contact

<!-- Add author name, email, and GitHub profile here -->

Project homepage: [github.com/agiresearch/multi-bit-adaptive-quantization](https://github.com/agiresearch/multi-bit-adaptive-quantization)

If you use the upstream agentic memory research, please cite:

```bibtex
@article{xu2025mem,
  title={Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
