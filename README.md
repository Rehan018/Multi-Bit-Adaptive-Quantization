# Multi-Bit Adaptive Quantization

Multi-Bit Adaptive Quantization is a Python toolkit for compressed embedding storage and memory retrieval in LLM applications. It combines agent-style memory notes with TurboQuant compression, adaptive bit-width selection, persistent compressed storage, and hybrid search.

The Python import package remains `agentic_memory` for backward compatibility. The repository focus is now the multi-bit compression and retrieval layer built around it.

Related papers and source material:

- [Upstream memory-system paper](https://arxiv.org/pdf/2502.12110)
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- Original memory-system reproduction repository: [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)

## What This Project Does

The project treats memory as more than a flat vector store. Each memory is a note with content and metadata:

- keywords
- context
- tags
- category
- timestamps
- retrieval count
- links to related memories
- evolution history

The basic flow is:

1. Add a memory note.
2. Store it in ChromaDB for retrieval.
3. Search by semantic similarity.
4. Optionally use an LLM to analyze related memories and update links, tags, or context.

The repo includes these compression and retrieval pieces:

- TurboQuant vector compression
- persistent compressed memory
- adaptive bit-width compression
- hybrid keyword + vector search
- a lightweight memory-evolution filter

## Installation

Create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

For tests:

```bash
pip install "pytest>=8.0"
```

## Quick Start

```python
from agentic_memory.memory_system import AgenticMemorySystem

memory = AgenticMemorySystem(
    model_name="all-MiniLM-L6-v2",
    llm_backend="openai",
    llm_model="gpt-4o-mini",
)

memory_id = memory.add_note(
    content="The user prefers concise technical explanations.",
    tags=["preference", "communication"],
    category="User Profile",
)

result = memory.read(memory_id)
print(result.content)

matches = memory.search_agentic("how should I explain technical topics?", k=5)
for item in matches:
    print(item["content"])
```

If no LLM backend is configured, basic memory CRUD and search still work. LLM-based metadata generation and evolution are skipped gracefully.

## Using a Custom Retriever

You can pass a custom retriever into `AgenticMemorySystem`. For example, to use TurboQuant-backed compressed metadata:

```python
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import TurboQuantRetriever

retriever = TurboQuantRetriever(
    collection_name="compressed_memories",
    bit_width=4,
    use_qjl=True,
    embedding_dimension=384,
)

memory = AgenticMemorySystem(retriever=retriever)
```

For normal ChromaDB usage, you do not need to pass a retriever.

## Core Components

### `AgenticMemorySystem`

The main user-facing class. It manages memory notes, metadata, search, updates, deletes, and optional evolution.

Path:

```text
agentic_memory/memory_system.py
```

### `ChromaRetriever`

The default in-memory ChromaDB retriever.

Path:

```text
agentic_memory/retrievers.py
```

### `PersistentChromaRetriever`

A ChromaDB retriever that survives process restarts.

### `TurboQuantRetriever`

Stores packed TurboQuant representations in metadata and can rank with compressed vectors through `search_with_compression()`.

## Compressed Memory Features

The TurboQuant integration compresses dense embeddings into packed byte arrays. For 384-dimensional `float32` vectors:

| Mode | Stored bytes | Approx. ratio |
| --- | ---: | ---: |
| 2-bit + QJL | 96 bytes | 16.0x |
| 3-bit + QJL | 144 bytes | 10.7x |
| 4-bit + QJL | 192 bytes | 8.0x |

This is useful for metadata snapshots, backups, and compressed persistence. It does not replace ChromaDB's internal vector index with a production compressed index.

## Documentation

The docs are split by feature:

- [TurboQuant guide](docs/turboquant-guide.md)
- [Persistent compressed memory](docs/persistent-compressed-memory.md)
- [Adaptive quantization](docs/adaptive-quantization.md)
- [Hybrid search](docs/hybrid-search.md)
- [Memory evolution](docs/memory-evolution.md)

## Running Tests

Use the repo virtual environment:

```bash
.venv/bin/pytest -q
```

You can also run focused tests:

```bash
.venv/bin/pytest -q tests/test_turboquant.py
.venv/bin/pytest -q tests/test_persistent_compressed.py
.venv/bin/pytest -q tests/test_hybrid_search.py
```

## Practical Notes

- OpenAI support requires `OPENAI_API_KEY`.
- Ollama support requires the `ollama` Python package and a running local Ollama setup.
- If no LLM backend is available, the system still stores and retrieves memories.
- TurboQuant code is NumPy CPU code. It is not a CUDA or Triton implementation.
- Compressed search currently scans compressed metadata in Python, so it is best treated as an experimental retrieval path rather than a large-scale vector database engine.

## Citation

If you use the original memory-system research, cite the upstream paper:

```bibtex
@article{xu2025mem,
  title={Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
