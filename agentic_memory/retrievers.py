import json
from pathlib import Path
from typing import Dict, List, Optional
import ast
import tempfile
import atexit
import numpy as np

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from nltk.tokenize import word_tokenize
from .turboquant import TurboQuantCompressor


def simple_tokenize(text):
    return word_tokenize(text)


def _clone_collection(
    src: chromadb.Collection,
    dest: chromadb.Collection,
    batch_size: int = 10
):
    """
    Copies one ChromaDB collection to another. 
    Enables duplicating of collections.
    This seemed to be the only (best) way to do this as the official ChromaDB
        docs also suggest this method:
    """
    existing_count = src.count()
    for i in range(0, existing_count, batch_size):
        batch = src.get(
            include=["metadatas", "documents", "embeddings"],
            limit=batch_size,
            offset=i)
        dest.add(
            ids=batch["ids"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            embeddings=batch["embeddings"])


class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""

    def __init__(
        self, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        self.collection.add(
            documents=[document], metadatas=[processed_metadata], ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        if (results is not None) and (results.get("metadatas", [])):
            results["metadatas"] = self._convert_metadata_types(
                results["metadatas"])
        
        return results

    def _convert_metadata_types(
        self, 
        metadatas: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Convert string metadata back to original types.
        
        Args:
            metadatas: List of metadata lists from query results
            
        Returns:
            Converted metadata structure
        """
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, List):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, Dict):
                        self._convert_metadata_dict(metadata_dict)
        return metadatas

    def _convert_metadata_dict(self, metadata: Dict) -> None:
        """Convert metadata values from strings to appropriate types in-place.
        
        Args:
            metadata: Single metadata dictionary to convert
        """
        for key, value in metadata.items():
            # only attempt to convert strings
            if not isinstance(value, str):
                continue
            else:
                try:
                    metadata[key] = ast.literal_eval(value)
                except Exception:
                    pass


class PersistentChromaRetriever(ChromaRetriever):
    """
    Persistent ChromaDB client/retriever to facilitate sharing of memory from
        multiple agents across sessions.
    Simply changes how the client and collection are initialized. Other
        functionality is inherited from ChromaRetriever.
    """

    def __init__(
        self, 
        directory: Optional[str] = None, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2",
        extend: bool = False
    ):
        """
        Initialize persistent ChromaDB retriever.
        
        :param directory: Directory path for ChromaDB storage. Defaults to
            '~/.chromadb' if None.
        :collection_name: Name of the ChromaDB collection.
        :model_name: SentenceTransformer model name for embeddings.
        :extend: If True, allows initializes client and retriever from
            collection if it exists. Raises error if False and collection
            already exists. This prevents accidental overwriting of
            existing collections.
        """
        if directory is None:
            directory = Path.home() / '.chromadb'
            directory.mkdir(parents=True, exist_ok=True)
        elif isinstance(directory, str):
            directory = Path(directory)

        try:
            directory.resolve(strict=True)
        except FileNotFoundError:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f'Error accessing directory: {e}')        

        # Use PersistentClient instead of regular Client
        self.client = chromadb.PersistentClient(path=str(directory))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name)
        
        existing_collections = [col.name for col in self.client.list_collections()]
        
        if collection_name in existing_collections:
            if extend:
                self.collection = self.client.get_collection(name=collection_name)
            else:
                raise ValueError(
                    f"Collection '{collection_name}' already exists. "
                    "Use extend=True to add to it."
                )
        else:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        self.collection_name = collection_name


class CopiedChromaRetriever(PersistentChromaRetriever):
    """
    ChromaDB retriever that creates a copy of an existing collection
        under to a temporary ChromaDB instance.
    Useful for creating isolated copies of shared starting memory collections.
    """

    def __init__(
        self,
        directory: Optional[str] = None, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2",
        _dest_collection_name: Optional[str] = None,
        _copy_batch_size: int = 10,
    ):
        """
        Initialize the CopiedChromaDB retriever.

        :param directory: Directory path for source ChromaDB storage. If None,
            defaults to '~/.chromadb'.
        :param collection_name: Name of the source ChromaDB collection to copy.
        :param model_name: SentenceTransformer model name for embeddings.
        :param _dest_collection_name: Optional name for the destination
            collection. If None, defaults to '{collection_name}__clone'.
            This parameter is marked as private as the class itself is meant
            for single use and discard db that exists in a temporary so naming
            the copied collection is most likely not needed. 
        :param _copy_batch_size: Number of documents to copy per batch.
            Shouldn't need to be changed normally. 
        """

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name)

        # ensure source is valid
        if directory is None:
            directory = Path.home() / '.chromadb'
            directory.mkdir(parents=True, exist_ok=True)
        elif isinstance(directory, str):
            directory = Path(directory)
        self._src_client = chromadb.PersistentClient(path=str(directory))

        self._src = self._src_client.get_collection(name=collection_name)
        existing_collections = [
            col.name for col in self._src_client.list_collections()]
        if collection_name not in existing_collections:
            raise ValueError(
                f"Collection '{collection_name}' to be copied does not exist."
            )        

        # use temp directory for destination collection
        try:
            self._tmpdir = tempfile.TemporaryDirectory(
                prefix='chromadb_ephemeral_')
            self._tmp_path = Path(self._tmpdir.name)
            self._dst_client = chromadb.PersistentClient(
                path=str(self._tmp_path)
            )
            self.collection_name = (
                _dest_collection_name 
                or f"{collection_name}__clone"
            )
            self.collection = self._dst_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=self._src.metadata
            )
        except Exception as e:
            raise ValueError(f"Error creating temporary ChromaDB: {e}")
        
        try:
            _clone_collection(
                src=self._src,
                dest=self.collection,
                batch_size=_copy_batch_size,
            )
        except Exception as e:
            raise ValueError(f"Error cloning ChromaDB collection: {e}")
        
        atexit.register(self.close)

    def close(self):
        """Cleanup temporary directory."""
        try:
            self._dst_client.delete_collection(self.collection_name)
        except Exception:
            pass
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TurboQuantRetriever(ChromaRetriever):
    """
    ChromaDB retriever with TurboQuant vector compression.
    
    Compresses embeddings from 32-bit float to 2-4 bits per coordinate,
    achieving 8-16x compression while maintaining retrieval quality.
    
    Based on: TurboQuant paper (arXiv:2504.19874v1)
    - Random rotation induces Beta distribution on coordinates
    - Optimal scalar quantization per coordinate
    - QJL for unbiased inner product estimation
    - Near-optimal distortion rate (within 2.7x of theoretical lower bound)
    """
    
    def __init__(
        self, 
        collection_name: str = "memories", 
        model_name: str = "all-MiniLM-L6-v2",
        bit_width: int = 4,
        use_qjl: bool = True,
        embedding_dimension: int = 384
    ):
        """
        Initialize TurboQuant-enabled ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model name
            bit_width: Bits per coordinate (2, 3, or 4 recommended)
            use_qjl: Whether to use QJL for unbiased inner products
            embedding_dimension: Expected embedding dimension (384 for all-MiniLM-L6-v2)
        """
        # Initialize base ChromaRetriever
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        
        # Initialize TurboQuant compressor
        self.compressor = TurboQuantCompressor(
            bit_width=bit_width,
            use_qjl=use_qjl,
            dimension=embedding_dimension
        )
        
        # Track compression statistics
        self.compression_stats = {
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'documents_compressed': 0
        }
        
        print(f"✓ TurboQuantRetriever initialized: {bit_width}-bit, QJL={use_qjl}, "
              f"compression_ratio={self.compressor.get_compression_ratio():.1f}x")
    
    def add_document(self, document: str, metadata: Dict, doc_id: str, embedding: Optional[np.ndarray] = None):
        """
        Add a document to ChromaDB with TurboQuant compression.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
            embedding: Pre-computed embedding (if None, will be computed by ChromaDB)
        """
        # Convert metadata to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # If embedding is provided, compress it with TurboQuant
        if embedding is not None:
            compressed = self.compressor.compress(embedding)
            
            # Store compressed representation in metadata
            processed_metadata['compressed_indices'] = json.dumps(compressed['indices'].tolist())
            processed_metadata['residual_norm'] = str(compressed['residual_norm'])
            processed_metadata['indices_packed'] = str(compressed.get('indices_packed', False))
            processed_metadata['mse_bit_width'] = str(compressed.get('mse_bit_width', self.compressor.mse_bit_width))
            processed_metadata['bit_width'] = str(compressed.get('bit_width', self.compressor.bit_width))
            processed_metadata['embedding_dimension'] = str(compressed.get('dimension', self.compressor.dimension))
            
            if 'qjl_bits' in compressed:
                processed_metadata['qjl_bits'] = json.dumps(compressed['qjl_bits'].tolist())
                processed_metadata['qjl_packed'] = str(compressed.get('qjl_packed', False))
            
            # Update compression statistics
            self.compression_stats['total_original_bytes'] += embedding.nbytes
            compressed_size = compressed['indices'].nbytes
            if 'qjl_bits' in compressed:
                compressed_size += compressed['qjl_bits'].nbytes
            self.compression_stats['total_compressed_bytes'] += compressed_size
            self.compression_stats['documents_compressed'] += 1
        
        # Add to ChromaDB
        self.collection.add(
            documents=[document], 
            metadatas=[processed_metadata], 
            ids=[doc_id]
        )
    
    def search_with_compression(self, query: str, k: int = 5) -> Dict:
        """
        Search for similar documents using TurboQuant compressed embeddings.
        
        If compressed vectors are available in metadata, this computes the
        query embedding and ranks documents with TurboQuant reconstruction.
        It falls back to ChromaDB search for collections without compressed
        vectors or if embedding generation is unavailable.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        try:
            all_docs = self.collection.get(include=["documents", "metadatas"])
            if not all_docs.get("ids"):
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            metadatas = all_docs.get("metadatas") or []
            documents = all_docs.get("documents") or []
            if not any(md and md.get("compressed_indices") for md in metadatas):
                raise ValueError("collection has no compressed embeddings")

            query_embedding = np.asarray(self.embedding_function([query])[0], dtype=np.float32)
            scored = []
            for doc_id, document, metadata in zip(all_docs["ids"], documents, metadatas):
                if not metadata or "compressed_indices" not in metadata:
                    continue
                converted = dict(metadata)
                self._convert_metadata_dict(converted)
                compressed = {
                    "indices": np.asarray(converted["compressed_indices"], dtype=np.uint8),
                    "residual_norm": float(converted["residual_norm"]),
                    "indices_packed": converted.get("indices_packed", True),
                    "mse_bit_width": int(converted.get("mse_bit_width", self.compressor.mse_bit_width)),
                    "bit_width": int(converted.get("bit_width", self.compressor.bit_width)),
                }
                if "qjl_bits" in converted:
                    compressed["qjl_bits"] = np.asarray(converted["qjl_bits"], dtype=np.uint8)
                    compressed["qjl_packed"] = converted.get("qjl_packed", True)

                score = self.compressor.inner_product_with_vector(compressed, query_embedding)
                scored.append((score, doc_id, document, converted))

            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:k]
            return {
                "ids": [[doc_id for _, doc_id, _, _ in top]],
                "documents": [[document for _, _, document, _ in top]],
                "metadatas": [[metadata for _, _, _, metadata in top]],
                # Chroma distances are lower-is-better; expose negative score
                # so existing callers can still sort ascending if needed.
                "distances": [[-score for score, _, _, _ in top]],
            }
        except Exception:
            results = self.collection.query(query_texts=[query], n_results=k)

            if (results is not None) and (results.get("metadatas", [])):
                results["metadatas"] = self._convert_metadata_types(
                    results["metadatas"])

            return results
    
    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression metrics
        """
        if self.compression_stats['documents_compressed'] == 0:
            return {
                'documents_compressed': 0,
                'compression_ratio': 0,
                'total_saved_bytes': 0
            }
        
        actual_ratio = (self.compression_stats['total_original_bytes'] / 
                       self.compression_stats['total_compressed_bytes'])
        saved_bytes = (self.compression_stats['total_original_bytes'] - 
                      self.compression_stats['total_compressed_bytes'])
        
        return {
            'documents_compressed': self.compression_stats['documents_compressed'],
            'original_size_mb': self.compression_stats['total_original_bytes'] / (1024 * 1024),
            'compressed_size_mb': self.compression_stats['total_compressed_bytes'] / (1024 * 1024),
            'compression_ratio': actual_ratio,
            'space_saved_mb': saved_bytes / (1024 * 1024),
            'theoretical_ratio': self.compressor.get_compression_ratio()
        }
    
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB."""
        self.collection.delete(ids=[doc_id])
