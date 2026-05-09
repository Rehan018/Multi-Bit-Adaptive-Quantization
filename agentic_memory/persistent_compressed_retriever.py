"""
Persistent Memory with TurboQuant Compression

Combines persistent storage (survives restarts) with TurboQuant compression
(8x-16x size reduction) for scalable, cost-effective long-term memory.

Key Features:
- Persistent storage across sessions
- Automatic compression on save
- Automatic decompression on load
- Compression statistics tracking
- Migration support for uncompressed data
- Space-efficient for large-scale deployments
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .turboquant import TurboQuantCompressor

logger = logging.getLogger(__name__)


class PersistentCompressedRetriever:
    """
    Persistent ChromaDB retriever with TurboQuant compression.
    
    Combines the benefits of:
    1. PersistentChromaRetriever: Survives restarts, shared across sessions
    2. TurboQuantRetriever: 8x-16x compression, reduced storage costs
    
    Ideal for:
    - Long-term memory storage
    - Multi-agent systems sharing memories
    - Cost-sensitive deployments
    - Large-scale memory banks (>10K memories)
    """
    
    def __init__(
        self,
        directory: Optional[str] = None,
        collection_name: str = "compressed_memories",
        model_name: str = "all-MiniLM-L6-v2",
        bit_width: int = 4,
        use_qjl: bool = True,
        embedding_dimension: int = 384,
        extend: bool = False,
        auto_compress: bool = True
    ):
        """
        Initialize persistent compressed retriever.
        
        Args:
            directory: Directory for ChromaDB storage. Defaults to ~/.chromadb
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model for embeddings
            bit_width: Bits per coordinate (2, 3, or 4)
            use_qjl: Whether to use QJL for unbiased inner products
            embedding_dimension: Expected embedding dimension (384 for all-MiniLM)
            extend: If True, extends existing collection. Raises error if False and exists
            auto_compress: If True, automatically compress embeddings on add
        """
        # Setup directory
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
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=str(directory))
        self.storage_path = str(directory)  # Store path for stats file
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection_name = collection_name
        
        # Get or create collection
        existing_collections = [col.name for col in self.client.list_collections()]
        
        if collection_name in existing_collections:
            if extend:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Extended existing collection: {collection_name}")
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
            logger.info(f"Created new collection: {collection_name}")
        
        # Initialize TurboQuant compressor
        self.compressor = TurboQuantCompressor(
            bit_width=bit_width,
            use_qjl=use_qjl,
            dimension=embedding_dimension
        )
        
        self.auto_compress = auto_compress
        self.bit_width = bit_width
        self.use_qjl = use_qjl
        
        # Compression statistics
        self.stats = {
            'total_documents': 0,
            'compressed_documents': 0,
            'uncompressed_documents': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0
        }
        
        # Load existing stats
        self._load_stats()
        
        logger.info(
            f"PersistentCompressedRetriever initialized: "
            f"bit_width={bit_width}, QJL={use_qjl}, "
            f"compression_ratio={self.compressor.get_compression_ratio():.1f}x, "
            f"auto_compress={auto_compress}"
        )
    
    def add_document(
        self,
        document: str,
        metadata: Dict,
        doc_id: str,
        embedding: Optional[np.ndarray] = None,
        force_compress: bool = None
    ):
        """
        Add a document with optional TurboQuant compression.
        
        Args:
            document: Text content
            metadata: Document metadata
            doc_id: Unique document ID
            embedding: Pre-computed embedding (if None, ChromaDB computes it)
            force_compress: Override auto_compress setting
        """
        should_compress = force_compress if force_compress is not None else self.auto_compress
        
        # Process metadata
        processed_metadata = self._process_metadata(metadata)
        
        # Compress embedding if provided and compression enabled
        if embedding is not None and should_compress:
            compressed = self.compressor.compress(embedding)
            
            # Store compressed representation
            processed_metadata['compressed_indices'] = json.dumps(
                compressed['indices'].tolist()
            )
            processed_metadata['residual_norm'] = str(compressed['residual_norm'])
            processed_metadata['indices_packed'] = str(compressed.get('indices_packed', False))
            processed_metadata['mse_bit_width'] = str(compressed.get('mse_bit_width', self.compressor.mse_bit_width))
            
            if 'qjl_bits' in compressed:
                processed_metadata['qjl_bits'] = json.dumps(
                    compressed['qjl_bits'].tolist()
                )
                processed_metadata['qjl_packed'] = str(compressed.get('qjl_packed', False))
            
            processed_metadata['compression_enabled'] = 'true'
            processed_metadata['bit_width'] = str(self.bit_width)
            
            # Update statistics
            self.stats['total_original_bytes'] += embedding.nbytes
            compressed_size = compressed['indices'].nbytes
            if 'qjl_bits' in compressed:
                compressed_size += compressed['qjl_bits'].nbytes
            self.stats['total_compressed_bytes'] += compressed_size
            self.stats['compressed_documents'] += 1
        
        elif embedding is not None:
            # Store uncompressed but mark it
            processed_metadata['compression_enabled'] = 'false'
            self.stats['uncompressed_documents'] += 1
        
        # Add to ChromaDB
        self.collection.add(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
        self.stats['total_documents'] += 1
        self._save_stats()
    
    def search(
        self,
        query: str,
        k: int = 5,
        decompress_results: bool = True
    ) -> Dict:
        """
        Search for documents with optional decompression.
        
        Args:
            query: Search query text
            k: Number of results to return
            decompress_results: If True, decompress embeddings in results
            
        Returns:
            Search results with documents, metadatas, ids, distances
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        if results and results.get("metadatas"):
            results["metadatas"] = self._convert_metadata_types(
                results["metadatas"]
            )
            
            # Optionally decompress embeddings
            if decompress_results and results.get('embeddings'):
                results['embeddings'] = self._decompress_batch(
                    results['metadatas'][0]
                )
        
        return results
    
    def get_document(
        self,
        doc_id: str,
        decompress: bool = True
    ) -> Optional[Dict]:
        """
        Retrieve a single document with optional decompression.
        
        Args:
            doc_id: Document ID
            decompress: If True, decompress embedding
            
        Returns:
            Document dict or None if not found
        """
        result = self.collection.get(ids=[doc_id])
        
        if not result['ids']:
            return None
        
        doc = {
            'id': result['ids'][0],
            'document': result['documents'][0],
            'metadata': result['metadatas'][0]
        }
        
        # Convert metadata types
        doc['metadata'] = self._convert_single_metadata(doc['metadata'])
        
        # Decompress if needed
        if decompress and result.get('embeddings'):
            doc['embedding'] = self._decompress_single(
                doc['metadata'],
                result['embeddings'][0] if result.get('embeddings') else None
            )
        
        return doc
    
    def delete_document(self, doc_id: str):
        """Delete a document."""
        self.collection.delete(ids=[doc_id])
        self.stats['total_documents'] -= 1
        self._save_stats()
    
    def get_compression_stats(self) -> Dict:
        """
        Get comprehensive compression statistics.
        
        Returns:
            Dictionary with compression metrics
        """
        if self.stats['compressed_documents'] == 0:
            return {
                'total_documents': self.stats['total_documents'],
                'compressed_documents': 0,
                'uncompressed_documents': self.stats['uncompressed_documents'],
                'compression_ratio': 0,
                'original_size_mb': 0,
                'compressed_size_mb': 0,
                'space_saved_mb': 0,
                'theoretical_ratio': self.compressor.get_compression_ratio()
            }
        
        actual_ratio = (self.stats['total_original_bytes'] / 
                       max(self.stats['total_compressed_bytes'], 1))
        saved_bytes = (self.stats['total_original_bytes'] - 
                      self.stats['total_compressed_bytes'])
        
        return {
            'total_documents': self.stats['total_documents'],
            'compressed_documents': self.stats['compressed_documents'],
            'uncompressed_documents': self.stats['uncompressed_documents'],
            'compression_percentage': (
                self.stats['compressed_documents'] / 
                max(self.stats['total_documents'], 1) * 100
            ),
            'compression_ratio': actual_ratio,
            'original_size_mb': self.stats['total_original_bytes'] / (1024 * 1024),
            'compressed_size_mb': self.stats['total_compressed_bytes'] / (1024 * 1024),
            'space_saved_mb': saved_bytes / (1024 * 1024),
            'theoretical_ratio': self.compressor.get_compression_ratio(),
            'efficiency': actual_ratio / self.compressor.get_compression_ratio() * 100
        }
    
    def migrate_to_compressed(
        self,
        batch_size: int = 100,
        progress_callback=None
    ) -> Dict:
        """
        Migrate existing uncompressed documents to compressed format.
        
        This re-processes all documents, compressing their embeddings.
        
        Args:
            batch_size: Number of documents to process per batch
            progress_callback: Optional callback function(batch_num, total_batches)
            
        Returns:
            Migration statistics
        """
        # Get all documents
        all_docs = self.collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        total_docs = len(all_docs['ids'])
        migrated_count = 0
        skipped_count = 0
        
        logger.info(f"Starting migration of {total_docs} documents...")
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            
            for j in range(i, batch_end):
                doc_id = all_docs['ids'][j]
                metadata = all_docs['metadatas'][j]
                
                # Skip if already compressed
                if metadata.get('compression_enabled') == 'true':
                    skipped_count += 1
                    continue
                
                # Get embedding
                embedding = all_docs['embeddings'][j] if all_docs.get('embeddings') is not None else None
                
                if embedding is None:
                    logger.warning(f"No embedding for {doc_id}, skipping")
                    skipped_count += 1
                    continue
                
                # Compress and update
                compressed = self.compressor.compress(np.array(embedding))
                
                # Update metadata with compressed data
                update_metadata = {
                    'compressed_indices': json.dumps(compressed['indices'].tolist()),
                    'residual_norm': str(compressed['residual_norm']),
                    'compression_enabled': 'true',
                    'bit_width': str(self.bit_width),
                    'indices_packed': str(compressed.get('indices_packed', False)),
                    'mse_bit_width': str(compressed.get('mse_bit_width', self.compressor.mse_bit_width))
                }
                
                if 'qjl_bits' in compressed:
                    update_metadata['qjl_bits'] = json.dumps(
                        compressed['qjl_bits'].tolist()
                    )
                    update_metadata['qjl_packed'] = str(compressed.get('qjl_packed', False))
                
                # Update in collection
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[update_metadata]
                )
                
                migrated_count += 1
                
                # Update stats
                emb_array = np.array(embedding)
                self.stats['total_original_bytes'] += emb_array.nbytes
                compressed_size = compressed['indices'].nbytes
                if 'qjl_bits' in compressed:
                    compressed_size += compressed['qjl_bits'].nbytes
                self.stats['total_compressed_bytes'] += compressed_size
                self.stats['compressed_documents'] += 1
            
            # Progress callback
            if progress_callback:
                batch_num = (i // batch_size) + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                progress_callback(batch_num, total_batches)
            
            logger.info(f"Migrated {migrated_count}/{total_docs} documents...")
        
        self._save_stats()
        
        migration_stats = {
            'total_documents': total_docs,
            'migrated': migrated_count,
            'skipped': skipped_count,
            'compression_stats': self.get_compression_stats()
        }
        
        logger.info(
            f"Migration complete: {migrated_count} migrated, "
            f"{skipped_count} skipped"
        )
        
        return migration_stats
    
    def _process_metadata(self, metadata: Dict) -> Dict:
        """Convert metadata to serializable format."""
        processed = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed[key] = json.dumps(value)
            else:
                processed[key] = str(value)
        return processed
    
    def _convert_metadata_types(
        self,
        metadatas: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Convert string metadata back to original types."""
        converted = []
        for query_metadatas in metadatas:
            query_converted = []
            for metadata_dict in query_metadatas:
                query_converted.append(
                    self._convert_single_metadata(metadata_dict)
                )
            converted.append(query_converted)
        return converted
    
    def _convert_single_metadata(self, metadata: Dict) -> Dict:
        """Convert a single metadata dict."""
        import ast
        converted = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    converted[key] = ast.literal_eval(value)
                except Exception:
                    converted[key] = value
            else:
                converted[key] = value
        return converted
    
    def _decompress_single(
        self,
        metadata: Dict,
        original_embedding: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Decompress a single embedding from metadata."""
        if metadata.get('compression_enabled') != 'true':
            return original_embedding
        
        try:
            indices = np.array(metadata['compressed_indices'])
            residual_norm = float(metadata['residual_norm'])
            
            compressed = {
                'indices': indices,
                'residual_norm': residual_norm,
                'indices_packed': metadata.get('indices_packed', True),
                'mse_bit_width': int(metadata.get('mse_bit_width', self.compressor.mse_bit_width))
            }
            
            if 'qjl_bits' in metadata:
                compressed['qjl_bits'] = np.array(metadata['qjl_bits'])
                compressed['qjl_packed'] = metadata.get('qjl_packed', True)
            
            return self.compressor.decompress(compressed)
        
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return original_embedding
    
    def _decompress_batch(
        self,
        metadatas: List[Dict]
    ) -> List[Optional[np.ndarray]]:
        """Decompress a batch of embeddings."""
        embeddings = []
        for metadata in metadatas:
            emb = self._decompress_single(metadata)
            embeddings.append(emb)
        return embeddings
    
    def _save_stats(self):
        """Save compression statistics to file."""
        stats_file = Path(self.storage_path) / f"{self.collection_name}_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save stats: {e}")
    
    def _load_stats(self):
        """Load compression statistics from file."""
        stats_file = Path(self.storage_path) / f"{self.collection_name}_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                    logger.info(f"Loaded stats: {self.stats['total_documents']} documents")
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}")


def create_persistent_compressed_retriever(
    directory: Optional[str] = None,
    collection_name: str = "memories",
    **kwargs
) -> PersistentCompressedRetriever:
    """
    Convenience function to create a persistent compressed retriever.
    
    Args:
        directory: Storage directory
        collection_name: Collection name
        **kwargs: Additional arguments passed to PersistentCompressedRetriever
        
    Returns:
        Configured PersistentCompressedRetriever instance
    """
    return PersistentCompressedRetriever(
        directory=directory,
        collection_name=collection_name,
        **kwargs
    )
