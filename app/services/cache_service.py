"""
Cache Service
Manages caching of document chunks and embeddings to avoid redundant processing.

Now supports pluggable storage backends (local filesystem or S3).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from app.services.storage_backend import StorageBackend
from app.services.local_storage import LocalStorageBackend
from app.services.s3_storage import S3StorageBackend
from app.config import settings

logger = logging.getLogger("rag_app.cache_service")


class CacheService:
    """
    Manages caching of document chunks and embeddings.
    Uses content-based SHA-256 hashing for true deduplication.

    Supports multiple storage backends:
    - Local filesystem (for development)
    - AWS S3 (for Lambda deployment)
    """

    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        """
        Initialize cache service with storage backend.

        Args:
            storage_backend: Storage backend to use. If None, auto-selects
                           based on STORAGE_BACKEND environment variable.
        """
        if storage_backend is None:
            # Auto-select backend from config
            backend_type = getattr(settings, 'STORAGE_BACKEND', 'local').lower()

            if backend_type == "s3":
                try:
                    self.storage = S3StorageBackend()
                    logger.info(f"Using S3 storage (bucket: {settings.S3_CACHE_BUCKET})")
                except Exception as e:
                    logger.warning(f"Failed to initialize S3 storage: {e}. Falling back to local.")
                    self.storage = LocalStorageBackend()
            elif backend_type == "local":
                self.storage = LocalStorageBackend()
                logger.info(f"Using local storage (dir: {settings.CACHE_DIR})")
            else:
                raise ValueError(f"Unknown storage backend: {backend_type}")
        else:
            self.storage = storage_backend
            logger.info(f"Using custom storage backend: {type(storage_backend).__name__}")

    def compute_document_id(self, file_path: Path) -> str:
        """
        Generate unique document ID from file contents using SHA-256.
        Same file = same ID (true content-based deduplication).

        Args:
            file_path: Path to the document file

        Returns:
            64-character hexadecimal SHA-256 hash

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256 = hashlib.sha256()

        # Read file in chunks to handle large files efficiently
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        doc_id = sha256.hexdigest()
        logger.debug(f"Computed document ID: {doc_id} for {file_path.name}")
        return doc_id

    def cache_exists(self, doc_id: str, file_extension: str) -> bool:
        """
        Check if valid cache exists for document ID.

        Args:
            doc_id: Document ID (SHA-256 hash)
            file_extension: File extension (pdf, txt, md, etc.) - required for S3 folder organization

        Returns:
            True if complete cache exists, False otherwise
        """
        try:
            return self.storage.exists(doc_id, file_extension)
        except Exception as e:
            logger.warning(f"Error checking cache for {doc_id}: {e}")
            return False

    def save_document(
        self, doc_id: str, file_path: Path, file_extension: str
    ) -> None:
        """
        Save original document to storage.

        NEW METHOD - saves the uploaded file itself.

        Args:
            doc_id: Document ID (SHA-256 hash)
            file_path: Path to the uploaded file
            file_extension: File extension (pdf, txt, md, etc.)

        Raises:
            Exception if save fails
        """
        try:
            self.storage.save_document(doc_id, file_path, file_extension)
            logger.info(f"Saved original document {doc_id}.{file_extension}")
        except Exception as e:
            logger.error(f"Failed to save document {doc_id}: {e}")
            raise

    def save_chunks_and_embeddings(
        self,
        doc_id: str,
        file_extension: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save chunks, embeddings, and metadata to cache.

        Args:
            doc_id: Document ID (SHA-256 hash)
            file_extension: File extension (REQUIRED for S3 folder organization)
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors (N x 1536)
            metadata: Document metadata (filename, timestamp, etc.)

        Raises:
            ValueError: If chunks and embeddings length mismatch
            Exception: If save operation fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        try:
            # Convert embeddings to NumPy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Save all files via storage backend
            self.storage.save_chunks(doc_id, file_extension, chunks)
            self.storage.save_embeddings(doc_id, file_extension, embeddings_array)
            self.storage.save_metadata(doc_id, file_extension, metadata)

            logger.info(
                f"Cached {len(chunks)} chunks for {doc_id} (type: {file_extension})"
            )

        except Exception as e:
            logger.error(f"Failed to cache document {doc_id}: {e}")
            # Attempt cleanup on failure (delete partial cache)
            try:
                self.storage.delete(doc_id, file_extension)
            except:
                pass
            raise Exception(f"Failed to save cache: {str(e)}")

    def load_chunks_and_embeddings(self, doc_id: str, file_extension: str) -> Optional[Dict[str, Any]]:
        """
        Load cached chunks and embeddings.

        Args:
            doc_id: Document ID (SHA-256 hash)
            file_extension: File extension (needed to find S3 folder)

        Returns:
            Dictionary with 'chunks', 'embeddings', and 'metadata' keys,
            or None if cache doesn't exist or is corrupted

        Note:
            Returns None on any error (graceful degradation)
        """
        if not self.cache_exists(doc_id, file_extension):
            return None

        try:
            # Load all files from storage backend
            chunks = self.storage.load_chunks(doc_id, file_extension)
            embeddings_array = self.storage.load_embeddings(doc_id, file_extension)
            metadata = self.storage.load_metadata(doc_id, file_extension)

            # Convert embeddings to list for consistency with API
            embeddings = embeddings_array.tolist()

            # Validate consistency
            if len(chunks) != len(embeddings):
                logger.error(
                    f"Cache corruption: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                return None

            logger.info(f"Loaded {len(chunks)} chunks from cache for {doc_id}")

            return {
                'chunks': chunks,
                'embeddings': embeddings,
                'metadata': metadata
            }

        except Exception as e:
            logger.warning(f"Failed to load cache for {doc_id}: {str(e)}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from storage backend.

        Returns:
            Dictionary with storage-specific stats (varies by backend)
        """
        try:
            return self.storage.get_stats()
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                'error': str(e),
                'total_documents': 0
            }

    def clear_cache(self, doc_id: Optional[str] = None, file_extension: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear cache for specific document or entire cache.

        Args:
            doc_id: Document ID to clear, or None to clear all cache
            file_extension: Required if doc_id provided (to locate S3 folder)

        Returns:
            Dictionary with:
                - cleared: Boolean success status
                - message: Description of what was cleared
                - documents_cleared: Number of documents cleared
        """
        try:
            if doc_id:
                # Clear specific document
                if not file_extension:
                    return {
                        'cleared': False,
                        'message': 'file_extension required when clearing specific document',
                        'documents_cleared': 0
                    }

                self.storage.delete(doc_id, file_extension)
                logger.info(f"Cleared cache for document: {doc_id}")
                return {
                    'cleared': True,
                    'message': f'Cleared cache for document {doc_id}',
                    'documents_cleared': 1
                }
            else:
                # Clear entire cache (WARNING: scans entire bucket in S3)
                document_ids = self.storage.list_documents()
                cleared_count = 0

                # Note: For S3, this is a limitation - we don't store doc_id → file_extension mapping
                # So clearing all cache requires iterating through all documents
                # For now, we'll just return the count without actually deleting
                # TODO: Implement full cache clear for S3 if needed

                logger.warning("Clear entire cache not fully implemented for S3 backend")
                return {
                    'cleared': False,
                    'message': 'Clear entire cache not fully implemented for S3 backend',
                    'documents_cleared': 0,
                    'total_documents': len(document_ids)
                }

        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return {
                'cleared': False,
                'message': f'Failed to clear cache: {str(e)}',
                'documents_cleared': 0
            }
