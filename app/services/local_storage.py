"""
Local filesystem storage backend (for development).

This module provides local file-based storage for document cache.
Use this during development for fast iteration without S3 costs.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from app.services.storage_backend import StorageBackend
from app.config import settings

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """
    Filesystem-based storage for local development.

    Stores files in: data/cached_chunks/{document_id}/
    Each document gets a folder with 4 files:
    - document.{ext} (original file)
    - chunks.json
    - embeddings.npy
    - metadata.json

    For local development, we don't organize by document type (simpler).
    """

    def __init__(self, cache_dir: Path = None):
        """
        Initialize local storage with cache directory.

        Args:
            cache_dir: Path to cache directory (defaults to settings.CACHE_DIR)
        """
        self.cache_dir = cache_dir or Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized with cache_dir: {self.cache_dir}")

    def _get_document_path(self, document_id: str) -> Path:
        """
        Get the folder path for a document.

        Args:
            document_id: SHA-256 hash of document

        Returns:
            Path to document folder
        """
        return self.cache_dir / document_id

    def exists(self, document_id: str, file_extension: str) -> bool:
        """
        Check if all cache files exist for this document.

        Note: We check for chunks, embeddings, and metadata.
        Original document is optional (backward compatibility with existing cache).

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used in local storage, but kept for interface compatibility)

        Returns:
            True if all required cache files exist
        """
        doc_path = self._get_document_path(document_id)

        # Check for required cache files (original document is optional for backward compatibility)
        required_files = [
            doc_path / "chunks.json",
            doc_path / "embeddings.npy",
            doc_path / "metadata.json"
        ]

        exists = all(f.exists() for f in required_files)

        if exists:
            logger.debug(f"Cache hit for document {document_id}")
        else:
            logger.debug(f"Cache miss for document {document_id}")

        return exists

    def save_document(self, document_id: str, file_path: Path, file_extension: str) -> None:
        """
        Save original document to local storage.

        Args:
            document_id: SHA-256 hash of document
            file_path: Path to the uploaded file
            file_extension: File extension (pdf, txt, md, etc.)
        """
        doc_path = self._get_document_path(document_id)
        doc_path.mkdir(parents=True, exist_ok=True)

        # Copy original file to cache folder
        destination = doc_path / f"document.{file_extension}"
        shutil.copy2(file_path, destination)
        logger.info(f"Saved original document to {destination}")

    def save_chunks(self, document_id: str, file_extension: str, chunks: List[Dict]) -> None:
        """
        Save chunks.json to local storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)
            chunks: List of document chunks
        """
        doc_path = self._get_document_path(document_id)
        doc_path.mkdir(parents=True, exist_ok=True)

        chunks_file = doc_path / "chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(chunks, f, indent=2)

        logger.debug(f"Saved {len(chunks)} chunks to {chunks_file}")

    def save_embeddings(self, document_id: str, file_extension: str, embeddings: np.ndarray) -> None:
        """
        Save embeddings.npy to local storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)
            embeddings: NumPy array of shape (num_chunks, 1536)
        """
        doc_path = self._get_document_path(document_id)
        doc_path.mkdir(parents=True, exist_ok=True)

        embeddings_file = doc_path / "embeddings.npy"
        np.save(embeddings_file, embeddings)

        logger.debug(f"Saved embeddings {embeddings.shape} to {embeddings_file}")

    def save_metadata(self, document_id: str, file_extension: str, metadata: Dict) -> None:
        """
        Save metadata.json to local storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)
            metadata: Document metadata
        """
        doc_path = self._get_document_path(document_id)
        doc_path.mkdir(parents=True, exist_ok=True)

        metadata_file = doc_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved metadata to {metadata_file}")

    def load_chunks(self, document_id: str, file_extension: str) -> List[Dict]:
        """
        Load chunks.json from local storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)

        Returns:
            List of document chunks

        Raises:
            FileNotFoundError if chunks file doesn't exist
        """
        chunks_file = self._get_document_path(document_id) / "chunks.json"

        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        with open(chunks_file) as f:
            chunks = json.load(f)

        logger.debug(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks

    def load_embeddings(self, document_id: str, file_extension: str) -> np.ndarray:
        """
        Load embeddings.npy from local storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)

        Returns:
            NumPy array of embeddings

        Raises:
            FileNotFoundError if embeddings file doesn't exist
        """
        embeddings_file = self._get_document_path(document_id) / "embeddings.npy"

        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        embeddings = np.load(embeddings_file)
        logger.debug(f"Loaded embeddings {embeddings.shape} from {embeddings_file}")
        return embeddings

    def load_metadata(self, document_id: str, file_extension: str) -> Dict:
        """
        Load metadata.json from local storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)

        Returns:
            Document metadata dictionary

        Raises:
            FileNotFoundError if metadata file doesn't exist
        """
        metadata_file = self._get_document_path(document_id) / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        logger.debug(f"Loaded metadata from {metadata_file}")
        return metadata

    def delete(self, document_id: str, file_extension: str) -> None:
        """
        Delete all cache files for a document.

        Deletes the entire folder including all files (document, chunks, embeddings, metadata).

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (not used, kept for interface)
        """
        doc_path = self._get_document_path(document_id)

        if doc_path.exists():
            shutil.rmtree(doc_path)
            logger.info(f"Deleted cache for document {document_id}")
        else:
            logger.warning(f"Attempted to delete non-existent document {document_id}")

    def list_documents(self) -> List[str]:
        """
        List all cached document IDs.

        Returns:
            List of document IDs (SHA-256 hashes)
        """
        if not self.cache_dir.exists():
            return []

        # List all directories in cache_dir (each is a document ID)
        document_ids = [d.name for d in self.cache_dir.iterdir() if d.is_dir()]

        logger.debug(f"Found {len(document_ids)} cached documents")
        return document_ids

    def get_stats(self) -> Dict:
        """
        Get local storage statistics.

        Returns:
            Dictionary with storage stats (backend, cache_dir, total_documents, total_files, total_size_mb)
        """
        total_size = 0
        total_files = 0
        documents_count = 0

        if self.cache_dir.exists():
            for doc_dir in self.cache_dir.iterdir():
                if doc_dir.is_dir():
                    documents_count += 1
                    for file in doc_dir.iterdir():
                        if file.is_file():
                            total_size += file.stat().st_size
                            total_files += 1

        stats = {
            "backend": "local",
            "cache_dir": str(self.cache_dir),
            "total_documents": documents_count,
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

        logger.info(f"Local storage stats: {stats}")
        return stats
