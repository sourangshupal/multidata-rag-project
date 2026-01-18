"""
Unit tests for storage backends (LocalStorage and S3Storage).

Tests both local filesystem and S3 storage implementations to ensure
they correctly implement the StorageBackend interface.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.local_storage import LocalStorageBackend
from app.services.s3_storage import S3StorageBackend


# Test fixtures

@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "text": "This is the first chunk of text.",
            "metadata": {"page": 1, "tokens": 7}
        },
        {
            "text": "This is the second chunk of text.",
            "metadata": {"page": 1, "tokens": 7}
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings array (2 chunks x 1536 dimensions)."""
    return np.random.rand(2, 1536).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Sample document metadata."""
    return {
        "filename": "test.pdf",
        "cached_at": "2026-01-18T00:00:00Z",
        "total_chunks": 2,
        "total_tokens": 14
    }


@pytest.fixture
def temp_document(tmp_path):
    """Create a temporary test document file."""
    doc_path = tmp_path / "test_document.pdf"
    doc_path.write_text("This is a test document content.")
    return doc_path


# Tests for LocalStorageBackend

class TestLocalStorageBackend:
    """Tests for local filesystem storage backend."""

    @pytest.fixture
    def local_storage(self, tmp_path):
        """Create a LocalStorageBackend with temporary directory."""
        return LocalStorageBackend(cache_dir=tmp_path)

    def test_initialization(self, local_storage, tmp_path):
        """Test that LocalStorageBackend initializes correctly."""
        assert local_storage.cache_dir == tmp_path
        assert local_storage.cache_dir.exists()

    def test_save_and_load_chunks(self, local_storage, sample_chunks):
        """Test saving and loading chunks.json."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save chunks
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Verify file exists
        chunks_file = local_storage._get_document_path(doc_id) / "chunks.json"
        assert chunks_file.exists()

        # Load and verify
        loaded_chunks = local_storage.load_chunks(doc_id, file_extension)
        assert loaded_chunks == sample_chunks

    def test_save_and_load_embeddings(self, local_storage, sample_embeddings):
        """Test saving and loading embeddings.npy."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save embeddings
        local_storage.save_embeddings(doc_id, file_extension, sample_embeddings)

        # Verify file exists
        embeddings_file = local_storage._get_document_path(doc_id) / "embeddings.npy"
        assert embeddings_file.exists()

        # Load and verify
        loaded_embeddings = local_storage.load_embeddings(doc_id, file_extension)
        assert np.array_equal(loaded_embeddings, sample_embeddings)

    def test_save_and_load_metadata(self, local_storage, sample_metadata):
        """Test saving and loading metadata.json."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save metadata
        local_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Verify file exists
        metadata_file = local_storage._get_document_path(doc_id) / "metadata.json"
        assert metadata_file.exists()

        # Load and verify
        loaded_metadata = local_storage.load_metadata(doc_id, file_extension)
        assert loaded_metadata == sample_metadata

    def test_save_and_load_document(self, local_storage, temp_document):
        """Test saving and loading original document."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save document
        local_storage.save_document(doc_id, temp_document, file_extension)

        # Verify file exists
        saved_doc = local_storage._get_document_path(doc_id) / f"document.{file_extension}"
        assert saved_doc.exists()

        # Verify content
        assert saved_doc.read_text() == temp_document.read_text()

    def test_exists_all_files(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test exists() returns True when all files present."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Initially should not exist
        assert not local_storage.exists(doc_id, file_extension)

        # Save all files
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)
        local_storage.save_embeddings(doc_id, file_extension, sample_embeddings)
        local_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Now should exist
        assert local_storage.exists(doc_id, file_extension)

    def test_exists_partial_files(self, local_storage, sample_chunks):
        """Test exists() returns False when only some files present."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save only chunks
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Should not exist (missing embeddings and metadata)
        assert not local_storage.exists(doc_id, file_extension)

    def test_delete(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test deleting all files for a document."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save all files
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)
        local_storage.save_embeddings(doc_id, file_extension, sample_embeddings)
        local_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Verify exists
        assert local_storage.exists(doc_id, file_extension)

        # Delete
        local_storage.delete(doc_id, file_extension)

        # Verify deleted
        assert not local_storage.exists(doc_id, file_extension)
        assert not local_storage._get_document_path(doc_id).exists()

    def test_list_documents(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test listing all cached documents."""
        # Initially empty
        assert local_storage.list_documents() == []

        # Add multiple documents
        for i in range(3):
            doc_id = f"doc_{i}"
            local_storage.save_chunks(doc_id, "pdf", sample_chunks)
            local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
            local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        # Verify list
        doc_list = local_storage.list_documents()
        assert len(doc_list) == 3
        assert "doc_0" in doc_list
        assert "doc_1" in doc_list
        assert "doc_2" in doc_list

    def test_get_stats(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test getting storage statistics."""
        # Add a document
        doc_id = "test_doc"
        local_storage.save_chunks(doc_id, "pdf", sample_chunks)
        local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        # Get stats
        stats = local_storage.get_stats()

        assert stats["backend"] == "local"
        assert stats["cache_dir"] == str(local_storage.cache_dir)
        assert stats["total_documents"] == 1
        assert stats["total_files"] == 3  # chunks, embeddings, metadata
        assert stats["total_size_mb"] > 0


# Tests for S3StorageBackend (using moto for mocking)

class TestS3StorageBackend:
    """Tests for S3 storage backend (mocked with moto)."""

    @pytest.fixture
    def s3_storage(self):
        """Create S3StorageBackend with mocked S3."""
        # Use moto to mock S3
        from moto import mock_aws
        import boto3

        with mock_aws():
            # Create mock S3 bucket
            s3_client = boto3.client('s3', region_name='us-east-1')
            s3_client.create_bucket(Bucket='test-bucket')

            # Create S3 storage backend
            storage = S3StorageBackend(bucket_name='test-bucket')
            yield storage

    def test_initialization(self, s3_storage):
        """Test S3StorageBackend initialization."""
        assert s3_storage.bucket_name == 'test-bucket'
        assert s3_storage.region == 'us-east-1'

    def test_save_and_load_chunks(self, s3_storage, sample_chunks):
        """Test saving and loading chunks to S3."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save chunks
        s3_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Load and verify
        loaded_chunks = s3_storage.load_chunks(doc_id, file_extension)
        assert loaded_chunks == sample_chunks

    def test_save_and_load_embeddings(self, s3_storage, sample_embeddings):
        """Test saving and loading embeddings to S3."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save embeddings
        s3_storage.save_embeddings(doc_id, file_extension, sample_embeddings)

        # Load and verify
        loaded_embeddings = s3_storage.load_embeddings(doc_id, file_extension)
        assert np.array_equal(loaded_embeddings, sample_embeddings)

    def test_save_and_load_metadata(self, s3_storage, sample_metadata):
        """Test saving and loading metadata to S3."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save metadata
        s3_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Load and verify
        loaded_metadata = s3_storage.load_metadata(doc_id, file_extension)
        assert loaded_metadata == sample_metadata

    def test_save_and_load_document(self, s3_storage, temp_document):
        """Test saving and loading original document to S3."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save document
        s3_storage.save_document(doc_id, temp_document, file_extension)

        # Verify object exists
        key = s3_storage._get_s3_key(doc_id, file_extension, f"document.{file_extension}")
        assert s3_storage._object_exists(key)

    def test_exists_all_files(self, s3_storage, sample_chunks, sample_embeddings, sample_metadata, temp_document):
        """Test exists() returns True when all S3 files present."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Initially should not exist
        assert not s3_storage.exists(doc_id, file_extension)

        # Save all files
        s3_storage.save_document(doc_id, temp_document, file_extension)
        s3_storage.save_chunks(doc_id, file_extension, sample_chunks)
        s3_storage.save_embeddings(doc_id, file_extension, sample_embeddings)
        s3_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Now should exist
        assert s3_storage.exists(doc_id, file_extension)

    def test_exists_partial_files(self, s3_storage, sample_chunks):
        """Test exists() returns False when only some S3 files present."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save only chunks
        s3_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Should not exist (missing other files)
        assert not s3_storage.exists(doc_id, file_extension)

    def test_delete(self, s3_storage, sample_chunks, sample_embeddings, sample_metadata, temp_document):
        """Test deleting all S3 files for a document."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save all files
        s3_storage.save_document(doc_id, temp_document, file_extension)
        s3_storage.save_chunks(doc_id, file_extension, sample_chunks)
        s3_storage.save_embeddings(doc_id, file_extension, sample_embeddings)
        s3_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Verify exists
        assert s3_storage.exists(doc_id, file_extension)

        # Delete
        s3_storage.delete(doc_id, file_extension)

        # Verify deleted
        assert not s3_storage.exists(doc_id, file_extension)

    def test_s3_folder_structure(self, s3_storage, sample_chunks, temp_document):
        """Test S3 folder structure is organized by document type."""
        doc_id = "test_s3_doc"
        file_extension = "pdf"

        # Save document and chunks
        s3_storage.save_document(doc_id, temp_document, file_extension)
        s3_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Verify keys follow pattern: {doc_type}/{doc_id}/{filename}
        doc_key = s3_storage._get_s3_key(doc_id, file_extension, f"document.{file_extension}")
        chunks_key = s3_storage._get_s3_key(doc_id, file_extension, "chunks.json")

        assert doc_key == f"pdf/{doc_id}/document.pdf"
        assert chunks_key == f"pdf/{doc_id}/chunks.json"

    def test_list_documents(self, s3_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test listing documents across all S3 folders."""
        # Add multiple documents
        for i in range(3):
            doc_id = f"doc_{i}"
            file_ext = "pdf" if i % 2 == 0 else "txt"
            s3_storage.save_chunks(doc_id, file_ext, sample_chunks)
            s3_storage.save_embeddings(doc_id, file_ext, sample_embeddings)
            s3_storage.save_metadata(doc_id, file_ext, sample_metadata)

        # List documents
        doc_list = s3_storage.list_documents()

        assert len(doc_list) == 3
        assert "doc_0" in doc_list
        assert "doc_1" in doc_list
        assert "doc_2" in doc_list

    def test_get_stats(self, s3_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test getting S3 storage statistics."""
        # Add a document
        doc_id = "test_doc"
        s3_storage.save_chunks(doc_id, "pdf", sample_chunks)
        s3_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        s3_storage.save_metadata(doc_id, "pdf", sample_metadata)

        # Get stats
        stats = s3_storage.get_stats()

        assert stats["backend"] == "s3"
        assert stats["bucket"] == "test-bucket"
        assert stats["region"] == "us-east-1"
        assert stats["total_documents"] == 1
        assert stats["total_objects"] >= 3  # At least chunks, embeddings, metadata
        assert stats["total_size_mb"] > 0
        assert "documents_by_type" in stats


# Integration tests

class TestStorageBackendCompatibility:
    """Test that both backends implement the same interface correctly."""

    @pytest.fixture(params=["local", "s3"])
    def storage_backend(self, request, tmp_path):
        """Parametrized fixture to test both backends."""
        if request.param == "local":
            return LocalStorageBackend(cache_dir=tmp_path)
        elif request.param == "s3":
            from moto import mock_aws
            import boto3

            with mock_aws():
                s3_client = boto3.client('s3', region_name='us-east-1')
                s3_client.create_bucket(Bucket='test-bucket')
                yield S3StorageBackend(bucket_name='test-bucket')

    def test_interface_compatibility(self, storage_backend, sample_chunks, sample_embeddings, sample_metadata, temp_document):
        """Test that both backends implement the same interface."""
        doc_id = "compat_test"
        file_ext = "pdf"

        # Test save operations
        storage_backend.save_document(doc_id, temp_document, file_ext)
        storage_backend.save_chunks(doc_id, file_ext, sample_chunks)
        storage_backend.save_embeddings(doc_id, file_ext, sample_embeddings)
        storage_backend.save_metadata(doc_id, file_ext, sample_metadata)

        # Test exists
        assert storage_backend.exists(doc_id, file_ext)

        # Test load operations
        loaded_chunks = storage_backend.load_chunks(doc_id, file_ext)
        loaded_embeddings = storage_backend.load_embeddings(doc_id, file_ext)
        loaded_metadata = storage_backend.load_metadata(doc_id, file_ext)

        # Verify loaded data
        assert loaded_chunks == sample_chunks
        assert np.array_equal(loaded_embeddings, sample_embeddings)
        assert loaded_metadata == sample_metadata

        # Test delete
        storage_backend.delete(doc_id, file_ext)
        assert not storage_backend.exists(doc_id, file_ext)
