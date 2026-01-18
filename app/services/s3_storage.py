"""
AWS S3 storage backend for Lambda deployment.

This stores documents in S3 organized by file type:
- s3://bucket/pdf/{hash}/document.pdf
- s3://bucket/pdf/{hash}/chunks.json
- s3://bucket/pdf/{hash}/embeddings.npy
- s3://bucket/pdf/{hash}/metadata.json

Why S3?
- AWS Lambda has ephemeral filesystem (only /tmp is writable)
- S3 provides persistent, durable storage
- Lambda can access S3 via IAM role (no credentials needed)
- ~200ms latency is acceptable for cache operations
"""

import json
import io
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from app.services.storage_backend import StorageBackend
from app.config import settings

logger = logging.getLogger(__name__)


class S3StorageBackend(StorageBackend):
    """
    S3-based storage for production Lambda deployment.

    Organizes documents by type in S3:
    - pdf/{doc_id}/document.pdf, chunks.json, embeddings.npy, metadata.json
    - txt/{doc_id}/document.txt, chunks.json, embeddings.npy, metadata.json
    - markdown/{doc_id}/document.md, chunks.json, embeddings.npy, metadata.json

    Each document gets 4 files: original document + cache files.
    """

    def __init__(self, bucket_name: str = None):
        """
        Initialize S3 storage.

        Args:
            bucket_name: S3 bucket name (defaults to settings.S3_CACHE_BUCKET)

        Raises:
            ValueError if bucket doesn't exist
            PermissionError if access denied to bucket
        """
        self.bucket_name = bucket_name or settings.S3_CACHE_BUCKET
        self.region = settings.AWS_REGION

        # Configure boto3 with retry logic (exponential backoff)
        boto_config = Config(
            region_name=self.region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'  # Handles throttling automatically
            }
        )

        # Create boto3 client (uses IAM role in Lambda, or AWS credentials locally)
        self.s3_client = boto3.client('s3', config=boto_config)

        # Check bucket exists on startup (fail fast if misconfigured)
        self._validate_bucket()

        logger.info(f"S3Storage initialized with bucket: {self.bucket_name} (region: {self.region})")

    def _validate_bucket(self) -> None:
        """
        Check if S3 bucket exists and is accessible.

        Raises:
            ValueError if bucket doesn't exist
            PermissionError if access denied
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                raise PermissionError(f"Access denied to S3 bucket '{self.bucket_name}'")
            raise

    def _get_s3_key(self, document_id: str, file_extension: str, filename: str) -> str:
        """
        Generate S3 key with folder structure.

        Pattern: {doc_type}/{doc_id}/{filename}

        Examples:
            pdf/abc123def456/document.pdf
            pdf/abc123def456/chunks.json
            txt/xyz789/document.txt

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (pdf, txt, md, etc.)
            filename: File name (document.pdf, chunks.json, etc.)

        Returns:
            S3 key string
        """
        return f"{file_extension}/{document_id}/{filename}"

    def _object_exists(self, key: str) -> bool:
        """
        Check if S3 object exists (using HEAD request - doesn't download).

        Args:
            key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            # Re-raise other errors (permissions, etc.)
            raise

    def exists(self, document_id: str, file_extension: str) -> bool:
        """
        Check if ALL 4 files exist in S3 for this document.

        Returns True only if all files present (all-or-nothing).

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (pdf, txt, md, etc.)

        Returns:
            True if all 4 files exist
        """
        required_files = [
            f"document.{file_extension}",  # Original file
            "chunks.json",
            "embeddings.npy",
            "metadata.json"
        ]

        # Check each file exists
        for filename in required_files:
            key = self._get_s3_key(document_id, file_extension, filename)
            if not self._object_exists(key):
                logger.debug(f"S3 cache miss for {document_id} (missing: {filename})")
                return False

        logger.debug(f"S3 cache hit for {document_id}")
        return True

    def save_document(self, document_id: str, file_path: Path, file_extension: str) -> None:
        """
        Upload original document to S3.

        Example: s3://bucket/pdf/{doc_id}/document.pdf

        Args:
            document_id: SHA-256 hash of document
            file_path: Path to the uploaded file
            file_extension: File extension (pdf, txt, md, etc.)

        Raises:
            Exception if upload fails
        """
        key = self._get_s3_key(document_id, file_extension, f"document.{file_extension}")

        try:
            # Read file and upload to S3
            with open(file_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=f.read(),
                    ServerSideEncryption='AES256'  # Encrypt at rest
                )
            logger.info(f"Uploaded original document to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to upload document to S3: {e}")
            raise

    def save_chunks(self, document_id: str, file_extension: str, chunks: List[Dict]) -> None:
        """
        Save chunks to S3 as JSON.

        Example: s3://bucket/pdf/{doc_id}/chunks.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            chunks: List of document chunks

        Raises:
            Exception if upload fails
        """
        key = self._get_s3_key(document_id, file_extension, "chunks.json")

        try:
            body = json.dumps(chunks, indent=2).encode('utf-8')

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType='application/json',
                ServerSideEncryption='AES256'  # Encrypt at rest
            )
            logger.debug(f"Saved {len(chunks)} chunks to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to save chunks to S3: {e}")
            raise

    def save_embeddings(self, document_id: str, file_extension: str, embeddings: np.ndarray) -> None:
        """
        Save embeddings to S3 as NumPy binary.

        Example: s3://bucket/pdf/{doc_id}/embeddings.npy

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            embeddings: NumPy array of shape (num_chunks, 1536)

        Raises:
            Exception if upload fails
        """
        key = self._get_s3_key(document_id, file_extension, "embeddings.npy")

        try:
            # Serialize NumPy array to bytes (in-memory)
            buffer = io.BytesIO()
            np.save(buffer, embeddings)
            buffer.seek(0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream',
                ServerSideEncryption='AES256'
            )
            logger.debug(f"Saved embeddings {embeddings.shape} to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to S3: {e}")
            raise

    def save_metadata(self, document_id: str, file_extension: str, metadata: Dict) -> None:
        """
        Save metadata to S3 as JSON.

        Example: s3://bucket/pdf/{doc_id}/metadata.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            metadata: Document metadata

        Raises:
            Exception if upload fails
        """
        key = self._get_s3_key(document_id, file_extension, "metadata.json")

        try:
            body = json.dumps(metadata, indent=2).encode('utf-8')

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )
            logger.debug(f"Saved metadata to S3: {key}")
        except Exception as e:
            logger.error(f"Failed to save metadata to S3: {e}")
            raise

    def load_chunks(self, document_id: str, file_extension: str) -> List[Dict]:
        """
        Load chunks from S3.

        Args:
            document_id: SHA-256 hash
            file_extension: File extension to locate correct folder

        Returns:
            List of document chunks

        Raises:
            Exception if file not found or load fails
        """
        key = self._get_s3_key(document_id, file_extension, "chunks.json")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            chunks = json.loads(response['Body'].read().decode('utf-8'))
            logger.debug(f"Loaded {len(chunks)} chunks from S3: {key}")
            return chunks
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Chunks file not found in S3: {key}")
            raise

    def load_embeddings(self, document_id: str, file_extension: str) -> np.ndarray:
        """
        Load embeddings from S3.

        Args:
            document_id: SHA-256 hash
            file_extension: File extension

        Returns:
            NumPy array of shape (num_chunks, 1536)

        Raises:
            Exception if file not found or load fails
        """
        key = self._get_s3_key(document_id, file_extension, "embeddings.npy")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)

            # Load NumPy array from S3 bytes
            buffer = io.BytesIO(response['Body'].read())
            embeddings = np.load(buffer)

            logger.debug(f"Loaded embeddings {embeddings.shape} from S3: {key}")
            return embeddings
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Embeddings file not found in S3: {key}")
            raise

    def load_metadata(self, document_id: str, file_extension: str) -> Dict:
        """
        Load metadata from S3.

        Args:
            document_id: SHA-256 hash
            file_extension: File extension

        Returns:
            Document metadata dictionary

        Raises:
            Exception if file not found or load fails
        """
        key = self._get_s3_key(document_id, file_extension, "metadata.json")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            logger.debug(f"Loaded metadata from S3: {key}")
            return metadata
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Metadata file not found in S3: {key}")
            raise

    def delete(self, document_id: str, file_extension: str) -> None:
        """
        Delete all 4 files for a document from S3.

        Deletes: document.{ext}, chunks.json, embeddings.npy, metadata.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension

        Raises:
            Exception if delete fails
        """
        keys_to_delete = [
            {'Key': self._get_s3_key(document_id, file_extension, f"document.{file_extension}")},
            {'Key': self._get_s3_key(document_id, file_extension, "chunks.json")},
            {'Key': self._get_s3_key(document_id, file_extension, "embeddings.npy")},
            {'Key': self._get_s3_key(document_id, file_extension, "metadata.json")}
        ]

        try:
            # Batch delete all files in single API call
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': keys_to_delete}
            )
            logger.info(f"Deleted S3 cache for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            raise

    def list_documents(self) -> List[str]:
        """
        List all cached document IDs from S3 across all document types.

        Returns:
            List of document IDs (SHA-256 hashes)

        Note: This scans the entire bucket (uses pagination for >1000 objects)
        """
        document_ids = set()

        try:
            # List ALL objects in the bucket (across pdf/, txt/, markdown/, etc.)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    # Extract document_id from key: {doc_type}/{doc_id}/{filename}
                    # Example: pdf/abc123/chunks.json → doc_id = abc123
                    key_parts = obj['Key'].split('/')
                    if len(key_parts) >= 2:
                        # key_parts[0] = document type (pdf, txt, etc.)
                        # key_parts[1] = document_id
                        document_ids.add(key_parts[1])

            logger.debug(f"Found {len(document_ids)} cached documents in S3")
            return list(document_ids)
        except Exception as e:
            logger.error(f"Failed to list documents from S3: {e}")
            return []

    def get_stats(self) -> Dict:
        """
        Get S3 cache statistics.

        Returns:
            Dictionary with stats (backend, bucket, region, total_documents, total_objects, total_size_mb, documents_by_type)

        Note: This scans the entire bucket to compute stats
        """
        total_size = 0
        total_objects = 0
        doc_type_counts = {}  # Count documents by type (pdf: 10, txt: 5, etc.)

        try:
            # Scan all objects in bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    total_size += obj['Size']
                    total_objects += 1

                    # Count by document type (pdf/, txt/, etc.)
                    doc_type = obj['Key'].split('/')[0] if '/' in obj['Key'] else 'unknown'
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

            stats = {
                "backend": "s3",
                "bucket": self.bucket_name,
                "region": self.region,
                "total_documents": len(self.list_documents()),
                "total_objects": total_objects,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "documents_by_type": doc_type_counts  # Shows pdf: 10, txt: 5, etc.
            }

            logger.info(f"S3 storage stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get S3 stats: {e}")
            return {
                "backend": "s3",
                "bucket": self.bucket_name,
                "error": str(e)
            }
