"""
Vector Service
Handles vector storage and retrieval using Pinecone.
"""

from typing import List, Dict, Any
import logging
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from app.config import settings

logger = logging.getLogger("rag_app.vector_service")


class VectorService:
    """Service for vector operations using Pinecone."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the vector service with Pinecone.

        Args:
            api_key: Pinecone API key (optional, uses settings if not provided)
        """
        self.api_key = api_key or settings.PINECONE_API_KEY
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY in .env file.")

        self.environment = settings.PINECONE_ENVIRONMENT
        self.index_name = settings.PINECONE_INDEX_NAME

        # Initialize Pinecone client with gRPC for better performance
        self.pc = PineconeGRPC(api_key=self.api_key)
        self.index = None

    def connect_to_index(self):
        """
        Connect to the Pinecone index.
        Creates the index if it doesn't exist.
        """
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]

            if self.index_name not in index_names:
                # Create index if it doesn't exist
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment.split("-")[0]  # Extract region from environment
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")

            # Connect to the index
            index_description = self.pc.describe_index(name=self.index_name)
            self.index = self.pc.Index(host=index_description.host)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            raise Exception(f"Failed to connect to Pinecone index: {str(e)}")

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        filename: str,
        namespace: str = "default"
    ):
        """
        Store document chunks with their embeddings in Pinecone.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors corresponding to chunks
            filename: Source filename for metadata
            namespace: Pinecone namespace for organization (default: "default")

        Raises:
            Exception: If upsert fails
        """
        if not self.index:
            self.connect_to_index()

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

        try:
            # Prepare vectors for upsert
            vectors_to_upsert = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create unique ID: filename + chunk_index
                vector_id = f"{filename}_{chunk['chunk_index']}"

                # Prepare metadata
                import json
                metadata = {
                    "filename": filename,
                    "chunk_index": chunk['chunk_index'],
                    "token_count": chunk['token_count'],
                    "text": chunk['text'][:1000],  # Limit text size in metadata (Pinecone has limits)
                    "start_char": chunk.get('start_char', 0),
                    "end_char": chunk.get('end_char', 0),
                    # NEW: Docling enhancements - store as JSON strings
                    "headings": json.dumps(chunk.get('headings', [])),
                    "page_numbers": json.dumps(chunk.get('page_numbers', [])),
                    "has_context": len(chunk.get('headings', [])) > 0  # Quick filter for context-aware chunks
                }

                # Create vector tuple: (id, values, metadata)
                vectors_to_upsert.append((vector_id, embedding, metadata))

            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )

            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")

        except Exception as e:
            raise Exception(f"Failed to add documents to Pinecone: {str(e)}")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        namespace: str = "default",
        filter_dict: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in Pinecone.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default: 3)
            namespace: Pinecone namespace to search (default: "default")
            filter_dict: Optional metadata filter

        Returns:
            Dictionary with search results:
                - query: The query vector (first 5 dims for reference)
                - chunks: List of matched chunks with metadata and scores
                - total_found: Number of results returned
        """
        if not self.index:
            self.connect_to_index()

        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                filter=filter_dict
            )

            # Format results
            chunks = []
            for match in results['matches']:
                chunks.append({
                    'id': match['id'],
                    'score': match['score'],
                    'text': match['metadata'].get('text', ''),
                    'metadata': {
                        'filename': match['metadata'].get('filename', ''),
                        'chunk_index': match['metadata'].get('chunk_index', 0),
                        'token_count': match['metadata'].get('token_count', 0),
                    }
                })

            return {
                'query_preview': query_embedding[:5],  # Just first 5 dims for reference
                'chunks': chunks,
                'total_found': len(chunks)
            }

        except Exception as e:
            raise Exception(f"Failed to search Pinecone: {str(e)}")

    def get_index_stats(self, namespace: str = "default") -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Args:
            namespace: Namespace to get stats for

        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            self.connect_to_index()

        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "namespaces": stats.get('namespaces', {}),
            }
        except Exception as e:
            raise Exception(f"Failed to get index stats: {str(e)}")

    def delete_by_filename(self, filename: str, namespace: str = "default"):
        """
        Delete all vectors associated with a filename.

        Args:
            filename: Filename to delete
            namespace: Namespace containing the vectors
        """
        if not self.index:
            self.connect_to_index()

        try:
            # Delete using metadata filter
            self.index.delete(
                filter={"filename": {"$eq": filename}},
                namespace=namespace
            )
            logger.info(f"Deleted all vectors for filename: {filename}")

        except Exception as e:
            raise Exception(f"Failed to delete vectors: {str(e)}")
