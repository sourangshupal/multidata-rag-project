"""
RAG (Retrieval-Augmented Generation) Service
Combines vector search with LLM generation to answer questions from documents.
"""

from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.config import settings
from app.services.vector_service import VectorService
from app.services.embedding_service import EmbeddingService


class RAGService:
    """Service for Retrieval-Augmented Generation."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the RAG service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

        # Initialize services
        self.embedding_service = EmbeddingService(api_key=self.api_key)
        self.vector_service = VectorService()
        self.llm_client = AsyncOpenAI(api_key=self.api_key)

        # LLM configuration
        self.model = "gpt-4-turbo-preview"
        self.temperature = 0.1
        self.max_tokens = 1000

    async def generate_answer(
        self,
        question: str,
        top_k: int = 3,
        namespace: str = "default",
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve relevant chunks and generate an answer.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default: 3)
            namespace: Pinecone namespace to search (default: "default")
            include_sources: Whether to include source citations (default: True)

        Returns:
            Dictionary containing:
                - question: The original question
                - answer: Generated answer from LLM
                - sources: List of source chunks used (if include_sources=True)
                - chunks_used: Number of chunks retrieved
                - model: LLM model used
        """
        try:
            # Step 1: Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(question)

            # Step 2: Search for relevant chunks in Pinecone
            search_results = await self.vector_service.search(
                query_embedding=query_embedding,
                top_k=top_k,
                namespace=namespace
            )

            chunks = search_results['chunks']

            if not chunks:
                return {
                    "question": question,
                    "answer": "I don't have enough information to answer that question. Please upload relevant documents first.",
                    "sources": [],
                    "chunks_used": 0,
                    "model": self.model
                }

            # Step 3: Build context from retrieved chunks
            context = self._build_context(chunks)

            # Step 4: Create prompt for LLM
            prompt = self._create_prompt(question, context)

            # Step 5: Generate answer using LLM
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. "
                                   "If the context doesn't contain enough information to answer the question, "
                                   "say so explicitly. Always base your answers on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content

            # Step 6: Format response
            result = {
                "question": question,
                "answer": answer,
                "chunks_used": len(chunks),
                "model": self.model
            }

            if include_sources:
                result["sources"] = self._format_sources(chunks)

            return result

        except Exception as e:
            raise Exception(f"RAG pipeline failed: {str(e)}")

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks with hierarchical heading context.

        Args:
            chunks: List of chunk dictionaries from vector search

        Returns:
            Formatted context string with heading hierarchy
        """
        import json
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            filename = chunk['metadata'].get('filename', 'Unknown')
            text = chunk.get('text', '')
            score = chunk.get('score', 0.0)

            # NEW: Extract heading hierarchy from Docling metadata
            headings_json = chunk['metadata'].get('headings', '[]')
            try:
                headings = json.loads(headings_json) if isinstance(headings_json, str) else headings_json
            except (json.JSONDecodeError, TypeError):
                headings = []

            # Build context with heading hierarchy if available
            if headings and len(headings) > 0:
                heading_context = " > ".join(headings)
                context_part = f"[Source {i}: {filename} (relevance: {score:.3f})]\n[Section: {heading_context}]\n{text}\n"
            else:
                context_part = f"[Source {i}: {filename} (relevance: {score:.3f})]\n{text}\n"

            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create the prompt for the LLM.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer that based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        return prompt

    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source information for response.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of source dictionaries
        """
        sources = []

        for chunk in chunks:
            sources.append({
                "filename": chunk['metadata'].get('filename', 'Unknown'),
                "chunk_index": chunk['metadata'].get('chunk_index', 0),
                "relevance_score": chunk.get('score', 0.0),
                "preview": chunk.get('text', '')[:200] + "..."  # First 200 chars
            })

        return sources

    async def get_similar_chunks(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Retrieve similar chunks without generating an answer.
        Useful for debugging or showing what documents were found.

        Args:
            question: Query text
            top_k: Number of chunks to retrieve
            namespace: Pinecone namespace

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(question)

            # Search for relevant chunks
            search_results = await self.vector_service.search(
                query_embedding=query_embedding,
                top_k=top_k,
                namespace=namespace
            )

            return {
                "question": question,
                "chunks": search_results['chunks'],
                "total_found": search_results['total_found']
            }

        except Exception as e:
            raise Exception(f"Failed to retrieve similar chunks: {str(e)}")
