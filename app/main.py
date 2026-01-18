"""
Multi-Source RAG + Text-to-SQL API
FastAPI application with document RAG and natural language to SQL capabilities.
"""

from typing import Optional
from fastapi import FastAPI, status, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import sys
import shutil

from app.config import settings
from app.logging_config import setup_logging, get_logger
from app.services.document_service import parse_document, chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.rag_service import RAGService
from app.services.sql_service import TextToSQLService
from app.services.router_service import QueryRouter
from app.services.cache_service import CacheService
from app.utils import (
    FileValidator, QueryValidator, ValidationError,
    ErrorResponse, format_file_size, truncate_text
)

# Initialize logging
logger = setup_logging(log_level="INFO")

# OPIK monitoring (optional - gracefully handles if not configured)
try:
    from opik import track
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    # Create a no-op decorator if OPIK is not installed
    def track(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

app = FastAPI(
    title="Multi-Source RAG + Text-to-SQL API",
    description="A system that combines document RAG with natural language to SQL conversion",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=settings.ROOT_PATH,  # For API Gateway: "/prod", for local: ""
)

# Global service instances (initialized on startup if API keys are available)
embedding_service: EmbeddingService | None = None
vector_service: VectorService | None = None
rag_service: RAGService | None = None
sql_service: TextToSQLService | None = None
cache_service: CacheService | None = None

# Upload directory (from config, supports both Lambda /tmp and local paths)
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
CACHE_DIR = Path(settings.CACHE_DIR)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running and check service connectivity.

    Returns:
        dict: Status information including timestamp, service state, and connectivity checks
    """
    global embedding_service, vector_service, rag_service, sql_service

    # Check service availability
    services_status = {
        "embedding_service": embedding_service is not None,
        "vector_service": vector_service is not None,
        "rag_service": rag_service is not None,
        "sql_service": sql_service is not None,
    }

    # Determine overall health
    any_service_available = any(services_status.values())
    health_status = "healthy" if any_service_available else "degraded"

    return {
        "status": health_status,
        "service": "Multi-Source RAG + Text-to-SQL API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "services": services_status,
        "features_available": {
            "document_rag": services_status["rag_service"],
            "text_to_sql": services_status["sql_service"],
            "query_routing": True,  # Always available (keyword-based)
        },
        "configuration": {
            "openai_configured": settings.OPENAI_API_KEY is not None,
            "pinecone_configured": settings.PINECONE_API_KEY is not None,
            "database_configured": settings.DATABASE_URL is not None,
            "opik_configured": settings.OPIK_API_KEY is not None if hasattr(settings, 'OPIK_API_KEY') else False,
        }
    }


@app.get("/info", status_code=status.HTTP_200_OK, tags=["Information"])
async def get_info():
    """
    Get system information and configuration details.

    Returns:
        dict: System information including Python version, environment, and features
    """
    return {
        "application": {
            "name": "Multi-Source RAG + Text-to-SQL",
            "version": "0.1.0",
            "environment": "development",  # Will be loaded from settings once .env exists
        },
        "features": {
            "document_rag": "Available - Phase 1 Complete",
            "text_to_sql": "Available - Phase 2 Complete",
            "query_routing": "Available - Phase 3 Complete",
            "evaluation_monitoring": "Available - Phase 4 Complete",
            "polish_documentation": "Available - Phase 5 Complete",
            "docker_deployment": "Available - Phase 6 Complete",
        },
        "deployment": {
            "docker": "Ready - Use docker-compose up",
            "dockerfile": "Multi-stage build optimized",
            "health_checks": "Enabled",
            "volumes": ["uploads", "vanna_chromadb"]
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health (enhanced with service checks)",
            "info": "/info",
            "stats": "/stats (system statistics)",
            "unified_query": "POST /query (recommended - intelligent routing)",
            "upload_document": "POST /upload (with validation)",
            "query_documents": "POST /query/documents (with validation)",
            "list_documents": "GET /documents",
            "generate_sql": "POST /query/sql/generate",
            "execute_sql": "POST /query/sql/execute",
            "pending_sql_queries": "GET /query/sql/pending",
        },
    }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with welcome message and quick links.

    Returns:
        dict: Welcome message and navigation links
    """
    return {
        "message": "Welcome to Multi-Source RAG + Text-to-SQL API",
        "version": "0.1.0",
        "status": "Phase 0 Complete - Development Ready",
        "documentation": "/docs",
        "health_check": "/health",
        "system_info": "/info",
    }


@app.post("/upload", status_code=status.HTTP_201_CREATED, tags=["Documents"])
@track(name="upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, DOCX, CSV, JSON, TXT).
    Pipeline: validate → save → [cache check] → parse → chunk → embed → cache → store

    NEW: Implements intelligent caching to avoid re-processing identical documents.

    Args:
        file: The document file to upload (max 50 MB)

    Returns:
        dict: Upload status with filename, chunks created, and cache_hit indicator

    Raises:
        HTTPException: If validation fails or services unavailable
    """
    global embedding_service, vector_service, cache_service

    # Validate file
    try:
        FileValidator.validate_file(file)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse.validation_error(str(e), field="file")
        )

    # Check if services are initialized
    if not embedding_service or not vector_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Document RAG services",
                "Please configure OPENAI_API_KEY and PINECONE_API_KEY in .env"
            )
        )

    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract file extension (IMPORTANT for S3 folder organization)
        file_extension = file_path.suffix.lstrip('.').lower()  # pdf, txt, md, docx, etc.

        # NEW: Compute unique document ID from file contents
        doc_id = None
        cache_hit = False
        chunks = None
        embeddings = None

        if cache_service:
            try:
                doc_id = cache_service.compute_document_id(file_path)
                logger.info(f"Document ID computed: {doc_id}")

                # NEW: Check if cache exists for this document (pass file_extension)
                if cache_service.cache_exists(doc_id, file_extension):
                    logger.info(f"Cache HIT for document: {file.filename} (ID: {doc_id[:8]}...)")

                    # Load from cache (pass file_extension)
                    cached_data = cache_service.load_chunks_and_embeddings(doc_id, file_extension)

                    if cached_data:
                        chunks = cached_data['chunks']
                        embeddings = cached_data['embeddings']
                        cache_hit = True
                        logger.info(f"Loaded {len(chunks)} chunks from cache, skipping processing")
                    else:
                        logger.warning("Cache load failed, falling back to full processing")
                else:
                    logger.info(f"Cache MISS for document: {file.filename} (ID: {doc_id[:8]}...)")

            except Exception as e:
                logger.warning(f"Cache check failed, proceeding with full processing: {e}")

        # If cache miss or cache unavailable, process document
        if chunks is None or embeddings is None:
            # Parse and chunk with context-aware approach (Docling with smart merging)
            logger.info(f"Parsing and chunking document with context awareness: {file.filename}")
            from app.services.document_service import parse_and_chunk_with_context
            chunks = parse_and_chunk_with_context(
                str(file_path),
                chunk_size=settings.CHUNK_SIZE,
                min_chunk_size=settings.MIN_CHUNK_SIZE
            )
            logger.info(f"Created {len(chunks)} context-aware chunks (target {settings.MIN_CHUNK_SIZE}-{settings.CHUNK_SIZE} tokens)")

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [chunk['text'] for chunk in chunks]
            embeddings, embedding_usage = await embedding_service.generate_embeddings(texts)

            # NEW: Save to cache if cache service is available
            if cache_service and doc_id:
                try:
                    # Save original document to storage (NEW)
                    cache_service.save_document(
                        doc_id=doc_id,
                        file_path=file_path,
                        file_extension=file_extension
                    )
                    logger.info(f"Saved original document to storage: {doc_id}")

                    # Prepare metadata
                    metadata = {
                        "document_id": doc_id,
                        "original_filename": file.filename,
                        "upload_timestamp": datetime.utcnow().isoformat() + "Z",
                        "file_size_bytes": file_path.stat().st_size,
                        "chunk_count": len(chunks),
                        "embedding_model": "text-embedding-3-small",
                        "chunk_size": settings.CHUNK_SIZE,
                        "chunk_overlap": settings.CHUNK_OVERLAP,
                        "file_extension": file_extension  # NEW: include file extension
                    }

                    # Save chunks, embeddings, and metadata to cache (pass file_extension)
                    cache_service.save_chunks_and_embeddings(
                        doc_id=doc_id,
                        file_extension=file_extension,  # NEW parameter
                        chunks=chunks,
                        embeddings=embeddings,
                        metadata=metadata
                    )
                    logger.info(f"Saved cache data (chunks, embeddings, metadata): {doc_id}")

                except Exception as e:
                    # Don't fail upload if caching fails
                    logger.warning(f"Failed to save to cache (continuing anyway): {e}")

        # Store in Pinecone (always, even on cache hit - in case vector DB was cleared)
        logger.info(f"Storing {len(chunks)} vectors in Pinecone...")
        vector_service.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename,
            namespace="default"
        )

        file_size = file_path.stat().st_size
        total_tokens = sum(chunk['token_count'] for chunk in chunks)
        # file_extension already defined earlier (line 218)

        # Update OPIK span with metadata for observability
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_span

                update_current_span(
                    tags=[
                        "document_upload",
                        f"extension_{file_extension}",
                        "cache_hit" if cache_hit else "cache_miss"
                    ],
                    metadata={
                        "filename": file.filename,
                        "file_size_bytes": file_size,
                        "file_size_human": format_file_size(file_size),
                        "file_extension": file_extension,
                        "chunk_count": len(chunks),
                        "total_tokens": total_tokens,
                        "cache_hit": cache_hit
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update OPIK span: {e}")

        return {
            "status": "success",
            "filename": file.filename,
            "document_id": doc_id[:16] + "..." if doc_id else None,  # Show first 16 chars
            "file_size": format_file_size(file_size),
            "file_size_bytes": file_size,
            "chunks_created": len(chunks),
            "total_tokens": total_tokens,
            "cache_hit": cache_hit,  # NEW: Indicate if cache was used
            "message": (
                f"Document loaded from cache and {len(chunks)} chunks stored in Pinecone"
                if cache_hit
                else f"Document processed and {len(chunks)} chunks stored in Pinecone"
            )
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse.validation_error(str(e))
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("upload document", e)
        )


@app.post("/query/documents", status_code=status.HTTP_200_OK, tags=["Query"])
@track(name="query_documents", type="llm")
async def query_documents(question: str, top_k: int = 3):
    """
    Query documents using RAG (Retrieval-Augmented Generation).
    Retrieves relevant chunks and generates an answer using GPT-4.

    Args:
        question: The question to answer (3-1000 characters)
        top_k: Number of document chunks to retrieve (1-10, default: 3)

    Returns:
        dict: Generated answer with sources and metadata

    Raises:
        HTTPException: If validation fails or service unavailable
    """
    global rag_service

    # Validate inputs
    try:
        question = QueryValidator.validate_question(question)
        top_k = QueryValidator.validate_top_k(top_k)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse.validation_error(str(e))
        )

    # Check if service is initialized
    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "RAG service",
                "Please configure OPENAI_API_KEY and PINECONE_API_KEY in .env"
            )
        )

    try:
        result = await rag_service.generate_answer(
            question=question,
            top_k=top_k,
            namespace="default",
            include_sources=True
        )

        # Update OPIK span with metadata and cost tracking
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_span

                usage_data = result.get('usage')

                span_update = {
                    "tags": ["document_query", f"top_k_{top_k}"],
                    "metadata": {
                        "question_length": len(question),
                        "top_k": top_k,
                        "chunks_retrieved": result.get('chunks_used', 0),
                        "model": result.get('model', 'unknown')
                    },
                    "model": "gpt-4-turbo-preview",
                    "provider": "openai"
                }

                # Add usage data for cost tracking
                if usage_data:
                    span_update["usage"] = {
                        "prompt_tokens": usage_data['embedding_tokens'] + usage_data['llm_prompt_tokens'],
                        "completion_tokens": usage_data['llm_completion_tokens'],
                        "total_tokens": usage_data['total_tokens']
                    }

                update_current_span(**span_update)
            except Exception as e:
                logger.warning(f"Failed to update OPIK span: {e}")

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("query documents", e)
        )


@app.get("/documents", status_code=status.HTTP_200_OK, tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents.

    Returns:
        dict: List of uploaded documents with metadata
    """
    try:
        documents = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                documents.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "uploaded_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/stats", status_code=status.HTTP_200_OK, tags=["Information"])
async def get_stats():
    """
    Get system statistics and usage information.

    Returns:
        dict: Statistics including document count, total size, and service status
    """
    global sql_service

    try:
        # Count uploaded documents
        documents = []
        total_size = 0

        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_size = file_path.stat().st_size
                documents.append({
                    "filename": file_path.name,
                    "size_bytes": file_size,
                })
                total_size += file_size

        # Get pending SQL queries count
        pending_sql_count = 0
        if sql_service:
            try:
                pending_sql_count = len(sql_service.get_pending_queries())
            except Exception:
                pass

        return {
            "documents": {
                "total_uploaded": len(documents),
                "total_size": format_file_size(total_size),
                "total_size_bytes": total_size,
            },
            "sql": {
                "pending_queries": pending_sql_count,
                "service_available": sql_service is not None,
            },
            "system": {
                "uptime_checked_at": datetime.utcnow().isoformat(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "configuration": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "max_file_size": format_file_size(FileValidator.MAX_FILE_SIZE),
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("get statistics", e)
        )


@app.get("/cache/stats", status_code=status.HTTP_200_OK, tags=["Cache"])
async def get_cache_stats():
    """
    Get cache statistics (total documents, size, etc.).

    Returns:
        dict: Cache statistics including document count and total size
    """
    global cache_service

    if not cache_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Cache service",
                "Cache service not initialized"
            )
        )

    try:
        stats = cache_service.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats,
            "message": f"Cache contains {stats['total_documents']} documents"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("get cache stats", e)
        )


@app.delete("/cache/clear", status_code=status.HTTP_200_OK, tags=["Cache"])
async def clear_cache(document_id: Optional[str] = None):
    """
    Clear cache for specific document or entire cache.

    Args:
        document_id: Optional document ID to clear (if not provided, clears all cache)

    Returns:
        dict: Result of cache clearing operation
    """
    global cache_service

    if not cache_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Cache service",
                "Cache service not initialized"
            )
        )

    try:
        result = cache_service.clear_cache(doc_id=document_id)

        return {
            "status": "success" if result['cleared'] else "failed",
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("clear cache", e)
        )


@app.post("/query", status_code=status.HTTP_200_OK, tags=["Query"])
@track(name="unified_query")
async def unified_query(question: str, auto_approve_sql: bool = False, top_k: int = 3):
    """
    Unified query endpoint with intelligent routing.
    Automatically routes queries to SQL, Documents, or both (HYBRID) based on intent.

    Args:
        question: The natural language question
        auto_approve_sql: If True, automatically execute SQL queries without approval (testing only)
        top_k: Number of document chunks to retrieve for RAG (default: 3)

    Returns:
        dict: Query results with routing information and answers
    """
    global rag_service, sql_service

    try:
        # Determine routing using QueryRouter
        route_type = QueryRouter.route(question)

        result = {
            "question": question,
            "route": route_type,
            "routing_explanation": QueryRouter.explain_routing(question)
        }

        # Update OPIK span with routing metadata
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_span

                update_current_span(
                    tags=[
                        "unified_query",
                        f"route_{route_type.lower()}",
                        "auto_approve" if auto_approve_sql else "manual_approve"
                    ],
                    metadata={
                        "question_length": len(question),
                        "route_type": route_type,
                        "auto_approve_sql": auto_approve_sql,
                        "top_k": top_k
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update OPIK span: {e}")

        # Route to SQL
        if route_type == "SQL":
            if not sql_service:
                raise HTTPException(
                    status_code=503,
                    detail="SQL service not initialized. Please configure DATABASE_URL in .env file."
                )

            # Generate SQL
            sql_result = await sql_service.generate_sql_for_approval(question)

            if auto_approve_sql:
                # Auto-execute for testing
                execution_result = await sql_service.execute_approved_query(
                    sql_result['query_id'],
                    approved=True
                )
                result.update({
                    "sql": execution_result['sql'],
                    "results": execution_result['results'],
                    "result_count": execution_result['result_count'],
                    "status": "executed",
                    "note": "SQL auto-executed (testing mode)"
                })
            else:
                # Return for approval
                result.update({
                    "query_id": sql_result['query_id'],
                    "sql": sql_result['sql'],
                    "explanation": sql_result['explanation'],
                    "status": "pending_approval",
                    "note": "Use POST /query/sql/execute with this query_id to execute"
                })

        # Route to Documents
        elif route_type == "DOCUMENTS":
            if not rag_service:
                raise HTTPException(
                    status_code=503,
                    detail="RAG service not initialized. Please configure API keys in .env file."
                )

            rag_result = await rag_service.generate_answer(
                question=question,
                top_k=top_k,
                namespace="default",
                include_sources=True
            )

            result.update({
                "answer": rag_result['answer'],
                "sources": rag_result.get('sources', []),
                "chunks_used": rag_result.get('chunks_used', 0),
                "status": "completed"
            })

        # Route to HYBRID (both SQL and Documents)
        elif route_type == "HYBRID":
            # Check both services are available
            if not sql_service or not rag_service:
                missing_services = []
                if not sql_service:
                    missing_services.append("SQL service")
                if not rag_service:
                    missing_services.append("RAG service")

                raise HTTPException(
                    status_code=503,
                    detail=f"HYBRID query requires both services. Missing: {', '.join(missing_services)}"
                )

            # Get SQL results
            sql_result = await sql_service.generate_sql_for_approval(question)

            if auto_approve_sql:
                execution_result = await sql_service.execute_approved_query(
                    sql_result['query_id'],
                    approved=True
                )
                sql_data = {
                    "sql": execution_result['sql'],
                    "results": execution_result['results'],
                    "result_count": execution_result['result_count'],
                    "status": "executed"
                }
            else:
                sql_data = {
                    "query_id": sql_result['query_id'],
                    "sql": sql_result['sql'],
                    "explanation": sql_result['explanation'],
                    "status": "pending_approval"
                }

            # Get document context
            rag_result = await rag_service.generate_answer(
                question=question,
                top_k=top_k,
                namespace="default",
                include_sources=True
            )

            # Combine both results
            result.update({
                "sql_component": sql_data,
                "document_component": {
                    "answer": rag_result['answer'],
                    "sources": rag_result.get('sources', []),
                    "chunks_used": rag_result.get('chunks_used', 0)
                },
                "status": "completed" if auto_approve_sql else "partial_pending_sql_approval",
                "note": "HYBRID query combines both SQL data and document context"
            })

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified query failed: {str(e)}")


@app.post("/query/sql/generate", status_code=status.HTTP_200_OK, tags=["SQL"])
@track(name="generate_sql", type="llm")
async def generate_sql(question: str):
    """
    Generate SQL from a natural language question using Vanna.ai.
    Returns the SQL for user review and approval before execution.

    Args:
        question: Natural language question about the database

    Returns:
        dict: Generated SQL with query_id for approval
    """
    global sql_service

    if not sql_service:
        raise HTTPException(
            status_code=503,
            detail="SQL service not initialized. Please configure DATABASE_URL in .env file."
        )

    try:
        result = await sql_service.generate_sql_for_approval(question)

        # Update OPIK span with metadata
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_span

                update_current_span(
                    tags=["sql_generation", "text_to_sql"],
                    metadata={
                        "question_length": len(question),
                        "sql_length": len(result.get('sql', '')),
                        "query_id": result.get('query_id'),
                        "model": settings.VANNA_MODEL
                    },
                    model=settings.VANNA_MODEL,
                    provider="openai"
                )
            except Exception as e:
                logger.warning(f"Failed to update OPIK span: {e}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")


@app.post("/query/sql/execute", status_code=status.HTTP_200_OK, tags=["SQL"])
@track(name="execute_sql")
async def execute_sql(query_id: str, approved: bool = True):
    """
    Execute a previously generated SQL query after user approval.

    Args:
        query_id: ID from the generate_sql endpoint
        approved: Whether to execute (True) or reject (False) the query

    Returns:
        dict: Query results or rejection message
    """
    global sql_service

    if not sql_service:
        raise HTTPException(
            status_code=503,
            detail="SQL service not initialized. Please configure DATABASE_URL in .env file."
        )

    try:
        result = await sql_service.execute_approved_query(query_id, approved)

        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))

        # Update OPIK span with metadata
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_span

                update_current_span(
                    tags=[
                        "sql_execution",
                        "approved" if approved else "rejected"
                    ],
                    metadata={
                        "query_id": query_id,
                        "approved": approved,
                        "result_count": result.get('result_count', 0),
                        "status": result.get('status')
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update OPIK span: {e}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")


@app.get("/query/sql/pending", status_code=status.HTTP_200_OK, tags=["SQL"])
async def list_pending_sql_queries():
    """
    List all SQL queries pending approval.

    Returns:
        dict: List of pending queries with their IDs and SQL
    """
    global sql_service

    if not sql_service:
        raise HTTPException(
            status_code=503,
            detail="SQL service not initialized."
        )

    try:
        pending = sql_service.get_pending_queries()
        return {
            "total_pending": len(pending),
            "pending_queries": pending
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pending queries: {str(e)}")


def initialize_services():
    """Initialize all services. Called directly on Lambda startup or via FastAPI startup event."""
    global embedding_service, vector_service, rag_service, sql_service, cache_service

    # Ensure upload and cache directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage directories initialized: {UPLOAD_DIR}, {CACHE_DIR}")

    logger.info("=" * 60)
    logger.info("Starting Multi-Source RAG + Text-to-SQL API...")
    logger.info("=" * 60)
    logger.info("Phase 0: Foundation Setup - COMPLETE")
    logger.info("Phase 1: Document RAG MVP - COMPLETE")
    logger.info("Phase 2: Text-to-SQL Foundation - COMPLETE")
    logger.info("Phase 3: Query Routing - COMPLETE")
    logger.info("Phase 4: Evaluation & Monitoring - COMPLETE")
    logger.info("Phase 5: Polish & Documentation - COMPLETE")
    logger.info("Phase 6: Docker Deployment - COMPLETE")
    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETE - PRODUCTION READY!")
    logger.info("=" * 60)

    # Initialize OPIK monitoring if available
    if OPIK_AVAILABLE:
        try:
            if settings.OPIK_API_KEY:
                logger.info("Initializing OPIK monitoring...")
                from opik import configure
                configure(api_key=settings.OPIK_API_KEY)
                logger.info("✓ OPIK monitoring initialized!")
            else:
                logger.warning("OPIK available but API key not configured.")
                logger.info("Monitoring will use local tracking only.")
        except Exception as e:
            logger.warning(f"Failed to initialize OPIK: {e}")
    else:
        logger.info("OPIK not available (package not installed).")

    # Initialize Document RAG services if API keys are available
    try:
        if settings.OPENAI_API_KEY and settings.PINECONE_API_KEY:
            logger.info("Initializing Document RAG services...")
            embedding_service = EmbeddingService()
            vector_service = VectorService()
            vector_service.connect_to_index()
            rag_service = RAGService()
            logger.info("✓ Document RAG services initialized!")
        else:
            logger.warning("OpenAI/Pinecone API keys not configured.")
            logger.warning("Document RAG features will be unavailable.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG services: {e}", exc_info=True)
        logger.warning("Document RAG features will be unavailable.")

    # Initialize Text-to-SQL service if database is available
    try:
        if settings.DATABASE_URL and settings.OPENAI_API_KEY:
            logger.info("Initializing Text-to-SQL service...")
            sql_service = TextToSQLService()
            logger.info("Training Vanna on database schema and examples...")
            sql_service.complete_training()
            logger.info("✓ Text-to-SQL service initialized and trained!")
        else:
            logger.warning("DATABASE_URL not configured.")
            logger.warning("Text-to-SQL features will be unavailable.")
    except Exception as e:
        logger.error(f"Failed to initialize SQL service: {e}", exc_info=True)
        logger.warning("Text-to-SQL features will be unavailable.")

    # Initialize cache service (always available, no API key needed)
    # Auto-selects storage backend based on STORAGE_BACKEND environment variable
    try:
        logger.info("Initializing cache service...")
        cache_service = CacheService()  # Auto-selects local or S3 backend
        logger.info("✓ Cache service initialized!")
    except Exception as e:
        logger.error(f"✗ Failed to initialize cache service: {e}")
        logger.warning("Document uploads will work but caching will be unavailable.")

    logger.info("=" * 60)
    logger.info("API is ready!")
    logger.info("=" * 60)


# Event handlers for startup/shutdown
@app.on_event("startup")
async def startup_event():
    """Execute tasks on application startup."""
    initialize_services()


@app.on_event("shutdown")
async def shutdown_event():
    """Execute cleanup tasks on application shutdown."""
    logger.info("Shutting down Multi-Source RAG + Text-to-SQL API...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
