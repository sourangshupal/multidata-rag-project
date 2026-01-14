"""
Docling Service
Provides context-aware document parsing and chunking using Docling's HybridChunker.
Preserves document structure and hierarchical heading context for better RAG quality.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("rag_app.docling_service")

# Import Docling components
try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    DOCLING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Docling not available: {e}")
    DOCLING_AVAILABLE = False


def convert_document(file_path: str):
    """
    Convert document using Docling's advanced layout analysis.

    Args:
        file_path: Path to the document file

    Returns:
        DoclingDocument: Structured document with hierarchy preserved

    Raises:
        ImportError: If Docling is not installed
        Exception: If conversion fails
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is not installed. Run: pip install docling docling-core")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Converting document with Docling: {Path(file_path).name}")
        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        logger.info(f"Document converted successfully: {len(doc.texts)} text elements")
        return doc

    except Exception as e:
        logger.error(f"Docling conversion failed: {str(e)}")
        raise Exception(f"Failed to convert document with Docling: {str(e)}")


def chunk_with_hybrid(doc, max_tokens: int = 512, min_tokens: int = 256) -> List[Dict[str, Any]]:
    """
    Chunk document using HybridChunker with context awareness.

    Uses OpenAI tokenizer (tiktoken) for consistency with existing embeddings.
    Preserves hierarchical heading context and semantic boundaries.
    Post-processes to merge small chunks for better RAG context.

    Args:
        doc: DoclingDocument from convert_document()
        max_tokens: Maximum tokens per chunk (default: 512)
        min_tokens: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata:
            - text: The chunk text
            - chunk_index: Sequential index
            - token_count: Actual token count
            - start_char: Starting character position
            - end_char: Ending character position
            - headings: List of hierarchical headings (e.g., ["Chapter 1", "Section 1.2"])
            - page_numbers: List of page numbers this chunk spans
            - doc_items: References to original document items
            - captions: Table/figure captions if applicable
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is not installed. Run: pip install docling docling-core")

    try:
        # Set up OpenAI tokenizer (tiktoken) with GPT-4 encoding
        import tiktoken
        tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken_encoder,  # Pass the actual tiktoken encoder instance
            max_tokens=max_tokens
        )

        # Create hybrid chunker with context awareness
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=True  # Merge chunks with same heading context
        )

        logger.info(f"Chunking with HybridChunker (max_tokens={max_tokens}, merge_peers=True)")

        # Generate chunks with structural context
        raw_chunks = list(chunker.chunk(dl_doc=doc))

        logger.info(f"Generated {len(raw_chunks)} raw semantic chunks, merging to target {min_tokens}-{max_tokens} tokens")

        # Post-process: Merge consecutive small chunks to reach target size
        merged_chunks = []
        current_merged = None

        for chunk in raw_chunks:
            token_count = len(tiktoken_encoder.encode(chunk.text))

            if current_merged is None:
                # Start a new merged chunk
                current_merged = chunk
            else:
                # Check if we should merge with current chunk
                current_tokens = len(tiktoken_encoder.encode(current_merged.text))
                combined_tokens = current_tokens + token_count

                # Merge if current chunk is undersized and combined won't exceed max
                if current_tokens < min_tokens and combined_tokens <= max_tokens:
                    # Merge chunks
                    current_merged.text = current_merged.text + "\n\n" + chunk.text

                    # Merge metadata
                    if chunk.meta and chunk.meta.headings:
                        if not current_merged.meta.headings:
                            current_merged.meta.headings = []
                        for h in chunk.meta.headings:
                            if h not in current_merged.meta.headings:
                                current_merged.meta.headings.append(h)

                    # Merge page numbers
                    if chunk.meta and chunk.meta.origin and hasattr(chunk.meta.origin, 'page_numbers'):
                        if chunk.meta.origin.page_numbers:
                            if not hasattr(current_merged.meta.origin, 'page_numbers') or not current_merged.meta.origin.page_numbers:
                                if current_merged.meta and current_merged.meta.origin:
                                    current_merged.meta.origin.page_numbers = []
                            if current_merged.meta and current_merged.meta.origin and current_merged.meta.origin.page_numbers is not None:
                                for pn in chunk.meta.origin.page_numbers:
                                    if pn not in current_merged.meta.origin.page_numbers:
                                        current_merged.meta.origin.page_numbers.append(pn)
                else:
                    # Current chunk is complete, save it and start new one
                    merged_chunks.append(current_merged)
                    current_merged = chunk

        # Don't forget the last chunk
        if current_merged is not None:
            merged_chunks.append(current_merged)

        logger.info(f"After merging: {len(merged_chunks)} chunks (avg {sum(len(tiktoken_encoder.encode(c.text)) for c in merged_chunks) / len(merged_chunks):.1f} tokens)")

        # Convert to format compatible with existing cache/vector storage
        result = []
        char_position = 0

        for idx, chunk in enumerate(merged_chunks):
            # Extract heading hierarchy
            headings = []
            if chunk.meta and chunk.meta.headings:
                headings = [h.text for h in chunk.meta.headings if hasattr(h, 'text')]

            # Extract page numbers
            page_numbers = []
            if chunk.meta and chunk.meta.origin and hasattr(chunk.meta.origin, 'page_numbers'):
                page_numbers = chunk.meta.origin.page_numbers or []

            # Extract captions (for tables/figures)
            captions = []
            if chunk.meta and hasattr(chunk.meta, 'captions') and chunk.meta.captions:
                captions = [str(c) for c in chunk.meta.captions]

            # Get document items (for grounding)
            doc_items = []
            if chunk.meta and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                # Store first 3 items as strings for reference
                doc_items = [str(item)[:100] for item in chunk.meta.doc_items[:3]]

            # Calculate token count using the underlying tiktoken encoder
            token_count = len(tiktoken_encoder.encode(chunk.text))

            # Calculate character positions
            chunk_text = chunk.text
            start_char = char_position
            end_char = start_char + len(chunk_text)
            char_position = end_char

            # Create enhanced chunk dictionary
            chunk_data = {
                'text': chunk_text,
                'chunk_index': idx,
                'token_count': token_count,
                'start_char': start_char,
                'end_char': end_char,
                # NEW: Rich metadata from Docling
                'headings': headings,
                'page_numbers': page_numbers,
                'doc_items': doc_items,
                'captions': captions
            }

            result.append(chunk_data)

        # Log sample of first chunk's metadata
        if result:
            first_chunk = result[0]
            logger.info(f"Sample chunk metadata - Headings: {first_chunk['headings']}, Pages: {first_chunk['page_numbers']}")

        return result

    except Exception as e:
        logger.error(f"HybridChunker failed: {str(e)}")
        raise Exception(f"Failed to chunk document with HybridChunker: {str(e)}")


def parse_and_chunk_document(file_path: str, chunk_size: int = 512, min_chunk_size: int = 256) -> List[Dict[str, Any]]:
    """
    Parse and chunk document using Docling's context-aware approach.

    This is the main entry point that replaces the old parse_document() + chunk_text() flow.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk (default: 512)
        min_chunk_size: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata

    Raises:
        Exception: If both Docling and fallback fail
    """
    if not DOCLING_AVAILABLE:
        logger.warning("Docling not available, cannot use context-aware chunking")
        raise ImportError("Docling is required for context-aware chunking. Run: pip install docling docling-core")

    try:
        # Step 1: Convert document with layout awareness
        doc = convert_document(file_path)

        # Step 2: Chunk with hybrid chunker (with merging)
        chunks = chunk_with_hybrid(doc, max_tokens=chunk_size, min_tokens=min_chunk_size)

        logger.info(f"Successfully processed {Path(file_path).name}: {len(chunks)} chunks with context")

        return chunks

    except Exception as e:
        logger.error(f"Docling processing failed for {Path(file_path).name}: {str(e)}")
        raise Exception(f"Failed to process document with Docling: {str(e)}")


def fallback_to_unstructured(file_path: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    Fallback to Unstructured.io for documents Docling cannot handle.

    This maintains compatibility but without context-aware chunking benefits.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk

    Returns:
        List of chunk dictionaries (without rich metadata)
    """
    logger.warning(f"Using Unstructured.io fallback for {Path(file_path).name}")

    try:
        from app.services.document_service import parse_document, chunk_text

        # Use old token-based chunking
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)

        # Add empty metadata fields for compatibility
        for chunk in chunks:
            chunk['headings'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.info(f"Fallback chunking complete: {len(chunks)} chunks (no context)")

        return chunks

    except Exception as e:
        logger.error(f"Fallback also failed: {str(e)}")
        raise Exception(f"Both Docling and Unstructured failed: {str(e)}")


def get_docling_status() -> Dict[str, Any]:
    """
    Check if Docling is available and functioning.

    Returns:
        Dictionary with status information
    """
    return {
        "docling_available": DOCLING_AVAILABLE,
        "features": {
            "context_aware_chunking": DOCLING_AVAILABLE,
            "heading_preservation": DOCLING_AVAILABLE,
            "table_structure": DOCLING_AVAILABLE,
            "layout_analysis": DOCLING_AVAILABLE
        }
    }
