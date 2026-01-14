"""
Document Processing Service
Handles parsing and chunking of various document formats (PDF, DOCX, CSV, JSON).

Now supports context-aware chunking with Docling for improved RAG quality.
"""

from typing import List, Dict, Any
import tiktoken
import logging
from unstructured.partition.auto import partition
from pathlib import Path

logger = logging.getLogger("rag_app.document_service")


def parse_document(file_path: str) -> str:
    """
    Parse any document type and return extracted text.
    Uses Unstructured.io to handle PDF, DOCX, CSV, JSON, and other formats.

    Args:
        file_path: Path to the document file

    Returns:
        str: Extracted text content from the document

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If parsing fails
    """
    # Verify file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Use Unstructured.io's auto partition to handle any file type
        elements = partition(filename=file_path)

        # Combine all elements into a single text string
        text = "\n\n".join([str(el) for el in elements])

        return text

    except Exception as e:
        raise Exception(f"Failed to parse document {file_path}: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"  # GPT-4 encoding
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        overlap: Number of overlapping tokens between chunks (default: 50)
        encoding_name: Tokenizer encoding to use (default: cl100k_base for GPT-4)

    Returns:
        List of dictionaries containing:
            - text: The chunk text
            - chunk_index: Index of the chunk
            - token_count: Number of tokens in the chunk
            - start_char: Starting character position
            - end_char: Ending character position
    """
    # Initialize tokenizer
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback to default encoding
        tokenizer = tiktoken.encoding_for_model("gpt-4")

    # Encode the entire text
    tokens = tokenizer.encode(text)

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Get chunk tokens
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)

        # Calculate character positions (approximate)
        if chunks:
            # For subsequent chunks, use the previous end position
            start_char = chunks[-1]['end_char'] - (overlap * 4)  # Rough estimate
            start_char = max(0, start_char)
        else:
            start_char = 0

        end_char = start_char + len(chunk_text)

        # Create chunk metadata
        chunk_data = {
            'text': chunk_text,
            'chunk_index': len(chunks),
            'token_count': len(chunk_tokens),
            'start_char': start_char,
            'end_char': end_char
        }

        chunks.append(chunk_data)

        # Move to next chunk with overlap
        start_idx += (chunk_size - overlap)

        # Break if we've reached the end
        if end_idx >= len(tokens):
            break

    return chunks


def get_document_stats(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a document.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary with document statistics
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse document
    text = parse_document(file_path)

    # Get token count
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(text)

    return {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_type": path.suffix,
        "character_count": len(text),
        "token_count": len(tokens),
        "estimated_chunks_512": (len(tokens) // 512) + 1
    }


def parse_and_chunk_with_context(file_path: str, chunk_size: int = 512, min_chunk_size: int = 256) -> List[Dict[str, Any]]:
    """
    Parse and chunk document using Docling's context-aware approach.

    This is the RECOMMENDED method that provides:
    - Semantic boundary detection (no mid-sentence splits)
    - Hierarchical heading context preservation
    - Rich metadata (page numbers, captions, document structure)
    - Smart merging to ensure chunks are 256-512 tokens (not too small)

    Falls back to traditional token-based chunking if Docling is unavailable.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk (default: 512)
        min_chunk_size: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata
    """
    try:
        # Try Docling first (context-aware chunking with merging)
        from app.services.docling_service import parse_and_chunk_document

        logger.info(f"Using Docling for context-aware chunking: {Path(file_path).name}")
        chunks = parse_and_chunk_document(file_path, chunk_size=chunk_size, min_chunk_size=min_chunk_size)

        logger.info(f"Docling chunking complete: {len(chunks)} chunks with heading context")
        return chunks

    except ImportError as e:
        logger.warning(f"Docling not available, falling back to token-based chunking: {e}")

        # Fallback to old method
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)

        # Add empty metadata fields for compatibility
        for chunk in chunks:
            chunk['headings'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.info(f"Token-based chunking complete: {len(chunks)} chunks (no context)")
        return chunks

    except Exception as e:
        logger.error(f"Docling failed, falling back to token-based chunking: {e}")

        # Fallback to old method
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)

        # Add empty metadata fields for compatibility
        for chunk in chunks:
            chunk['headings'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.warning(f"Using fallback chunking: {len(chunks)} chunks (no context)")
        return chunks
