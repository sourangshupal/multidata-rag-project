# Multi-Source RAG + Text-to-SQL Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with system dependencies
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for python-magic
    libmagic1 \
    # Required for PDF processing
    poppler-utils \
    # Required for OCR in document processing
    tesseract-ocr \
    # Build tools for compiling Python packages
    gcc \
    g++ \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage 3: Final runtime image
FROM base as runtime

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ ./app/
COPY data/sql/ ./data/sql/
COPY data/generate_sample_data.py ./data/
COPY evaluate.py .
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p data/uploads data/vanna_chromadb

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
