"""
Configuration management using Pydantic Settings.
Loads environment variables from .env file.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Multi-Source RAG + Text-to-SQL"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    ROOT_PATH: str = ""  # Set to "/prod" for API Gateway, empty for local development

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None  # Required for embeddings and RAG

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None  # Required for vector storage
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "rag-documents"

    # Supabase/PostgreSQL Configuration
    DATABASE_URL: Optional[str] = None  # Required for Text-to-SQL

    # OPIK Monitoring
    OPIK_API_KEY: Optional[str] = None  # Optional for monitoring
    OPIK_PROJECT_NAME: str = "Multi-Source-RAG"  # Add this line with your custom project name

    # Vanna 2.0 Configuration (Text-to-SQL)
    VANNA_MODEL: str = "gpt-4o"  # OpenAI model for SQL generation
    VANNA_PINECONE_INDEX: str = "vanna-sql-training"  # Dedicated Pinecone index for SQL training
    VANNA_NAMESPACE: str = "sql-agent"  # Namespace within Pinecone index

    # Text Chunking Configuration
    CHUNK_SIZE: int = 512
    MIN_CHUNK_SIZE: int = 256  # Minimum chunk size - smaller chunks will be merged
    CHUNK_OVERLAP: int = 50

    # Storage paths (defaults to /tmp for Lambda, can be overridden for local dev)
    # UPLOAD_DIR: str = "/tmp/uploads"
    # CACHE_DIR: str = "/tmp/cached_chunks"
    UPLOAD_DIR: str = "data/uploads"
    CACHE_DIR: str = "data/cached_chunks"
    @property
    def is_lambda(self) -> bool:
        """Check if running in AWS Lambda environment."""
        return os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
