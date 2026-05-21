"""Domain-specific exception hierarchy."""


class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline errors."""


class ConfigurationError(RAGPipelineError):
    """Configuration loading or validation error."""


class IngestionError(RAGPipelineError):
    """Document ingestion error (unreadable file, unsupported format)."""


class ChunkingError(IngestionError):
    """Text chunking error."""


class EmbeddingError(RAGPipelineError):
    """Embedding model inference error."""


class VectorDBError(RAGPipelineError):
    """Vector database operation error."""


class ConnectionError(VectorDBError):
    """Vector database connection error."""


class CollectionNotFoundError(VectorDBError):
    """Requested collection does not exist."""


class RetrievalError(RAGPipelineError):
    """Retrieval operation error."""


class RerankerError(RetrievalError):
    """Reranker model inference error."""


class LLMError(RAGPipelineError):
    """LLM provider error (API call failure, rate limit, etc.)."""


class GenerationError(RAGPipelineError):
    """Answer generation error."""


class ValidationError(RAGPipelineError):
    """Request/response validation error."""
