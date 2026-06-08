"""领域专用异常层级。"""


class RAGPipelineError(Exception):
    """所有 RAG pipeline 错误的基类。"""


class ConfigurationError(RAGPipelineError):
    """配置加载或校验错误。"""


class IngestionError(RAGPipelineError):
    """文档摄取错误（文件不可读、格式不支持）。"""


class ChunkingError(IngestionError):
    """文本分块错误。"""


class EmbeddingError(RAGPipelineError):
    """Embedding 模型推理错误。"""


class VectorDBError(RAGPipelineError):
    """向量数据库操作错误。"""


class ConnectionError(VectorDBError):
    """向量数据库连接错误。"""


class CollectionNotFoundError(VectorDBError):
    """请求的 collection 不存在。"""


class RetrievalError(RAGPipelineError):
    """检索操作错误。"""


class RerankerError(RetrievalError):
    """Reranker 模型推理错误。"""


class LLMError(RAGPipelineError):
    """LLM provider 错误（API 调用失败、限流等）。"""


class ValidationError(RAGPipelineError):
    """请求/响应校验错误。"""
