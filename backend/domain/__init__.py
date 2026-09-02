"""Domain entities package."""

from backend.domain.agent import Agent
from backend.domain.agent_trace import (
    AgentTrace,
    ExecutionEvent,
    ExecutionEventType,
    TerminationReason,
)
from backend.domain.base import DomainEntity
from backend.domain.dataset import Dataset, Task
from backend.domain.environment import Environment, EnvironmentStatus
from backend.domain.evaluation import Evaluation
from backend.domain.knowledge import (
    # Enums
    ChunkStrategy,
    DocumentStatus,
    EvidenceRelevance,
    FusionStrategy,
    RetrievalMethod,
    RetrievalRunStatus,
    # Storage entities
    Chunk,
    Document,
    DocumentVersion,
    KnowledgeCollection,
    KnowledgeSource,
    # Authorization
    RetrievalContext,
    # Configuration
    RetrievalConfiguration,
    # Retrieval event
    RetrievalQuery,
    RetrievalResult,
    RetrievalRun,
    # Evidence
    Evidence,
    # Protocols
    EmbeddingProvider,
    RerankingService,
    RetrievalService,
    # Errors
    AuthorizationError,
    ChunkingError,
    DatabaseUnavailableError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EvidenceConstructionError,
    IngestionError,
    KnowledgeError,
    MalformedDocumentError,
    ParserError,
    RerankerError,
    RetrievalTimeoutError,
)
from backend.domain.project import Project
from backend.domain.result import EvaluationResult, VerificationResult
from backend.domain.run import Run, RunStatus
from backend.domain.trial import Trial

__all__ = [
    "DomainEntity",
    "Project",
    "Agent",
    "Dataset",
    "Task",
    "Evaluation",
    "Run",
    "RunStatus",
    "Environment",
    "EnvironmentStatus",
    "Trial",
    "VerificationResult",
    "EvaluationResult",
    "AgentTrace",
    "ExecutionEvent",
    "ExecutionEventType",
    "TerminationReason",
    # Knowledge & RAG — Enums
    "ChunkStrategy",
    "DocumentStatus",
    "EvidenceRelevance",
    "FusionStrategy",
    "RetrievalMethod",
    "RetrievalRunStatus",
    # Knowledge & RAG — Storage entities
    "Chunk",
    "Document",
    "DocumentVersion",
    "KnowledgeCollection",
    "KnowledgeSource",
    # Knowledge & RAG — Authorization
    "RetrievalContext",
    # Knowledge & RAG — Configuration
    "RetrievalConfiguration",
    # Knowledge & RAG — Retrieval event
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalRun",
    # Knowledge & RAG — Evidence
    "Evidence",
    # Knowledge & RAG — Protocols
    "EmbeddingProvider",
    "RerankingService",
    "RetrievalService",
    # Knowledge & RAG — Errors
    "AuthorizationError",
    "ChunkingError",
    "DatabaseUnavailableError",
    "EmbeddingProviderError",
    "EmbeddingRateLimitError",
    "EmbeddingTimeoutError",
    "EvidenceConstructionError",
    "IngestionError",
    "KnowledgeError",
    "MalformedDocumentError",
    "ParserError",
    "RerankerError",
    "RetrievalTimeoutError",
]
