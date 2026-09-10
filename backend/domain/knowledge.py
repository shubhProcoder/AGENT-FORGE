"""Knowledge domain models and retrieval contract.

This module defines the domain-level abstractions for the Knowledge & RAG
evaluation capability.  All types here are pure domain models — no embedding
dimensions, no vector stores, no vendor SDKs.  Retrieval infrastructure
implementations will live in backend/infrastructure/retrieval/.

Architecture
============

AgentForge treats RAG as an **observable, evaluable agent capability**.
The domain layer is designed *evaluation-first*: every retrieval operation
produces provenance records sufficient to answer five evaluation invariants.

Five Evaluation Invariants
==========================

1. **Traceability** — every Evidence item links back to a specific Project,
   Collection, Document, DocumentVersion, Chunk, RetrievalQuery,
   RetrievalConfiguration, and RetrievalRun.

2. **Reproducibility** — every RetrievalRun captures a frozen
   RetrievalConfiguration snapshot (embedding model/version, methods, fusion
   strategy, reranker, filters, top_k) and a knowledge_snapshot_hash so that
   identical inputs under identical config produce identical results.

3. **Authorization** — retrieval is gated by a RetrievalContext that
   establishes the allowed knowledge scope *before* any results are fetched.
   The agent never sees chunks from unauthorized collections.

4. **Evidence ≠ Chunk** — a Chunk is stored knowledge; an Evidence item is a
   historical retrieval event recording *what was retrieved, why, under which
   configuration, at what rank and score, at what time*.

5. **Measurable Quality** — retrieval quality (Recall@K, Precision@K, MRR,
   NDCG), generation quality (groundedness, citation support), and agent
   usage quality (sequencing, completeness) are independently measurable.

Entity Hierarchy (Knowledge Storage)
=====================================

    KnowledgeSource           (project-scoped permission boundary)
          │
          └── KnowledgeCollection
                    │
                    └── Document
                           │
                           └── DocumentVersion  (immutable after seal())
                                  │
                                  └── Chunk

Retrieval Flow (Evaluation Event)
=================================

    RetrievalContext           (authorization boundary)
          ↓
    RetrievalQuery             (what the agent asked)
          ↓
    RetrievalConfiguration     (how retrieval was configured)
          ↓
    RetrievalRun               (the evaluation event)
          │
          ├── RetrievalResult[] (ranked hits)
          │
          └── Evidence[]       (provenance records for evaluation)

Design Rationale
================
RAG inside AgentForge is NOT a chatbot feature.  It is an *observable agent
capability* — another way the agent can fail (wrong document, irrelevant chunks,
missing evidence, stale information, conflicting documents, unsupported claims,
wrong policy, citation mismatch).  AgentForge evaluates whether the agent's
retrieval-augmented reasoning can be trusted.

We design the evidence and evaluation model first, then build the RAG system
that produces the evidence.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from backend.domain.base import DomainEntity, now_utc


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class DocumentStatus(StrEnum):
    """Lifecycle states for a document."""

    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class ChunkStrategy(StrEnum):
    """How the document was segmented into chunks."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    PAGE = "page"
    CUSTOM = "custom"


class RetrievalMethod(StrEnum):
    """Which retrieval strategy produced a result."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    FULL_TEXT = "full_text"
    RERANKED = "reranked"


class FusionStrategy(StrEnum):
    """How keyword and vector results are combined."""

    RECIPROCAL_RANK = "reciprocal_rank"
    WEIGHTED_SUM = "weighted_sum"
    INTERLEAVE = "interleave"


class RetrievalRunStatus(StrEnum):
    """Status of a retrieval run."""

    STARTED = "started"
    AUTHORIZED = "authorized"
    AUTHORIZATION_DENIED = "authorization_denied"
    RETRIEVING = "retrieving"
    FUSING = "fusing"
    RERANKING = "reranking"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class EvidenceRelevance(StrEnum):
    """Evaluator-assigned relevance judgement on a piece of evidence."""

    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"
    CONTRADICTORY = "contradictory"
    STALE = "stale"


# ─────────────────────────────────────────────────────────────────────────────
# Entity models — KnowledgeSource → KnowledgeCollection → Document →
#                  DocumentVersion → Chunk
# ─────────────────────────────────────────────────────────────────────────────


class Chunk(DomainEntity):
    """An atomic unit of retrievable content derived from a document version.

    The chunk is the granularity at which retrieval results and evidence are
    reported.  Chunk metadata (page, section, position) enables the evaluator
    to trace a retrieval hit back to the exact source location.

    Chunks are IMMUTABLE after their parent DocumentVersion is sealed.
    """

    document_version_id: uuid.UUID
    document_id: uuid.UUID

    # Content
    text: str
    sequence_index: int = 0  # position within the parent version (0-based)

    # Source provenance
    page: int | None = None
    section: str | None = None
    start_char: int | None = None
    end_char: int | None = None

    # Chunking metadata
    strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    token_count: int = 0
    char_count: int = 0

    # Content integrity
    content_hash: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_content_hash(self) -> str:
        """Deterministic hash of chunk text for deduplication and versioning."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def seal(self) -> None:
        """Freeze the chunk's content hash.  Called by DocumentVersion.seal()."""
        self.content_hash = self.compute_content_hash()
        self.char_count = len(self.text)


class DocumentVersion(DomainEntity):
    """A specific, immutable snapshot of a document's content.

    Documents change over time (policy updates, knowledge-base edits).
    Versioning ensures that evaluation results are reproducible: the exact
    version of each document that was retrievable during an evaluation run
    is recorded.

    IMMUTABILITY CONTRACT: after seal() is called, this version and its
    chunks must never be mutated.  A new document change creates a NEW
    DocumentVersion with NEW chunks and NEW embeddings.
    """

    document_id: uuid.UUID
    version_tag: str = "1.0.0"
    source_uri: str = ""  # original file, URL, or object-storage path

    # Raw and processed content
    raw_text: str = ""
    content_type: str = "text/plain"  # MIME type
    language: str = "en"

    # Chunk manifest
    chunks: list[Chunk] = Field(default_factory=list)
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    total_chunks: int = 0
    total_tokens: int = 0
    total_chars: int = 0

    # Integrity
    content_hash: str = ""
    is_current: bool = True
    is_sealed: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_content_hash(self) -> str:
        """SHA-256 prefix of the raw text for change detection."""
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()[:16]

    def seal(self) -> None:
        """Compute derived fields and freeze the version.

        After sealing:
        - total_chunks, total_chars, total_tokens are finalized
        - content_hash is computed
        - each chunk's content_hash is computed
        - is_sealed is set to True
        - no further mutation is permitted
        """
        if self.is_sealed:
            raise ValueError(
                f"DocumentVersion {self.id} is already sealed. "
                "Create a new DocumentVersion instead of mutating."
            )
        for chunk in self.chunks:
            chunk.seal()
        self.total_chunks = len(self.chunks)
        self.total_chars = sum(c.char_count for c in self.chunks)
        self.total_tokens = sum(c.token_count for c in self.chunks)
        self.content_hash = self.compute_content_hash()
        self.is_sealed = True


class Document(DomainEntity):
    """A logical document in the knowledge base.

    A document is the unit of authorship and governance.  It may have many
    versions; only one version is considered "current" at any point in time.
    """

    collection_id: uuid.UUID
    title: str
    description: str = ""
    source: str = ""  # e.g. "confluence", "notion", "s3://bucket/path"
    status: DocumentStatus = DocumentStatus.ACTIVE
    category: str = ""
    tags: list[str] = Field(default_factory=list)

    # Version lineage (most recent first)
    versions: list[DocumentVersion] = Field(default_factory=list)
    current_version_id: uuid.UUID | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def current_version(self) -> DocumentVersion | None:
        """Return the current (latest active) version, if any."""
        if self.current_version_id:
            for v in self.versions:
                if v.id == self.current_version_id:
                    return v
        # Fallback: first version marked is_current
        for v in self.versions:
            if v.is_current:
                return v
        return None


class KnowledgeCollection(DomainEntity):
    """A curated collection of documents available for retrieval.

    A collection is the retrieval boundary: when a task specifies which
    knowledge the agent may access, it references a collection.  This enables
    the evaluator to control exactly what knowledge is available during an
    evaluation and to detect when the agent accesses stale, wrong, or
    unexpected sources.
    """

    source_id: uuid.UUID = Field(default_factory=uuid.uuid4)  # FK to KnowledgeSource
    project_id: uuid.UUID | None = None
    name: str
    description: str = ""
    version: str = "1.0.0"

    documents: list[Document] = Field(default_factory=list)

    # Collection-level configuration defaults
    default_chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    default_chunk_size: int = 500  # target tokens per chunk
    default_chunk_overlap: int = 75  # tokens of overlap between chunks

    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_version_hash(self) -> str:
        """Composite hash over all document content hashes for pinning."""
        doc_hashes: list[str] = []
        for doc in self.documents:
            cv = doc.current_version
            if cv:
                doc_hashes.append(cv.compute_content_hash())
        doc_hashes.sort()
        composite = f"{self.name}:{self.version}:" + ",".join(doc_hashes)
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def total_chunks(self) -> int:
        return sum(
            (doc.current_version.total_chunks if doc.current_version else 0)
            for doc in self.documents
        )


class KnowledgeSource(DomainEntity):
    """A project-scoped permission boundary for knowledge.

    Every KnowledgeSource belongs to exactly one project.  This is the
    authorization unit: retrieval must verify that the requesting context
    has access to this source before returning any results.

    KnowledgeSource → KnowledgeCollection → Document is the ownership chain.
    """

    project_id: uuid.UUID
    name: str
    description: str = ""

    # Owned collections
    collections: list[KnowledgeCollection] = Field(default_factory=list)

    # Source-level defaults (inherited by collections unless overridden)
    default_chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    default_chunk_size: int = 500
    default_chunk_overlap: int = 75

    # Permission descriptor — who may access this source
    # (extensible: start with project-level, add tenant/role later)
    allowed_project_ids: list[uuid.UUID] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_authorized_for_project(self, project_id: uuid.UUID) -> bool:
        """Check if the given project has access to this knowledge source."""
        # The owning project always has access
        if project_id == self.project_id:
            return True
        return project_id in self.allowed_project_ids


# ─────────────────────────────────────────────────────────────────────────────
# Authorization context — established BEFORE retrieval
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalContext(BaseModel):
    """Authorization boundary for a retrieval request.

    This context is established BEFORE any retrieval executes.
    The retrieval layer guarantees:

        Every returned chunk ∈ authorized knowledge scope

    This becomes an evaluation invariant — AgentForge can create security
    tests that verify:
        "Given Agent A has access to Collection X,
         When Agent A searches for 'salary policy',
         Then Evidence must never contain chunks from Collection Y."
    """

    project_id: uuid.UUID
    agent_id: uuid.UUID | None = None
    tenant_id: str | None = None

    # Explicitly allowed scope — populated by the authorization layer
    allowed_source_ids: list[uuid.UUID] = Field(default_factory=list)
    allowed_collection_ids: list[uuid.UUID] = Field(default_factory=list)

    # The requesting principal (for audit trail)
    requesting_principal: str = ""

    def is_collection_authorized(self, collection_id: uuid.UUID) -> bool:
        """Check whether retrieval from this collection is permitted."""
        return collection_id in self.allowed_collection_ids

    def is_source_authorized(self, source_id: uuid.UUID) -> bool:
        """Check whether retrieval from this source is permitted."""
        return source_id in self.allowed_source_ids


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval configuration snapshot — for reproducibility
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalConfiguration(BaseModel):
    """Frozen snapshot of how retrieval was configured for a run.

    This is the reproducibility invariant.  When a retrieval is replayed,
    comparing the configuration snapshot tells you *what* changed:
    - embedding model changed?
    - fusion strategy changed?
    - reranker changed?
    - filters changed?
    - top_k changed?

    This snapshot is stored on RetrievalRun and is IMMUTABLE.
    """

    # Embedding
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_version: str = ""

    # Retrieval pipeline
    retrieval_methods: list[str] = Field(default_factory=lambda: ["keyword", "semantic"])
    fusion_strategy: FusionStrategy = FusionStrategy.RECIPROCAL_RANK
    top_k: int = 5
    min_score: float = 0.0

    # Reranker
    reranker: str = ""  # empty = no reranker
    reranker_version: str = ""

    # Filters
    filters: dict[str, Any] = Field(default_factory=dict)

    # Knowledge state
    knowledge_snapshot_hash: str = ""  # collection version hash at retrieval time

    def compute_config_hash(self) -> str:
        """Deterministic hash of the retrieval configuration."""
        data = {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_version": self.embedding_version,
            "retrieval_methods": sorted(self.retrieval_methods),
            "fusion_strategy": self.fusion_strategy.value if hasattr(self.fusion_strategy, "value") else self.fusion_strategy,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "reranker": self.reranker,
            "reranker_version": self.reranker_version,
            "filters": self.filters,
            "knowledge_snapshot_hash": self.knowledge_snapshot_hash,
        }
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────────────────────────────────────
# Query / Result / Run — the retrieval evaluation event
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalQuery(BaseModel):
    """A structured query submitted to the retrieval engine.

    The agent runtime or evaluation pipeline constructs this.
    The retrieval engine resolves it into ranked RetrievalResult items
    within a RetrievalRun.
    """

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    collection_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    trial_id: uuid.UUID | None = None
    top_k: int = 5
    method: RetrievalMethod = RetrievalMethod.HYBRID
    filters: dict[str, Any] = Field(default_factory=dict)
    min_score: float = 0.0
    include_metadata: bool = True

    model_config = {"use_enum_values": True}

    def compute_query_hash(self) -> str:
        """Deterministic hash of the query for reproducibility tracking."""
        data = {
            "text": self.text,
            "collection_id": str(self.collection_id) if self.collection_id else "",
            "top_k": self.top_k,
            "method": self.method,
            "filters": self.filters,
            "min_score": self.min_score,
        }
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class RetrievalResult(BaseModel):
    """A single ranked retrieval hit returned by the retrieval engine.

    This is a raw result *within* a RetrievalRun.  The evaluator inspects
    these to determine whether the retrieval was appropriate.
    """

    retrieval_run_id: uuid.UUID | None = None

    document_id: uuid.UUID
    document_version_id: uuid.UUID | None = None
    chunk_id: uuid.UUID
    text: str
    chunk_hash: str = ""  # chunk content hash for integrity verification
    score: float = 0.0
    rank: int = 0

    # Provenance
    source: str = ""  # document title or URI
    page: int | None = None
    section: str | None = None
    category: str = ""
    tags: list[str] = Field(default_factory=list)

    # Retrieval metadata
    retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRun(DomainEntity):
    """A single retrieval operation — the evaluation event.

    A RetrievalRun captures the complete context of one retrieval:
    - WHO requested it (RetrievalContext — project, agent, principal)
    - WHAT was asked (RetrievalQuery — text, filters, top_k)
    - HOW it was configured (RetrievalConfiguration — models, methods, fusion)
    - WHAT was returned (RetrievalResult[] — ranked hits)
    - WHAT evidence was produced (Evidence[] — provenance records)
    - WHAT happened (status, duration, errors)

    This is the unit that the evaluation engine inspects.  One agent turn
    may produce multiple RetrievalRuns if the agent makes multiple
    knowledge requests.

    The trace becomes:
        Agent Run
           └── Knowledge Request
                  └── RetrievalRun
                         ├── Result 1
                         ├── Result 2
                         ├── Result 3
                         └── Evidence[]
    """

    # Correlation — links back to the agent execution
    project_id: uuid.UUID
    trial_id: uuid.UUID | None = None
    agent_run_id: uuid.UUID | None = None

    # Authorization context snapshot (frozen at request time)
    context: RetrievalContext

    # What was asked
    query: RetrievalQuery

    # How retrieval was configured (frozen snapshot)
    configuration: RetrievalConfiguration = Field(
        default_factory=RetrievalConfiguration
    )

    # Query reproducibility
    query_hash: str = ""
    config_hash: str = ""

    # Results
    results: list[RetrievalResult] = Field(default_factory=list)
    total_results: int = 0

    # Status and timing
    status: RetrievalRunStatus = RetrievalRunStatus.STARTED
    started_at: datetime = Field(default_factory=now_utc)
    completed_at: datetime | None = None
    duration_ms: float = 0.0

    # Errors
    error: str | None = None
    error_type: str | None = None

    # Authorization audit
    authorization_checked: bool = False
    authorization_passed: bool = False

    def seal(self) -> None:
        """Finalize the retrieval run after results are collected."""
        self.query_hash = self.query.compute_query_hash()
        self.config_hash = self.configuration.compute_config_hash()
        self.total_results = len(self.results)
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000

    def mark_completed(self) -> None:
        """Mark the run as completed."""
        self.status = RetrievalRunStatus.COMPLETED
        self.completed_at = now_utc()
        self.seal()

    def mark_failed(self, error: str, error_type: str = "") -> None:
        """Mark the run as failed with error details."""
        self.status = RetrievalRunStatus.FAILED
        self.error = error
        self.error_type = error_type
        self.completed_at = now_utc()
        self.seal()

    def mark_authorization_denied(self, reason: str = "") -> None:
        """Mark the run as denied due to authorization failure."""
        self.status = RetrievalRunStatus.AUTHORIZATION_DENIED
        self.authorization_checked = True
        self.authorization_passed = False
        self.error = reason or "Authorization denied"
        self.error_type = "authorization_denied"
        self.completed_at = now_utc()
        self.seal()


# ─────────────────────────────────────────────────────────────────────────────
# Evidence — the formal provenance record for evaluation
# ─────────────────────────────────────────────────────────────────────────────


class Evidence(DomainEntity):
    """A formal evidence item connecting an agent claim to retrieved knowledge.

    Evidence ≠ Chunk.

    A Chunk is stored knowledge.
    An Evidence item is a HISTORICAL RETRIEVAL EVENT recording:
    - what was retrieved
    - from which exact version
    - why it was retrieved (query, configuration)
    - under which permissions (authorization context)
    - at what rank and score
    - at what time

    The evaluator uses Evidence objects to determine whether:
    - the agent cited the right document
    - the cited passage supports the claim
    - the evidence is from the current version (not stale)
    - the agent's reasoning is faithful to the source text
    - the retrieval was authorized
    - the result is reproducible
    """

    # Link to the retrieval event
    retrieval_run_id: uuid.UUID | None = None
    trial_id: uuid.UUID | None = None
    query_id: str = ""

    # What the agent said
    claim: str = ""
    citation: str = ""

    # What was actually retrieved — full provenance chain
    collection_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    retrieved_text: str = ""
    chunk_hash: str = ""  # integrity: hash of the chunk text at retrieval time
    retrieval_score: float = 0.0
    retrieval_rank: int = 0

    # Source provenance
    document_title: str = ""
    source: str = ""
    page: int | None = None
    section: str | None = None

    # Retrieval method and configuration
    retrieval_method: RetrievalMethod | None = None
    embedding_model: str = ""
    embedding_version: str = ""

    # Timestamp
    retrieved_at: datetime = Field(default_factory=now_utc)

    # Evaluator annotations (filled by RAG verifiers, not by the agent)
    relevance: EvidenceRelevance | None = None
    faithfulness_score: float | None = None  # 0.0–1.0
    is_supported: bool | None = None
    is_current_version: bool | None = None
    is_authorized: bool | None = None  # was this from an authorized collection?
    evaluator_notes: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
        *,
        retrieval_run_id: uuid.UUID | None = None,
        collection_id: uuid.UUID | None = None,
        claim: str = "",
        citation: str = "",
        trial_id: uuid.UUID | None = None,
        query_id: str = "",
        embedding_model: str = "",
        embedding_version: str = "",
        retrieval_method: RetrievalMethod | None = None,
    ) -> Evidence:
        """Construct an Evidence object from a raw retrieval hit.

        This factory ensures that every Evidence item carries full provenance
        back to the retrieval event and the stored chunk.
        """
        return cls(
            retrieval_run_id=retrieval_run_id,
            trial_id=trial_id,
            query_id=query_id,
            claim=claim,
            citation=citation,
            collection_id=collection_id,
            document_id=result.document_id,
            document_version_id=result.document_version_id,
            chunk_id=result.chunk_id,
            retrieved_text=result.text,
            chunk_hash=result.chunk_hash,
            retrieval_score=result.score,
            retrieval_rank=result.rank,
            document_title=result.source,
            source=result.source,
            page=result.page,
            section=result.section,
            retrieval_method=retrieval_method or (
                result.retrieval_method
                if isinstance(result.retrieval_method, RetrievalMethod)
                else None
            ),
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Protocols — the contracts that infrastructure implementations satisfy
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class RetrievalService(Protocol):
    """Provider-agnostic interface for knowledge retrieval.

    Implementations MUST:
    - Accept a RetrievalQuery AND a RetrievalContext.
    - Verify authorization BEFORE executing retrieval.
    - Return a RetrievalRun (the complete evaluation event).
    - Never expose embedding infrastructure details to callers.
    - Never return chunks from unauthorized collections.

    The application layer depends on this protocol.
    The application layer never imports vector-store or embedding classes.
    """

    async def retrieve(
        self,
        query: RetrievalQuery,
        context: RetrievalContext,
    ) -> RetrievalRun:
        """Execute a retrieval query within the given authorization context.

        Returns a RetrievalRun containing ranked results and evidence.
        Authorization is checked BEFORE retrieval.
        """
        ...

    async def index_collection(self, collection: KnowledgeCollection) -> None:
        """Index (or re-index) all chunks in a collection for retrieval."""
        ...


@runtime_checkable
class RerankingService(Protocol):
    """Interface for result reranking.

    Phase 14 default: deterministic lexical/score reranking (no LLM call).
    A cross-encoder or LLM reranker can be added later without changing
    RetrievalService.
    """

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank results and return them in new order with updated scores."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding generation.

    Implementations:
    - OpenAIEmbeddingProvider (production)
    - FakeEmbeddingProvider (tests)

    The domain layer does not depend on any specific embedding provider.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    @property
    def model_name(self) -> str:
        """The model identifier (e.g. 'text-embedding-3-small')."""
        ...

    @property
    def dimensions(self) -> int:
        """The dimensionality of the output vectors."""
        ...

    @property
    def provider_name(self) -> str:
        """The provider identifier (e.g. 'openai')."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Failure types — explicit, typed errors for the retrieval pipeline
# ─────────────────────────────────────────────────────────────────────────────


class KnowledgeError(Exception):
    """Base class for all knowledge/retrieval pipeline errors."""

    def __init__(self, message: str, error_type: str = "") -> None:
        super().__init__(message)
        self.error_type = error_type


class AuthorizationError(KnowledgeError):
    """Retrieval denied due to authorization failure."""

    def __init__(self, message: str = "Authorization denied") -> None:
        super().__init__(message, error_type="authorization_denied")


class EmbeddingProviderError(KnowledgeError):
    """Embedding provider failure (timeout, rate limit, unavailable)."""

    def __init__(self, message: str, error_type: str = "embedding_failure") -> None:
        super().__init__(message, error_type=error_type)


class EmbeddingTimeoutError(EmbeddingProviderError):
    """Embedding provider request timed out."""

    def __init__(self, message: str = "Embedding request timed out") -> None:
        super().__init__(message, error_type="embedding_timeout")


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Embedding provider rate limit exceeded."""

    def __init__(self, message: str = "Embedding rate limit exceeded") -> None:
        super().__init__(message, error_type="embedding_rate_limit")


class RetrievalTimeoutError(KnowledgeError):
    """Retrieval operation timed out."""

    def __init__(self, message: str = "Retrieval timed out") -> None:
        super().__init__(message, error_type="retrieval_timeout")


class RerankerError(KnowledgeError):
    """Reranker failure."""

    def __init__(self, message: str = "Reranker failed") -> None:
        super().__init__(message, error_type="reranker_failure")


class IngestionError(KnowledgeError):
    """Document ingestion failure (parse, chunk, embed, persist)."""

    def __init__(self, message: str, error_type: str = "ingestion_failure") -> None:
        super().__init__(message, error_type=error_type)


class ParserError(IngestionError):
    """Document parsing failure."""

    def __init__(self, message: str = "Document parsing failed") -> None:
        super().__init__(message, error_type="parser_failure")


class ChunkingError(IngestionError):
    """Chunking failure."""

    def __init__(self, message: str = "Chunking failed") -> None:
        super().__init__(message, error_type="chunking_failure")


class EvidenceConstructionError(KnowledgeError):
    """Evidence construction failure."""

    def __init__(self, message: str = "Evidence construction failed") -> None:
        super().__init__(message, error_type="evidence_construction_failure")


class DatabaseUnavailableError(KnowledgeError):
    """Knowledge database is unavailable."""

    def __init__(self, message: str = "Knowledge database unavailable") -> None:
        super().__init__(message, error_type="database_unavailable")


class MalformedDocumentError(IngestionError):
    """Document is malformed or corrupt."""

    def __init__(self, message: str = "Document is malformed") -> None:
        super().__init__(message, error_type="malformed_document")


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    # Enums
    "DocumentStatus",
    "ChunkStrategy",
    "RetrievalMethod",
    "FusionStrategy",
    "RetrievalRunStatus",
    "EvidenceRelevance",
    # Storage entities
    "Chunk",
    "DocumentVersion",
    "Document",
    "KnowledgeCollection",
    "KnowledgeSource",
    # Authorization
    "RetrievalContext",
    # Configuration
    "RetrievalConfiguration",
    # Retrieval event
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalRun",
    # Evidence
    "Evidence",
    # Protocols
    "RetrievalService",
    "RerankingService",
    "EmbeddingProvider",
    # Errors
    "KnowledgeError",
    "AuthorizationError",
    "EmbeddingProviderError",
    "EmbeddingTimeoutError",
    "EmbeddingRateLimitError",
    "RetrievalTimeoutError",
    "RerankerError",
    "IngestionError",
    "ParserError",
    "ChunkingError",
    "EvidenceConstructionError",
    "DatabaseUnavailableError",
    "MalformedDocumentError",
]
