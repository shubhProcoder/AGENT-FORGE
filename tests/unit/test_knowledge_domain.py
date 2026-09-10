"""Unit tests for TASK 14.1 — Knowledge Domain + Retrieval Contract.

Validates the Five Evaluation Invariants:

1. Traceability — every Evidence links back through Chunk, DocumentVersion, Document,
   KnowledgeCollection, KnowledgeSource, RetrievalQuery, RetrievalConfiguration, and RetrievalRun.
2. Reproducibility — RetrievalConfiguration captures frozen snapshot and computes config hash;
   RetrievalQuery computes query hash; DocumentVersion/Collection compute version hashes.
3. Authorization — RetrievalContext establishes allowed knowledge scope BEFORE retrieval;
   KnowledgeSource verifies project-level access.
4. Evidence ≠ Chunk — Chunk is stored knowledge; Evidence is a historical retrieval event
   capturing query, configuration, scores, and evaluator judgements.
5. Measurable Quality — Evidence supports evaluator relevance, faithfulness, version freshness,
   and authorization annotations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
import pytest

from backend.domain.knowledge import (
    AuthorizationError,
    Chunk,
    ChunkStrategy,
    ChunkingError,
    DatabaseUnavailableError,
    Document,
    DocumentStatus,
    DocumentVersion,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    Evidence,
    EvidenceConstructionError,
    EvidenceRelevance,
    FusionStrategy,
    IngestionError,
    KnowledgeCollection,
    KnowledgeError,
    KnowledgeSource,
    MalformedDocumentError,
    ParserError,
    RerankerError,
    RerankingService,
    RetrievalConfiguration,
    RetrievalContext,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    RetrievalRun,
    RetrievalRunStatus,
    RetrievalService,
    RetrievalTimeoutError,
)


# ============================================================================
# Entity hierarchy construction helpers
# ============================================================================

def _make_chunk(doc_version_id: uuid.UUID, doc_id: uuid.UUID, text: str, index: int = 0) -> Chunk:
    return Chunk(
        document_version_id=doc_version_id,
        document_id=doc_id,
        text=text,
        sequence_index=index,
        char_count=len(text),
        token_count=len(text.split()),
        strategy=ChunkStrategy.PARAGRAPH,
    )


def _make_document_version(doc_id: uuid.UUID, raw_text: str, tag: str = "1.0.0") -> DocumentVersion:
    dv = DocumentVersion(
        document_id=doc_id,
        version_tag=tag,
        raw_text=raw_text,
        content_type="text/plain",
    )
    # Simulate chunking: one chunk per paragraph
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        dv.chunks.append(_make_chunk(dv.id, doc_id, para, i))
    dv.seal()
    return dv


def _make_document(collection_id: uuid.UUID, title: str, text: str) -> Document:
    doc = Document(collection_id=collection_id, title=title, source="test")
    dv = _make_document_version(doc.id, text)
    doc.versions.append(dv)
    doc.current_version_id = dv.id
    return doc


def _make_collection(
    name: str = "Policy KB",
    texts: dict[str, str] | None = None,
    source_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
) -> KnowledgeCollection:
    source_id = source_id or uuid.uuid4()
    coll = KnowledgeCollection(source_id=source_id, project_id=project_id, name=name)
    if texts:
        for title, text in texts.items():
            coll.documents.append(_make_document(coll.id, title, text))
    return coll


def _make_source(
    name: str = "Corporate Knowledge",
    project_id: uuid.UUID | None = None,
    allowed_project_ids: list[uuid.UUID] | None = None,
) -> KnowledgeSource:
    return KnowledgeSource(
        project_id=project_id or uuid.uuid4(),
        name=name,
        allowed_project_ids=allowed_project_ids or [],
    )


# ============================================================================
# Tests: Chunk
# ============================================================================

class TestChunk:
    def test_chunk_fields(self):
        doc_id = uuid.uuid4()
        dv_id = uuid.uuid4()
        chunk = _make_chunk(dv_id, doc_id, "Refund policy requires receipt.", 0)

        assert chunk.document_id == doc_id
        assert chunk.document_version_id == dv_id
        assert chunk.text == "Refund policy requires receipt."
        assert chunk.sequence_index == 0
        assert chunk.char_count == len("Refund policy requires receipt.")
        assert chunk.token_count == 4
        assert chunk.strategy == ChunkStrategy.PARAGRAPH

    def test_content_hash_deterministic(self):
        dv_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        c1 = _make_chunk(dv_id, doc_id, "same text")
        c2 = _make_chunk(dv_id, doc_id, "same text")
        assert c1.compute_content_hash() == c2.compute_content_hash()

    def test_content_hash_changes_with_text(self):
        dv_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        c1 = _make_chunk(dv_id, doc_id, "text A")
        c2 = _make_chunk(dv_id, doc_id, "text B")
        assert c1.compute_content_hash() != c2.compute_content_hash()


# ============================================================================
# Tests: DocumentVersion
# ============================================================================

class TestDocumentVersion:
    def test_seal_computes_derived_fields(self):
        doc_id = uuid.uuid4()
        dv = _make_document_version(doc_id, "First paragraph.\n\nSecond paragraph.")

        assert dv.total_chunks == 2
        assert dv.total_chars > 0
        assert dv.total_tokens > 0
        assert len(dv.content_hash) == 16
        assert dv.is_sealed is True

    def test_content_hash_deterministic(self):
        doc_id = uuid.uuid4()
        dv1 = _make_document_version(doc_id, "Deterministic text.")
        dv2 = _make_document_version(doc_id, "Deterministic text.")
        assert dv1.compute_content_hash() == dv2.compute_content_hash()

    def test_content_hash_changes_on_edit(self):
        doc_id = uuid.uuid4()
        dv1 = _make_document_version(doc_id, "Version 1 text.")
        dv2 = _make_document_version(doc_id, "Version 2 text — updated.")
        assert dv1.compute_content_hash() != dv2.compute_content_hash()

    def test_cannot_seal_twice(self):
        doc_id = uuid.uuid4()
        dv = _make_document_version(doc_id, "Immutable text.")
        with pytest.raises(ValueError, match="already sealed"):
            dv.seal()


# ============================================================================
# Tests: Document
# ============================================================================

class TestDocument:
    def test_current_version(self):
        coll_id = uuid.uuid4()
        doc = _make_document(coll_id, "Refund Policy", "Refunds are allowed within 30 days.")

        cv = doc.current_version
        assert cv is not None
        assert cv.version_tag == "1.0.0"
        assert "Refunds are allowed" in cv.raw_text

    def test_document_status_default(self):
        coll_id = uuid.uuid4()
        doc = Document(collection_id=coll_id, title="Test")
        assert doc.status == DocumentStatus.ACTIVE

    def test_current_version_none_when_empty(self):
        coll_id = uuid.uuid4()
        doc = Document(collection_id=coll_id, title="Empty")
        assert doc.current_version is None


# ============================================================================
# Tests: KnowledgeSource
# ============================================================================

class TestKnowledgeSource:
    def test_ownership_and_authorization(self):
        owner_proj = uuid.uuid4()
        collab_proj = uuid.uuid4()
        stranger_proj = uuid.uuid4()

        source = _make_source("HR Knowledge", project_id=owner_proj, allowed_project_ids=[collab_proj])

        assert source.is_authorized_for_project(owner_proj) is True
        assert source.is_authorized_for_project(collab_proj) is True
        assert source.is_authorized_for_project(stranger_proj) is False

    def test_source_with_collections(self):
        source = _make_source("Support Knowledge")
        c1 = KnowledgeCollection(source_id=source.id, name="KB-1")
        c2 = KnowledgeCollection(source_id=source.id, name="KB-2")
        source.collections.extend([c1, c2])

        assert len(source.collections) == 2
        assert source.collections[0].source_id == source.id


# ============================================================================
# Tests: KnowledgeCollection
# ============================================================================

class TestKnowledgeCollection:
    def test_collection_with_documents(self):
        coll = _make_collection("Support KB", {
            "Refund Policy": "Refunds require a receipt.\n\nRefunds are processed in 5 business days.",
            "Shipping Policy": "Free shipping over $50.\n\nExpress shipping is $9.99.",
        })

        assert coll.total_documents == 2
        assert coll.total_chunks == 4  # 2 paragraphs per document

    def test_version_hash_deterministic(self):
        texts = {"Doc A": "Content A.\n\nMore A.", "Doc B": "Content B."}
        c1 = _make_collection("KB", texts)
        c2 = _make_collection("KB", texts)
        assert c1.compute_version_hash() == c2.compute_version_hash()

    def test_version_hash_changes_on_content_change(self):
        c1 = _make_collection("KB", {"Doc": "Version 1 text."})
        c2 = _make_collection("KB", {"Doc": "Version 2 text — different."})
        assert c1.compute_version_hash() != c2.compute_version_hash()

    def test_empty_collection(self):
        coll = _make_collection("Empty")
        assert coll.total_documents == 0
        assert coll.total_chunks == 0
        assert len(coll.compute_version_hash()) == 16


# ============================================================================
# Tests: RetrievalContext (Authorization Boundary)
# ============================================================================

class TestRetrievalContext:
    def test_authorization_scope(self):
        proj_id = uuid.uuid4()
        c1_id = uuid.uuid4()
        c2_id = uuid.uuid4()
        c3_id = uuid.uuid4()
        s1_id = uuid.uuid4()

        ctx = RetrievalContext(
            project_id=proj_id,
            allowed_source_ids=[s1_id],
            allowed_collection_ids=[c1_id, c2_id],
            requesting_principal="agent-executor-1",
        )

        assert ctx.is_collection_authorized(c1_id) is True
        assert ctx.is_collection_authorized(c2_id) is True
        assert ctx.is_collection_authorized(c3_id) is False
        assert ctx.is_source_authorized(s1_id) is True
        assert ctx.is_source_authorized(uuid.uuid4()) is False
        assert ctx.requesting_principal == "agent-executor-1"


# ============================================================================
# Tests: RetrievalConfiguration (Reproducibility Snapshot)
# ============================================================================

class TestRetrievalConfiguration:
    def test_config_hash_deterministic(self):
        cfg1 = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            fusion_strategy=FusionStrategy.RECIPROCAL_RANK,
            top_k=5,
            knowledge_snapshot_hash="abc123hash",
        )
        cfg2 = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            fusion_strategy=FusionStrategy.RECIPROCAL_RANK,
            top_k=5,
            knowledge_snapshot_hash="abc123hash",
        )
        assert cfg1.compute_config_hash() == cfg2.compute_config_hash()

    def test_config_hash_detects_changes(self):
        cfg_base = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            top_k=5,
        )
        cfg_diff_model = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            top_k=5,
        )
        cfg_diff_k = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            top_k=10,
        )
        cfg_diff_fusion = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            top_k=5,
        )

        h_base = cfg_base.compute_config_hash()
        assert h_base != cfg_diff_model.compute_config_hash()
        assert h_base != cfg_diff_k.compute_config_hash()
        assert h_base != cfg_diff_fusion.compute_config_hash()


# ============================================================================
# Tests: RetrievalQuery
# ============================================================================

class TestRetrievalQuery:
    def test_defaults(self):
        q = RetrievalQuery(text="What is the refund policy?")
        assert q.text == "What is the refund policy?"
        assert q.top_k == 5
        assert q.method == RetrievalMethod.HYBRID.value
        assert q.min_score == 0.0

    def test_scoped_to_collection(self):
        coll_id = uuid.uuid4()
        q = RetrievalQuery(text="refund", collection_id=coll_id, top_k=3)
        assert q.collection_id == coll_id
        assert q.top_k == 3

    def test_correlation_ids(self):
        run_id = uuid.uuid4()
        trial_id = uuid.uuid4()
        q = RetrievalQuery(text="query", run_id=run_id, trial_id=trial_id)
        assert q.run_id == run_id
        assert q.trial_id == trial_id

    def test_query_hash_deterministic(self):
        coll_id = uuid.uuid4()
        q1 = RetrievalQuery(text="shipping cost", collection_id=coll_id, top_k=5)
        q2 = RetrievalQuery(text="shipping cost", collection_id=coll_id, top_k=5)
        assert q1.compute_query_hash() == q2.compute_query_hash()

    def test_query_hash_changes_with_text(self):
        q1 = RetrievalQuery(text="shipping cost")
        q2 = RetrievalQuery(text="return policy")
        assert q1.compute_query_hash() != q2.compute_query_hash()


# ============================================================================
# Tests: RetrievalResult
# ============================================================================

class TestRetrievalResult:
    def test_fields(self):
        run_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        r = RetrievalResult(
            retrieval_run_id=run_id,
            document_id=doc_id,
            chunk_id=chunk_id,
            text="Refunds require a receipt.",
            chunk_hash="abc123hash",
            score=0.92,
            rank=1,
            source="Refund Policy",
            page=3,
            section="Returns",
        )
        assert r.retrieval_run_id == run_id
        assert r.text == "Refunds require a receipt."
        assert r.chunk_hash == "abc123hash"
        assert r.score == 0.92
        assert r.rank == 1
        assert r.source == "Refund Policy"
        assert r.page == 3

    def test_default_method(self):
        r = RetrievalResult(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            text="test",
        )
        assert r.retrieval_method == RetrievalMethod.HYBRID


# ============================================================================
# Tests: RetrievalRun (The Evaluation Event)
# ============================================================================

class TestRetrievalRun:
    def test_run_lifecycle_completed(self):
        proj_id = uuid.uuid4()
        ctx = RetrievalContext(project_id=proj_id)
        query = RetrievalQuery(text="return window")
        cfg = RetrievalConfiguration(embedding_model="text-embedding-3-small")

        run = RetrievalRun(
            project_id=proj_id,
            context=ctx,
            query=query,
            configuration=cfg,
        )
        assert run.status == RetrievalRunStatus.STARTED
        assert run.completed_at is None

        # Add a result and mark completed
        result = RetrievalResult(
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            text="Items can be returned within 30 days.",
            rank=1,
            score=0.95,
        )
        run.results.append(result)
        run.mark_completed()

        assert run.status == RetrievalRunStatus.COMPLETED
        assert run.total_results == 1
        assert run.completed_at is not None
        assert run.query_hash == query.compute_query_hash()
        assert run.config_hash == cfg.compute_config_hash()

    def test_run_lifecycle_failed(self):
        proj_id = uuid.uuid4()
        ctx = RetrievalContext(project_id=proj_id)
        query = RetrievalQuery(text="return window")

        run = RetrievalRun(project_id=proj_id, context=ctx, query=query)
        run.mark_failed("Embedding service timeout", error_type="embedding_timeout")

        assert run.status == RetrievalRunStatus.FAILED
        assert run.error == "Embedding service timeout"
        assert run.error_type == "embedding_timeout"
        assert run.completed_at is not None

    def test_run_lifecycle_authorization_denied(self):
        proj_id = uuid.uuid4()
        ctx = RetrievalContext(project_id=proj_id)
        query = RetrievalQuery(text="restricted salary doc")

        run = RetrievalRun(project_id=proj_id, context=ctx, query=query)
        run.mark_authorization_denied("Collection not in authorized scope")

        assert run.status == RetrievalRunStatus.AUTHORIZATION_DENIED
        assert run.authorization_checked is True
        assert run.authorization_passed is False
        assert "not in authorized scope" in (run.error or "")


# ============================================================================
# Tests: Evidence
# ============================================================================

class TestEvidence:
    def test_from_retrieval_result_with_full_provenance(self):
        run_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        version_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        trial_id = uuid.uuid4()
        coll_id = uuid.uuid4()

        result = RetrievalResult(
            retrieval_run_id=run_id,
            document_id=doc_id,
            document_version_id=version_id,
            chunk_id=chunk_id,
            text="Refunds require a receipt.",
            chunk_hash="chunkhash16chars",
            score=0.92,
            rank=1,
            source="Refund Policy",
            page=3,
            section="Returns",
        )

        evidence = Evidence.from_retrieval_result(
            result,
            retrieval_run_id=run_id,
            collection_id=coll_id,
            claim="Customer is eligible for a refund.",
            citation="per the refund policy",
            trial_id=trial_id,
            query_id="q-123",
            embedding_model="text-embedding-3-small",
            embedding_version="v1",
        )

        assert evidence.retrieval_run_id == run_id
        assert evidence.trial_id == trial_id
        assert evidence.query_id == "q-123"
        assert evidence.claim == "Customer is eligible for a refund."
        assert evidence.citation == "per the refund policy"
        assert evidence.collection_id == coll_id
        assert evidence.document_id == doc_id
        assert evidence.document_version_id == version_id
        assert evidence.chunk_id == chunk_id
        assert evidence.retrieved_text == "Refunds require a receipt."
        assert evidence.chunk_hash == "chunkhash16chars"
        assert evidence.retrieval_score == 0.92
        assert evidence.retrieval_rank == 1
        assert evidence.page == 3
        assert evidence.section == "Returns"
        assert evidence.embedding_model == "text-embedding-3-small"
        assert evidence.embedding_version == "v1"

    def test_evaluator_annotations_default_none(self):
        evidence = Evidence()
        assert evidence.relevance is None
        assert evidence.faithfulness_score is None
        assert evidence.is_supported is None
        assert evidence.is_current_version is None
        assert evidence.is_authorized is None

    def test_evaluator_can_annotate(self):
        evidence = Evidence(
            claim="Free shipping on all orders.",
            retrieved_text="Free shipping on orders over $50.",
            relevance=EvidenceRelevance.PARTIALLY_RELEVANT,
            faithfulness_score=0.4,
            is_supported=False,
            is_current_version=True,
            is_authorized=True,
            evaluator_notes="Claim overgeneralises; policy has a $50 threshold.",
        )
        assert evidence.relevance == EvidenceRelevance.PARTIALLY_RELEVANT
        assert evidence.faithfulness_score == 0.4
        assert evidence.is_supported is False
        assert evidence.is_current_version is True
        assert evidence.is_authorized is True
        assert "overgeneralises" in evidence.evaluator_notes


# ============================================================================
# Tests: Protocols (RetrievalService, RerankingService, EmbeddingProvider)
# ============================================================================

class TestProtocols:
    def test_retrieval_service_conformance(self):
        class ValidRetriever:
            async def retrieve(
                self,
                query: RetrievalQuery,
                context: RetrievalContext,
            ) -> RetrievalRun:
                return RetrievalRun(
                    project_id=context.project_id,
                    context=context,
                    query=query,
                )

            async def index_collection(self, collection: KnowledgeCollection) -> None:
                pass

        assert isinstance(ValidRetriever(), RetrievalService)

    def test_retrieval_service_non_conformance(self):
        class InvalidRetriever:
            def search(self, text: str) -> list:
                return []

        assert not isinstance(InvalidRetriever(), RetrievalService)

    def test_reranking_service_conformance(self):
        class ValidReranker:
            async def rerank(
                self,
                query: str,
                results: list[RetrievalResult],
            ) -> list[RetrievalResult]:
                return results

        assert isinstance(ValidReranker(), RerankingService)

    def test_embedding_provider_conformance(self):
        class ValidEmbedder:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 1536 for _ in texts]

            @property
            def model_name(self) -> str:
                return "text-embedding-3-small"

            @property
            def dimensions(self) -> int:
                return 1536

            @property
            def provider_name(self) -> str:
                return "fake"

        assert isinstance(ValidEmbedder(), EmbeddingProvider)


# ============================================================================
# Tests: Error Hierarchy
# ============================================================================

class TestKnowledgeErrors:
    def test_error_inheritance_and_types(self):
        err_auth = AuthorizationError("Tenant not allowed")
        assert isinstance(err_auth, KnowledgeError)
        assert err_auth.error_type == "authorization_denied"

        err_timeout = EmbeddingTimeoutError()
        assert isinstance(err_timeout, EmbeddingProviderError)
        assert isinstance(err_timeout, KnowledgeError)
        assert err_timeout.error_type == "embedding_timeout"

        err_rate = EmbeddingRateLimitError()
        assert isinstance(err_rate, EmbeddingProviderError)
        assert err_rate.error_type == "embedding_rate_limit"

        err_retrieval = RetrievalTimeoutError()
        assert isinstance(err_retrieval, KnowledgeError)
        assert err_retrieval.error_type == "retrieval_timeout"

        err_parse = ParserError("Failed to parse PDF")
        assert isinstance(err_parse, IngestionError)
        assert isinstance(err_parse, KnowledgeError)
        assert err_parse.error_type == "parser_failure"

        err_chunk = ChunkingError("Zero chunks produced")
        assert isinstance(err_chunk, IngestionError)
        assert err_chunk.error_type == "chunking_failure"

        err_db = DatabaseUnavailableError()
        assert isinstance(err_db, KnowledgeError)
        assert err_db.error_type == "database_unavailable"

        err_malformed = MalformedDocumentError()
        assert isinstance(err_malformed, IngestionError)
        assert err_malformed.error_type == "malformed_document"

        err_evidence = EvidenceConstructionError()
        assert isinstance(err_evidence, KnowledgeError)
        assert err_evidence.error_type == "evidence_construction_failure"

        err_reranker = RerankerError()
        assert isinstance(err_reranker, KnowledgeError)
        assert err_reranker.error_type == "reranker_failure"


# ============================================================================
# Tests: Five Evaluation Invariants Integration
# ============================================================================

class TestFiveEvaluationInvariants:
    """Simulates an end-to-end knowledge evaluation flow demonstrating all 5 invariants."""

    def test_end_to_end_knowledge_provenance_and_evaluation_flow(self):
        # 1. SETUP KNOWLEDGE HIERARCHY
        project_id = uuid.uuid4()
        source = _make_source("Company Docs", project_id=project_id)
        coll = _make_collection("HR Policies", source_id=source.id, project_id=project_id)
        source.collections.append(coll)

        doc = _make_document(
            coll.id,
            "Leave Policy",
            "Employees are entitled to 20 annual leave days.\n\nCarry over is up to 5 days.",
        )
        coll.documents.append(doc)
        doc_version = doc.current_version
        assert doc_version is not None
        chunk_0 = doc_version.chunks[0]

        # 2. INVARIANT 3: AUTHORIZATION BEFORE RETRIEVAL
        context = RetrievalContext(
            project_id=project_id,
            allowed_source_ids=[source.id],
            allowed_collection_ids=[coll.id],
            requesting_principal="agent-hr-bot",
        )
        assert context.is_collection_authorized(coll.id) is True

        # Unauthorized collection check
        unauthorized_coll_id = uuid.uuid4()
        assert context.is_collection_authorized(unauthorized_coll_id) is False

        # 3. INVARIANT 2: REPRODUCIBILITY (Query + Configuration snapshot)
        query = RetrievalQuery(
            text="How many annual leave days do employees get?",
            collection_id=coll.id,
            top_k=3,
        )
        config = RetrievalConfiguration(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            embedding_version="2024-05",
            fusion_strategy=FusionStrategy.RECIPROCAL_RANK,
            top_k=3,
            knowledge_snapshot_hash=coll.compute_version_hash(),
        )

        trial_id = uuid.uuid4()
        run = RetrievalRun(
            project_id=project_id,
            trial_id=trial_id,
            context=context,
            query=query,
            configuration=config,
            authorization_checked=True,
            authorization_passed=True,
        )

        # 4. RETRIEVAL EXECUTION → RETRIEVAL RESULTS
        result = RetrievalResult(
            retrieval_run_id=run.id,
            document_id=doc.id,
            document_version_id=doc_version.id,
            chunk_id=chunk_0.id,
            text=chunk_0.text,
            chunk_hash=chunk_0.content_hash,
            score=0.96,
            rank=1,
            source=doc.title,
            section="Annual Leave",
        )
        run.results.append(result)
        run.mark_completed()

        assert run.status == RetrievalRunStatus.COMPLETED
        assert run.query_hash == query.compute_query_hash()
        assert run.config_hash == config.compute_config_hash()

        # 5. INVARIANT 4: EVIDENCE ≠ CHUNK (Agent cites evidence in claim)
        agent_claim = "You are entitled to 20 annual leave days each year."
        agent_citation = "per the Leave Policy"

        evidence = Evidence.from_retrieval_result(
            result,
            retrieval_run_id=run.id,
            collection_id=coll.id,
            claim=agent_claim,
            citation=agent_citation,
            trial_id=trial_id,
            query_id=query.query_id,
            embedding_model=config.embedding_model,
            embedding_version=config.embedding_version,
        )

        # 6. INVARIANT 1: TRACEABILITY VERIFICATION
        assert evidence.retrieval_run_id == run.id
        assert evidence.trial_id == trial_id
        assert evidence.collection_id == coll.id
        assert evidence.document_id == doc.id
        assert evidence.document_version_id == doc_version.id
        assert evidence.chunk_id == chunk_0.id
        assert evidence.chunk_hash == chunk_0.content_hash
        assert evidence.retrieved_text == chunk_0.text

        # 7. INVARIANT 5: MEASURABLE QUALITY (Evaluator annotations)
        evidence.relevance = EvidenceRelevance.RELEVANT
        evidence.faithfulness_score = 1.0
        evidence.is_supported = True
        evidence.is_current_version = (doc_version.id == doc.current_version_id)
        evidence.is_authorized = context.is_collection_authorized(evidence.collection_id)
        evidence.evaluator_notes = "Exact factual match to section 1."

        assert evidence.is_supported is True
        assert evidence.is_current_version is True
        assert evidence.is_authorized is True
        assert evidence.faithfulness_score == 1.0
