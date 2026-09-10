"""Tests for the Document Ingestion Pipeline."""

import uuid
import pytest

from backend.domain.knowledge import Document, ChunkStrategy
from backend.infrastructure.retrieval.embeddings import FakeEmbeddingProvider
from backend.application.knowledge.ingestion_service import (
    SimpleChunkingEngine,
    DocumentIngestionPipeline,
)


def test_simple_chunking_engine_paragraph():
    engine = SimpleChunkingEngine()
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = engine.chunk(text, strategy=ChunkStrategy.PARAGRAPH)
    
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph 1."
    assert chunks[1] == "Paragraph 2."
    assert chunks[2] == "Paragraph 3."


def test_simple_chunking_engine_fixed_size():
    engine = SimpleChunkingEngine(chunk_size=10, chunk_overlap=2)
    text = "0123456789012345"
    chunks = engine.chunk(text, strategy=ChunkStrategy.FIXED_SIZE)
    
    assert len(chunks) == 2
    # 0 to 10
    assert chunks[0] == "0123456789"
    # overlap is 2, so starts at 8, length 10 -> up to end
    assert chunks[1] == "89012345"


@pytest.mark.asyncio
async def test_document_ingestion_pipeline():
    provider = FakeEmbeddingProvider(dimensions=10)
    engine = SimpleChunkingEngine(chunk_size=100, chunk_overlap=0)
    pipeline = DocumentIngestionPipeline(
        embedding_provider=provider,
        chunking_engine=engine
    )

    doc = Document(
        source_id=uuid.uuid4(),
        collection_id=uuid.uuid4(),
        title="Test Doc"
    )

    text = "This is a test document.\n\nIt has multiple sentences."
    
    version, embeddings = await pipeline.ingest_document(
        document=doc,
        raw_text=text,
        chunk_strategy=ChunkStrategy.PARAGRAPH,
        metadata={"author": "AI"}
    )

    assert version.document_id == doc.id
    assert version.raw_text == text
    assert version.chunk_strategy == ChunkStrategy.PARAGRAPH
    assert version.is_sealed is True
    assert version.content_hash != ""
    assert version.metadata["author"] == "AI"

    assert len(version.chunks) == 2
    assert version.chunks[0].text == "This is a test document."
    assert version.chunks[1].text == "It has multiple sentences."
    
    assert version.chunks[0].content_hash != ""
    assert version.chunks[1].content_hash != ""

    assert version.chunks[0].sequence_index == 0
    assert version.chunks[1].sequence_index == 1

    # Check embeddings
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 10
