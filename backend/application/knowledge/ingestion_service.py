"""Document Ingestion Service for Knowledge & RAG.

This module provides the pipeline for taking raw documents, chunking them,
generating embeddings, and coordinating their preparation for persistence.
"""

import uuid
from typing import Any, Protocol

from backend.domain.knowledge import (
    Chunk,
    ChunkStrategy,
    Document,
    DocumentVersion,
    EmbeddingProvider,
)


class ChunkingEngine(Protocol):
    """Protocol for chunking documents into smaller pieces."""

    def chunk(self, text: str, strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE) -> list[str]:
        """Split text into chunks based on the given strategy."""
        ...


class SimpleChunkingEngine:
    """A basic chunking engine supporting fixed-size and paragraph chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE) -> list[str]:
        if not text:
            return []

        if strategy == ChunkStrategy.PARAGRAPH:
            # Simple double-newline split
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            return paragraphs if paragraphs else [text]
        
        # Default to FIXED_SIZE
        chunks = []
        start = 0
        text_len = len(text)
        
        # Ensure we always make progress
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size - 1
            
        while start < text_len:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= text_len:
                break
            start = end - self.chunk_overlap
                
        return chunks


class DocumentIngestionPipeline:
    """Coordinates the ingestion of documents into the knowledge base."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunking_engine: ChunkingEngine,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chunking_engine = chunking_engine

    async def ingest_document(
        self,
        document: Document,
        raw_text: str,
        chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[DocumentVersion, list[list[float]]]:
        """Create a sealed DocumentVersion with chunks and generate embeddings.
        
        Returns:
            A tuple of (DocumentVersion, embeddings) where embeddings is a list
            of vectors corresponding 1-to-1 with the version's chunks.
        """
        
        # 1. Create DocumentVersion
        version = DocumentVersion(
            document_id=document.id,
            raw_text=raw_text,
            chunk_strategy=chunk_strategy,
            metadata=metadata or {}
        )

        # 2. Chunking
        chunk_texts = self.chunking_engine.chunk(raw_text, strategy=chunk_strategy)
        
        chunks = []
        for i, text in enumerate(chunk_texts):
            chunk = Chunk(
                document_version_id=version.id,
                document_id=document.id,
                text=text,
                sequence_index=i,
                strategy=chunk_strategy
            )
            chunks.append(chunk)
            
        version.chunks = chunks

        # 3. Sealing (computes hashes and finalizes counts)
        version.seal()

        # 4. Generate Embeddings
        embeddings: list[list[float]] = []
        if chunks:
            embeddings = await self.embedding_provider.embed([c.text for c in chunks])
            
        return version, embeddings
