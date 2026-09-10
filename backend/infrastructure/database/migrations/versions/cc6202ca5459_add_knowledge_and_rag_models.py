"""Add knowledge and RAG models

Revision ID: cc6202ca5459
Revises: 
Create Date: 2026-09-10 12:37:53.429683

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cc6202ca5459'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Ensure vector extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # knowledge_sources
    op.create_table('knowledge_sources',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=256), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source_type', sa.String(length=64), nullable=False),
        sa.Column('configuration', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_sources_project_id'), 'knowledge_sources', ['project_id'], unique=False)

    # knowledge_collections
    op.create_table('knowledge_collections',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=256), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_collections_project_id'), 'knowledge_collections', ['project_id'], unique=False)

    # knowledge_documents
    op.create_table('knowledge_documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('source_id', sa.UUID(), nullable=False),
        sa.Column('collection_id', sa.UUID(), nullable=True),
        sa.Column('title', sa.String(length=512), nullable=False),
        sa.Column('external_reference', sa.String(length=1024), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['collection_id'], ['knowledge_collections.id'], ),
        sa.ForeignKeyConstraint(['source_id'], ['knowledge_sources.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # knowledge_document_versions
    op.create_table('knowledge_document_versions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('content_hash', sa.String(length=128), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=True),
        sa.Column('sealed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['knowledge_documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # knowledge_chunks
    op.create_table('knowledge_chunks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_version_id', sa.UUID(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_version_id'], ['knowledge_document_versions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.execute("CREATE INDEX ix_knowledge_chunks_content_gin ON knowledge_chunks USING gin (to_tsvector('english', content));")

    # knowledge_embeddings
    import pgvector.sqlalchemy
    op.create_table('knowledge_embeddings',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('chunk_id', sa.UUID(), nullable=False),
        sa.Column('provider_name', sa.String(length=64), nullable=False),
        sa.Column('model_name', sa.String(length=128), nullable=False),
        sa.Column('dimensions', sa.Integer(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], ['knowledge_chunks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chunk_id')
    )
    op.execute("CREATE INDEX ix_knowledge_embeddings_vector_hnsw ON knowledge_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);")

    # knowledge_retrieval_runs
    op.create_table('knowledge_retrieval_runs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column('agent_id', sa.UUID(), nullable=True),
        sa.Column('execution_id', sa.UUID(), nullable=True),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('query_metadata', sa.JSON(), nullable=True),
        sa.Column('configuration', sa.JSON(), nullable=True),
        sa.Column('authorization_context', sa.JSON(), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # knowledge_evidence
    op.create_table('knowledge_evidence',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('retrieval_run_id', sa.UUID(), nullable=False),
        sa.Column('chunk_id', sa.UUID(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('document_title', sa.String(length=512), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], ['knowledge_chunks.id'], ),
        sa.ForeignKeyConstraint(['retrieval_run_id'], ['knowledge_retrieval_runs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('knowledge_evidence')
    op.drop_table('knowledge_retrieval_runs')
    op.execute("DROP INDEX IF EXISTS ix_knowledge_embeddings_vector_hnsw;")
    op.drop_table('knowledge_embeddings')
    op.execute("DROP INDEX IF EXISTS ix_knowledge_chunks_content_gin;")
    op.drop_table('knowledge_chunks')
    op.drop_table('knowledge_document_versions')
    op.drop_table('knowledge_documents')
    op.drop_index(op.f('ix_knowledge_collections_project_id'), table_name='knowledge_collections')
    op.drop_table('knowledge_collections')
    op.drop_index(op.f('ix_knowledge_sources_project_id'), table_name='knowledge_sources')
    op.drop_table('knowledge_sources')
