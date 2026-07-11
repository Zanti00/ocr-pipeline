"""create embeddings table

Revision ID: 001
Revises: 
Create Date: 2026-07-06 16:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.create_table('receipt_embeddings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('receipt_id', sa.Integer(), nullable=False),
        sa.Column('source_service', sa.String(length=50), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=384), nullable=True),
        sa.Column('receipt_text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_embeddings_created_at', 'receipt_embeddings', ['created_at'], unique=False)
    op.create_index('idx_embeddings_source_service', 'receipt_embeddings', ['source_service'], unique=False)
    op.execute("CREATE INDEX idx_embeddings_vector ON receipt_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

def downgrade() -> None:
    op.drop_index('idx_embeddings_vector', table_name='receipt_embeddings', postgresql_using='ivfflat')
    op.drop_index('idx_embeddings_source_service', table_name='receipt_embeddings')
    op.drop_index('idx_embeddings_created_at', table_name='receipt_embeddings')
    op.drop_table('receipt_embeddings')
