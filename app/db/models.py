from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone

Base = declarative_base()

class ReceiptEmbedding(Base):
    __tablename__ = "receipt_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    receipt_id = Column(Integer, nullable=False)
    source_service = Column(String(50), nullable=False)
    embedding = Column(Vector(384))
    receipt_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index("idx_embeddings_source_service", "source_service"),
        Index("idx_embeddings_created_at", "created_at"),
    )
