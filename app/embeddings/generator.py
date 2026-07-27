"""Sentence embeddings for duplicate detection and category similarity.

``sentence_transformers`` is imported lazily. It is a heavy optional dependency,
and importing it at module scope meant a missing or broken model prevented the
entire pipeline module from loading - turning an optional feature into a hard
dependency for unrelated work.
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            # Cached during the image build by scripts/download_model.py.
            cls._model = SentenceTransformer(settings.embedding_model)
        return cls._model

    @classmethod
    def generate(cls, text: str) -> list[float]:
        model = cls.get_model()
        return model.encode(text).tolist()

    @classmethod
    def available(cls) -> bool:
        """Is the embedding model usable in this environment?"""
        try:
            cls.get_model()
            return True
        except Exception as exc:
            logger.debug("Embedding model unavailable: %s", exc)
            return False
