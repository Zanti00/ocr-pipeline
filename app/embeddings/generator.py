from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingGenerator:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Model is already cached by download_model.py in Dockerfile
            cls._model = SentenceTransformer(settings.embedding_model)
        return cls._model

    @classmethod
    def generate(cls, text: str) -> list[float]:
        model = cls.get_model()
        # SentenceTransformers returns a numpy array, convert to list
        embedding = model.encode(text)
        return embedding.tolist()
