from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.rag.config import DEFAULT_EMBEDDING_MODEL

class EmbeddingManager:
    """Manages sentence transformer embeddings."""
    _instance = None
    _model_cache = {}

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    @classmethod
    def _load_model(cls, model_name: str) -> SentenceTransformer:
        if model_name not in cls._model_cache:
            cls._model_cache[model_name] = SentenceTransformer(model_name)
        return cls._model_cache[model_name]

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.get_dimension()))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
