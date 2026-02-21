# backend/utils/embeddings.py

from sentence_transformers import SentenceTransformer
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

_embedding_model = None


def get_embedding_model():
    """
    Lazy-load the embedding model.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of texts into embedding vectors.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings
