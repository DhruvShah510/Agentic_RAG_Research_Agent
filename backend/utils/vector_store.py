# backend/utils/vector_store.py

import faiss
import numpy as np
from typing import List, Tuple


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[str] = []

    def add(self, embeddings: np.ndarray, documents: List[str]):
        """
        Add embeddings and corresponding documents to the index.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents length mismatch")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for top-k similar documents.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))

        return results
