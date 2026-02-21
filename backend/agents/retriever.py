# backend/agents/retriever.py

from typing import List, Dict, Any
import numpy as np

from backend.utils.embeddings import embed_texts
from backend.utils.vector_store import FAISSVectorStore


class RetrieverAgent:
    """
    Agent responsible for retrieving relevant document chunks
    using vector similarity search.
    """

    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def run(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant document chunks for a query.
        """
        # Embed query
        query_embedding = embed_texts([query])[0]
        query_embedding = np.array(query_embedding)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        # Structure results
        retrieved_chunks = []
        for text, distance in results:
            retrieved_chunks.append({
                "content": text,
                "score": distance
            })

        return retrieved_chunks
