from typing import List, Dict, Any
from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, top_k=top_k)
