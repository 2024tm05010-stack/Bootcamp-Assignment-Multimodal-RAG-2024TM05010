from typing import List, Dict, Any
from .vector_store import VectorStore
import numpy as np


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents with optional re-ranking.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            rerank: Whether to apply re-ranking for better results

        Returns:
            List of relevant documents with scores
        """
        # Initial retrieval - get more candidates than needed for re-ranking
        candidates = self.vector_store.search(query, top_k=top_k * 2)

        if not candidates:
            return []

        if rerank and len(candidates) > top_k:
            # Apply re-ranking based on multiple factors
            reranked = self._rerank_documents(candidates, query)
            return reranked[:top_k]

        return candidates[:top_k]

    def _rerank_documents(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on multiple relevance factors:
        - Semantic similarity (original score)
        - Document type priority
        - Content quality
        - Query-document term overlap
        """
        reranked_docs = []

        for doc in documents:
            score = doc.get("score", 0.0)
            metadata = doc["metadata"]
            content = doc["content"]

            # Type-based scoring boost
            type_boost = self._get_type_boost(metadata.get("type", "text"))
            score += type_boost

            # Content quality boost (longer, more specific content)
            quality_boost = self._calculate_quality_score(content)
            score += quality_boost * 0.1

            # Query term overlap boost
            overlap_boost = self._calculate_query_overlap(query, content)
            score += overlap_boost * 0.2

            # Recency boost (earlier pages might be more important)
            page = metadata.get("page", 1)
            recency_boost = max(0, (100 - page) / 1000.0)  # Small boost for earlier pages
            score += recency_boost

            doc["score"] = score
            reranked_docs.append(doc)

        # Sort by final score
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        return reranked_docs

    def _get_type_boost(self, doc_type: str) -> float:
        """Get relevance boost based on document type."""
        type_boosts = {
            "table": 0.3,        # Tables often contain structured, valuable info
            "text": 0.1,         # Standard text content
            "image_description": 0.2,  # VLM descriptions are valuable
            "image_ocr": 0.0     # OCR text might be noisy
        }
        return type_boosts.get(doc_type, 0.0)

    def _calculate_quality_score(self, content: str) -> float:
        """Calculate content quality score based on length and specificity."""
        if not content:
            return 0.0

        length_score = min(len(content.split()) / 50.0, 1.0)  # Prefer substantial content

        # Check for specific indicators of quality
        has_numbers = any(char.isdigit() for char in content)
        has_capitals = any(char.isupper() for char in content)
        has_punctuation = any(char in '.,;:!?' for char in content)

        specificity_score = (0.3 if has_numbers else 0) + \
                           (0.2 if has_capitals else 0) + \
                           (0.1 if has_punctuation else 0)

        return (length_score * 0.6) + (specificity_score * 0.4)

    def _calculate_query_overlap(self, query: str, content: str) -> float:
        """Calculate term overlap between query and content."""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)  # Normalized overlap score
