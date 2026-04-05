import openai
from typing import List, Dict, Any
from .retriever import Retriever
from ..config import settings


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        openai.api_key = settings.OPENAI_API_KEY

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "confidence": 0.0
            }

        context = self._prepare_context(retrieved_docs)
        answer = self._generate_answer(query, context)
        confidence = self._calculate_confidence(retrieved_docs)
        sources = self._format_sources(retrieved_docs)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            metadata = doc["metadata"]
            doc_type = metadata.get("type", "unknown")
            page = metadata.get("page", "unknown")
            source = metadata.get("source", "unknown")
            context_parts.append(f"[Document {i}] Type: {doc_type}, Page: {page}, Source: {source}\n{content}\n")
        return "\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        prompt = f"""
You are a helpful assistant that answers questions based on the provided context from multimodal documents.
The context may include text, tables, and image descriptions extracted from PDF documents.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information to answer the question, say so
- Be concise but comprehensive
- Include specific details from the documents when relevant
- If referring to tables or images, mention their location (page, document)

Answer:"""

        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("test_key"):
            return self._local_answer(query, context)

        try:
            response = openai.ChatCompletion.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for document Q&A."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self._local_answer(query, context)

    def _local_answer(self, query: str, context: str) -> str:
        lines = context.splitlines()
        if not lines:
            return "No relevant information found in the documents."
        first_block = "\n".join(lines[:5]).strip()
        return f"Based on the retrieved document chunks, here is the relevant information:\n{first_block}"

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0
        scores = [doc.get("score", 0.0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        return min(avg_score, 1.0)

    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for doc in documents:
            metadata = doc["metadata"]
            sources.append({
                "content": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                "page": metadata.get("page"),
                "type": metadata.get("type"),
                "source": metadata.get("source"),
                "score": doc.get("score", 0.0)
            })
        return sources
