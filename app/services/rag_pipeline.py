import openai
from typing import List, Dict, Any
from .vector_store import VectorStore
from ..config import settings

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        openai.api_key = settings.OPENAI_API_KEY

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform retrieval-augmented generation.
        Returns answer, sources, and confidence score.
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "confidence": 0.0
            }

        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)

        # Generate answer using LLM
        answer = self._generate_answer(query, context)

        # Calculate confidence based on retrieval scores
        confidence = self._calculate_confidence(retrieved_docs)

        # Format sources
        sources = self._format_sources(retrieved_docs)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            metadata = doc["metadata"]

            # Format document with metadata
            doc_type = metadata.get("type", "unknown")
            page = metadata.get("page", "unknown")
            source = metadata.get("source", "unknown")

            context_part = f"[Document {i}] Type: {doc_type}, Page: {page}, Source: {source}\n{content}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using OpenAI GPT or local fallback if OpenAI is unavailable."""
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
        """Fallback answer generation when OpenAI is not available."""
        lines = context.splitlines()
        if not lines:
            return "No relevant information found in the documents."

        # Use the first relevant context block and mention the query.
        first_block = "\n".join(lines[:5]).strip()
        return f"Based on the retrieved document chunks, here is the relevant information:\n{first_block}"

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not documents:
            return 0.0

        # Simple confidence calculation based on average score
        scores = [doc.get("score", 0.0) for doc in documents]
        avg_score = sum(scores) / len(scores)

        # Normalize to 0-1 range (assuming scores are similarity measures)
        confidence = min(avg_score, 1.0)
        return confidence

    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for the response"""
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