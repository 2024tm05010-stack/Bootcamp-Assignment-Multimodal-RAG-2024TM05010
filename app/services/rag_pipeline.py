import openai
from typing import List, Dict, Any, Optional
from .retriever import Retriever
from ..config import settings
from datetime import datetime


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        openai.api_key = settings.OPENAI_API_KEY

    def query(self, query: str, top_k: int = 5, query_type: str = "general") -> Dict[str, Any]:
        """
        Execute RAG pipeline with custom prompt templates based on query type.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            query_type: Type of query (general, factual, analytical, comparative, etc.)

        Returns:
            Dictionary with answer, sources, confidence, and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "confidence": 0.0,
                "query_type": query_type,
                "processing_time": 0.0
            }

        start_time = datetime.now()

        # Prepare context with intelligent chunking
        context = self._prepare_context(retrieved_docs)

        # Generate answer using custom prompt template
        answer = self._generate_answer(query, context, query_type)

        # Calculate confidence with multiple factors
        confidence = self._calculate_confidence(retrieved_docs, answer, query)

        # Format sources with relevance scores
        sources = self._format_sources(retrieved_docs)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "query_type": query_type,
            "processing_time": processing_time,
            "chunks_retrieved": len(retrieved_docs)
        }

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare context with intelligent chunking and prioritization.
        Groups documents by type and prioritizes most relevant content.
        """
        # Group documents by type
        text_docs = [d for d in documents if d["metadata"].get("type") == "text"]
        table_docs = [d for d in documents if d["metadata"].get("type") == "table"]
        image_docs = [d for d in documents if d["metadata"].get("type") in ["image_description", "image_ocr"]]

        context_parts = []

        # Add header with document overview
        total_docs = len(documents)
        context_parts.append(f"Retrieved {total_docs} relevant document chunks:")
        if text_docs:
            context_parts.append(f"- {len(text_docs)} text segments")
        if table_docs:
            context_parts.append(f"- {len(table_docs)} tables")
        if image_docs:
            context_parts.append(f"- {len(image_docs)} images")
        context_parts.append("")

        # Process each document type with appropriate formatting
        doc_counter = 1

        # Tables first (often contain structured information)
        for doc in table_docs:
            context_parts.append(self._format_document_chunk(doc, doc_counter, "TABLE"))
            doc_counter += 1

        # Text documents
        for doc in text_docs:
            context_parts.append(self._format_document_chunk(doc, doc_counter, "TEXT"))
            doc_counter += 1

        # Image descriptions
        for doc in image_docs:
            context_parts.append(self._format_document_chunk(doc, doc_counter, "IMAGE"))
            doc_counter += 1

        return "\n".join(context_parts)

    def _format_document_chunk(self, doc: Dict[str, Any], index: int, doc_type: str) -> str:
        """Format a single document chunk with rich metadata."""
        content = doc["content"]
        metadata = doc["metadata"]
        score = doc.get("score", 0.0)

        # Create header with metadata
        header_parts = [f"[{doc_type} {index}]"]
        if metadata.get("page"):
            header_parts.append(f"Page {metadata['page']}")
        if metadata.get("source"):
            header_parts.append(f"Source: {metadata['source']}")
        if score > 0:
            header_parts.append(f"Relevance: {score:.3f}")

        header = " | ".join(header_parts)

        # Format content based on type
        if doc_type == "TABLE":
            formatted_content = f"```\n{content}\n```"
        elif doc_type == "IMAGE":
            formatted_content = f"[Image Description]\n{content}"
        else:
            formatted_content = content

        return f"{header}\n{formatted_content}\n"

    def _generate_answer(self, query: str, context: str, query_type: str) -> str:
        """
        Generate answer using custom prompt templates based on query type.
        """
        prompt_template = self._get_prompt_template(query_type)

        # Prepare the full prompt
        full_prompt = prompt_template.format(
            context=context,
            query=query,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("test_key"):
            return self._local_answer(query, context, query_type)

        try:
            response = openai.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(query_type)},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1500,
                temperature=self._get_temperature(query_type)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._local_answer(query, context, query_type)

    def _get_prompt_template(self, query_type: str) -> str:
        """Get custom prompt template based on query type."""
        templates = {
            "general": """
Based on the following document context, please answer the question comprehensively and accurately.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be comprehensive but concise
- Include specific details, quotes, and references from the documents
- If the context contains tables or images, reference them appropriately
- If information is insufficient, clearly state what you don't know
- Structure your answer logically with clear sections if needed

ANSWER:""",

            "factual": """
You are a fact-checking assistant. Based on the document context provided, answer the following question with precise, factual information.

CONTEXT:
{context}

QUESTION: {query}

REQUIREMENTS:
- Provide ONLY factual information from the documents
- Include specific page numbers and document sources
- If the information appears in tables, quote the exact table content
- If information is contradictory, note the discrepancies
- If no factual answer exists, state "No factual information found"
- Be precise and avoid interpretation

FACTUAL ANSWER:""",

            "analytical": """
You are an analytical assistant. Analyze the following question using the document context and provide a structured analysis.

CONTEXT:
{context}

QUESTION: {query}

ANALYSIS FRAMEWORK:
1. **Key Findings**: Extract the most relevant information
2. **Patterns & Trends**: Identify any patterns in the data
3. **Implications**: Discuss what this information means
4. **Limitations**: Note any gaps or uncertainties in the context
5. **Conclusions**: Provide clear, evidence-based conclusions

STRUCTURED ANALYSIS:""",

            "comparative": """
You are a comparative analysis assistant. Compare and contrast information from the document context to answer the question.

CONTEXT:
{context}

QUESTION: {query}

COMPARISON STRUCTURE:
- **Similarities**: What common elements exist?
- **Differences**: What distinctions are evident?
- **Strengths/Advantages**: What are the positive aspects?
- **Weaknesses/Limitations**: What are the drawbacks?
- **Recommendations**: Based on the comparison, what would you recommend?

COMPARATIVE ANALYSIS:""",

            "summarization": """
You are a summarization specialist. Provide a concise yet comprehensive summary of the relevant information from the documents.

CONTEXT:
{context}

QUESTION: {query}

SUMMARY REQUIREMENTS:
- Capture all key points and main ideas
- Maintain factual accuracy
- Organize information logically
- Include important details, dates, and figures
- Keep the summary focused and relevant to the question
- Aim for clarity and readability

EXECUTIVE SUMMARY:"""
        }

        return templates.get(query_type, templates["general"])

    def _get_system_prompt(self, query_type: str) -> str:
        """Get system prompt based on query type."""
        system_prompts = {
            "general": "You are a helpful document analysis assistant. Provide accurate, well-structured answers based on the provided context.",
            "factual": "You are a precise fact-checker. Provide only verified information with clear citations.",
            "analytical": "You are an analytical expert. Break down complex information into clear, structured insights.",
            "comparative": "You are a comparative analyst. Excel at identifying similarities, differences, and trade-offs.",
            "summarization": "You are a summarization expert. Create clear, concise summaries that capture essential information."
        }

        return system_prompts.get(query_type, system_prompts["general"])

    def _get_temperature(self, query_type: str) -> float:
        """Get temperature setting based on query type."""
        temperatures = {
            "factual": 0.1,      # Low temperature for factual accuracy
            "analytical": 0.3,   # Moderate temperature for balanced analysis
            "general": 0.2,      # Low-moderate for general questions
            "comparative": 0.2,  # Balanced for comparisons
            "summarization": 0.1 # Low for consistent summaries
        }

        return temperatures.get(query_type, 0.2)

    def _local_answer(self, query: str, context: str, query_type: str) -> str:
        """Fallback answer generation when OpenAI is not available."""
        if not context or context.strip() == "":
            return "No relevant information found in the documents."

        # Extract key information from context
        lines = context.split('\n')
        relevant_lines = [line for line in lines if len(line.strip()) > 20][:10]  # Get substantial lines

        if not relevant_lines:
            return "Based on the retrieved document chunks, here is the relevant information:\n\n[Context provided but no clear answer could be extracted]"

        # Format based on query type
        if query_type == "factual":
            answer = "FACTS FROM DOCUMENTS:\n" + "\n".join(f"• {line}" for line in relevant_lines[:5])
        elif query_type == "analytical":
            answer = "KEY FINDINGS:\n" + "\n".join(f"• {line}" for line in relevant_lines[:5])
        else:
            answer = f"Based on the retrieved document chunks, here is the relevant information:\n\n" + "\n".join(relevant_lines[:5])

        return answer

    def _calculate_confidence(self, documents: List[Dict[str, Any]], answer: str, query: str) -> float:
        """
        Calculate confidence score based on multiple factors:
        - Retrieval scores
        - Answer quality
        - Document relevance
        """
        if not documents:
            return 0.0

        # Base confidence from retrieval scores
        scores = [doc.get("score", 0.0) for doc in documents]
        avg_retrieval_score = sum(scores) / len(scores)

        # Document type bonus (tables and text are more reliable than images)
        type_weights = {
            "text": 1.0,
            "table": 1.2,
            "image_description": 0.8,
            "image_ocr": 0.7
        }

        type_score = sum(type_weights.get(doc["metadata"].get("type", "text"), 1.0) for doc in documents) / len(documents)

        # Answer quality indicators
        answer_length = len(answer.split())
        has_sources = any(f"page {doc['metadata'].get('page')}" in answer.lower() for doc in documents)
        has_specifics = any(char.isdigit() for char in answer)  # Numbers often indicate specific info

        quality_score = min(1.0, (answer_length / 100.0) + (0.2 if has_sources else 0) + (0.1 if has_specifics else 0))

        # Combine factors
        confidence = (avg_retrieval_score * 0.5) + (type_score * 0.2) + (quality_score * 0.3)

        return min(confidence, 1.0)

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
