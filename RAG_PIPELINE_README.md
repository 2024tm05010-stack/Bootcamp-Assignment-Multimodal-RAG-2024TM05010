# Enhanced RAG Pipeline Documentation

## Overview

The multimodal RAG system now features an advanced retrieval-augmented generation pipeline with custom prompt templates, intelligent retrieval, and multi-modal content processing.

## Key Features

### 1. Custom Prompt Templates

The system supports different query types with specialized prompt templates:

- **General**: Comprehensive answers based on document context
- **Factual**: Precise, citation-heavy responses for factual queries
- **Analytical**: Structured analysis with key findings and implications
- **Comparative**: Side-by-side comparisons and trade-off analysis
- **Summarization**: Concise executive summaries of key information

### 2. Intelligent Retrieval & Re-ranking

- **Multi-factor re-ranking**: Combines semantic similarity, document type priority, content quality, and query overlap
- **Document type prioritization**: Tables and structured content get higher relevance scores
- **Quality scoring**: Longer, more specific content is preferred
- **Query-aware ranking**: Better term overlap improves ranking

### 3. Enhanced Context Preparation

- **Structured context**: Documents grouped by type (text, tables, images)
- **Rich metadata**: Includes page numbers, sources, relevance scores, and document types
- **Intelligent chunking**: Prioritizes most relevant content types

### 4. Advanced Confidence Scoring

Confidence scores now consider:
- Retrieval similarity scores
- Document type reliability
- Answer quality indicators
- Source specificity

## API Usage

### Query with Custom Types

```python
import requests

# General query
response = requests.post("http://localhost:8000/query", json={
    "query": "What are the main regulations?",
    "query_type": "general",
    "top_k": 5
})

# Factual query
response = requests.post("http://localhost:8000/query", json={
    "query": "What is the penalty amount?",
    "query_type": "factual",
    "top_k": 3
})

# Analytical query
response = requests.post("http://localhost:8000/query", json={
    "query": "Analyze the waste management framework",
    "query_type": "analytical",
    "top_k": 5
})
```

### Response Format

```json
{
  "answer": "Comprehensive answer based on retrieved context...",
  "sources": [
    {
      "content": "Source content preview...",
      "page": 15,
      "type": "table",
      "source": "document.pdf",
      "score": 0.85
    }
  ],
  "confidence": 0.73,
  "query_type": "analytical",
  "processing_time": 0.45,
  "chunks_retrieved": 5
}
```

## Supported Query Types

1. **general**: Default comprehensive answering
2. **factual**: Precise facts with citations
3. **analytical**: Structured analysis framework
4. **comparative**: Comparison and contrast analysis
5. **summarization**: Executive summaries

## Technical Implementation

### Components

- **RAGPipeline**: Orchestrates retrieval, context preparation, and answer generation
- **Retriever**: Handles document retrieval with re-ranking
- **Custom Prompts**: Type-specific prompt engineering
- **Confidence Scoring**: Multi-factor confidence calculation

### Performance Features

- Processing time tracking
- Chunk retrieval counts
- Relevance scoring for all sources
- Fallback handling for API failures

## Example Queries

### General Query
```
Query: "What are the waste management rules?"
Type: general
Result: Comprehensive overview with key points and references
```

### Factual Query
```
Query: "What is the fine for littering?"
Type: factual
Result: Exact amounts, specific sections, precise citations
```

### Analytical Query
```
Query: "Analyze the environmental impact regulations"
Type: analytical
Result: Structured analysis with findings, implications, limitations
```