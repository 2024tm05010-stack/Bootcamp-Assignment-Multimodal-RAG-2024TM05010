# Multimodal RAG System for Sustainability Documents

A comprehensive multimodal Retrieval-Augmented Generation (RAG) system designed specifically for processing and analyzing sustainability documents. This system combines advanced PDF processing, Vision Language Model (VLM) integration, and domain-aware RAG pipelines to provide intelligent querying capabilities for environmental compliance, regulatory documents, and sustainability reports.

## Key Features

- **Multimodal PDF Processing**: Extracts text, tables, and images from complex PDF documents using Docling and PyMuPDF
- **Vision Language Model Integration**: Uses GPT-4V to analyze and describe images within documents
- **Domain-Aware RAG Pipeline**: Specialized query types for compliance, regulatory, and sustainability content
- **Vector Database Storage**: Efficient similarity search using FAISS/ChromaDB with multimodal embeddings
- **REST API**: FastAPI-based web service with comprehensive endpoints
- **Docker Deployment**: Containerized solution for easy deployment and scaling

## Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **AI/ML**: OpenAI GPT-4V, Sentence Transformers, CLIP
- **Vector DB**: FAISS, ChromaDB
- **PDF Processing**: Docling, PyMuPDF
- **Deployment**: Docker, Docker Compose

## Use Cases

- Environmental compliance document analysis
- Regulatory requirement extraction
- Sustainability report processing
- Multi-modal document search and retrieval
- Automated compliance checking

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/Bootcamp-Assignment-Multimodal-RAG-2024TM05010.git
cd Bootcamp-Assignment-Multimodal-RAG-2024TM05010

# Start with Docker
docker-compose up --build
```

For detailed setup instructions, see [README.md](README.md).