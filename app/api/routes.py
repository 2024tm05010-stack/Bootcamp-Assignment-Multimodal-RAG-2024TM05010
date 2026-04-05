import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..config import settings
from ..schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    DocumentsResponse,
)
from ..services.parser import PDFParser
from ..services.embeddings import EmbeddingService
from ..services.vector_store import VectorStore
from ..services.retriever import Retriever
from ..services.rag_pipeline import RAGPipeline
from ..services.vlm_service import VLMService

router = APIRouter()

vlm_service = VLMService()
pdf_parser = PDFParser(vlm_service)
embedding_service = EmbeddingService()
vector_store = VectorStore(embedding_service)
retriever = Retriever(vector_store)
rag_pipeline = RAGPipeline(retriever)


@router.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to the Multimodal RAG API",
        "health": "/health",
        "ingest": "/ingest",
        "query": "/query",
        "documents": "/documents",
        "documentation": "/documentation",
        "api_docs": "/docs"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="multimodal-rag",
        indexed_documents=len(vector_store.documents)
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf') and file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            buffer.write(content)

        documents = pdf_parser.process_pdf(temp_path)
        if not documents:
            raise HTTPException(status_code=422, detail="PDF was processed but no content could be extracted")

        vector_store.add_documents(documents)

        try:
            os.remove(temp_path)
        except OSError:
            pass

        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            chunks_processed=len(documents)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text must be provided")

    if len(vector_store.documents) == 0:
        raise HTTPException(status_code=404, detail="No documents have been ingested yet")

    try:
        result = rag_pipeline.query(request.query, request.top_k, request.query_type)
        if not result["sources"]:
            raise HTTPException(status_code=404, detail="No relevant information found for the query")
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    try:
        documents = vector_store.list_documents()
        if not documents:
            raise HTTPException(status_code=404, detail="No ingested documents found")
        return DocumentsResponse(documents=documents)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/docs", include_in_schema=False)
async def api_docs():
    return JSONResponse({
        "title": "Multimodal RAG API Documentation",
        "description": "Available API endpoints for health, ingestion, query, and document listing.",
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check for the API service"
            },
            {
                "path": "/ingest",
                "method": "POST",
                "description": "Upload a PDF document for ingestion"
            },
            {
                "path": "/query",
                "method": "POST",
                "description": "Query the ingested documents with a natural language question"
            },
            {
                "path": "/documents",
                "method": "GET",
                "description": "List ingested document chunks"
            }
        ],
        "swagger_ui": "/documentation",
        "openapi_schema": "/openapi.json"
    })
