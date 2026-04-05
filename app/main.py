from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
from .services.pdf_processor import PDFProcessor
from .services.vector_store import VectorStore
from .services.rag_pipeline import RAGPipeline
from .config import settings

app = FastAPI(
    title="Multimodal RAG API",
    description="API for multimodal document processing and retrieval-augmented generation",
    version="1.0.0",
    docs_url="/documentation",
    redoc_url=None,
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
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

@app.get("/docs", include_in_schema=False)
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

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStore()
rag_pipeline = RAGPipeline(vector_store)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "multimodal-rag",
        "indexed_documents": len(vector_store.documents)
    }

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a multimodal PDF document"""
    if not file.filename.lower().endswith('.pdf') and file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            buffer.write(content)

        # Process the PDF
        documents = pdf_processor.process_pdf(temp_path)

        if not documents:
            raise HTTPException(status_code=422, detail="PDF was processed but no content could be extracted")

        # Add to vector store
        vector_store.add_documents(documents)

        # Clean up
        try:
            os.remove(temp_path)
        except OSError:
            pass

        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks_processed": len(documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text must be provided")

    if len(vector_store.documents) == 0:
        raise HTTPException(status_code=404, detail="No documents have been ingested yet")

    try:
        result = rag_pipeline.query(request.query, request.top_k)
        if not result["sources"]:
            raise HTTPException(status_code=404, detail="No relevant information found for the query")
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all ingested documents"""
    try:
        documents = vector_store.list_documents()
        if not documents:
            raise HTTPException(status_code=404, detail="No ingested documents found")
        return {"documents": documents}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)