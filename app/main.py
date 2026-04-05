from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the PDF
        documents = pdf_processor.process_pdf(temp_path)

        # Add to vector store
        vector_store.add_documents(documents)

        # Clean up
        os.remove(temp_path)

        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks_processed": len(documents)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = rag_pipeline.query(request.query, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all ingested documents"""
    try:
        documents = vector_store.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)