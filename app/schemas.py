from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    indexed_documents: int


class IngestResponse(BaseModel):
    message: str
    chunks_processed: int


class SourceItem(BaseModel):
    content: str
    page: Optional[int] = None
    type: Optional[str] = None
    source: Optional[str] = None
    score: float


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    query_type: str = "general"  # general, factual, analytical, comparative, summarization


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    confidence: float
    query_type: str
    processing_time: float
    chunks_retrieved: int


class DocumentPreview(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    type: str
    content_preview: str


class DocumentsResponse(BaseModel):
    documents: List[DocumentPreview]
