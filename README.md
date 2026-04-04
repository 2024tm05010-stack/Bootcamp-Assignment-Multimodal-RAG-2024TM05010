# Bootcamp-Assignment-Multimodal-RAG-2024TM05010
Bootcamp Assignment from WILP, BITS Pilani, Hyderabad Campus on Multimodal RAG
# Multimodal RAG for Sustainability

Prototype FastAPI server that ingests multimodal PDFs (text, tables, images),
builds a FAISS vector index, and serves RAG queries.

## Endpoints
- GET /health
- POST /ingest  (multipart form file)
- POST /query   (JSON { "query": "...", "top_k": 5 })
- GET /docs     (Swagger UI)

## Quickstart
1. pip install -r requirements.txt
2. uvicorn app.main:app --reload --port 8000
3. Visit http://localhost:8000/docs
