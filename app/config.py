import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss or chromadb
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "768"))  # For sentence-transformers

    # Embedding Model Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # Document Processing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Storage Settings
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    VECTOR_STORE_PATH: str = os.path.join(DATA_DIR, "vector_store")

    # OCR Settings
    TESSERACT_CONFIG: str = os.getenv("TESSERACT_CONFIG", '--oem 3 --psm 6')

settings = Settings()