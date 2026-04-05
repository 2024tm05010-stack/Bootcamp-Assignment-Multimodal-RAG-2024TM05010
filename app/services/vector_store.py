import os
import numpy as np
from typing import List, Dict, Any
import faiss
import chromadb
from .embeddings import EmbeddingService
from ..config import settings


class VectorStore:
    def __init__(self, embedding_service: EmbeddingService):
        self.store_type = settings.VECTOR_STORE_TYPE
        self.embedding_service = embedding_service
        self.dimension = self.embedding_service.dimension

        os.makedirs(settings.DATA_DIR, exist_ok=True)
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)

        if self.store_type == "faiss":
            self._init_faiss()
        elif self.store_type == "chromadb":
            self._init_chromadb()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        self.documents = []

    def _init_faiss(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_vectors = []
        self.image_vectors = []

    def _init_chromadb(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
        self.text_collection = self.chroma_client.get_or_create_collection(name="text_documents")
        self.image_collection = self.chroma_client.get_or_create_collection(name="image_documents")

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]

            if metadata["type"] == "text":
                vector = self.embedding_service.text_embedding(content)
                self._add_text_vector(vector, content, metadata)

            elif metadata["type"] in ["image_ocr", "table", "image_description"]:
                vector = self.embedding_service.text_embedding(content)
                self._add_text_vector(vector, content, metadata)

                if metadata["type"] == "image_ocr" and "image_path" in metadata:
                    try:
                        clip_vector = self.embedding_service.image_embedding(metadata["image_path"])
                        self._add_image_vector(clip_vector, content, metadata)
                    except Exception as e:
                        print(f"Failed to generate CLIP embedding: {e}")

            self.documents.append(doc)

    def _add_text_vector(self, vector: np.ndarray, content: str, metadata: Dict):
        if self.store_type == "faiss":
            self.text_vectors.append({
                "vector": vector,
                "content": content,
                "metadata": metadata
            })
            normalized_vector = vector / np.linalg.norm(vector)
            self.index.add(np.array([normalized_vector], dtype=np.float32))

        elif self.store_type == "chromadb":
            self.text_collection.add(
                embeddings=[vector.tolist()],
                documents=[content],
                metadatas=[metadata],
                ids=[f"text_{len(self.text_collection.get()['ids']) + 1}"]
            )

    def _add_image_vector(self, vector: np.ndarray, content: str, metadata: Dict):
        if self.store_type == "faiss":
            self.image_vectors.append({
                "vector": vector,
                "content": content,
                "metadata": metadata
            })
        elif self.store_type == "chromadb":
            self.image_collection.add(
                embeddings=[vector.tolist()],
                documents=[content],
                metadatas=[metadata],
                ids=[f"image_{len(self.image_collection.get()['ids']) + 1}"]
            )

    def search(self, query: str, top_k: int = 5, search_type: str = "text") -> List[Dict[str, Any]]:
        if search_type == "text":
            query_vector = self.embedding_service.text_embedding(query)
        elif search_type == "image":
            query_vector = self.embedding_service.text_embedding(query)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

        if self.store_type == "faiss":
            return self._search_faiss(query_vector, top_k)
        elif self.store_type == "chromadb":
            return self._search_chromadb(query_vector, top_k, search_type)

    def _search_faiss(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        normalized_query = query_vector / np.linalg.norm(query_vector)
        scores, indices = self.index.search(np.array([normalized_query], dtype=np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.text_vectors):
                doc = self.text_vectors[idx]
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })

        return results

    def _search_chromadb(self, query_vector: np.ndarray, top_k: int, search_type: str) -> List[Dict[str, Any]]:
        collection = self.text_collection if search_type == "text" else self.image_collection
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )

        search_results = []
        for i in range(len(results['documents'])):
            search_results.append({
                "content": results['documents'][i],
                "metadata": results['metadatas'][i],
                "score": results['distances'][i] if 'distances' in results else 0.0
            })
        return search_results

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": i,
                "source": doc["metadata"]["source"],
                "page": doc["metadata"].get("page"),
                "type": doc["metadata"]["type"],
                "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            }
            for i, doc in enumerate(self.documents)
        ]

    def save(self):
        if self.store_type == "faiss":
            faiss.write_index(self.index, os.path.join(settings.VECTOR_STORE_PATH, "faiss_index.idx"))
            np.save(os.path.join(settings.VECTOR_STORE_PATH, "documents.npy"), self.documents)

    def load(self):
        if self.store_type == "faiss":
            index_path = os.path.join(settings.VECTOR_STORE_PATH, "faiss_index.idx")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            docs_path = os.path.join(settings.VECTOR_STORE_PATH, "documents.npy")
            if os.path.exists(docs_path):
                self.documents = np.load(docs_path, allow_pickle=True).tolist()
