import os
import numpy as np
from typing import List, Dict, Any
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from ..config import settings

class VectorStore:
    def __init__(self):
        self.store_type = settings.VECTOR_STORE_TYPE
        self.text_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self.text_model.get_sentence_embedding_dimension()
        self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)

        # Create data directory
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
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.text_vectors = []
        self.image_vectors = []

    def _init_chromadb(self):
        """Initialize ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
        self.text_collection = self.chroma_client.get_or_create_collection(name="text_documents")
        self.image_collection = self.chroma_client.get_or_create_collection(name="image_documents")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]

            if metadata["type"] == "text":
                # Generate text embedding
                vector = self._generate_text_embedding(content)
                self._add_text_vector(vector, content, metadata)

            elif metadata["type"] in ["image_ocr", "table"]:
                # For images and tables, use text embedding but also try CLIP if image available
                vector = self._generate_text_embedding(content)
                self._add_text_vector(vector, content, metadata)

                # If it's an image with OCR, also add CLIP embedding
                if metadata["type"] == "image_ocr" and "image_path" in metadata:
                    try:
                        clip_vector = self._generate_clip_embedding(metadata["image_path"])
                        self._add_image_vector(clip_vector, content, metadata)
                    except Exception as e:
                        print(f"Failed to generate CLIP embedding: {e}")

            self.documents.append(doc)

    def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using sentence transformer"""
        return self.text_model.encode(text, convert_to_numpy=True)

    def _generate_clip_embedding(self, image_path: str) -> np.ndarray:
        """Generate image embedding using CLIP"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features.squeeze().numpy()

    def _add_text_vector(self, vector: np.ndarray, content: str, metadata: Dict):
        """Add text vector to the store"""
        if self.store_type == "faiss":
            self.text_vectors.append({
                "vector": vector,
                "content": content,
                "metadata": metadata
            })
            # Add to FAISS index
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
        """Add image vector to the store"""
        if self.store_type == "faiss":
            self.image_vectors.append({
                "vector": vector,
                "content": content,
                "metadata": metadata
            })
            # Note: FAISS doesn't support mixed modalities well, so we'll keep separate

        elif self.store_type == "chromadb":
            self.image_collection.add(
                embeddings=[vector.tolist()],
                documents=[content],
                metadatas=[metadata],
                ids=[f"image_{len(self.image_collection.get()['ids']) + 1}"]
            )

    def search(self, query: str, top_k: int = 5, search_type: str = "text") -> List[Dict[str, Any]]:
        """Search the vector store"""
        if search_type == "text":
            query_vector = self._generate_text_embedding(query)
        elif search_type == "image":
            # For image search, we'd need an image, but for now use text
            query_vector = self._generate_text_embedding(query)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

        if self.store_type == "faiss":
            return self._search_faiss(query_vector, top_k)
        elif self.store_type == "chromadb":
            return self._search_chromadb(query_vector, top_k, search_type)

    def _search_faiss(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search FAISS index"""
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
        """Search ChromaDB"""
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
        """List all documents in the store"""
        return [
            {
                "id": i,
                "source": doc["metadata"]["source"],
                "page": doc["metadata"]["page"],
                "type": doc["metadata"]["type"],
                "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            }
            for i, doc in enumerate(self.documents)
        ]

    def save(self):
        """Save the vector store to disk"""
        if self.store_type == "faiss":
            faiss.write_index(self.index, os.path.join(settings.VECTOR_STORE_PATH, "faiss_index.idx"))
            # Save document metadata
            np.save(os.path.join(settings.VECTOR_STORE_PATH, "documents.npy"), self.documents)

    def load(self):
        """Load the vector store from disk"""
        if self.store_type == "faiss":
            index_path = os.path.join(settings.VECTOR_STORE_PATH, "faiss_index.idx")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)

            docs_path = os.path.join(settings.VECTOR_STORE_PATH, "documents.npy")
            if os.path.exists(docs_path):
                self.documents = np.load(docs_path, allow_pickle=True).tolist()