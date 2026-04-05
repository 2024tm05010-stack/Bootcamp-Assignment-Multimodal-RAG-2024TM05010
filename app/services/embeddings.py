import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from ..config import settings


class EmbeddingService:
    def __init__(self):
        self.text_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self.text_model.get_sentence_embedding_dimension()
        self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)

    def text_embedding(self, text: str) -> np.ndarray:
        return self.text_model.encode(text, convert_to_numpy=True)

    def image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.squeeze().numpy()
