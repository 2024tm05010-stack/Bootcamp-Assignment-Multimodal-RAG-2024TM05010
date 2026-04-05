import base64
import openai
from PIL import Image
import io
from typing import Optional
from ..config import settings


class VLMService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = "gpt-4o-mini"  # Using GPT-4o-mini for vision capabilities

    def generate_image_summary(self, image_path: str, context: Optional[str] = None) -> str:
        """
        Generate a text summary/description of an image using GPT-4V.

        Args:
            image_path: Path to the image file
            context: Optional context about where the image came from (e.g., page number, document title)

        Returns:
            Text summary of the image content
        """
        try:
            # Load and encode the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large (GPT-4V has token limits)
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Prepare the prompt
            system_prompt = """You are an expert at analyzing images and providing detailed, accurate descriptions.
            Focus on the key visual elements, text content (if any), and overall meaning.
            Be concise but comprehensive - aim for 2-4 sentences that capture the essence of the image."""

            user_prompt = "Please analyze this image and provide a detailed description of its content:"
            if context:
                user_prompt += f"\n\nContext: This image is from {context}"

            # Call GPT-4V
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            print(f"Error generating VLM summary for image {image_path}: {e}")
            return f"Image content (VLM processing failed: {str(e)})"

    def is_available(self) -> bool:
        """Check if the VLM service is available and configured."""
        return bool(settings.OPENAI_API_KEY)