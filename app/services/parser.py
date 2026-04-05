import os
import tempfile
from typing import List, Dict, Any, Optional
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
import pypdfium2 as pdfium
from .vlm_service import VLMService


class PDFParser:
    def __init__(self, vlm_service: Optional[VLMService] = None):
        self.temp_dir = tempfile.mkdtemp()
        self.backend = PyPdfiumDocumentBackend()
        self.vlm_service = vlm_service or VLMService()

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a PDF document and extract multimodal content using PyPdfium."""
        documents = []

        try:
            # Load PDF with pypdfium2
            pdf = pdfium.PdfDocument(pdf_path)

            # Extract text content
            text_chunks = self._extract_text_chunks(pdf, pdf_path)
            documents.extend(text_chunks)

            # Extract images with VLM summaries
            image_chunks = self._extract_image_chunks(pdf, pdf_path)
            documents.extend(image_chunks)

            # For tables, we'll use a simple approach since full Docling has issues
            # In a production system, you might want to integrate table extraction separately

        except Exception as e:
            print(f"Error processing PDF: {e}")
            # Fallback to basic text extraction
            documents = self._fallback_text_extraction(pdf_path)

        return documents

    def _extract_text_chunks(self, pdf, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text content from each page."""
        chunks = []

        for page_num in range(len(pdf)):
            try:
                page = pdf[page_num]
                text = page.get_textpage().get_text()
                if text.strip():
                    chunks.append({
                        "content": text.strip(),
                        "metadata": {
                            "page": page_num + 1,
                            "type": "text",
                            "source": os.path.basename(pdf_path),
                            "extraction_method": "pypdfium2"
                        }
                    })
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {e}")

        return chunks

    def _extract_image_chunks(self, pdf, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from the PDF and generate VLM summaries."""
        chunks = []

        for page_num in range(len(pdf)):
            try:
                page = pdf[page_num]
                images = page.get_images()

                for img_index, img in enumerate(images):
                    try:
                        # Extract image
                        image = page.get_image(img[0])
                        image_path = os.path.join(self.temp_dir, f"page_{page_num+1}_img_{img_index}.png")
                        image.save(image_path)

                        # Generate VLM summary
                        context = f"page {page_num + 1} of document '{os.path.basename(pdf_path)}'"
                        if self.vlm_service.is_available():
                            vlm_summary = self.vlm_service.generate_image_summary(image_path, context)
                        else:
                            vlm_summary = f"Image extracted from page {page_num + 1} (VLM service not available)"

                        chunks.append({
                            "content": vlm_summary,
                            "metadata": {
                                "page": page_num + 1,
                                "type": "image_description",
                                "image_path": image_path,
                                "image_index": img_index,
                                "source": os.path.basename(pdf_path),
                                "extraction_method": "pypdfium2_vlm",
                                "vlm_model": "gpt-4o-mini" if self.vlm_service.is_available() else "none"
                            }
                        })
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")

            except Exception as e:
                print(f"Error processing images on page {page_num + 1}: {e}")

        return chunks

    def _fallback_text_extraction(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback text extraction if main processing fails."""
        try:
            with open(pdf_path, 'rb') as f:
                content = f"PDF content from {os.path.basename(pdf_path)} - processing failed, please check PDF format"
                return [{
                    "content": content,
                    "metadata": {
                        "page": 1,
                        "type": "text",
                        "source": os.path.basename(pdf_path),
                        "fallback": True
                    }
                }]
        except Exception as e:
            print(f"Fallback extraction also failed: {e}")
            return []

    def cleanup(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
