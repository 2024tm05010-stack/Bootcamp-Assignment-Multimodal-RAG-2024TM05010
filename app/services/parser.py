import os
import tempfile
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


class PDFParser:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        # Configure Docling pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True  # Enable OCR for images
        self.pipeline_options.do_table_structure = True  # Extract table structure

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a PDF document and extract multimodal content using Docling."""
        documents = []

        try:
            # Initialize Docling converter
            doc_converter = DocumentConverter(
                format_options={InputFormat.PDF: self.pipeline_options}
            )

            # Convert the PDF
            result = doc_converter.convert(pdf_path)
            doc = result.document

            # Extract different content types
            text_chunks = self._extract_text_chunks(doc, pdf_path)
            documents.extend(text_chunks)

            table_chunks = self._extract_table_chunks(doc, pdf_path)
            documents.extend(table_chunks)

            image_chunks = self._extract_image_chunks(doc, pdf_path)
            documents.extend(image_chunks)

        except Exception as e:
            print(f"Error processing PDF with Docling: {e}")
            # Fallback to basic text extraction if Docling fails
            documents = self._fallback_text_extraction(pdf_path)

        return documents

    def _extract_text_chunks(self, doc, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text content as separate chunks."""
        chunks = []

        for item in doc.texts:
            if item.text.strip():
                chunks.append({
                    "content": item.text.strip(),
                    "metadata": {
                        "page": getattr(item, 'page_no', 1),
                        "type": "text",
                        "source": os.path.basename(pdf_path),
                        "docling_type": "text"
                    }
                })

        return chunks

    def _extract_table_chunks(self, doc, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables as markdown chunks."""
        chunks = []

        for table_index, table in enumerate(doc.tables):
            try:
                # Convert table to markdown
                table_md = table.export_to_markdown()
                if table_md.strip():
                    chunks.append({
                        "content": table_md.strip(),
                        "metadata": {
                            "page": getattr(table, 'page_no', 1),
                            "type": "table",
                            "table_index": table_index,
                            "source": os.path.basename(pdf_path),
                            "docling_type": "table"
                        }
                    })
            except Exception as e:
                print(f"Error extracting table {table_index}: {e}")

        return chunks

    def _extract_image_chunks(self, doc, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images and their OCR text as separate chunks."""
        chunks = []

        for img_index, image in enumerate(doc.pictures):
            try:
                # Get image metadata
                image_path = getattr(image, 'image_path', None)
                if image_path and os.path.exists(image_path):
                    # Copy image to temp directory for persistence
                    temp_image_path = os.path.join(self.temp_dir, f"docling_img_{img_index}.png")
                    with open(image_path, "rb") as src, open(temp_image_path, "wb") as dst:
                        dst.write(src.read())

                    # Extract OCR text if available
                    ocr_text = getattr(image, 'ocr_text', '')
                    if ocr_text.strip():
                        chunks.append({
                            "content": ocr_text.strip(),
                            "metadata": {
                                "page": getattr(image, 'page_no', 1),
                                "type": "image_ocr",
                                "image_path": temp_image_path,
                                "image_index": img_index,
                                "source": os.path.basename(pdf_path),
                                "docling_type": "image"
                            }
                        })
            except Exception as e:
                print(f"Error extracting image {img_index}: {e}")

        return chunks

    def _fallback_text_extraction(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback text extraction if Docling fails."""
        try:
            # Simple fallback using basic file reading
            with open(pdf_path, 'rb') as f:
                # This is a very basic fallback - in practice, you might want to use a simpler PDF library
                content = f"PDF content from {os.path.basename(pdf_path)} - Docling processing failed"
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
