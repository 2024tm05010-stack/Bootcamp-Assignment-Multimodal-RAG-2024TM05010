import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
import numpy as np
from typing import List, Dict, Any
import os
import tempfile
from ..config import settings

class PDFProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF document and extract multimodal content.
        Returns a list of document chunks with metadata.
        """
        documents = []

        # Extract text and images using PyMuPDF
        text_chunks = self._extract_text_and_images(pdf_path)
        documents.extend(text_chunks)

        # Extract tables using pdfplumber
        table_chunks = self._extract_tables(pdf_path)
        documents.extend(table_chunks)

        return documents

    def _extract_text_and_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text and images from PDF using PyMuPDF"""
        chunks = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    chunks.append({
                        "content": text.strip(),
                        "metadata": {
                            "page": page_num + 1,
                            "type": "text",
                            "source": os.path.basename(pdf_path)
                        }
                    })

                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Save image temporarily
                    image_path = os.path.join(self.temp_dir, f"page_{page_num+1}_img_{img_index}.{image_ext}")
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    # Extract text from image using OCR
                    ocr_text = self._extract_text_from_image(image_path)

                    if ocr_text.strip():
                        chunks.append({
                            "content": ocr_text.strip(),
                            "metadata": {
                                "page": page_num + 1,
                                "type": "image_ocr",
                                "image_path": image_path,
                                "source": os.path.basename(pdf_path)
                            }
                        })

        return chunks

    def _extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber"""
        chunks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()

                for table_index, table in enumerate(tables):
                    if table:
                        # Convert table to markdown format
                        table_md = self._table_to_markdown(table)

                        chunks.append({
                            "content": table_md,
                            "metadata": {
                                "page": page_num + 1,
                                "type": "table",
                                "table_index": table_index,
                                "source": os.path.basename(pdf_path)
                            }
                        })

        return chunks

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Use pytesseract for OCR
            text = pytesseract.image_to_string(image, config=settings.TESSERACT_CONFIG)
            return text
        except Exception as e:
            print(f"Error extracting text from image {image_path}: {e}")
            return ""

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert table data to markdown format"""
        if not table:
            return ""

        # Clean table data
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)

        # Create markdown table
        markdown_lines = []

        # Header row
        if cleaned_table:
            header = cleaned_table[0]
            markdown_lines.append("| " + " | ".join(header) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

            # Data rows
            for row in cleaned_table[1:]:
                markdown_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(markdown_lines)

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)