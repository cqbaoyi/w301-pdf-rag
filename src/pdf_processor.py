"""PDF processor for extracting text, tables, and images."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class TextExtract:
    """Extracted text with metadata."""

    text: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class TableExtract:
    """Extracted table with metadata."""

    table_data: List[List[str]]
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class ImageExtract:
    """Extracted image with metadata."""

    image_bytes: bytes
    image_format: str  # PNG, JPEG, etc.
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image_index: int  # Index of image on the page


class PDFProcessor:
    """Processor for extracting content from PDF files."""

    def __init__(self, prefer_pdfplumber: bool = True):
        """Initialize PDF processor.

        Args:
            prefer_pdfplumber: If True, use pdfplumber for table extraction,
                otherwise use PyMuPDF for everything
        """
        self.prefer_pdfplumber = prefer_pdfplumber

    def process(self, pdf_path: Path) -> Tuple[List[TextExtract], List[TableExtract], List[ImageExtract]]:
        """Process PDF file and extract text, tables, and images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (text_extracts, table_extracts, image_extracts)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Processing PDF: {pdf_path}")

        # Extract text and images using PyMuPDF
        text_extracts = []
        image_extracts = []

        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)

            for page_num in range(num_pages):
                page = doc[page_num]

                # Extract text with bounding boxes
                text_dict = page.get_text("dict")
                page_text = page.get_text()
                if page_text.strip():
                    # Get page dimensions for bbox
                    rect = page.rect
                    text_extracts.append(
                        TextExtract(
                            text=page_text,
                            page_number=page_num + 1,
                            bbox=(0, 0, rect.width, rect.height),
                        )
                    )

                # Extract images
                image_list = page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Get image position on page
                        image_rects = page.get_image_rects(xref)
                        if image_rects:
                            rect = image_rects[0]
                            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                        else:
                            # Fallback to full page
                            rect = page.rect
                            bbox = (0, 0, rect.width, rect.height)

                        image_extracts.append(
                            ImageExtract(
                                image_bytes=image_bytes,
                                image_format=image_ext.upper(),
                                page_number=page_num + 1,
                                bbox=bbox,
                                image_index=img_idx,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error extracting image {img_idx} from page {page_num + 1}: {e}"
                        )

            doc.close()

        except Exception as e:
            logger.error(f"Error processing PDF with PyMuPDF: {e}")
            raise

        # Extract tables using pdfplumber (more accurate for tables)
        table_extracts = []
        if self.prefer_pdfplumber:
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        tables = page.extract_tables()
                        for table in tables:
                            if table and len(table) > 0:
                                # Get table bounding box (approximate)
                                # pdfplumber doesn't always provide exact bbox
                                bbox = (
                                    0,
                                    0,
                                    page.width,
                                    page.height,
                                )

                                table_extracts.append(
                                    TableExtract(
                                        table_data=table,
                                        page_number=page_num,
                                        bbox=bbox,
                                    )
                                )
            except Exception as e:
                logger.warning(
                    f"Error extracting tables with pdfplumber: {e}. Using PyMuPDF results only."
                )

        logger.info(
            f"Extracted {len(text_extracts)} text blocks, "
            f"{len(table_extracts)} tables, "
            f"{len(image_extracts)} images"
        )

        return text_extracts, table_extracts, image_extracts

    def extract_text_only(self, pdf_path: Path) -> str:
        """Extract all text from PDF as a single string.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Concatenated text from all pages
        """
        text_extracts, _, _ = self.process(pdf_path)
        return "\n\n".join([extract.text for extract in text_extracts])

    def save_image(self, image_extract: ImageExtract, output_path: Path) -> bool:
        """Save extracted image to file.

        Args:
            image_extract: ImageExtract object
            output_path: Path to save image

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(image_extract.image_bytes)

            logger.debug(f"Saved image to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    def get_image_pil(self, image_extract: ImageExtract) -> Optional[Image.Image]:
        """Get PIL Image object from ImageExtract.

        Args:
            image_extract: ImageExtract object

        Returns:
            PIL Image object or None if conversion fails
        """
        try:
            return Image.open(io.BytesIO(image_extract.image_bytes))
        except Exception as e:
            logger.error(f"Error converting image to PIL: {e}")
            return None

