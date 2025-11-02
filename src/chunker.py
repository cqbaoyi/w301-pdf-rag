"""Chunking module for splitting documents into retrievable units."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from .pdf_processor import TextExtract, TableExtract, ImageExtract

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of document content."""

    content: str
    chunk_type: str  # "text", "table", "image"
    page_number: int
    chunk_id: str
    source: str
    metadata: dict


class Chunker:
    """Chunker for splitting documents into retrievable chunks."""

    def __init__(
        self,
        text_chunk_size: int = 512,
        text_chunk_overlap: int = 50,
        table_chunk_mode: str = "per_table",
        max_table_size: int = 1000,
    ):
        """Initialize chunker.

        Args:
            text_chunk_size: Maximum size of text chunks (in characters or tokens)
            text_chunk_overlap: Overlap between consecutive chunks
            table_chunk_mode: "per_table" or "split"
            max_table_size: Maximum table size before splitting
        """
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap = text_chunk_overlap
        self.table_chunk_mode = table_chunk_mode
        self.max_table_size = max_table_size

    def chunk_text(
        self, text_extracts: List[TextExtract], source: str
    ) -> List[Chunk]:
        """Chunk text extracts into smaller units.

        Args:
            text_extracts: List of text extracts
            source: Source PDF filename

        Returns:
            List of text chunks
        """
        chunks = []
        chunk_counter = 0

        for extract in text_extracts:
            text = extract.text.strip()
            if not text:
                continue

            # Split by sentences first for better chunk boundaries
            sentences = self._split_sentences(text)

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                # If adding this sentence would exceed chunk size
                if current_length + sentence_length > self.text_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_content = " ".join(current_chunk)
                    chunks.append(
                        Chunk(
                            content=chunk_content,
                            chunk_type="text",
                            page_number=extract.page_number,
                            chunk_id=f"{source}_text_page{extract.page_number}_chunk{chunk_counter}",
                            source=source,
                            metadata={"bbox": extract.bbox},
                        )
                    )
                    chunk_counter += 1

                    # Start new chunk with overlap (last few sentences)
                    if self.text_chunk_overlap > 0 and current_chunk:
                        overlap_chunk = []
                        overlap_length = 0
                        for sent in reversed(current_chunk):
                            if overlap_length + len(sent) <= self.text_chunk_overlap:
                                overlap_chunk.insert(0, sent)
                                overlap_length += len(sent) + 1
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0

                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space

            # Add remaining content as final chunk
            if current_chunk:
                chunks.append(
                    Chunk(
                        content=" ".join(current_chunk),
                        chunk_type="text",
                        page_number=extract.page_number,
                        chunk_id=f"{source}_text_page{extract.page_number}_chunk{chunk_counter}",
                        source=source,
                        metadata={"bbox": extract.bbox},
                    )
                )

        logger.info(f"Created {len(chunks)} text chunks from {len(text_extracts)} extracts")
        return chunks

    def chunk_tables(
        self, table_extracts: List[TableExtract], source: str
    ) -> List[Chunk]:
        """Chunk table extracts.

        Args:
            table_extracts: List of table extracts
            source: Source PDF filename

        Returns:
            List of table chunks
        """
        chunks = []

        for idx, extract in enumerate(table_extracts):
            table_data = extract.table_data
            if not table_data:
                continue

            if self.table_chunk_mode == "per_table":
                # One chunk per table
                table_text = self._table_to_text(table_data)
                chunk_id = f"{source}_table_page{extract.page_number}_idx{idx}"
                chunks.append(
                    Chunk(
                        content=table_text,
                        chunk_type="table",
                        page_number=extract.page_number,
                        chunk_id=chunk_id,
                        source=source,
                        metadata={
                            "bbox": extract.bbox,
                            "table_data": table_data,
                            "num_rows": len(table_data),
                            "num_cols": len(table_data[0]) if table_data else 0,
                        },
                    )
                )
            else:  # split mode
                # Split large tables
                table_text = self._table_to_text(table_data)
                if len(table_text) > self.max_table_size:
                    # Split by rows
                    chunks.extend(
                        self._split_table(table_data, extract, source, idx)
                    )
                else:
                    chunk_id = f"{source}_table_page{extract.page_number}_idx{idx}"
                    chunks.append(
                        Chunk(
                            content=table_text,
                            chunk_type="table",
                            page_number=extract.page_number,
                            chunk_id=chunk_id,
                            source=source,
                            metadata={
                                "bbox": extract.bbox,
                                "table_data": table_data,
                            },
                        )
                    )

        logger.info(f"Created {len(chunks)} table chunks from {len(table_extracts)} extracts")
        return chunks

    def prepare_images(
        self, image_extracts: List[ImageExtract], source: str
    ) -> List[Chunk]:
        """Prepare image extracts for captioning.

        Note: Images themselves are not chunked, but their captions will be.
        This method creates placeholder chunks that will be filled with captions later.

        Args:
            image_extracts: List of image extracts
            source: Source PDF filename

        Returns:
            List of image chunks (with empty content, to be filled with captions)
        """
        chunks = []

        for idx, extract in enumerate(image_extracts):
            chunk_id = f"{source}_image_page{extract.page_number}_idx{extract.image_index}"
            # Content will be filled with caption later
            chunks.append(
                Chunk(
                    content="",  # Will be filled with caption
                    chunk_type="image",
                    page_number=extract.page_number,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={
                        "bbox": extract.bbox,
                        "image_index": extract.image_index,
                        "image_format": extract.image_format,
                        "image_bytes": extract.image_bytes,  # Store for captioning
                    },
                )
            )

        logger.info(f"Prepared {len(chunks)} image chunks from {len(image_extracts)} extracts")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting by period, exclamation, question mark
        # followed by space or newline
        sentences = re.split(r"([.!?]+\s+)", text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        if not result:
            # Fallback: split by newlines
            result = [s.strip() for s in text.split("\n") if s.strip()]
        return result

    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text representation.

        Args:
            table_data: 2D list representing table

        Returns:
            Text representation of table
        """
        if not table_data:
            return ""

        # Convert to markdown-like format
        lines = []
        for row in table_data:
            # Clean cells and join with separator
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            lines.append(" | ".join(clean_row))

        return "\n".join(lines)

    def _split_table(
        self,
        table_data: List[List[str]],
        extract: TableExtract,
        source: str,
        base_idx: int,
    ) -> List[Chunk]:
        """Split a large table into multiple chunks.

        Args:
            table_data: Table data
            extract: Original table extract
            source: Source PDF filename
            base_idx: Base index for chunk IDs

        Returns:
            List of table chunks
        """
        chunks = []
        chunk_size = self.max_table_size // 2  # Rough estimate

        # Split by rows
        current_chunk_data = []
        current_text_length = 0
        chunk_idx = 0

        for row in table_data:
            row_text = self._table_to_text([row])
            row_length = len(row_text)

            if (
                current_text_length + row_length > chunk_size
                and current_chunk_data
            ):
                # Save current chunk
                chunk_text = self._table_to_text(current_chunk_data)
                chunk_id = f"{source}_table_page{extract.page_number}_idx{base_idx}_part{chunk_idx}"
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_type="table",
                        page_number=extract.page_number,
                        chunk_id=chunk_id,
                        source=source,
                        metadata={
                            "bbox": extract.bbox,
                            "table_data": current_chunk_data,
                            "is_partial": True,
                        },
                    )
                )
                chunk_idx += 1
                current_chunk_data = [row]
                current_text_length = row_length
            else:
                current_chunk_data.append(row)
                current_text_length += row_length

        # Add remaining rows
        if current_chunk_data:
            chunk_text = self._table_to_text(current_chunk_data)
            chunk_id = f"{source}_table_page{extract.page_number}_idx{base_idx}_part{chunk_idx}"
            chunks.append(
                Chunk(
                    content=chunk_text,
                    chunk_type="table",
                    page_number=extract.page_number,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={
                        "bbox": extract.bbox,
                        "table_data": current_chunk_data,
                        "is_partial": chunk_idx > 0,
                    },
                )
            )

        return chunks

