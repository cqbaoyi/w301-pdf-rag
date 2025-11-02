"""Indexing pipeline for processing PDFs and storing in ElasticSearch."""

import logging
from pathlib import Path
from typing import List
from .pdf_processor import PDFProcessor
from .chunker import Chunker
from .image_captioner import ImageCaptioner
from .embeddings import EmbeddingService
from .elasticsearch_client import ElasticSearchClient
from .config import Config

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Pipeline for indexing PDF documents."""

    def __init__(self, config: Config):
        """Initialize indexing pipeline.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        es_config = config.get_elasticsearch_config()
        self.es_client = ElasticSearchClient(
            host=es_config.get("host", "127.0.0.1"),
            port=es_config.get("port", 9200),
            username=es_config.get("username"),
            password=es_config.get("password"),
            use_ssl=es_config.get("use_ssl", False),
            verify_certs=es_config.get("verify_certs", False),
        )

        chunking_config = config.get_chunking_config()
        self.chunker = Chunker(
            text_chunk_size=chunking_config.get("text_chunk_size", 512),
            text_chunk_overlap=chunking_config.get("text_chunk_overlap", 50),
            table_chunk_mode=chunking_config.get("table_chunk_mode", "per_table"),
            max_table_size=chunking_config.get("max_table_size", 1000),
        )

        embedding_config = config.get_embedding_config()
        self.embedding_service = EmbeddingService(
            embedding_url=embedding_config.get("url") or embedding_config.get("embedding_url", ""),
            api_key=embedding_config.get("api_key"),
            batch_size=embedding_config.get("batch_size", 32),
        )

        image_config = config.get_image_captioning_config()
        self.image_captioner = ImageCaptioner(
            image_model_url=image_config.get("url") or image_config.get("image_model_url", ""),
            api_key=image_config.get("api_key"),
            timeout=image_config.get("timeout", 30),
            max_image_size=image_config.get("max_image_size", 1024),
        )

        self.pdf_processor = PDFProcessor()

    def index_pdf(self, pdf_path: Path) -> bool:
        """Index a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if indexing was successful, False otherwise
        """
        pdf_path = Path(pdf_path)
        source_name = pdf_path.stem
        logger.info(f"Indexing PDF: {pdf_path}")

        try:
            # Extract and chunk content
            text_extracts, table_extracts, image_extracts = self.pdf_processor.process(pdf_path)
            text_chunks = self.chunker.chunk_text(text_extracts, source_name)
            table_chunks = self.chunker.chunk_tables(table_extracts, source_name)
            image_chunks = self.chunker.prepare_images(image_extracts, source_name)

            # Generate captions for images
            for chunk in image_chunks:
                if image_bytes := chunk.metadata.get("image_bytes"):
                    caption = self.image_captioner.caption_image(
                        image_bytes, chunk.metadata.get("image_format", "PNG")
                    )
                    chunk.content = caption or f"Image from page {chunk.page_number}"

            all_chunks = text_chunks + table_chunks + image_chunks
            if not all_chunks:
                logger.warning(f"No chunks extracted from {pdf_path}")
                return False

            # Generate embeddings
            embeddings = self.embedding_service.embed_texts([chunk.content for chunk in all_chunks])

            # Prepare documents (skip chunks with failed embeddings)
            documents = [
                {
                    "text_content": chunk.content,
                    "embedding": embedding,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type,
                    "chunk_id": chunk.chunk_id,
                    "metadata": self._prepare_metadata(chunk.metadata),
                }
                for chunk, embedding in zip(all_chunks, embeddings)
                if embedding is not None
            ]

            if not documents:
                logger.error("No documents to index after embedding generation")
                return False

            # Ensure index exists
            index_name = self.config.get_elasticsearch_config().get("index_name", "pdf_rag_index")
            if not self.es_client.index_exists(index_name):
                embedding_dim = len(embeddings[0]) if embeddings[0] else 1024
                self.es_client.create_index(index_name, embedding_dimension=embedding_dim)

            # Index documents
            success = self.es_client.index_documents(index_name, documents, batch_size=100)
            if success:
                logger.info(f"Successfully indexed {len(documents)} chunks from {pdf_path}")
            return success

        except Exception as e:
            logger.error(f"Error indexing PDF {pdf_path}: {e}", exc_info=True)
            return False

    def index_directory(self, directory: Path) -> int:
        """Index all PDF files in a directory.

        Args:
            directory: Directory containing PDF files

        Returns:
            Number of successfully indexed files
        """
        pdf_files = list(Path(directory).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        success_count = sum(1 for pdf_file in pdf_files if self.index_pdf(pdf_file))
        logger.info(f"Successfully indexed {success_count}/{len(pdf_files)} PDF files")
        return success_count

    def _prepare_metadata(self, metadata: dict) -> dict:
        """Prepare metadata for ElasticSearch (convert tuples to lists)."""
        result = {}
        for key, value in metadata.items():
            if isinstance(value, tuple):
                result[key] = [float(v) for v in value]
            else:
                result[key] = value
        return result

