"""ElasticSearch client for storing and retrieving documents."""

import logging
from typing import Any, Dict, List, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import RequestError

logger = logging.getLogger(__name__)


class ElasticSearchClient:
    """Client for ElasticSearch operations."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9200,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        verify_certs: bool = False,
    ):
        """Initialize ElasticSearch client.

        Args:
            host: ElasticSearch host
            port: ElasticSearch port
            username: Username for authentication
            password: Password for authentication
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify SSL certificates
        """
        self.host = host
        self.port = port

        # Build connection URL
        scheme = "https" if use_ssl else "http"
        url = f"{scheme}://{host}:{port}"

        # Build auth tuple if credentials provided
        auth = None
        if username and password:
            auth = (username, password)

        self.client = Elasticsearch(
            [url],
            basic_auth=auth,
            verify_certs=verify_certs,
            request_timeout=30,
        )

    def create_index(
        self,
        index_name: str,
        embedding_dimension: int = 1024,
        overwrite: bool = False,
    ) -> bool:
        """Create index with proper mapping for hybrid search.

        Args:
            index_name: Name of the index to create
            embedding_dimension: Dimension of embedding vectors
            overwrite: Whether to delete existing index if it exists

        Returns:
            True if index was created successfully, False otherwise
        """
        # Check if index exists
        if self.client.indices.exists(index=index_name):
            if overwrite:
                logger.info(f"Deleting existing index: {index_name}")
                self.client.indices.delete(index=index_name)
            else:
                logger.info(f"Index {index_name} already exists")
                return True

        # Define mapping for hybrid search
        mapping = {
            "mappings": {
                "properties": {
                    "text_content": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "source": {
                        "type": "keyword",
                    },
                    "page_number": {
                        "type": "integer",
                    },
                    "chunk_type": {
                        "type": "keyword",  # text, table, image
                    },
                    "chunk_id": {
                        "type": "keyword",
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True,
                    },
                }
            }
        }

        try:
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created index: {index_name}")
            return True
        except RequestError as e:
            logger.error(f"Error creating index: {e}")
            return False

    def index_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
        max_retries: int = 3,
    ) -> bool:
        """Index documents in bulk."""
        if not documents:
            return True

        remaining_docs = documents
        for attempt in range(max_retries):
            def doc_generator():
                for doc in remaining_docs:
                    yield {"_index": index_name, "_source": doc}

            try:
                results = bulk(
                    self.client,
                    doc_generator(),
                    chunk_size=batch_size,
                    request_timeout=60,
                    raise_on_error=False,
                )

                failed = results[1] or []
                if not failed:
                    logger.info(f"Successfully indexed {results[0]} documents")
                    return True

                logger.warning(f"Attempt {attempt + 1}: {len(failed)} documents failed")
                if attempt < max_retries - 1:
                    # Extract failed documents for retry
                    remaining_docs = [
                        item["index"]["_source"]
                        for item in failed
                        if "index" in item and "_source" in item.get("index", {})
                    ]
                    if not remaining_docs:
                        break
                    continue

            except Exception as e:
                logger.error(f"Error during bulk indexing: {e}")
                if attempt == max_retries - 1:
                    return False

        logger.error(f"Failed to index {len(remaining_docs)} documents after {max_retries} attempts")
        return False

    def delete_index(self, index_name: str) -> bool:
        """Delete an index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if index was deleted successfully, False otherwise
        """
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted index: {index_name}")
                return True
            else:
                logger.info(f"Index {index_name} does not exist")
                return False
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        return self.client.indices.exists(index=index_name)

    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an index.

        Args:
            index_name: Name of the index

        Returns:
            Dictionary with index information or None if index doesn't exist
        """
        try:
            if not self.client.indices.exists(index=index_name):
                return None

            stats = self.client.indices.stats(index=index_name)
            mapping = self.client.indices.get_mapping(index=index_name)

            return {
                "stats": stats,
                "mapping": mapping,
            }
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return None

    def health_check(self) -> bool:
        """Check if ElasticSearch is healthy.

        Returns:
            True if ElasticSearch is healthy, False otherwise
        """
        try:
            health = self.client.cluster.health()
            return health.get("status") in ["green", "yellow"]
        except Exception as e:
            logger.error(f"ElasticSearch health check failed: {e}")
            return False

