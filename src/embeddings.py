"""Embedding generation using qwen3-embedding-0.6b service."""

import logging
from typing import List, Optional
import requests
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service client using qwen3-embedding-0.6b."""

    def __init__(
        self,
        embedding_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        batch_size: int = 32,
    ):
        """Initialize embedding service.

        Args:
            embedding_url: URL for embedding API endpoint
            api_key: API key if required
            timeout: Request timeout in seconds
            batch_size: Batch size for batch requests
        """
        self.embedding_url = embedding_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if embedding fails
        """
        embeddings = self.embed_texts([text])
        if embeddings:
            return embeddings[0]
        return None

    def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        if not texts:
            return []

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        try:
            # Filter out empty or whitespace-only texts
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
                else:
                    valid_indices.append(None)  # Mark as invalid
            
            if not valid_texts:
                logger.warning("Batch contains no valid texts")
                return [None] * len(texts)

            # Prepare request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # API expects "texts" field, not "input"
            payload = {
                "texts": valid_texts,
            }

            # Make request
            response = requests.post(
                self.embedding_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            
            # Better error logging for debugging
            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    logger.error(f"Embedding API error {response.status_code}: {error_detail}")
                except:
                    # Try to decode as text, but avoid printing binary data
                    try:
                        error_text = response.text[:200]
                        # Check if it's likely binary (contains many non-printable chars)
                        if any(ord(c) < 32 and c not in '\n\r\t' for c in error_text[:50]):
                            logger.error(f"Embedding API error {response.status_code}: [Binary or non-text response]")
                        else:
                            logger.error(f"Embedding API error {response.status_code}: {error_text}")
                    except:
                        logger.error(f"Embedding API error {response.status_code}: [Unable to decode response]")
            
            response.raise_for_status()

            result = response.json()
            
            # Handle the API response format: {"code": 0, "message": "success", "data": {"text_vectors": [[...], [...]]}}
            embeddings = None
            if "data" in result and isinstance(result["data"], dict):
                # Format: {"data": {"text_vectors": [[...], [...]]}}
                if "text_vectors" in result["data"]:
                    embeddings = result["data"]["text_vectors"]
            elif "data" in result and isinstance(result["data"], list):
                # OpenAI format: {"data": [{"embedding": [...], ...}, ...]}
                embeddings = [item["embedding"] if isinstance(item, dict) else item for item in result["data"]]
            elif isinstance(result, list):
                # Direct list of embeddings
                embeddings = result
            elif "text_vectors" in result:
                # Direct text_vectors in result
                embeddings = result["text_vectors"]
            
            if embeddings is None:
                logger.warning(f"Unexpected response format. Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return [None] * len(texts)

            # Map embeddings back to original positions (accounting for filtered texts)
            result_embeddings = [None] * len(texts)
            embedding_idx = 0
            for i, valid_idx in enumerate(valid_indices):
                if valid_idx is not None:
                    if embedding_idx < len(embeddings):
                        result_embeddings[i] = embeddings[embedding_idx]
                        embedding_idx += 1

            # Ensure we have the right number of embeddings
            if embedding_idx != len(embeddings):
                logger.warning(
                    f"Expected {len(valid_texts)} embeddings for valid texts, got {len(embeddings)}"
                )

            return result_embeddings

        except Exception as e:
            # Only log exception type and message, not full exception (may contain binary data)
            error_msg = str(e)[:200] if len(str(e)) <= 200 else str(e)[:200] + "..."
            logger.error(f"Error generating embeddings: {type(e).__name__}: {error_msg}")
            import traceback
            logger.debug(traceback.format_exc())
            return [None] * len(texts)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension (default 1024 for qwen3-embedding-0.6b)
        """
        # Test with a dummy text to get dimension
        test_embedding = self.embed_text("test")
        if test_embedding:
            return len(test_embedding)
        # Default dimension for qwen3-embedding-0.6b
        return 1024

