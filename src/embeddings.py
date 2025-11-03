"""Embedding generation using qwen3-embedding-0.6b service."""

import logging
from typing import List, Optional, Tuple
import requests

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
        """Generate embeddings for a batch of texts."""
        try:
            valid_texts, valid_indices = self._filter_valid_texts(texts)
            if not valid_texts:
                logger.warning("Batch contains no valid texts")
                return [None] * len(texts)

            response = self._make_embedding_request(valid_texts)
            embeddings = self._parse_embedding_response(response, len(texts))
            
            if embeddings is None:
                return [None] * len(texts)

            return self._map_embeddings_to_positions(embeddings, valid_indices, len(texts))

        except Exception as e:
            error_msg = str(e)[:200] if len(str(e)) <= 200 else str(e)[:200] + "..."
            logger.error(f"Error generating embeddings: {type(e).__name__}: {error_msg}")
            import traceback
            logger.debug(traceback.format_exc())
            return [None] * len(texts)

    def _filter_valid_texts(self, texts: List[str]) -> Tuple[List[str], List[Optional[int]]]:
        """Filter out empty or whitespace-only texts."""
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                valid_indices.append(None)
        return valid_texts, valid_indices

    def _make_embedding_request(self, valid_texts: List[str]) -> requests.Response:
        """Make API request for embeddings."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"texts": valid_texts}
        response = requests.post(
            self.embedding_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        
        self._log_embedding_error(response)
        response.raise_for_status()
        return response

    def _log_embedding_error(self, response: requests.Response):
        """Log embedding API errors safely."""
        if response.status_code == 200:
            return
            
        try:
            error_detail = response.json()
            logger.error(f"Embedding API error {response.status_code}: {error_detail}")
        except:
            try:
                error_text = response.text[:200]
                if any(ord(c) < 32 and c not in '\n\r\t' for c in error_text[:50]):
                    logger.error(f"Embedding API error {response.status_code}: [Binary or non-text response]")
                else:
                    logger.error(f"Embedding API error {response.status_code}: {error_text}")
            except:
                logger.error(f"Embedding API error {response.status_code}: [Unable to decode response]")

    def _parse_embedding_response(self, response: requests.Response, num_texts: int) -> Optional[List]:
        """Parse embedding response from API."""
        result = response.json()
        
        embeddings = None
        if "data" in result and isinstance(result["data"], dict):
            if "text_vectors" in result["data"]:
                embeddings = result["data"]["text_vectors"]
        elif "data" in result and isinstance(result["data"], list):
            embeddings = [item["embedding"] if isinstance(item, dict) else item for item in result["data"]]
        elif isinstance(result, list):
            embeddings = result
        elif "text_vectors" in result:
            embeddings = result["text_vectors"]
        
        if embeddings is None:
            logger.warning(f"Unexpected response format. Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        return embeddings

    def _map_embeddings_to_positions(self, embeddings: List, valid_indices: List[Optional[int]], 
                                     num_texts: int) -> List[Optional[List[float]]]:
        """Map embeddings back to original text positions."""
        result_embeddings = [None] * num_texts
        embedding_idx = 0
        for i, valid_idx in enumerate(valid_indices):
            if valid_idx is not None and embedding_idx < len(embeddings):
                result_embeddings[i] = embeddings[embedding_idx]
                embedding_idx += 1

        if embedding_idx != len(embeddings):
            logger.warning(f"Expected {len([i for i in valid_indices if i is not None])} embeddings, got {len(embeddings)}")

        return result_embeddings


