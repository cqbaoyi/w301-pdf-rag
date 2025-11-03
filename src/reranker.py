"""Reranking module using qwen3-reranker-0.6b service."""

import logging
from typing import List, Optional
import requests
from .retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker using qwen3-reranker-0.6b service."""

    def __init__(
        self,
        rerank_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize reranker.

        Args:
            rerank_url: URL for reranking API endpoint
            api_key: API key if required
            timeout: Request timeout in seconds
        """
        self.rerank_url = rerank_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def rerank(
        self, query: str, documents: List[RetrievedDocument], top_k: int = 5
    ) -> List[RetrievedDocument]:
        """Rerank documents using reranking service.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top documents to return (should match config search.final_top_k)

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        # Enforce API limit of 100 documents
        MAX_RERANK_DOCS = 100
        if len(documents) > MAX_RERANK_DOCS:
            logger.warning(
                f"Limiting {len(documents)} documents to {MAX_RERANK_DOCS} for reranker API "
                f"(API maximum is {MAX_RERANK_DOCS})"
            )
            documents = documents[:MAX_RERANK_DOCS]

        try:
            # Prepare request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # API expects documents as array of strings (not objects)
            doc_texts = [doc.content for doc in documents]
            
            # Create mapping from content to original document (for matching results)
            # Use index as fallback since content matching might fail with whitespace differences
            content_to_doc = {doc.content: doc for doc in documents}
            index_to_doc = {i: doc for i, doc in enumerate(documents)}

            payload = {
                "model": "qwen3-reranker-0.6b",
                "query": query,
                "documents": doc_texts,  # Array of strings
                "top_n": min(top_k, len(documents)),
            }

            # Make request
            response = requests.post(
                self.rerank_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            
            # Better error logging
            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    logger.error(f"Reranking API error {response.status_code}: {error_detail}")
                except:
                    logger.error(f"Reranking API error {response.status_code}: {response.text[:200]}")
            
            response.raise_for_status()

            result = response.json()

            # Parse reranked results
            # Response format: {"scores": [...], "ranked_documents": [{"document": "...", "score": ..., "index": ...}]}
            reranked_results = None
            if "ranked_documents" in result:
                reranked_results = result["ranked_documents"]
            elif "scores" in result:
                # If only scores are returned, we need to reconstruct from scores and original order
                scores = result["scores"]
                reranked_results = [
                    {
                        "document": doc_texts[i],
                        "score": scores[i],
                        "index": i
                    }
                    for i in range(len(scores))
                ]
            elif isinstance(result, list):
                reranked_results = result
            else:
                logger.warning(f"Unexpected rerank response format: {list(result.keys())}")
                return documents[:top_k]

            # Build reranked document list by matching results back to original documents
            reranked_docs = []
            for item in reranked_results:
                # Try to get document text and score from response
                doc_text = item.get("document") or item.get("text")
                score = item.get("score", 0.0)
                original_index = item.get("index")
                
                # Match back to original document
                # Priority: index > exact content match > fuzzy content match
                original_doc = None
                if original_index is not None and original_index in index_to_doc:
                    # Match by original index (most reliable)
                    original_doc = index_to_doc[original_index]
                elif doc_text and doc_text in content_to_doc:
                    # Match by exact content
                    original_doc = content_to_doc[doc_text]
                elif doc_text:
                    # Try fuzzy matching by content (handle whitespace differences)
                    for orig_doc in documents:
                        if orig_doc.content.strip() == doc_text.strip():
                            original_doc = orig_doc
                            break
                
                if original_doc:
                    # Create reranked document with updated score
                    reranked_doc = RetrievedDocument(
                        chunk_id=original_doc.chunk_id,
                        content=original_doc.content,
                        source=original_doc.source,
                        page_number=original_doc.page_number,
                        chunk_type=original_doc.chunk_type,
                        score=score,
                        metadata=original_doc.metadata,
                    )
                    reranked_docs.append(reranked_doc)
                else:
                    logger.warning(f"Could not match reranked document to original: index={original_index}, text={doc_text[:50] if doc_text else 'unknown'}")

            # Limit to top_k results
            reranked_docs = reranked_docs[:top_k]
            
            logger.info(
                f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}"
            )
            return reranked_docs

        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return original documents if reranking fails
            return documents[:top_k]

