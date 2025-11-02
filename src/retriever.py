"""Hybrid search retriever combining dense and sparse search."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    chunk_id: str
    content: str
    source: str
    page_number: int
    chunk_type: str
    score: float
    metadata: Dict[str, Any]


class HybridRetriever:
    """Hybrid retriever combining dense vector and sparse BM25 search."""

    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        embedding_dimension: int = 1024,
    ):
        """Initialize hybrid retriever.

        Args:
            es_client: ElasticSearch client
            index_name: Name of the index to search
            dense_weight: Weight for dense vector search
            sparse_weight: Weight for sparse BM25 search
            embedding_dimension: Dimension of embedding vectors
        """
        self.es_client = es_client
        self.index_name = index_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.embedding_dimension = embedding_dimension

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 50,
    ) -> List[RetrievedDocument]:
        """Retrieve documents using hybrid search."""
        if not query_embedding:
            logger.warning("Empty query embedding, falling back to sparse search only")
            return self._sparse_search_documents(query, top_k)

        dense_results = self._dense_search(query_embedding, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)
        return self._combine_results(dense_results, sparse_results, top_k)

    def _dense_search(self, query_embedding: List[float], top_k: int) -> Dict[str, Dict]:
        """Perform dense vector search."""
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding},
                            },
                        }
                    },
                    "_source": True,
                },
            )
            return self._parse_search_results(response)
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return {}

    def _sparse_search(self, query: str, top_k: int) -> Dict[str, Dict]:
        """Perform sparse BM25 search."""
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "size": top_k,
                    "query": {"match": {"text_content": query}},
                    "_source": True,
                },
            )
            return self._parse_search_results(response)
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return {}

    def _sparse_search_documents(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Perform sparse search and return documents directly."""
        results = self._sparse_search(query, top_k)
        return self._convert_to_documents(results)

    def _parse_search_results(self, response: Dict) -> Dict[str, Dict]:
        """Parse ElasticSearch search results."""
        return {
            hit["_source"].get("chunk_id", hit["_id"]): {
                "score": hit["_score"],
                "source": hit["_source"],
            }
            for hit in response.get("hits", {}).get("hits", [])
        }

    def _combine_results(
        self,
        dense_results: Dict[str, Dict],
        sparse_results: Dict[str, Dict],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """Combine dense and sparse search results with weighted scores."""
        # Extract scores and normalize
        dense_scores = {k: r["score"] for k, r in dense_results.items()}
        sparse_scores = {k: r["score"] for k, r in sparse_results.items()}
        
        dense_max = max(dense_scores.values()) if dense_scores else 1.0
        sparse_max = max(sparse_scores.values()) if sparse_scores else 1.0

        # Combine normalized scores
        combined = {}
        all_chunk_ids = set(dense_results.keys()) | set(sparse_results.keys())
        
        for chunk_id in all_chunk_ids:
            dense_norm = dense_scores.get(chunk_id, 0) / dense_max if dense_max > 0 else 0
            sparse_norm = sparse_scores.get(chunk_id, 0) / sparse_max if sparse_max > 0 else 0
            combined_score = self.dense_weight * dense_norm + self.sparse_weight * sparse_norm
            
            source = (dense_results.get(chunk_id) or sparse_results.get(chunk_id))["source"]
            combined[chunk_id] = {"score": combined_score, "source": source}

        # Sort by score and convert to documents
        sorted_chunks = dict(sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k])
        return self._convert_to_documents(sorted_chunks)

    def _convert_to_documents(self, results: Dict[str, Dict]) -> List[RetrievedDocument]:
        """Convert search results to RetrievedDocument objects."""
        return [
            RetrievedDocument(
                chunk_id=chunk_id,
                content=src.get("text_content", ""),
                source=src.get("source", ""),
                page_number=src.get("page_number", 0),
                chunk_type=src.get("chunk_type", "text"),
                score=data["score"],
                metadata=src.get("metadata", {}),
            )
            for chunk_id, data in results.items()
            if (src := data["source"])
        ]

