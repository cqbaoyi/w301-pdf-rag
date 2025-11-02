"""Result fusion module for combining results from multiple query variations."""

import logging
from typing import List, Dict
from .retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class ResultFusion:
    """Fuse results from multiple query variations using RRF."""

    def __init__(self, rrf_k: int = 60):
        """Initialize result fusion.

        Args:
            rrf_k: RRF constant (typically 60)
        """
        self.rrf_k = rrf_k

    def fuse(
        self,
        query_results: List[List[RetrievedDocument]],
    ) -> List[RetrievedDocument]:
        """Fuse results from multiple queries using Reciprocal Rank Fusion.

        Args:
            query_results: List of result lists, one per query variation

        Returns:
            Fused and ranked list of documents
        """
        # Create document score map
        doc_scores: Dict[str, Dict] = {}

        for rank, results in enumerate(query_results):
            for position, doc in enumerate(results, start=1):
                chunk_id = doc.chunk_id

                if chunk_id not in doc_scores:
                    doc_scores[chunk_id] = {
                        "doc": doc,
                        "score": 0.0,
                    }

                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + position)
                doc_scores[chunk_id]["score"] += rrf_score

        # Sort by combined RRF score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # Create final list with updated scores
        fused_results = []
        for item in sorted_docs:
            doc = item["doc"]
            # Update score to fused score
            fused_doc = RetrievedDocument(
                chunk_id=doc.chunk_id,
                content=doc.content,
                source=doc.source,
                page_number=doc.page_number,
                chunk_type=doc.chunk_type,
                score=item["score"],
                metadata=doc.metadata,
            )
            fused_results.append(fused_doc)

        logger.info(
            f"Fused {len(query_results)} result sets into {len(fused_results)} unique documents"
        )
        return fused_results

