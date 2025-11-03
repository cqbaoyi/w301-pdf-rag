"""Main pipeline for query processing and response generation."""

import logging
from typing import Optional, List
from .config import Config
from .query_fusion import QueryFusion
from .embeddings import EmbeddingService
from .hybrid_retriever import HybridRetriever
from .result_fusion import ResultFusion
from .reranker import Reranker
from .response_generator import ResponseGenerator
from .elasticsearch_client import ElasticSearchClient

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Pipeline for processing queries and generating responses."""

    def __init__(self, config: Config):
        """Initialize query pipeline.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize ElasticSearch client
        es_config = config.get_elasticsearch_config()
        self.es_client = ElasticSearchClient(
            host=es_config.get("host", "127.0.0.1"),
            port=es_config.get("port", 9200),
            username=es_config.get("username"),
            password=es_config.get("password"),
            use_ssl=es_config.get("use_ssl", False),
            verify_certs=es_config.get("verify_certs", False),
        )

        # Initialize embedding service
        embedding_config = config.get_embedding_config()
        self.embedding_service = EmbeddingService(
            embedding_url=embedding_config.get("url", ""),
            api_key=embedding_config.get("api_key"),
        )

        # Initialize hybrid retriever
        search_config = config.get_search_config()
        embedding_dim = embedding_config.get("dimension", 1024)
        index_name = es_config.get("index_name", "pdf_rag_index")
        
        self.retriever = HybridRetriever(
            es_client=self.es_client.client,
            index_name=index_name,
            dense_weight=search_config.get("dense_weight", 0.6),
            sparse_weight=search_config.get("sparse_weight", 0.4),
            embedding_dimension=embedding_dim,
        )

        # Initialize query fusion
        query_fusion_config = config.get_query_fusion_config()
        
        self.query_fusion = QueryFusion(
            llm_base_url=query_fusion_config.get("base_url", "https://api.openai.com/v1"),
            llm_model_name=query_fusion_config.get("model_name", "gpt-5"),
            api_key=query_fusion_config.get("api_key"),
            max_variations=query_fusion_config.get("max_variations", 5),
            min_similarity_threshold=query_fusion_config.get("min_similarity_threshold", 0.85),
            min_variation_similarity=query_fusion_config.get("min_variation_similarity", 0.90),
            temperature=query_fusion_config.get("temperature", 0.7),
        )

        # Initialize result fusion
        result_fusion_config = config.get_result_fusion_config()
        self.result_fusion = ResultFusion(
            rrf_k=result_fusion_config.get("rrf_k", 60),
        )

        # Initialize reranker
        reranking_config = config.get_reranking_config()
        self.reranker = Reranker(
            rerank_url=reranking_config.get("url") or reranking_config.get("rerank_url", ""),
            api_key=reranking_config.get("api_key"),
        )

        # Initialize response generator
        generation_config = config.get_generation_config()
        self.generator = ResponseGenerator(
            llm_base_url=generation_config.get("base_url", ""),
            llm_model_name=generation_config.get("model_name", "gemma-2b"),
            api_key=generation_config.get("api_key"),
            max_tokens=generation_config.get("max_tokens", 2048),
            temperature=generation_config.get("temperature", 0.7),
            top_p=generation_config.get("top_p", 0.9),
        )

    def query(self, user_query: str) -> str:
        """Process a user query and generate response."""
        logger.info(f"Processing query: {user_query}")

        try:
            query_variations = self._generate_and_log_variations(user_query)
            query_embeddings = self.embedding_service.embed_texts(query_variations)
            
            all_query_results = self._retrieve_for_variations(query_variations, query_embeddings)
            if not all_query_results:
                return "Error: Failed to retrieve any documents."

            fused_results = self.result_fusion.fuse(all_query_results)
            logger.info(f"Fused results: {len(fused_results)} unique documents from {len(all_query_results)} query variations")

            reranked_results = self._rerank_results(user_query, fused_results)
            self._log_ranked_results(reranked_results)

            response = self.generator.generate(user_query, reranked_results)
            logger.info("Successfully generated response")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"Error processing query: {str(e)}"

    def _generate_and_log_variations(self, user_query: str) -> List[str]:
        """Generate query variations and log them."""
        query_variations = self.query_fusion.generate_variations(user_query)
        logger.info(f"Generated {len(query_variations)} query variations")
        print("\n" + "=" * 80)
        print("QUERY VARIATIONS:")
        print("=" * 80)
        for idx, variation in enumerate(query_variations, start=1):
            print(f"{idx}. {variation}")
        print("=" * 80 + "\n")
        return query_variations

    def _retrieve_for_variations(self, query_variations: List[str], 
                                 query_embeddings: List) -> List[List]:
        """Retrieve documents for each query variation."""
        search_config = self.config.get_search_config()
        top_k_per_query = search_config.get("top_k_per_query", 20)
        
        all_query_results = []
        for query_text, query_embedding in zip(query_variations, query_embeddings):
            if query_embedding is None:
                logger.warning(f"Failed to embed query: {query_text}")
                continue

            results = self.retriever.retrieve(
                query=query_text,
                query_embedding=query_embedding,
                top_k=top_k_per_query,
            )
            all_query_results.append(results)
        return all_query_results

    def _rerank_results(self, user_query: str, fused_results: List) -> List:
        """Rerank fused results."""
        reranking_config = self.config.get_reranking_config()
        search_config = self.config.get_search_config()
        max_rerank_input = reranking_config.get("max_rerank_input", 100)
        final_top_k = search_config.get("final_top_k", 5)
        
        if len(fused_results) > max_rerank_input:
            logger.info(
                f"Limiting {len(fused_results)} fused results to {max_rerank_input} "
                f"before reranking (API limit)"
            )
            fused_results = fused_results[:max_rerank_input]
        
        return self.reranker.rerank(user_query, fused_results, top_k=final_top_k)

    def _log_ranked_results(self, reranked_results: List):
        """Log top ranked results."""
        search_config = self.config.get_search_config()
        final_top_k = search_config.get("final_top_k", 5)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"TOP {final_top_k} RANKED RESULTS SENT TO GENERATOR:")
        logger.info("=" * 80)
        for idx, doc in enumerate(reranked_results[:final_top_k], start=1):
            content_preview = doc.content[:150].replace("\n", " ") if doc.content else "(empty)"
            if len(doc.content) > 150:
                content_preview += "..."
            logger.info(
                f"[{idx}] Type: {doc.chunk_type} | "
                f"Source: {doc.source} | Page: {doc.page_number} | "
                f"Score: {doc.score:.4f}"
            )
            logger.info(f"    Content: {content_preview}")
            logger.info("")
        logger.info("=" * 80 + "\n")

