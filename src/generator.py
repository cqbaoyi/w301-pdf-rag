"""Response generation module with citations."""

import logging
from typing import List, Optional
from openai import OpenAI
from .retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses with citations using Gemma-2B."""

    def __init__(
        self,
        llm_base_url: str,
        llm_model_name: str = "gemma-2b",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize response generator.

        Args:
            llm_base_url: Base URL for LLM API
            llm_model_name: Model name
            api_key: API key if required
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
        """
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key or "sk-no-key-required",
        )
        self.llm_model_name = llm_model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(
        self, query: str, retrieved_docs: List[RetrievedDocument]
    ) -> str:
        """Generate response with citations.

        Args:
            query: User query
            retrieved_docs: Retrieved documents with metadata

        Returns:
            Generated response with citations
        """
        # Format context with citations
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            citation = f"[{idx}]"
            context_parts.append(
                f"{citation} Source: {doc.source}, Page: {doc.page_number}\n"
                f"Content: {doc.content}"
            )

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.
Use citations [1], [2], etc. to reference the source documents when using information from them.

Context Documents:
{context}

Question: {query}

Answer the question using information from the context documents. Include citations [1], [2], etc. in your response when referencing specific documents. At the end, provide a "References:" section listing all cited sources with their details."""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate, well-cited answers based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()

            # Ensure references section is present
            if "References:" not in answer and "references:" not in answer:
                answer += "\n\nReferences:\n"
                for idx, doc in enumerate(retrieved_docs, start=1):
                    answer += f"[{idx}] {doc.source}, Page {doc.page_number} ({doc.chunk_type})\n"

            logger.info(f"Generated response for query: {query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

