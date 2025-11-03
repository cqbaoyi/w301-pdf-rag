"""Response generation module with citations."""

import logging
import re
from typing import List, Optional, Set
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

    def _extract_cited_references(self, text: str) -> Set[int]:
        """Extract citation numbers from the response text.
        
        Args:
            text: Response text containing citations like [1], [2], etc.
            
        Returns:
            Set of cited reference numbers
        """
        # Find all citation patterns like [1], [2], etc. in the text
        # Match patterns like [1], [2], [10], etc., but not in the References section
        citations = set()
        
        # Split text into main content and references section
        ref_section_start = -1
        for marker in ["References:", "references:"]:
            idx = text.find(marker)
            if idx != -1 and (ref_section_start == -1 or idx < ref_section_start):
                ref_section_start = idx
        
        # Only search in the main content (before references section)
        search_text = text[:ref_section_start] if ref_section_start != -1 else text
        
        # Find all [number] patterns
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, search_text)
        
        for match in matches:
            try:
                citations.add(int(match))
            except ValueError:
                continue
        
        return citations

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
        # Filter out documents with empty or very short content
        MIN_CONTENT_LENGTH = 10
        valid_docs = [
            doc for doc in retrieved_docs 
            if doc.content and len(doc.content.strip()) >= MIN_CONTENT_LENGTH
        ]
        
        # Format context with citations - make it more readable and structured
        context_parts = []
        for idx, doc in enumerate(valid_docs, start=1):
            citation = f"[{idx}]"
            # Include chunk type for better context understanding
            chunk_type_info = f" ({doc.chunk_type})" if doc.chunk_type != "text" else ""
            context_parts.append(
                f"{citation} Source: {doc.source}, Page: {doc.page_number}{chunk_type_info}\n"
                f"{doc.content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Log the context being sent to LLM for diagnostics
        logger.info("\n" + "=" * 80)
        logger.info("CONTEXT BEING SENT TO GENERATOR:")
        logger.info("=" * 80)
        logger.info(f"Number of documents: {len(valid_docs)}")
        logger.info(f"Query: {query}")
        logger.info("\nFormatted Context:")
        logger.info(context)
        logger.info("=" * 80 + "\n")

        # Build prompt optimized for instruction-tuned models
        # Using a clearer structure that works better with smaller models like gemma-2b-it
        # The prompt structure uses clear labels and imperative instructions
        prompt = f"""Answer the question using the information provided in the context documents below.

CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

Your task:
1. Read each context document carefully
2. Find information that directly answers the question
3. Write a clear answer using that information
4. Cite sources using [1], [2], etc. after each fact
5. Be specific - quote or paraphrase exact details from the context
6. If you find relevant information in the context, you MUST use it - do not say there is no information

Write your answer:"""

        try:
            # Use system message for better instruction following (if supported)
            # Some models work better with system messages
            messages = []
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()
            
            # Log the raw response for diagnostics
            logger.info(f"\nRaw LLM response (first 500 chars): {answer[:500]}")

            # Extract which citations were actually used in the response
            cited_refs = self._extract_cited_references(answer)
            
            # Remove any existing References section to rebuild it
            answer = re.sub(r'\n\nReferences:.*', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\n\nreferences:.*', '', answer, flags=re.DOTALL)
            
            # Build references section only for actually cited documents
            if cited_refs:
                answer += "\n\nReferences:\n"
                for ref_num in sorted(cited_refs):
                    # ref_num is 1-indexed, so we need to map it back to valid_docs
                    if 1 <= ref_num <= len(valid_docs):
                        doc = valid_docs[ref_num - 1]
                        answer += f"[{ref_num}] {doc.source}, Page {doc.page_number} ({doc.chunk_type})\n"

            logger.info(f"Generated response for query: {query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

