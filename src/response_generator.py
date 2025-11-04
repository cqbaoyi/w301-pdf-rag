"""Response generation module with citations."""

import logging
import re
from typing import List, Optional, Set
from openai import OpenAI
from .hybrid_retriever import RetrievedDocument

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
            text: Response text containing citations like [1], [2], document 1, etc.
            
        Returns:
            Set of cited reference numbers
        """
        # Find all citation patterns like [1], [2], document 1, doc 1, etc.
        citations = set()
        
        # Split text into main content and references section
        ref_section_start = -1
        for marker in ["References:", "references:"]:
            idx = text.find(marker)
            if idx != -1 and (ref_section_start == -1 or idx < ref_section_start):
                ref_section_start = idx
        
        # Only search in the main content (before references section)
        search_text = text[:ref_section_start] if ref_section_start != -1 else text
        
        # Pattern 1: [1], [2], etc.
        bracket_pattern = r'\[(\d+)\]'
        bracket_matches = re.findall(bracket_pattern, search_text)
        for match in bracket_matches:
            try:
                citations.add(int(match))
            except ValueError:
                continue
        
        # Pattern 2: "document 1", "document 2", "context document 1", etc.
        doc_patterns = [
            r'(?:context\s+)?document\s+(\d+)',
            r'doc\s+(\d+)',
            r'document\s+number\s+(\d+)',
        ]
        for pattern in doc_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            for match in matches:
                try:
                    citations.add(int(match))
                except ValueError:
                    continue
        
        return citations

    def generate(
        self, query: str, retrieved_docs: List[RetrievedDocument]
    ) -> str:
        """Generate response with citations."""
        valid_docs = self._filter_valid_docs(retrieved_docs)
        context = self._format_context(valid_docs)
        self._log_context(query, valid_docs, context)
        
        prompt = self._build_prompt(query, context)
        
        try:
            answer = self._call_llm(prompt)
            logger.info(f"\nRaw LLM response (first 500 chars): {answer[:500]}")
            
            cited_refs = self._extract_cited_references(answer)
            answer = self._clean_references_section(answer)
            answer = self._add_references_section(answer, cited_refs, valid_docs)
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def _filter_valid_docs(self, retrieved_docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Filter out documents with empty or very short content."""
        MIN_CONTENT_LENGTH = 10
        return [
            doc for doc in retrieved_docs 
            if doc.content and len(doc.content.strip()) >= MIN_CONTENT_LENGTH
        ]

    def _format_context(self, valid_docs: List[RetrievedDocument]) -> str:
        """Format context with citations."""
        context_parts = []
        for idx, doc in enumerate(valid_docs, start=1):
            citation = f"[{idx}]"
            chunk_type_info = f" ({doc.chunk_type})" if doc.chunk_type != "text" else ""
            context_parts.append(
                f"{citation} Source: {doc.source}, Page: {doc.page_number}{chunk_type_info}\n"
                f"{doc.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _log_context(self, query: str, valid_docs: List[RetrievedDocument], context: str):
        """Log context being sent to LLM."""
        logger.info("\n" + "=" * 80)
        logger.info("CONTEXT BEING SENT TO GENERATOR:")
        logger.info("=" * 80)
        logger.info(f"Number of documents: {len(valid_docs)}")
        logger.info(f"Query: {query}")
        logger.info("\nFormatted Context:")
        logger.info(context)
        logger.info("=" * 80 + "\n")

    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM."""
        return f"""Answer the question using the information provided in the context documents below.

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

    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate answer."""
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _clean_references_section(self, answer: str) -> str:
        """Remove any existing References section."""
        answer = re.sub(r'\n\nReferences:.*', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\n\nreferences:.*', '', answer, flags=re.DOTALL)
        return answer

    def _add_references_section(self, answer: str, cited_refs: Set[int], 
                                valid_docs: List[RetrievedDocument]) -> str:
        """Add references section for cited documents with content.
        
        If no citations are found but documents were provided, show all documents
        since they were all used to generate the response.
        """
        # If no citations found but we have documents, show all documents
        # (they were all used to generate the response)
        if not cited_refs and valid_docs:
            cited_refs = set(range(1, len(valid_docs) + 1))
        
        if not cited_refs:
            return answer
            
        answer += "\n\nReferences:\n"
        for ref_num in sorted(cited_refs):
            if 1 <= ref_num <= len(valid_docs):
                doc = valid_docs[ref_num - 1]
                chunk_type_info = f" ({doc.chunk_type})" if doc.chunk_type != "text" else ""
                answer += f"[{ref_num}] {doc.source}, Page {doc.page_number}{chunk_type_info}\n"
                if doc.content:
                    # Include the actual content, with indentation for readability
                    content = doc.content.strip()
                    # Truncate very long content (first 500 chars) with smart truncation
                    if len(content) > 500:
                        # Try to truncate at a sentence or word boundary
                        truncated = content[:500]
                        last_period = truncated.rfind('.')
                        last_space = truncated.rfind(' ')
                        cut_point = max(last_period, last_space) if (last_period > 0 or last_space > 0) else 500
                        if cut_point > 400:  # Only truncate if we found a good break point
                            content = content[:cut_point] + "..."
                        else:
                            content = content[:500] + "..."
                    # Indent the content for readability
                    indented_content = '\n'.join(f"    {line}" for line in content.split('\n'))
                    answer += f"{indented_content}\n"
        return answer

