"""Response generation module with citations."""

import json
import logging
import re
from typing import List, Optional, Set, Tuple
from openai import OpenAI, APIError
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
        max_context_length: int = 4096,
    ):
        """Initialize response generator.

        Args:
            llm_base_url: Base URL for LLM API
            llm_model_name: Model name
            api_key: API key if required
            max_tokens: Maximum tokens in response (will be adjusted if needed)
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            max_context_length: Maximum context length for the model (default: 4096)
        """
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key or "sk-no-key-required",
        )
        self.llm_model_name = llm_model_name
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
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
        # Find the earliest occurrence of References section (case-insensitive)
        ref_pattern = re.compile(r'references:', re.IGNORECASE)
        ref_match = ref_pattern.search(text)
        search_text = text[:ref_match.start()] if ref_match else text
        
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
            
            # If response references context but has no citations, add them automatically
            if not cited_refs and self._references_context_but_no_citations(answer):
                answer = self._add_automatic_citations(answer, valid_docs)
                cited_refs = self._extract_cited_references(answer)
                logger.info("Added automatic citations to response")
            
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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses a conservative approximation: ~3 characters per token for English text.
        This is conservative to avoid overestimating available space.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Conservative approximation: 1 token ≈ 3 characters for English
        # Add overhead for formatting and special tokens
        return len(text) // 3 + 100  # Add 100 token buffer for message formatting

    def _parse_token_error(self, error_message: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse token error message to extract actual input tokens and context limit.
        
        Error format: "'max_tokens' or 'max_completion_tokens' is too large: 2048. 
        This model's maximum context length is 4096 tokens and your request has 2177 input tokens"
        
        Args:
            error_message: The error message from the API
            
        Returns:
            Tuple of (input_tokens, max_context_length) or (None, None) if parsing fails
        """
        try:
            # Extract input tokens: "your request has 2177 input tokens"
            input_match = re.search(r'your request has (\d+) input tokens', error_message)
            # Extract max context: "maximum context length is 4096 tokens"
            context_match = re.search(r'maximum context length is (\d+) tokens', error_message)
            
            if input_match and context_match:
                input_tokens = int(input_match.group(1))
                max_context = int(context_match.group(1))
                return input_tokens, max_context
        except (ValueError, AttributeError) as e:
            logger.debug(f"Failed to parse token error: {e}")
        
        return None, None

    def _calculate_max_tokens(self, prompt: str, known_input_tokens: Optional[int] = None) -> int:
        """Calculate safe max_tokens based on prompt length and context window.
        
        Args:
            prompt: The prompt text to be sent to the LLM
            known_input_tokens: If provided, use this instead of estimating
            
        Returns:
            Adjusted max_tokens that fits within available context
        """
        if known_input_tokens is not None:
            input_tokens = known_input_tokens
        else:
            input_tokens = self._estimate_tokens(prompt)
        
        available_tokens = self.max_context_length - input_tokens
        
        # Ensure we have at least 50 tokens for response (minimal safety margin)
        safe_available = max(50, available_tokens - 50)
        
        # Use the minimum of requested max_tokens and available tokens
        adjusted_max_tokens = min(self.max_tokens, safe_available)
        
        if adjusted_max_tokens < self.max_tokens:
            logger.warning(
                f"Reducing max_tokens from {self.max_tokens} to {adjusted_max_tokens} "
                f"(input: {input_tokens} tokens, "
                f"context limit: {self.max_context_length}, available: {available_tokens})"
            )
        
        return adjusted_max_tokens

    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate answer with automatic retry on token limit errors."""
        # Initial attempt with conservative estimate
        adjusted_max_tokens = self._calculate_max_tokens(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=adjusted_max_tokens,
            )
            return response.choices[0].message.content.strip()
            
        except APIError as e:
            # Check if this is a token limit error (400 status)
            error_str = ""
            status_code = None
            
            # Try to get status code from various attributes
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            elif hasattr(e, 'code'):
                status_code = e.code
            
            if status_code == 400:
                # Extract error message from various possible locations
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    try:
                        error_data = e.response.json()
                        if 'error' in error_data and 'message' in error_data['error']:
                            error_str = error_data['error']['message']
                    except:
                        pass
                
                # Fallback to string representation
                if not error_str:
                    error_str = str(e)
                    # Try to extract from dict-like string format
                    # Handle cases like: "Error code: 400 - {'error': {'message': '...'}}"
                    if "'error'" in error_str or '"error"' in error_str:
                        try:
                            # Try to find dict in the string
                            start_idx = error_str.find('{')
                            if start_idx != -1:
                                # Find the matching closing brace
                                brace_count = 0
                                end_idx = start_idx
                                for i, char in enumerate(error_str[start_idx:], start_idx):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            end_idx = i + 1
                                            break
                                
                                if end_idx > start_idx:
                                    dict_str = error_str[start_idx:end_idx]
                                    # Convert single quotes to double quotes for JSON
                                    dict_str = dict_str.replace("'", '"')
                                    error_data = json.loads(dict_str)
                                    if 'error' in error_data and 'message' in error_data['error']:
                                        error_str = error_data['error']['message']
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
                
                # Try to parse the error to get actual token counts
                input_tokens, max_context = self._parse_token_error(error_str)
                
                if input_tokens and max_context:
                    # Update our understanding of the context limit
                    if max_context != self.max_context_length:
                        logger.info(f"Updating max_context_length from {self.max_context_length} to {max_context}")
                        self.max_context_length = max_context
                    
                    # Calculate correct max_tokens based on actual input
                    corrected_max_tokens = self._calculate_max_tokens(prompt, known_input_tokens=input_tokens)
                    
                    logger.warning(
                        f"Retrying with corrected max_tokens: {corrected_max_tokens} "
                        f"(actual input: {input_tokens} tokens)"
                    )
                    
                    # Retry with corrected max_tokens
                    try:
                        response = self.client.chat.completions.create(
                            model=self.llm_model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_tokens=corrected_max_tokens,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as retry_error:
                        logger.error(f"Retry also failed: {retry_error}")
                        raise
                else:
                    # Couldn't parse error, log and re-raise
                    logger.error(f"Token limit error but couldn't parse details: {error_str}")
                    raise
            else:
                # Not a token limit error, re-raise
                raise

    def _clean_references_section(self, answer: str) -> str:
        """Remove any existing References section."""
        answer = re.sub(r'\n\nReferences:.*', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\n\nreferences:.*', '', answer, flags=re.DOTALL)
        return answer

    def _references_context_but_no_citations(self, answer: str) -> bool:
        """Check if response references context but lacks citation markers.
        
        Args:
            answer: LLM response text
            
        Returns:
            True if response seems to reference context but has no citations
        """
        # Phrases that indicate the response is using context information
        context_indicators = [
            r'according to (?:the )?(?:context |document|source|text)',
            r'based on (?:the )?(?:context |document|source|text)',
            r'(?:the |these |those )?(?:context |document|source)',
            r'from (?:the )?(?:context |document|source)',
            r'as (?:mentioned |stated |described )in (?:the )?(?:context |document|source)',
        ]
        
        answer_lower = answer.lower()
        for pattern in context_indicators:
            if re.search(pattern, answer_lower):
                return True
        
        return False

    def _add_automatic_citations(self, answer: str, valid_docs: List[RetrievedDocument]) -> str:
        """Add citation [1] to response when it references context but has no citations.
        
        Args:
            answer: LLM response text
            valid_docs: List of valid documents (we'll cite the first one as [1])
            
        Returns:
            Answer with citation added
        """
        if not valid_docs:
            return answer
        
        # Try to find a good place to add citation - after phrases that reference context
        # Look for phrases that indicate context usage
        context_patterns = [
            r'according to (?:the )?(?:context )?documents?',
            r'based on (?:the )?(?:context )?documents?',
            r'(?:the |these )?(?:context )?documents? (?:say|state|indicate|show|mention)',
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                # Find the end of the sentence containing this phrase
                remaining_text = answer[match.end():]
                sentence_end = re.search(r'[.!?]', remaining_text)
                if sentence_end:
                    insert_pos = match.end() + sentence_end.start()
                    # Insert citation before the punctuation
                    return answer[:insert_pos] + ' [1]' + answer[insert_pos:]
                # No sentence ending found, fall through to add at end
        
        # Add citation at the end of the response (fallback for all cases)
        return answer.rstrip() + ' [1]'

    def _add_references_section(self, answer: str, cited_refs: Set[int], 
                                valid_docs: List[RetrievedDocument]) -> str:
        """Add references section for cited documents with content.
        
        Only includes documents that are explicitly cited in the response text.
        """
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
                        # Use the cut point if it's reasonable (found a good break), otherwise use 500
                        content = content[:cut_point] + "..."
                    # Indent the content for readability
                    indented_content = '\n'.join(f"    {line}" for line in content.split('\n'))
                    answer += f"{indented_content}\n"
        return answer

