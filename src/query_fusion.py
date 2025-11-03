"""Query fusion module for RAG Fusion - generating query variations."""

import logging
import re
from typing import List, Optional
from difflib import SequenceMatcher
from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryFusion:
    """Generate multiple query variations from a single user query."""

    def __init__(
        self,
        llm_base_url: str,
        llm_model_name: str = "gemma-2b",
        api_key: Optional[str] = None,
        max_variations: int = 5,
        min_similarity_threshold: float = 0.85,
        min_variation_similarity: float = 0.90,
        temperature: float = 0.7,
    ):
        """Initialize query fusion.

        Args:
            llm_base_url: Base URL for LLM API
            llm_model_name: Model name
            api_key: API key if required
            max_variations: Maximum number of candidate variations to generate
            min_similarity_threshold: Reject variations too similar to original (0.0-1.0)
            min_variation_similarity: Reject variations too similar to each other (0.0-1.0)
            temperature: Temperature for generation
        """
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key or "sk-no-key-required",
        )
        self.llm_model_name = llm_model_name
        self.max_variations = max_variations
        self.min_similarity_threshold = min_similarity_threshold
        self.min_variation_similarity = min_variation_similarity
        self.temperature = temperature

    def _should_skip_variations(self, user_query: str) -> bool:
        """Determine if query is too simple for variations.
        
        Args:
            user_query: Original user query
            
        Returns:
            True if variations should be skipped, False otherwise
        """
        words = user_query.strip().split()
        # Skip for very short queries (1-2 words) - variations unlikely to help
        if len(words) <= 2:
            return True
        return False

    def generate_variations(self, user_query: str) -> List[str]:
        """Generate query variations dynamically based on quality.

        Args:
            user_query: Original user query

        Returns:
            List of query variations (always includes original as first)
        """
        # Check if we should skip variations for simple queries
        if self._should_skip_variations(user_query):
            logger.info("Query too simple for variations, using original only")
            return [user_query]
        
        # Generate candidate variations with explicit format requirements
        prompt = f"""You must output ONLY the query variations, nothing else. No explanations, no labels, no introductory text.

Original query: {user_query}

Output exactly {self.max_variations} search query variations. Each variation must:
- Be a complete search query (not just words)
- Explore a different aspect or angle of the original
- Use different wording while maintaining core intent
- Be written as plain text (no markdown, bullets, numbering, or formatting)

Output format: Write each variation on its own line. Output ONLY the queries, one per line, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()
            
            # Parse variations - LLM should output clean queries, but do minimal cleaning
            raw_variations = []
            for line in content.split("\n"):
                # Basic cleaning: strip whitespace and remove obvious non-query content
                cleaned = line.strip()
                
                # Skip empty lines
                if not cleaned:
                    continue
                
                # Remove leading numbering/bullets if present
                cleaned = re.sub(r'^\s*[\d\-•*#.]+\s+', '', cleaned)
                
                # Skip if it's clearly not a query (too short, just formatting)
                if len(cleaned) < 5:
                    continue
                
                # Skip lines that are just labels (ending with colon and short)
                if cleaned.endswith(':') and len(cleaned.split()) <= 4:
                    continue
                
                # Skip obvious non-query text
                cleaned_lower = cleaned.lower()
                if any(skip in cleaned_lower for skip in ['variation', 'variations:', 'here are', 'below are', 'the query']):
                    continue
                
                raw_variations.append(cleaned)
            
            # Filter by quality: remove too similar to original
            quality_variations = []
            for var in raw_variations:
                similarity = SequenceMatcher(None, user_query.lower(), var.lower()).ratio()
                if similarity < self.min_similarity_threshold:
                    quality_variations.append(var)
                else:
                    logger.debug(f"Rejected variation too similar to original (similarity: {similarity:.2f}): {var[:50]}")
            
            # Remove duplicates and near-duplicates among variations
            unique_variations = []
            seen = set()
            for var in quality_variations:
                var_lower = var.lower()
                is_duplicate = False
                for seen_var in seen:
                    similarity = SequenceMatcher(None, var_lower, seen_var).ratio()
                    if similarity >= self.min_variation_similarity:
                        is_duplicate = True
                        logger.debug(f"Rejected duplicate variation (similarity: {similarity:.2f}): {var[:50]}")
                        break
                
                if not is_duplicate:
                    unique_variations.append(var)
                    seen.add(var_lower)
            
            # Always include original as first
            result = [user_query]
            result.extend(unique_variations)
            
            logger.info(
                f"Generated {len(unique_variations)} quality variations "
                f"(from {len(raw_variations)} candidates, filtered from {len(quality_variations)} quality) "
                f"for: {user_query[:50]}..."
            )
            
            return result

        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            # Fallback to original query
            return [user_query]
