"""Query fusion module for RAG Fusion - generating query variations."""

import logging
from typing import List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryFusion:
    """Generate multiple query variations from a single user query."""

    def __init__(
        self,
        llm_base_url: str,
        llm_model_name: str = "gemma-2b",
        api_key: Optional[str] = None,
        num_variations: int = 5,
        temperature: float = 0.7,
    ):
        """Initialize query fusion.

        Args:
            llm_base_url: Base URL for LLM API
            llm_model_name: Model name
            api_key: API key if required
            num_variations: Number of query variations to generate
            temperature: Temperature for generation
        """
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key or "sk-no-key-required",
        )
        self.llm_model_name = llm_model_name
        self.num_variations = num_variations
        self.temperature = temperature

    def generate_variations(self, user_query: str) -> List[str]:
        """Generate multiple query variations from user query.

        Args:
            user_query: Original user query

        Returns:
            List of query variations
        """
        prompt = f"""You are a helpful assistant that generates diverse search query variations.

Given the following user query, generate {self.num_variations} different search query variations.
Each variation should:
1. Explore a different aspect or angle of the original query
2. Use different wording while maintaining the core intent
3. Be a complete, standalone search query

Original query: {user_query}

Generate {self.num_variations} variations, one per line, without numbering or bullets:"""

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
            
            # Parse variations (one per line)
            variations = [
                line.strip()
                for line in content.split("\n")
                if line.strip()
            ]
            
            # Add original query as first variation
            variations.insert(0, user_query)
            
            # Limit to requested number
            variations = variations[: self.num_variations + 1]

            logger.info(
                f"Generated {len(variations)} query variations for: {user_query[:50]}..."
            )
            return variations

        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            # Fallback to original query
            return [user_query]

