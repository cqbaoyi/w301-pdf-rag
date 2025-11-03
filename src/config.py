"""Configuration management for PDF RAG system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""

    def __init__(self, config_path: Optional[Path] = None, env_path: Optional[Path] = None):
        """Initialize configuration."""
        # Load environment variables first
        if env_path:
            load_dotenv(env_path)
        else:
            # Try to load .env files in order of priority
            project_root = Path(__file__).parent.parent
            
            # First, try elastic-start-local/.env (for ElasticSearch passwords)
            elastic_env = project_root / "elastic-start-local" / ".env"
            if elastic_env.exists():
                load_dotenv(elastic_env, override=False)  # Don't override if already set
            
            # Then, try .env from project root
            root_env = project_root / ".env"
            if root_env.exists():
                load_dotenv(root_env, override=False)  # Don't override if already set

        # Load YAML config
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to config/config.yaml
            project_root = Path(__file__).parent.parent
            self.config_path = project_root / "config" / "config.yaml"

        self.config = self._load_config()

        # Override with environment variables
        self._apply_env_overrides()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}

    def _apply_env_overrides(self):
        """Override config values with environment variables."""
        # ElasticSearch
        if os.getenv("ES_LOCAL_HOST"):
            self._set_nested("elasticsearch.host", os.getenv("ES_LOCAL_HOST"))
        if os.getenv("ES_LOCAL_PORT"):
            self._set_nested("elasticsearch.port", int(os.getenv("ES_LOCAL_PORT")))
        if os.getenv("ES_LOCAL_USERNAME"):
            self._set_nested("elasticsearch.username", os.getenv("ES_LOCAL_USERNAME"))
        if os.getenv("ES_LOCAL_PASSWORD"):
            self._set_nested("elasticsearch.password", os.getenv("ES_LOCAL_PASSWORD"))
        if os.getenv("ES_INDEX_NAME"):
            self._set_nested("elasticsearch.index_name", os.getenv("ES_INDEX_NAME"))

        # Service URLs - prioritize environment variables
        # Embedding URL
        if os.getenv("EMBEDDING_URL"):
            self.config["embedding"] = self.config.get("embedding", {})
            self.config["embedding"]["url"] = os.getenv("EMBEDDING_URL")
        else:
            # If env var not set and YAML has null, keep it as None
            self.config["embedding"] = self.config.get("embedding", {})
            if self.config["embedding"].get("url") is None:
                logger.warning("EMBEDDING_URL not set in environment or config")

        # Reranking URL
        if os.getenv("RERANK_URL"):
            self.config["reranking"] = self.config.get("reranking", {})
            self.config["reranking"]["url"] = os.getenv("RERANK_URL")
        else:
            self.config["reranking"] = self.config.get("reranking", {})
            if self.config["reranking"].get("url") is None:
                logger.warning("RERANK_URL not set in environment or config")

        # Image Captioning URL
        if os.getenv("IMAGE_MODEL_URL"):
            self.config["image_captioning"] = self.config.get("image_captioning", {})
            self.config["image_captioning"]["url"] = os.getenv("IMAGE_MODEL_URL")
        else:
            self.config["image_captioning"] = self.config.get("image_captioning", {})
            if self.config["image_captioning"].get("url") is None:
                logger.warning("IMAGE_MODEL_URL not set in environment or config")

        # LLM (Generation)
        if os.getenv("LLM_BASE_URL"):
            self.config["generation"] = self.config.get("generation", {})
            self.config["generation"]["base_url"] = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_API_KEY"):
            self.config["generation"] = self.config.get("generation", {})
            self.config["generation"]["api_key"] = os.getenv("LLM_API_KEY")
        # Always check OPENAI_API_KEY as fallback if api_key is None or placeholder
        self.config["generation"] = self.config.get("generation", {})
        if (self.config["generation"].get("api_key") is None or 
            self.config["generation"].get("api_key") == "sk-no-key-required") and os.getenv("OPENAI_API_KEY"):
            self.config["generation"]["api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("LLM_MODEL_NAME"):
            self.config["generation"] = self.config.get("generation", {})
            self.config["generation"]["model_name"] = os.getenv("LLM_MODEL_NAME")

        # Query Fusion LLM
        if os.getenv("QUERY_FUSION_BASE_URL"):
            self.config["query_fusion"] = self.config.get("query_fusion", {})
            self.config["query_fusion"]["base_url"] = os.getenv("QUERY_FUSION_BASE_URL")
        if os.getenv("QUERY_FUSION_MODEL_NAME"):
            self.config["query_fusion"] = self.config.get("query_fusion", {})
            self.config["query_fusion"]["model_name"] = os.getenv("QUERY_FUSION_MODEL_NAME")
        if os.getenv("QUERY_FUSION_API_KEY"):
            self.config["query_fusion"] = self.config.get("query_fusion", {})
            self.config["query_fusion"]["api_key"] = os.getenv("QUERY_FUSION_API_KEY")
        # Always check OPENAI_API_KEY as fallback if api_key is None or placeholder
        self.config["query_fusion"] = self.config.get("query_fusion", {})
        if (self.config["query_fusion"].get("api_key") is None or 
            self.config["query_fusion"].get("api_key") == "sk-no-key-required") and os.getenv("OPENAI_API_KEY"):
            self.config["query_fusion"]["api_key"] = os.getenv("OPENAI_API_KEY")

    def _set_nested(self, key_path: str, value: Any):
        """Set a nested configuration value."""
        keys = key_path.split(".")
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_elasticsearch_config(self) -> Dict[str, Any]:
        return self.config.get("elasticsearch", {})

    def get_chunking_config(self) -> Dict[str, Any]:
        return self.config.get("chunking", {})

    def get_embedding_config(self) -> Dict[str, Any]:
        return self.config.get("embedding", {})

    def get_image_captioning_config(self) -> Dict[str, Any]:
        return self.config.get("image_captioning", {})

    def get_search_config(self) -> Dict[str, Any]:
        return self.config.get("search", {})

    def get_query_fusion_config(self) -> Dict[str, Any]:
        return self.config.get("query_fusion", {})

    def get_result_fusion_config(self) -> Dict[str, Any]:
        return self.config.get("result_fusion", {})

    def get_reranking_config(self) -> Dict[str, Any]:
        return self.config.get("reranking", {})

    def get_generation_config(self) -> Dict[str, Any]:
        return self.config.get("generation", {})

    def get_pdf_processing_config(self) -> Dict[str, Any]:
        return self.config.get("pdf_processing", {})

