#!/usr/bin/env python3
"""Test script for embedding API."""

import sys
import json
import requests
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingService
from src.config import Config


def test_api_direct(embedding_url):
    """Test API directly with correct format."""
    print(f"\nTesting embedding API: {embedding_url}")
    
    payload = {
        "texts": ["This is a test sentence.", "This is another sentence."]
    }
    
    try:
        response = requests.post(
            embedding_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and "text_vectors" in result["data"]:
                vectors = result["data"]["text_vectors"]
                print(f"✓ Success! Got {len(vectors)} embeddings of dimension {len(vectors[0])}")
                return True
            else:
                print(f"❌ Unexpected response format: {list(result.keys())}")
        else:
            print(f"❌ Status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False


def test_embedding_service(embedding_url):
    """Test using EmbeddingService class."""
    print(f"\nTesting EmbeddingService class...")
    
    try:
        service = EmbeddingService(embedding_url=embedding_url, timeout=30)
        
        # Test single text
        result = service.embed_text("Test sentence")
        if result and len(result) == 1024:
            print(f"✓ Single text embedding: dimension {len(result)}")
        else:
            print(f"❌ Single text embedding failed")
            return False
        
        # Test batch
        results = service.embed_texts(["First text", "Second text", "Third text"])
        if results and all(r is not None for r in results):
            print(f"✓ Batch embeddings: {len(results)}/{len(results)} successful")
            return True
        else:
            print(f"❌ Batch embeddings failed: {sum(1 for r in results if r)}/{len(results)}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Embedding API Test")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    
    config = Config(config_path=args.config)
    embedding_config = config.get_embedding_config()
    embedding_url = embedding_config.get("url", "")
    
    if not embedding_url:
        print("❌ Error: Embedding URL not found in configuration.")
        return 1
    
    print("=" * 60)
    print("Embedding API Test")
    print("=" * 60)
    
    success1 = test_api_direct(embedding_url)
    success2 = test_embedding_service(embedding_url)
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
