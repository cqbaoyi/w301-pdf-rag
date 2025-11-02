#!/usr/bin/env python3
"""Test script for reranking API."""

import sys
import json
import requests
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


def test_api_direct(rerank_url):
    """Test API directly with correct format (array of strings)."""
    print(f"\nTesting reranking API: {rerank_url}")
    
    query = "machine learning algorithms"
    documents = [
        f"This document discusses {query} in great detail. It covers all aspects.",
        f"Information about {query} can be found here. This is relevant content.",
        f"The topic of {query} is briefly mentioned in this document.",
        f"This document talks about something else entirely.",
        f"Completely unrelated content with no connection to {query}.",
    ]
    
    # Correct format: documents as array of strings
    payload = {
        "model": "qwen3-reranker-0.6b",
        "query": query,
        "documents": documents,  # Array of strings, not objects
        "top_n": 5,
    }
    
    try:
        response = requests.post(
            rerank_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            if "ranked_documents" in result:
                ranked = result["ranked_documents"]
                print(f"✓ Success! Got {len(ranked)} reranked documents")
                print(f"Top 3 results:")
                for i, doc in enumerate(ranked[:3]):
                    print(f"  {i+1}. Score: {doc.get('score', 0):.4f} - {doc.get('document', '')[:60]}...")
                return True
            elif "scores" in result:
                scores = result["scores"]
                print(f"✓ Success! Got {len(scores)} scores")
                print(f"Scores: {[f'{s:.4f}' for s in scores[:3]]}")
                return True
            else:
                print(f"❌ Unexpected response format: {list(result.keys())}")
        else:
            print(f"❌ Status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False


def test_response_format(rerank_url):
    """Test that response format matches expectations."""
    print(f"\nTesting response format...")
    
    query = "artificial intelligence"
    documents = [
        f"Deep dive into {query} and neural networks.",
        f"Brief mention of {query} concepts.",
        f"Unrelated content about other topics.",
    ]
    
    payload = {
        "model": "qwen3-reranker-0.6b",
        "query": query,
        "documents": documents,
        "top_n": 3,
    }
    
    try:
        response = requests.post(
            rerank_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            
            # Check format
            if "ranked_documents" in result:
                ranked = result["ranked_documents"]
                if isinstance(ranked, list) and len(ranked) > 0:
                    sample = ranked[0]
                    print(f"Sample document keys: {list(sample.keys())}")
                    if "document" in sample and "score" in sample:
                        print(f"✓ Response format correct: ranked_documents with document/text and score")
                        return True
            
            if "scores" in result:
                print(f"✓ Response format: scores array")
                return True
            
            print(f"❌ Unexpected response format")
        else:
            print(f"❌ Status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Reranking API Test")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    
    config = Config(config_path=args.config)
    reranking_config = config.get_reranking_config()
    rerank_url = reranking_config.get("url", "")
    
    if not rerank_url:
        print("❌ Error: Reranking URL not found in configuration.")
        return 1
    
    print("=" * 60)
    print("Reranking API Test")
    print("=" * 60)
    
    success1 = test_api_direct(rerank_url)
    success2 = test_response_format(rerank_url)
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
