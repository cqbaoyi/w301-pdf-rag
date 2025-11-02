#!/usr/bin/env python3
"""Test script for image captioning API."""

import sys
import json
import requests
import argparse
import base64
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_captioner import ImageCaptioner
from src.config import Config


def create_test_image(width=200, height=200, color=(255, 0, 0)):
    """Create a simple test image."""
    img = Image.new("RGB", (width, height), color=color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()


def image_to_base64_data_url(image_bytes, image_format="PNG"):
    """Convert image bytes to base64 data URL."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{image_b64}"


def test_api_direct(image_model_url):
    """Test API directly with correct format."""
    print(f"\nTesting image captioning API: {image_model_url}")
    
    image_bytes = create_test_image(200, 200, (255, 0, 0))
    data_url = image_to_base64_data_url(image_bytes, "PNG")
    
    chat_url = f"{image_model_url}/chat/completions"
    payload = {
        "model": "internvl-internlm2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail. Be concise and focus on key elements, text, and layout.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "max_tokens": 200,
    }
    
    try:
        response = requests.post(
            chat_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                caption = result["choices"][0]["message"]["content"]
                print(f"✓ Success! Caption: {caption[:100]}...")
                return True
            else:
                print(f"❌ Unexpected response format")
        else:
            print(f"❌ Status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False


def test_image_captioner_class(image_model_url):
    """Test using ImageCaptioner class."""
    print(f"\nTesting ImageCaptioner class...")
    
    try:
        captioner = ImageCaptioner(
            image_model_url=image_model_url,
            timeout=30,
            max_image_size=1024,
        )
        
        image_bytes = create_test_image(200, 200, (0, 255, 0))
        result = captioner.caption_image(image_bytes, "PNG")
        
        if result:
            print(f"✓ Success! Caption: {result[:100]}...")
            return True
        else:
            print(f"❌ Failed to get caption")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Image Captioning API Test")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    args = parser.parse_args()
    
    config = Config(config_path=args.config)
    image_config = config.get_image_captioning_config()
    image_model_url = image_config.get("url", "")
    
    if not image_model_url:
        print("❌ Error: Image captioning URL not found in configuration.")
        return 1
    
    print("=" * 60)
    print("Image Captioning API Test")
    print("=" * 60)
    
    success1 = test_api_direct(image_model_url)
    success2 = test_image_captioner_class(image_model_url)
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
