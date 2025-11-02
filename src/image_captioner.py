"""Image captioning module using internvl-internlm2."""

import logging
import base64
from typing import Optional
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """Image captioner using internvl-internlm2 via OpenAI-compatible API."""

    def __init__(
        self,
        image_model_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_image_size: int = 1024,
    ):
        """Initialize image captioner.

        Args:
            image_model_url: Base URL for image captioning API
            api_key: API key if required
            timeout: Request timeout in seconds
            max_image_size: Maximum image dimension (will resize if larger)
        """
        self.image_model_url = image_model_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_image_size = max_image_size

    def caption_image(
        self, image_bytes: bytes, image_format: str = "PNG"
    ) -> Optional[str]:
        """Generate caption for an image.

        Args:
            image_bytes: Image bytes
            image_format: Image format (PNG, JPEG, etc.)

        Returns:
            Caption string or None if captioning fails
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes))

            # Resize if too large
            if max(image.size) > self.max_image_size:
                image.thumbnail(
                    (self.max_image_size, self.max_image_size),
                    Image.Resampling.LANCZOS,
                )

            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format=image_format)
            image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Prepare request
            data_url = f"data:image/{image_format.lower()};base64,{image_b64}"

            # Call OpenAI-compatible vision API
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": "internvl-internlm2",  # Model name may vary
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

            # Make request
            chat_url = f"{self.image_model_url}/chat/completions"
            response = requests.post(
                chat_url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            caption = result["choices"][0]["message"]["content"].strip()

            logger.debug(f"Generated caption: {caption[:100]}...")
            return caption

        except Exception as e:
            logger.error(f"Error captioning image: {e}")
            return None

    def caption_images_batch(
        self, image_bytes_list: list[bytes], image_formats: list[str]
    ) -> list[Optional[str]]:
        """Generate captions for multiple images.

        Args:
            image_bytes_list: List of image bytes
            image_formats: List of image formats

        Returns:
            List of captions (None for failed captioning)
        """
        captions = []
        for image_bytes, image_format in zip(image_bytes_list, image_formats):
            caption = self.caption_image(image_bytes, image_format)
            captions.append(caption)
        return captions

