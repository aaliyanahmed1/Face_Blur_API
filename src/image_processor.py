"""Image processing utilities for face blurring."""

import logging
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processor for face detection and blurring."""

    def __init__(self, face_detector: FaceDetector, blur_intensity: int = 51):
        """
        Initialize the image processor.

        Args:
            face_detector: FaceDetector instance for face detection.
            blur_intensity: Gaussian blur kernel size (must be odd). Default is 51.

        Raises:
            ValueError: If blur_intensity is not positive or not odd.
        """
        if blur_intensity <= 0:
            raise ValueError("blur_intensity must be positive")
        if blur_intensity % 2 == 0:
            raise ValueError("blur_intensity must be odd")

        self.face_detector = face_detector
        self.blur_intensity = blur_intensity
        logger.info(f"ImageProcessor initialized with blur_intensity: {blur_intensity}")

    def load_image(self, image_data: bytes) -> NDArray[np.uint8]:
        """
        Load image from bytes.

        Args:
            image_data: Image data as bytes.

        Returns:
            Image as numpy array in BGR format.

        Raises:
            ValueError: If image cannot be decoded.
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            logger.debug(f"Image loaded successfully: shape={image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Invalid image data: {e}") from e

    def encode_image(self, image: NDArray[np.uint8], format: str = "JPEG") -> bytes:
        """
        Encode image to bytes.

        Args:
            image: Image as numpy array in BGR format.
            format: Output format ("JPEG" or "PNG"). Default is "JPEG".

        Returns:
            Encoded image as bytes.

        Raises:
            ValueError: If format is not supported or encoding fails.
        """
        if format.upper() not in ["JPEG", "JPG", "PNG"]:
            raise ValueError(f"Unsupported format: {format}")

        try:
            # Convert BGR to RGB for PIL
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Encode to bytes
            buffer = BytesIO()
            pil_image.save(buffer, format=format.upper())
            encoded_image = buffer.getvalue()

            logger.debug(f"Image encoded successfully: format={format}, size={len(encoded_image)} bytes")
            return encoded_image
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise ValueError(f"Failed to encode image: {e}") from e

    def blur_face_region(
        self, image: NDArray[np.uint8], x: int, y: int, width: int, height: int
    ) -> NDArray[np.uint8]:
        """
        Blur a specific region of an image.

        Args:
            image: Input image as numpy array.
            x: X coordinate of the top-left corner of the region.
            y: Y coordinate of the top-left corner of the region.
            width: Width of the region.
            height: Height of the region.

        Returns:
            Image with blurred region.
        """
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)

        if width <= 0 or height <= 0:
            logger.warning("Invalid region dimensions, skipping blur")
            return image

        # Extract face region
        face_region = image[y:y+height, x:x+width]

        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(
            face_region,
            (self.blur_intensity, self.blur_intensity),
            0
        )

        # Replace original region with blurred region
        result = image.copy()
        result[y:y+height, x:x+width] = blurred_region

        return result

    def process_image(self, image_data: bytes) -> Tuple[bytes, int]:
        """
        Process image: detect faces and blur them.

        Args:
            image_data: Image data as bytes.

        Returns:
            Tuple of (processed_image_bytes, number_of_faces_detected).

        Raises:
            ValueError: If image cannot be processed.
        """
        try:
            # Load image
            image = self.load_image(image_data)

            # Detect faces
            faces = self.face_detector.detect_faces(image)
            logger.info(f"Detected {len(faces)} faces in image")

            # Blur each detected face
            processed_image = image.copy()
            for (x, y, width, height) in faces:
                processed_image = self.blur_face_region(
                    processed_image, x, y, width, height
                )

            # Encode processed image
            # Determine format from original image data
            format = "JPEG"
            if image_data[:8].startswith(b"\x89PNG"):
                format = "PNG"

            encoded_image = self.encode_image(processed_image, format=format)

            return encoded_image, len(faces)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Failed to process image: {e}") from e

    def get_image_info(self, image_data: bytes) -> dict:
        """
        Get information about an image.

        Args:
            image_data: Image data as bytes.

        Returns:
            Dictionary with image information (width, height, format).
        """
        try:
            image = self.load_image(image_data)
            h, w = image.shape[:2]
            format = "JPEG"
            if image_data[:8].startswith(b"\x89PNG"):
                format = "PNG"

            return {
                "width": w,
                "height": h,
                "format": format,
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            raise ValueError(f"Failed to get image info: {e}") from e

