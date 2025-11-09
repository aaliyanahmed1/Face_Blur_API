"""Integration tests for API endpoints."""

import pytest
import cv2
import numpy as np
import zipfile
from io import BytesIO
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    # Create a simple image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image.fill(255)
    
    # Encode as JPEG
    _, encoded_image = cv2.imencode(".jpg", image)
    return ("test_image.jpg", encoded_image.tobytes(), "image/jpeg")


@pytest.fixture
def sample_png_file():
    """Create a sample PNG image file for testing."""
    # Create a simple image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image.fill(255)
    
    # Encode as PNG
    _, encoded_image = cv2.imencode(".png", image)
    return ("test_image.png", encoded_image.tobytes(), "image/png")


@pytest.fixture
def invalid_image_file():
    """Create an invalid image file for testing."""
    return ("test.txt", b"not an image", "text/plain")


class TestAPI:
    """Test cases for API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["service"] == "face-blur-api"

    def test_blur_image_valid_jpeg(self, sample_image_file):
        """Test blur-image endpoint with valid JPEG image."""
        filename, content, content_type = sample_image_file
        response = client.post(
            "/blur-image",
            files={"file": (filename, content, content_type)}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")
        assert "X-Faces-Detected" in response.headers
        assert int(response.headers["X-Faces-Detected"]) >= 0

    def test_blur_image_valid_png(self, sample_png_file):
        """Test blur-image endpoint with valid PNG image."""
        filename, content, content_type = sample_png_file
        response = client.post(
            "/blur-image",
            files={"file": (filename, content, content_type)}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "X-Faces-Detected" in response.headers

    def test_blur_image_invalid_format(self, invalid_image_file):
        """Test blur-image endpoint with invalid file format."""
        filename, content, content_type = invalid_image_file
        response = client.post(
            "/blur-image",
            files={"file": (filename, content, content_type)}
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_blur_image_no_file(self):
        """Test blur-image endpoint with no file."""
        response = client.post("/blur-image")
        assert response.status_code == 422  # Validation error

    def test_blur_image_large_file(self, sample_image_file):
        """Test blur-image endpoint with large file (should fail if exceeds limit)."""
        filename, _, content_type = sample_image_file
        # Create a large file (exceeds 10MB)
        large_content = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/blur-image",
            files={"file": (filename, large_content, content_type)}
        )
        # Should either fail with 400 or process (depending on actual size check)
        assert response.status_code in [400, 500]

    def test_blur_batch_valid_images(self, sample_image_file, sample_png_file):
        """Test blur-batch endpoint with valid images."""
        filename1, content1, content_type1 = sample_image_file
        filename2, content2, content_type2 = sample_png_file
        
        response = client.post(
            "/blur-batch",
            files=[
                ("files", (filename1, content1, content_type1)),
                ("files", (filename2, content2, content_type2))
            ]
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "X-Total-Images" in response.headers
        assert "X-Total-Faces" in response.headers
        assert "X-Zip-Size" in response.headers
        assert int(response.headers["X-Total-Images"]) == 2
        assert int(response.headers["X-Total-Faces"]) >= 0
        # Verify it's a valid zip file
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        assert len(zip_file.namelist()) == 2

    def test_blur_batch_no_files(self):
        """Test blur-batch endpoint with no files."""
        response = client.post("/blur-batch", files=[])
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_blur_batch_mixed_valid_invalid(self, sample_image_file, invalid_image_file):
        """Test blur-batch endpoint with mixed valid and invalid images."""
        filename1, content1, content_type1 = sample_image_file
        filename2, content2, content_type2 = invalid_image_file
        
        response = client.post(
            "/blur-batch",
            files=[
                ("files", (filename1, content1, content_type1)),
                ("files", (filename2, content2, content_type2))
            ]
        )
        # Should process valid images and skip invalid ones
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert int(response.headers["X-Total-Images"]) >= 1

    def test_blur_batch_too_many_files(self, sample_image_file):
        """Test blur-batch endpoint with too many files."""
        filename, content, content_type = sample_image_file
        files = [("files", (f"test_{i}.jpg", content, content_type)) for i in range(51)]
        
        response = client.post("/blur-batch", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_blur_image_with_face_pattern(self):
        """Test blur-image endpoint with an image containing face-like patterns."""
        # Create an image with face-like patterns
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image.fill(220)
        
        # Draw oval face
        cv2.ellipse(image, (150, 150), (80, 100), 0, 0, 360, (200, 180, 160), -1)
        
        # Draw eyes
        cv2.circle(image, (130, 130), 8, (0, 0, 0), -1)
        cv2.circle(image, (170, 130), 8, (0, 0, 0), -1)
        
        # Draw nose
        cv2.ellipse(image, (150, 150), (5, 15), 0, 0, 360, (150, 120, 100), -1)
        
        # Draw mouth
        cv2.ellipse(image, (150, 180), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        
        # Encode as JPEG
        _, encoded_image = cv2.imencode(".jpg", image)
        
        response = client.post(
            "/blur-image",
            files={"file": ("face_image.jpg", encoded_image.tobytes(), "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")

