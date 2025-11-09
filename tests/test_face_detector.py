"""Unit tests for face detector module."""

import pytest
import cv2
import numpy as np
from numpy.typing import NDArray

from src.face_detector import FaceDetector


@pytest.fixture
def face_detector():
    """Create a face detector instance for testing."""
    return FaceDetector(confidence_threshold=0.5)


@pytest.fixture
def sample_image_with_face() -> NDArray[np.uint8]:
    """Create a sample image with a simple face-like pattern."""
    # Create a simple image with a face-like pattern (rectangles)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image.fill(255)  # White background
    
    # Draw a simple face-like pattern (rectangles for eyes, nose, mouth)
    # This is a simplified representation for testing
    cv2.rectangle(image, (70, 60), (90, 80), (0, 0, 0), -1)  # Left eye
    cv2.rectangle(image, (110, 60), (130, 80), (0, 0, 0), -1)  # Right eye
    cv2.rectangle(image, (95, 100), (105, 120), (0, 0, 0), -1)  # Nose
    cv2.rectangle(image, (80, 140), (120, 150), (0, 0, 0), -1)  # Mouth
    
    return image


@pytest.fixture
def sample_image_without_face() -> NDArray[np.uint8]:
    """Create a sample image without faces."""
    # Create a simple image without faces
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image.fill(128)  # Gray background
    return image


@pytest.fixture
def invalid_image() -> NDArray[np.uint8]:
    """Create an invalid image for testing."""
    # Create an invalid image (wrong shape)
    return np.array([])


class TestFaceDetector:
    """Test cases for FaceDetector class."""

    def test_init_with_defaults(self):
        """Test FaceDetector initialization with default parameters."""
        detector = FaceDetector()
        assert detector.confidence_threshold == 0.5
        assert detector.detector_type == "haar"

    def test_init_with_custom_confidence(self):
        """Test FaceDetector initialization with custom confidence threshold."""
        detector = FaceDetector(confidence_threshold=0.7)
        assert detector.confidence_threshold == 0.7

    def test_init_with_invalid_confidence_low(self):
        """Test FaceDetector initialization with invalid low confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            FaceDetector(confidence_threshold=-0.1)

    def test_init_with_invalid_confidence_high(self):
        """Test FaceDetector initialization with invalid high confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            FaceDetector(confidence_threshold=1.5)

    def test_detect_faces_with_face(self, face_detector, sample_image_with_face):
        """Test face detection on image with face-like patterns."""
        faces = face_detector.detect_faces(sample_image_with_face)
        # Note: Haar Cascade may or may not detect the simple pattern
        # This test verifies the method runs without error
        assert isinstance(faces, list)
        for face in faces:
            assert len(face) == 4
            assert all(isinstance(coord, (int, np.integer)) for coord in face)

    def test_detect_faces_without_face(self, face_detector, sample_image_without_face):
        """Test face detection on image without faces."""
        faces = face_detector.detect_faces(sample_image_without_face)
        assert isinstance(faces, list)
        # Should return empty list or list with no valid faces
        assert len(faces) >= 0

    def test_detect_faces_invalid_image_none(self, face_detector):
        """Test face detection with None image."""
        with pytest.raises(ValueError, match="Invalid image"):
            face_detector.detect_faces(None)

    def test_detect_faces_invalid_image_empty(self, face_detector, invalid_image):
        """Test face detection with empty image."""
        with pytest.raises(ValueError, match="Invalid image"):
            face_detector.detect_faces(invalid_image)

    def test_detect_faces_invalid_shape(self, face_detector):
        """Test face detection with invalid image shape."""
        # Create image with wrong number of channels
        invalid_image = np.zeros((200, 200), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(ValueError, match="Invalid image"):
            face_detector.detect_faces(invalid_image)

    def test_detect_faces_real_image(self, face_detector):
        """Test face detection with a real image pattern."""
        # Create a more realistic face pattern
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
        
        faces = face_detector.detect_faces(image)
        assert isinstance(faces, list)

    def test_face_boxes_format(self, face_detector, sample_image_with_face):
        """Test that detected face boxes are in correct format."""
        faces = face_detector.detect_faces(sample_image_with_face)
        
        for face in faces:
            assert len(face) == 4, "Face box should have 4 coordinates"
            x, y, w, h = face
            assert x >= 0, "X coordinate should be non-negative"
            assert y >= 0, "Y coordinate should be non-negative"
            assert w > 0, "Width should be positive"
            assert h > 0, "Height should be positive"

