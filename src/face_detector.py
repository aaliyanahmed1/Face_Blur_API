"""Face detection module using OpenCV."""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using OpenCV's DNN-based face detection model."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = 0.5):
        """
        Initialize the face detector.

        Args:
            model_path: Path to the face detection model files. If None, uses Haar Cascades.
            confidence_threshold: Minimum confidence threshold for face detection (0.0-1.0).

        Raises:
            FileNotFoundError: If model files are not found.
            ValueError: If confidence_threshold is not in valid range.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.confidence_threshold = confidence_threshold
        self.model_path = model_path

        if model_path is None:
            # Use Haar Cascades as fallback
            self._init_haar_cascade()
            self.detector_type = "haar"
        else:
            # Use DNN model
            self._init_dnn_model(model_path)
            self.detector_type = "dnn"

        logger.info(f"Face detector initialized with type: {self.detector_type}")

    def _init_haar_cascade(self) -> None:
        """Initialize Haar Cascade classifier for face detection."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                raise FileNotFoundError("Haar Cascade file not found")

            logger.info("Haar Cascade classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Haar Cascade: {e}")
            raise

    def _init_dnn_model(self, model_path: str) -> None:
        """
        Initialize DNN-based face detection model.

        Args:
            model_path: Path to directory containing model files.

        Raises:
            FileNotFoundError: If model files are not found.
        """
        try:
            model_dir = Path(model_path)
            prototxt_path = model_dir / "deploy.prototxt"
            caffemodel_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

            if not prototxt_path.exists() or not caffemodel_path.exists():
                logger.warning(
                    "DNN model files not found, falling back to Haar Cascades"
                )
                self._init_haar_cascade()
                self.detector_type = "haar"
                return

            self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
            logger.info("DNN model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load DNN model: {e}, falling back to Haar Cascades")
            self._init_haar_cascade()
            self.detector_type = "haar"

    def detect_faces(self, image: NDArray[np.uint8]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of tuples (x, y, width, height) representing face bounding boxes.

        Raises:
            ValueError: If image is empty or invalid.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Invalid image: must be a 3-channel BGR image")

        try:
            if self.detector_type == "dnn":
                return self._detect_faces_dnn(image)
            else:
                return self._detect_faces_haar(image)
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            raise

    def _detect_faces_haar(self, image: NDArray[np.uint8]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade classifier.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of tuples (x, y, width, height) representing face bounding boxes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Convert numpy array to list of tuples
        face_boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        logger.debug(f"Detected {len(face_boxes)} faces using Haar Cascade")
        return face_boxes

    def _detect_faces_dnn(self, image: NDArray[np.uint8]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN model.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of tuples (x, y, width, height) representing face bounding boxes.
        """
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        face_boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                width = x1 - x
                height = y1 - y
                face_boxes.append((x, y, width, height))

        logger.debug(f"Detected {len(face_boxes)} faces using DNN model")
        return face_boxes

