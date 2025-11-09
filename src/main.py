"""FastAPI application for face blur API service."""

import logging
import os
from io import BytesIO
from typing import List
from zipfile import ZipFile

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from src.face_detector import FaceDetector
from src.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Blur API",
    description="A FastAPI service for detecting and blurring faces in images",
    version="1.0.0"
)

# Initialize face detector and image processor
face_detector = FaceDetector(confidence_threshold=0.5)
blur_intensity = int(os.getenv("BLUR_INTENSITY", "51"))
if blur_intensity % 2 == 0:
    blur_intensity += 1  # Ensure odd number
image_processor = ImageProcessor(face_detector, blur_intensity=blur_intensity)

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    service: str


def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file.

    Args:
        file: Uploaded file.

    Raises:
        HTTPException: If file is invalid.
    """
    # Check file extension
    if not any(file.filename.endswith(ext) for ext in SUPPORTED_FORMATS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check content type
    if file.content_type and not any(
        file.content_type.startswith(f"image/{fmt}")
        for fmt in ["jpeg", "jpg", "png"]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Expected image/jpeg or image/png"
        )


@app.get("/", response_model=dict)
async def root() -> dict:
    """
    Root endpoint.

    Returns:
        Welcome message and API information.
    """
    return {
        "message": "Welcome to Face Blur API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "blur_image": "/blur-image",
            "blur_batch": "/blur-batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status of the service.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="face-blur-api"
    )


@app.post("/blur-image", response_class=Response)
async def blur_image(file: UploadFile = File(...)) -> Response:
    """
    Blur faces in a single uploaded image.

    Args:
        file: Uploaded image file.

    Returns:
        Processed image with blurred faces.

    Raises:
        HTTPException: If file is invalid or processing fails.
    """
    try:
        # Validate file
        validate_image_file(file)

        # Read file content
        contents = await file.read()

        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024}MB"
            )

        logger.info(f"Processing image: {file.filename}, size: {len(contents)} bytes")

        # Process image
        processed_image, faces_detected = image_processor.process_image(contents)

        logger.info(f"Processed image: {file.filename}, faces detected: {faces_detected}")

        # Determine content type
        content_type = "image/jpeg"
        if file.filename and file.filename.lower().endswith(".png"):
            content_type = "image/png"

        # Return processed image
        return Response(
            content=processed_image,
            media_type=content_type,
            headers={
                "X-Faces-Detected": str(faces_detected),
                "X-Original-Filename": file.filename or "image"
            }
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error processing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error processing image: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process image: {str(e)}"
        )


@app.post("/blur-batch")
async def blur_batch(files: List[UploadFile] = File(...)) -> Response:
    """
    Blur faces in multiple uploaded images.

    Args:
        files: List of uploaded image files.

    Returns:
        Zip file containing processed images with blurred faces.

    Raises:
        HTTPException: If files are invalid or processing fails.
    """
    try:
        # Validate number of files
        if len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )

        if len(files) > 50:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 files allowed per batch"
            )

        logger.info(f"Processing batch of {len(files)} images")

        # Validate and process each file
        processed_images = []
        total_faces = 0
        processed_count = 0

        for file in files:
            try:
                # Validate file
                validate_image_file(file)

                # Read file content
                contents = await file.read()

                # Check file size
                if len(contents) > MAX_FILE_SIZE:
                    logger.warning(f"File {file.filename} exceeds size limit, skipping")
                    continue

                # Process image
                processed_image, faces_detected = image_processor.process_image(contents)
                total_faces += faces_detected
                processed_count += 1

                # Store processed image
                original_filename = file.filename or "image"
                name, ext = os.path.splitext(original_filename)
                processed_filename = f"{name}_blurred{ext}"
                processed_images.append((processed_filename, processed_image))

                logger.info(f"Processed image: {original_filename}, faces: {faces_detected}")

            except HTTPException:
                logger.warning(f"Failed to process file {file.filename}, skipping")
                continue
            except Exception as e:
                logger.warning(f"Error processing file {file.filename}: {e}, skipping")
                continue

        if processed_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid images were processed"
            )

        # Create zip file
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zip_file:
            for filename, image_data in processed_images:
                zip_file.writestr(filename, image_data)

        zip_data = zip_buffer.getvalue()
        zip_size = len(zip_data)

        logger.info(f"Batch processing complete: {processed_count} images, {total_faces} faces")

        # Return zip file as response
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=blurred_images.zip",
                "X-Total-Images": str(processed_count),
                "X-Total-Faces": str(total_faces),
                "X-Zip-Size": str(zip_size)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process batch: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)

