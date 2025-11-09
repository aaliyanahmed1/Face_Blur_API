# Real-time Face Blur API Service

A production-ready FastAPI service for detecting and blurring faces in images. This service provides RESTful API endpoints for privacy protection through face detection and blurring capabilities.

## Features

- 🔍 **Face Detection**: Uses OpenCV's Haar Cascades for reliable face detection
- 🎨 **Face Blurring**: Applies Gaussian blur to detected faces with configurable intensity
- 📦 **Batch Processing**: Process multiple images in a single request
- 🚀 **FastAPI**: Modern, fast web framework with automatic API documentation
- 🐳 **Docker Support**: Multi-stage Docker build for optimized production deployment
- ✅ **Comprehensive Testing**: Unit and integration tests with 80%+ code coverage
- 🔄 **CI/CD Pipeline**: Automated testing, building, and deployment with GitHub Actions
- 🔒 **Security Scanning**: Automated vulnerability scanning with Trivy and Safety

## Project Structure

```
project-root/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── face_detector.py        # Face detection logic
│   └── image_processor.py      # Image processing utilities
├── tests/
│   ├── __init__.py
│   ├── test_face_detector.py   # Unit tests for face detection
│   ├── test_image_processor.py # Unit tests for image processing
│   └── test_api.py             # Integration tests for API
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Multi-stage Docker build
├── pytest.ini                  # Pytest configuration
├── .flake8                     # Flake8 linting configuration
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Requirements

- Python 3.9+
- OpenCV (installed via requirements.txt)
- FastAPI
- uvicorn

## Installation

### Local Development

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CICD
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libopencv-dev python3-opencv
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Docker

1. **Build the Docker image:**
   ```bash
   docker build -t face-blur-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 face-blur-api
   ```

## Usage

### Running the Application

**Local development:**
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Using Docker:**
```bash
docker run -p 8000:8000 face-blur-api
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "face-blur-api"
}
```

#### 2. Blur Single Image
```http
POST /blur-image
Content-Type: multipart/form-data
```

**Request:**
- `file`: Image file (JPG, PNG)

**Response:**
- Processed image with blurred faces
- Headers:
  - `X-Faces-Detected`: Number of faces detected
  - `X-Original-Filename`: Original filename

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/blur-image" \
  -F "file=@path/to/image.jpg" \
  -o output.jpg
```

#### 3. Blur Batch Images
```http
POST /blur-batch
Content-Type: multipart/form-data
```

**Request:**
- `files`: Multiple image files (JPG, PNG) - maximum 50 files

**Response:**
- Zip file containing processed images with blurred faces
- Headers:
  - `X-Total-Images`: Number of images processed
  - `X-Total-Faces`: Total number of faces detected
  - `X-Zip-Size`: Size of the zip file in bytes

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/blur-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -o batch_result.zip
```

### Configuration

Environment variables:

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `BLUR_INTENSITY`: Gaussian blur kernel size (default: 51, must be odd)
- `CONFIDENCE_THRESHOLD`: Face detection confidence threshold (default: 0.5)
- `LOG_LEVEL`: Logging level (default: INFO)

## Testing

### Run Tests

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=html
```

**Run specific test file:**
```bash
pytest tests/test_api.py
```

**Run with verbose output:**
```bash
pytest -v
```

### Test Coverage

The project maintains a minimum of 80% code coverage. Coverage reports are generated in HTML format and can be viewed in the `htmlcov/` directory.

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline with GitHub Actions that includes:

### Stages

1. **Linting & Code Quality**
   - Runs on Python 3.9, 3.10, 3.11
   - Flake8 linting (max line length: 120)
   - Black code formatting check
   - MyPy type checking

2. **Testing**
   - Runs pytest with coverage
   - Requires minimum 80% code coverage
   - Uploads coverage reports as artifacts

3. **Build Docker Image**
   - Builds Docker image on main branch
   - Tags with commit SHA and 'latest'
   - Runs smoke tests on built image

4. **Security Scanning**
   - Safety check on dependencies
   - Trivy vulnerability scanning
   - Uploads results to GitHub Security

5. **Deployment**
   - Automatic deployment to staging on main branch
   - Manual deployment to production (workflow_dispatch)
   - Health checks after deployment

### Pipeline Status

[![CI/CD Pipeline](https://github.com/your-username/your-repo/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/ci-cd.yml)

## Development

### Code Quality

**Linting:**
```bash
flake8 src tests --max-line-length=120
```

**Formatting:**
```bash
black src tests --line-length 120
```

**Type Checking:**
```bash
mypy src --ignore-missing-imports
```

### Adding New Features

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass and coverage is maintained
5. Run linting and formatting
6. Submit a pull request

## Deployment

### Docker Deployment

1. **Build the image:**
   ```bash
   docker build -t face-blur-api:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 \
     -e BLUR_INTENSITY=51 \
     -e CONFIDENCE_THRESHOLD=0.5 \
     face-blur-api:latest
   ```

### Production Considerations

- Use a reverse proxy (nginx, Traefik) in front of the API
- Set up proper logging and monitoring
- Configure environment variables for production
- Use secrets management for sensitive data
- Set up health check endpoints for load balancers
- Consider rate limiting for API endpoints
- Use HTTPS in production

## Limitations

- Maximum file size: 10MB per image
- Maximum batch size: 50 images per request
- Supported formats: JPG, JPEG, PNG
- Face detection accuracy depends on image quality and face visibility

## Troubleshooting

### Common Issues

1. **OpenCV not found:**
   - Install system dependencies: `sudo apt-get install libopencv-dev python3-opencv`
   - Or use Docker which includes all dependencies

2. **Low face detection accuracy:**
   - Adjust `CONFIDENCE_THRESHOLD` environment variable
   - Ensure images have good lighting and face visibility
   - Consider using DNN-based models for better accuracy

3. **Docker build fails:**
   - Ensure Docker has enough memory allocated
   - Check that all files are present in the build context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenCV for face detection capabilities
- FastAPI for the web framework
- All contributors to this project

## Support

For issues and questions, please open an issue on the GitHub repository.

