# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# The CUDA build of torchcodec from the PyTorch index needs NPP libraries we do not ship;
# audio decoding happens on the CPU anyway, so use the CPU build from PyPI
RUN pip install --no-cache-dir --no-deps --force-reinstall --index-url https://pypi.org/simple "torchcodec==0.7.*"

# Copy application code
COPY whisper-web.py .

# Create directories for temporary files
RUN mkdir -p /tmp/uploads

# Set environment variables
ENV FLASK_APP=whisper-web.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "whisper-web.py"]
