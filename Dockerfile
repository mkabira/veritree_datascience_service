# Stage 1: Base image with system dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update --allow-insecure-repositories && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    pip3 install --upgrade pip

# Copy only requirements files first to leverage caching
COPY requirements.txt /app/

# Install ML dependencies
RUN pip3 install -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim AS final

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3", "src/main.py", "pipeline", "pipeline_api"]
