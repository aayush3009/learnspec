FROM python:3.9-buster

# Set working directory
WORKDIR /app

# Add non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with security flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in development mode
RUN pip install -e . && \
    # Fix permissions
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add HEALTHCHECK if your app has a health endpoint
# HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:port/health || exit 1