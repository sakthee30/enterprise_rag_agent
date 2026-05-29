# Dockerfile

# Base image — Python 3.11 slim (smaller size, faster builds)
FROM python:3.11-slim

# Why set PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED?
# PYTHONDONTWRITEBYTECODE: don't create .pyc files (keeps container clean)
# PYTHONUNBUFFERED: print logs immediately (important for debugging in cloud)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — before copying code
# Why? Docker caches layers. If requirements haven't changed,
# Docker skips the pip install step on rebuild. Saves minutes.
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data /app/chroma_db

# Expose port 8000 — tells Docker this container listens on 8000
EXPOSE 8000

# Command to run when container starts
# Uses uvicorn directly (not python -m api.main) for production
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}