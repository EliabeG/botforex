# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        pkg-config \
        gcc \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# Copy requirements first for better caching
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models

# Command to run the application
CMD ["python", "main.py"]