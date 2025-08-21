# Multi-stage build for optimal image size
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Add labels for metadata
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.source="https://github.com/grafana-ps/cardinality-analyser" \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.title="Cardinality Analyser" \
      org.opencontainers.image.description="A tool to investigate metric spike issues in Grafana Cloud Mimir" \
      org.opencontainers.image.vendor="Grafana Professional Services" \
      org.opencontainers.image.licenses="Apache-2.0"

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash analyzer

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=analyzer:analyzer /root/.local /home/analyzer/.local

# Copy application files
COPY --chown=analyzer:analyzer cardinality-analyzer.py .
COPY --chown=analyzer:analyzer cardinality_analyzer_ai_analysis.py .

# Make sure scripts are executable
RUN chmod +x cardinality-analyzer.py

# Create output directory with proper permissions
RUN mkdir -p /output && chown analyzer:analyzer /output

# Update PATH to include user's local bin
ENV PATH=/home/analyzer/.local/bin:$PATH \
    PYTHONPATH=/home/analyzer/.local/lib/python3.11/site-packages:$PYTHONPATH

# Switch to non-root user
USER analyzer

# Set working directory to output for file generation
WORKDIR /output

# Set the entrypoint
ENTRYPOINT ["python", "/app/cardinality-analyzer.py"]

# Default command (can be overridden)
CMD ["--help"]