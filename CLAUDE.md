# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cardinality Analyzer - A tool to investigate metric spike issues in Grafana Cloud Mimir by analyzing cardinality changes across time windows. This is a security analysis tool for understanding metric cardinality patterns and diagnosing ingestion volume increases.

## Key Commands

### Development Setup
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

### Running the Analyzer
```bash
# Basic usage - analyze last hour
./cardinality-analyzer.py -w 1h

# Analyze specific metric
./cardinality-analyzer.py -w 1h -m my_application_requests_total

# Compare time windows (before/after)
./cardinality-analyzer.py -w 1h --compare --compare-window 1h --compare-start-time 2024-01-15T10:00:00

# With AI analysis (requires OPENAI_API_KEY)
./cardinality-analyzer.py -w 1h --ai-analysis
```

### Docker Operations
```bash
# Build Docker image locally
docker build -t cardinality-analyser .

# Run with Docker
docker run --rm --env-file .env -v $(pwd):/output ghcr.io/grafana-ps/cardinality-analyser:latest -w 1h
```

## Architecture

### Core Components

1. **cardinality-analyzer.py** - Main analyzer script containing:
   - `CardinalityAnalyzer` class for Prometheus/Mimir queries
   - Time window parsing and comparison logic
   - HTML/CSV/CLI output formatters
   - Metric cardinality calculation algorithms

2. **cardinality_analyzer_ai_analysis.py** - AI analysis module:
   - OpenAI integration for analyzing cardinality patterns
   - Generates insights from metric data
   - Requires OPENAI_API_KEY environment variable

### Configuration

Environment variables (in `.env`):
- `PROMETHEUS_ENDPOINT` - Mimir/Prometheus endpoint URL
- `PROMETHEUS_USERNAME` - Basic auth username
- `PROMETHEUS_API_KEY` - API key for authentication
- `OPENAI_API_KEY` - Optional, for AI analysis
- `OPENAI_MODEL` - Optional, defaults to gpt-5
- `OPENAI_REASONING_EFFORT` - Optional, low|medium|high

### Output Formats

- **HTML**: Interactive dashboard with charts and sortable tables
- **CSV**: Exportable data for analysis
- **CLI**: Console output with formatted tables
- **All**: Generates all formats simultaneously

### CI/CD

GitHub Actions workflow (`.github/workflows/docker-publish.yml`):
- Builds multi-architecture Docker images (amd64, arm64)
- Publishes to `ghcr.io/grafana-ps/cardinality-analyser`
- Runs security scans with Trivy
- Triggers on: main branch pushes, tags, and weekly schedule