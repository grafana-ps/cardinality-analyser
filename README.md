## Support

This project is not actively supported by Grafana Labs. It comes with no warranty of any kind and is designed for Grafana Professional Services teams to run

---

# Cardinality Analyzer

In addition to the DPM finder, this repository includes a **Cardinality Analyzer** tool for investigating metric spike issues in Grafana Cloud Mimir by analyzing cardinality changes across time windows.

## Overview

The `cardinality-analyzer.py` script helps diagnose "what's the root cause of my recent increase in ingest volume" by:

1. **Analyzing cardinality by label** - Shows which labels have the most unique values
2. **Comparing time windows** - Identifies what changed between "before" and "after" periods
3. **Interactive reporting** - Generates HTML reports with sortable tables and charts
4. **Multiple output formats** - CLI, CSV, or HTML outputs for different use cases

## Quick Start

### Using Docker (Recommended)

The easiest way to run the Cardinality Analyzer is using the pre-built Docker image:

```bash
# Pull the latest image
docker pull ghcr.io/grafana-ps/cardinality-analyser:latest

# Run with environment variables
docker run --rm \
  -e PROMETHEUS_ENDPOINT="https://prometheus-prod-13-prod-us-east-0.grafana.net" \
  -e PROMETHEUS_USERNAME="1234567" \
  -e PROMETHEUS_API_KEY="glc_key-example-..." \
  -v $(pwd):/output \
  ghcr.io/grafana-ps/cardinality-analyser:latest \
  -w 1h -o html

# Or using an env file
docker run --rm \
  --env-file .env \
  -v $(pwd):/output \
  ghcr.io/grafana-ps/cardinality-analyser:latest \
  -w 1h --compare --compare-window 1h --compare-start-time 2024-01-15T10:00:00
```

**Available Docker tags:**
- `latest` - Latest stable release from main branch
- `vX.Y.Z` - Specific version releases
- `main-SHA` - Specific commits from main branch
- `nightly` - Nightly builds (may be unstable)

### Local Installation

#### 1. Set up environment

If you've already set up a virtual environment for dpm-finder, you can use the same one:

```bash
source ./venv/bin/activate  # On Windows: venv\Scripts\activate
# No additional dependencies needed - uses the same requirements.txt as dpm-finder
```

If you haven't set up a venv yet:

```bash
python3 -m venv venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

#### 2. Configure credentials

Use the same `.env` file as dpm-finder with your Grafana Cloud credentials:

```bash
PROMETHEUS_ENDPOINT="https://prometheus-prod-13-prod-us-east-0.grafana.net"
PROMETHEUS_USERNAME="1234567"
PROMETHEUS_API_KEY="glc_key-example-..."
```

#### 3. Run the analyzer

**Basic usage - analyze last hour:**
```bash
# Analyzes metrics from (now - 1 hour) to now
./cardinality-analyzer.py -w 1h
```

**Analyze specific metric:**
```bash
# Focus on a single metric over the last hour
./cardinality-analyzer.py -w 1h -m my_application_requests_total
```

**Compare time windows (automatic before/after):**
```bash
# Compares the hour before vs the last hour
# Great for detecting recent changes
./cardinality-analyzer.py -w 1h --compare --compare-window 1h
```

**Analyze a specific historical incident:**
```bash
# Investigate what happened on Jan 10th (24-hour window starting at midnight)
./cardinality-analyzer.py -w 24h -s 2024-01-10T00:00:00

# Compare with the previous day to see what changed
./cardinality-analyzer.py -w 24h -s 2024-01-10T00:00:00 \
  --compare --compare-window 24h --compare-start-time 2024-01-09T00:00:00
```

## Command Line Options

```
Required:
-w, --window, --duration    Duration of the time window to analyze (e.g., 30m, 1h, 24h, 7d)

Optional:
-s, --start-time, --from    Start time in ISO format (e.g., 2024-01-15T10:00:00 for local time,
                            2024-01-15T10:00:00Z for UTC). Default: current time - window duration
                            This sets the beginning of the analysis window.
-m, --metric                Specific metric to analyze. If not provided, analyzes top N metrics
--top-n                     Number of top metrics to analyze when no specific metric is provided (default: 20)
-o, --output                Output format: cli, csv, html, all (default: html)
-h, --help                  Show help message and exit

Comparison options:
--compare                   Enable comparison mode to analyze changes between two time periods
--compare-window,           Duration of the comparison window (e.g., 1h, same format as --window)
--compare-duration          Required if --compare is used
--compare-start-time,       Start time for the comparison window (same format as --start-time)
--compare-from              If not provided, defaults to immediately before the main analysis window

AI Analysis:
--ai-analysis               Generate AI-powered analysis using OpenAI API
                           (requires OPENAI_API_KEY env var)
```

### Understanding Time Windows

The script analyzes metrics over two potential time windows:

1. **Main Analysis Window**: The primary time period you want to investigate
   - Set by `-w` (duration) and optionally `-s` (start time)
   - If no start time is given, it defaults to "now minus duration"

2. **Comparison Window** (optional): A reference time period to compare against
   - Enabled with `--compare`
   - Set by `--compare-window` (duration) and optionally `--compare-start-time`
   - If no comparison start time is given, it defaults to immediately before the main window

**Time Flow Visualization:**
```
[Comparison Window] → [Main Analysis Window] → [Now]
      (before)              (after)
```

## Timezone Handling

- When no start time is specified, the script uses current UTC time
- Start times without timezone info are interpreted as local time
- Use 'Z' suffix or '+00:00' for UTC times (e.g., 2024-01-15T10:00:00Z)
- All queries to Mimir are performed using Unix timestamps (timezone-agnostic)

## Output Formats

### HTML (default)
- Interactive dashboard with sortable tables
- Charts showing top cardinality contributors
- Filterable results
- Before/after comparison views
- Usage instructions included

### CLI
- Console output with formatted tables
- Shows top labels by cardinality
- Highlights changes in comparison mode

### CSV
- Exportable data for further analysis
- Includes metric name, label, cardinality, and top values

## Complete Examples Guide

### Time Window Basics

The script operates on time windows, which are periods of time defined by:
- **Duration**: How long the window is (`-w` or `--window`)
- **Start Time**: When the window begins (`-s` or `--start-time`, optional)

#### How Time Windows Work:
```
Window Duration: -w 1h
Start Time: -s 2024-01-15T14:00:00 (optional)

If start time provided:
  [14:00] -------- 1 hour --------→ [15:00]
  
If no start time (default):
  [Now - 1h] ---- 1 hour --------→ [Now]
```

## Examples

### 1. Analyze the last hour (most common use case)
```bash
# Analyze metrics from the last hour
./cardinality-analyzer.py -w 1h

# Same as above but explicit about timing (if current time is 2024-01-15 15:00:00 UTC)
# This analyzes: 2024-01-15 14:00:00 to 2024-01-15 15:00:00
./cardinality-analyzer.py -w 1h
```

### 2. Compare current issues with yesterday (automatic comparison)
```bash
# Compare last hour with the hour before it
# If now is 15:00, compares [13:00-14:00] vs [14:00-15:00]
./cardinality-analyzer.py -w 1h --compare --compare-window 1h

# Compare last hour with same time yesterday
# If now is 2024-01-15 15:00, compares:
# [2024-01-14 14:00-15:00] vs [2024-01-15 14:00-15:00]
./cardinality-analyzer.py -w 1h --compare --compare-window 1h --compare-start-time 2024-01-14T14:00:00
```

### 3. Investigate a specific incident time
```bash
# Incident happened at 2PM yesterday (2024-01-14)
# Analyze the incident hour
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00

# Compare incident with the hour before to see what changed
# Compares [13:00-14:00] vs [14:00-15:00] on 2024-01-14
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00 --compare --compare-window 1h

# Compare incident with same time on previous day
# Compares [2024-01-13 14:00-15:00] vs [2024-01-14 14:00-15:00]
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00 \
  --compare --compare-window 1h --compare-start-time 2024-01-13T14:00:00
```

### 4. Analyze a specific metric
```bash
# Focus on a single problematic metric over 4 hours
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info

# Compare how this metric changed between yesterday and today
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info \
  --compare --compare-window 4h --compare-start-time 2024-01-14T00:00:00
```

### 5. Export data for further analysis
```bash
# Get top 50 metrics as CSV for spreadsheet analysis
./cardinality-analyzer.py -w 30m --top-n 50 -o csv

# Generate all output formats (HTML, CSV, and CLI)
./cardinality-analyzer.py -w 1h -o all
```

### 6. Debugging deployment issues
```bash
# Your deployment happened at 10:30 AM UTC
# Compare 30 minutes before and after deployment
./cardinality-analyzer.py -w 30m -s 2024-01-15T10:30:00 \
  --compare --compare-window 30m --compare-start-time 2024-01-15T10:00:00

# With AI analysis to get insights
./cardinality-analyzer.py -w 30m -s 2024-01-15T10:30:00 \
  --compare --compare-window 30m --ai-analysis
```

### 7. Weekly pattern analysis
```bash
# Compare this Monday with last Monday (business hours)
./cardinality-analyzer.py -w 8h -s 2024-01-15T09:00:00 \
  --compare --compare-window 8h --compare-start-time 2024-01-08T09:00:00
```

## Complete CLI Arguments Reference with Examples

### Window Duration (`-w`, `--window`, `--duration`)
**Required**: Specifies how long the time window should be.

```bash
# Different duration formats
./cardinality-analyzer.py -w 30m    # 30 minutes
./cardinality-analyzer.py -w 1h     # 1 hour
./cardinality-analyzer.py -w 24h    # 24 hours (1 day)
./cardinality-analyzer.py -w 7d     # 7 days (1 week)
./cardinality-analyzer.py -w 2w     # 2 weeks

# What it analyzes (if current time is 2024-01-15 15:00:00 UTC):
-w 1h  → Analyzes 14:00:00 to 15:00:00
-w 24h → Analyzes 2024-01-14 15:00:00 to 2024-01-15 15:00:00
```

### Start Time (`-s`, `--start-time`, `--from`)
**Optional**: Sets the beginning of the analysis window.

```bash
# ISO format (local timezone if not specified)
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00

# ISO format with UTC timezone
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00Z

# ISO format with timezone offset
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00+05:30

# What it analyzes:
-w 1h -s 2024-01-15T14:00:00 → Analyzes 14:00:00 to 15:00:00 on Jan 15
-w 24h -s 2024-01-10T00:00:00 → Analyzes entire day of Jan 10
```

### Metric Selection (`-m`, `--metric`)
**Optional**: Focus on a specific metric instead of top N metrics.

```bash
# Analyze specific metric
./cardinality-analyzer.py -w 1h -m kubernetes_pod_info

# Analyze specific metric with comparison
./cardinality-analyzer.py -w 1h -m http_requests_total --compare --compare-window 1h

# With AI analysis for deep insights
./cardinality-analyzer.py -w 1h -m problematic_metric --ai-analysis
```

### Top N Metrics (`--top-n`)
**Optional**: When no specific metric is provided, analyze the top N metrics by cardinality (default: 20).

```bash
# Analyze top 10 metrics only
./cardinality-analyzer.py -w 1h --top-n 10

# Analyze top 50 metrics for comprehensive view
./cardinality-analyzer.py -w 1h --top-n 50

# Just analyze the single highest cardinality metric
./cardinality-analyzer.py -w 1h --top-n 1
```

### Output Format (`-o`, `--output`)
**Optional**: Choose output format (default: html).

```bash
# HTML output (default) - Interactive web dashboard
./cardinality-analyzer.py -w 1h -o html
# Creates: cardinality_analysis.html

# CLI output - Terminal-friendly tables
./cardinality-analyzer.py -w 1h -o cli
# Displays results directly in terminal

# CSV output - For spreadsheet analysis
./cardinality-analyzer.py -w 1h -o csv
# Creates: cardinality_analysis_TIMESTAMP.csv

# All formats at once
./cardinality-analyzer.py -w 1h -o all
# Creates HTML and CSV files, plus shows CLI output
```

### Comparison Mode (`--compare`)
**Optional**: Enable before/after comparison between two time windows.

```bash
# Basic comparison (auto-selects comparison window before main window)
./cardinality-analyzer.py -w 1h --compare --compare-window 1h
# If now is 15:00, compares:
#   Comparison: 13:00-14:00 (baseline)
#   Main:       14:00-15:00 (current)
```

### Comparison Window Duration (`--compare-window`, `--compare-duration`)
**Required with --compare**: Duration of the comparison window.

```bash
# Same duration comparison
./cardinality-analyzer.py -w 1h --compare --compare-window 1h

# Different duration comparison (compare 30min baseline with 1h spike)
./cardinality-analyzer.py -w 1h --compare --compare-window 30m

# Compare day vs week
./cardinality-analyzer.py -w 1d --compare --compare-window 7d
```

### Comparison Start Time (`--compare-start-time`, `--compare-from`)
**Optional with --compare**: When the comparison window begins.

```bash
# Compare with yesterday at same time
./cardinality-analyzer.py -w 1h --compare --compare-window 1h \
  --compare-start-time 2024-01-14T14:00:00

# Compare with last week
./cardinality-analyzer.py -w 24h -s 2024-01-15T00:00:00 --compare \
  --compare-window 24h --compare-start-time 2024-01-08T00:00:00

# If not specified, defaults to immediately before main window:
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00 --compare --compare-window 1h
# Comparison: 13:00-14:00, Main: 14:00-15:00
```

### AI Analysis (`--ai-analysis`)
**Optional**: Generate AI-powered insights (requires OPENAI_API_KEY).

```bash
# Basic AI analysis
./cardinality-analyzer.py -w 1h --ai-analysis

# AI analysis with comparison
./cardinality-analyzer.py -w 1h --compare --compare-window 1h --ai-analysis

# AI analysis for specific metric
./cardinality-analyzer.py -w 1h -m kubernetes_pod_info --ai-analysis
```

### Verbose Logging (`-v`, `--verbose`)
**Optional**: Enable detailed debug output.

```bash
# See detailed query information
./cardinality-analyzer.py -w 1h -v

# Debug connection issues
./cardinality-analyzer.py -w 1h -m my_metric -v
```

## Common Use Case Combinations

### "What changed in the last hour?"
```bash
./cardinality-analyzer.py -w 1h --compare --compare-window 1h
```
Shows you what metrics/labels increased between the previous hour and the current hour.

### "What happened during the incident at 2 PM?"
```bash
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00 --compare --compare-window 1h
```
Analyzes the incident hour and compares with the hour before.

### "Is this metric behaving differently than yesterday?"
```bash
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info --compare \
  --compare-window 4h --compare-start-time 2024-01-14T10:00:00
```
Compares the metric's behavior over 4-hour windows.

### "Full investigation with all bells and whistles"
```bash
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00 \
  --compare --compare-window 1h --compare-start-time 2024-01-15T12:00:00 \
  --top-n 30 --ai-analysis -o all -v
```
Analyzes 2-3 PM, compares with 12-1 PM, includes top 30 metrics, AI insights, all output formats, and verbose logging.

### "Quick check of current state"
```bash
./cardinality-analyzer.py -w 30m --top-n 5 -o cli
```
Fast 30-minute analysis of top 5 metrics, displayed in terminal.

### "Export data for presentation"
```bash
./cardinality-analyzer.py -w 24h --compare --compare-window 24h \
  --ai-analysis -o html
```
Creates a comprehensive HTML report with AI insights comparing last 24 hours with the previous 24 hours.

## AI-Powered Analysis (NEW)

The cardinality analyzer now includes optional AI-powered analysis using OpenAI's GPT models to provide insights and recommendations.

### Setting up AI Analysis

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs the OpenAI SDK along with the standard dependencies.

2. **Configure OpenAI credentials**:
   Add these to your `.env` file:
   ```bash
   OPENAI_API_KEY="sk-..."  # Your OpenAI API key
   # Optional overrides:
   OPENAI_MODEL="gpt-5-mini"     # Defaults to gpt-5-mini
   OPENAI_REASONING_EFFORT="high"  # low|medium|high
   ```

3. **Enable AI analysis**:
   Add the `--ai-analysis` flag to any command:
   ```bash
   # Basic analysis with AI insights
   ./cardinality-analyzer.py -w 1h --ai-analysis
   
   # Comparison with AI recommendations
   ./cardinality-analyzer.py -w 1h --compare --compare-window 1h --compare-start-time 2024-01-14T10:00:00 --ai-analysis
   ```

### What the AI Analyzes

The AI provides:
- **Key Findings**: Detailed observations about cardinality patterns and changes
- **Detailed Analysis**: Comprehensive metric-by-metric and label-by-label breakdown
- **Changes Between Windows**: When comparing time periods, specific differences with numbers
- **Pattern Recognition**: Identifies which labels have highest cardinality and their distributions
- **Data Explanation**: Clear descriptions of what the metrics show, not recommendations

### AI Analysis Features

- **Detailed Breakdowns**: Provides comprehensive explanations of cardinality distribution
- **Change Analysis**: When comparing time windows, explains exactly what changed with numbers
- **Pattern Recognition**: Identifies which labels contribute most to cardinality
- **Data-Focused**: Concentrates on explaining the data rather than providing solutions
- **Markdown Formatting**: Results are properly formatted with headers, bold text, and bullet points

### Example AI Output

The AI analysis appears in:
- **HTML reports**: As a dedicated section with formatted recommendations
- **CLI output**: After the standard analysis tables
- **All formats**: When using `-o all`, AI insights are included in each format

### Cost Considerations

- Each analysis uses OpenAI API tokens (typically 2000-4000 tokens per analysis)
- Larger datasets and comparison analyses use more tokens
- Monitor your OpenAI usage to manage costs
- Use `OPENAI_MODEL="gpt-3.5-turbo"` for lower-cost analysis

### Troubleshooting AI Analysis

If AI analysis fails:
1. Check that `OPENAI_API_KEY` is set correctly in `.env`
2. Verify the API key has sufficient credits
3. Ensure you've installed dependencies: `pip install -r requirements.txt`
4. Check logs for specific error messages

## Building the Docker Image

### Building Locally

To build the Docker image locally:

```bash
# Build for your current platform
docker build -t cardinality-analyser .

# Build multi-platform image (requires Docker Buildx)
docker buildx build --platform linux/amd64,linux/arm64 -t cardinality-analyser .

# Run locally built image
docker run --rm --env-file .env -v $(pwd):/output cardinality-analyser -w 1h
```

### GitHub Actions CI/CD

The project includes automated Docker image building and publishing via GitHub Actions:

- **Automatic builds** on push to main branch
- **Multi-architecture support** (linux/amd64 and linux/arm64)
- **Container registry** at `ghcr.io/grafana-ps/cardinality-analyser`
- **Security scanning** with Trivy for vulnerability detection
- **Nightly builds** every Monday at 2 AM UTC

The workflow automatically:
1. Builds native images for each architecture in parallel
2. Creates and pushes multi-arch manifest
3. Tags releases appropriately (latest, version tags, SHA tags)
4. Runs security scans on scheduled builds

## Understanding the Results

The analyzer helps identify:
- **High cardinality labels**: Labels with many unique values that consume resources
- **Cardinality changes**: Which labels increased/decreased between time periods
- **Top contributors**: Which label values contribute most to cardinality

Use this information to:
1. Add recording rules for high-cardinality metrics
2. Implement label filtering in your applications
3. Configure metric relabeling in your collection pipeline
4. Set up alerts for cardinality thresholds
