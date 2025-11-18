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
5. **Optimized file sizes** - NEW: HTML reports now 97% smaller using lazy loading (500 KB vs 17 MB)

## Quick Start

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

# This creates two files:
#   - cardinality_analysis.html (~500 KB) - Open this in your browser
#   - cardinality_analysis_data.json (~17 MB) - Keep this alongside the HTML
```

**Analyze specific metric:**
```bash
# Focus on a single metric over the last hour
./cardinality-analyzer.py -w 1h -m my_application_requests_total
```

**Analyze multiple metrics from a file:**
```bash
# Analyze metrics listed in a file (one per line)
./cardinality-analyzer.py -w 1h -mf metrics.txt

# Combine file metrics with a single metric
./cardinality-analyzer.py -w 1h -m custom_metric -mf metrics.txt
```

**Compare time windows (automatic before/after):**
```bash
# Compares the hour before vs the last hour
# Great for detecting recent changes
./cardinality-analyzer.py -w 1h --compare --compare-window 1h
```

**Analyze a specific historical incident:**
```bash
# Investigate what happened on Jan 10th (24-hour window starting at midnight UTC)
./cardinality-analyzer.py -w 24h -s 2024-01-10T00:00:00Z

# Compare with the previous day to see what changed
./cardinality-analyzer.py -w 24h -s 2024-01-10T00:00:00Z \
  --compare --compare-window 24h --compare-start-time 2024-01-09T00:00:00Z
```

### Metrics File Format

When using `-mf` / `--metrics-file`, create a text file with one metric name per line:

```
# Production metrics
http_requests_total
http_request_duration_seconds

# Database metrics
db_connections_active
db_query_duration_seconds

# Application metrics
app_processing_time_seconds
```

**Features:**
- One metric name per line
- Lines starting with `#` are treated as comments
- Blank lines are ignored
- Leading/trailing whitespace is automatically stripped
- Can be combined with `-m` option to add additional metrics

**Use cases:**
- Analyzing large sets of metrics without overwhelming the system
- Automating analysis for specific metric groups
- Avoiding "too many chunks" errors by analyzing metrics individually
- Creating reusable metric lists for regular analysis

### Using Docker (Alternative)

You can also build and run the tool locally using Docker if you prefer containerization:

#### 1. Build the Docker image

```bash
# Build for your current platform
docker build -t cardinality-analyser .
```

#### 2. Run with Docker

```bash
# Run with environment variables
docker run --rm \
  -e PROMETHEUS_ENDPOINT="https://prometheus-prod-13-prod-us-east-0.grafana.net" \
  -e PROMETHEUS_USERNAME="1234567" \
  -e PROMETHEUS_API_KEY="glc_key-example-..." \
  -v $(pwd):/output \
  cardinality-analyser \
  -w 1h -o html

# Or using an env file
docker run --rm \
  --env-file .env \
  -v $(pwd):/output \
  cardinality-analyser \
  -w 1h --compare --compare-window 1h
```

## Command Line Options

```
Required:
-w, --window, --duration    Duration of the time window to analyze (e.g., 30m, 1h, 24h, 7d)

Optional:
-s, --start-time, --from    Start time in ISO format (e.g., 2024-01-15T10:00:00Z for UTC).
                            Default: current UTC time - window duration
                            This sets the beginning of the analysis window.
-m, --metric                Specific metric to analyze. If not provided, analyzes top N metrics
-mf, --metrics-file         Path to file containing metric names (one per line).
                            Supports # comments and blank lines. Can be combined with -m option
--top-n                     Number of top metrics to analyze when no specific metric is provided (default: 20)
-o, --output                Output format: cli, csv, html, all (default: html)
-h, --help                  Show help message and exit

Performance tuning options:
--step                      Query step interval in seconds. Auto-calculated if not specified.
                            Larger values reduce data points and prevent chunk limit errors.
                            Examples: 60 (1min), 300 (5min), 3600 (1hour)
--max-points                Target number of data points per series for auto-calculation (default: 150).
                            Lower values use larger steps and reduce chunk errors.

HTML optimization options:
--top-n-embed               Number of top label values to embed in HTML file (default: 20).
                            Remaining data is stored in a separate JSON file and loaded on-demand.
                            Higher values increase HTML size but show more data initially.
                            Set to -1 to embed all data (legacy mode, not recommended for large datasets).

Comparison options:
--compare                   Enable comparison mode to analyze changes between two time periods
--compare-window,           Duration of the comparison window (e.g., 1h, same format as --window)
--compare-duration          Required if --compare is used
--compare-start-time,       Start time for the comparison window (e.g., 2024-01-15T10:00:00Z)
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

## Timezone Handling and 422 Errors

**This tool operates in UTC** because Grafana Mimir's API only works with UTC timestamps.

### Specifying Timestamps

Timestamps can be provided in several formats:
- `2024-01-15T10:00:00` - Interpreted as UTC
- `2024-01-15T10:00:00Z` - Explicit UTC (recommended for clarity)
- `2024-01-15T10:00:00+00:00` - Explicit UTC with offset notation
- When no start time is provided, uses current UTC time

### Common Issue: 422 Errors

**Error**: `422 Client Error: Unprocessable Entity`

**Cause**: Mimir rejects queries with timestamps in the future. This can occur due to:
- System clock drift (your clock is ahead of actual UTC time)
- Timezone confusion (entering local time instead of UTC)
- Incorrect timestamp calculation

**Solution**:
- Ensure your system clock is accurate (check with `date -u`)
- Always use UTC timestamps
- Use the `-v` flag for detailed timestamp debugging

**Example**:
```bash
# Check current UTC time
date -u

# Use UTC in queries
./cardinality-analyzer.py -w 1h -s 2024-01-15T10:00:00Z
```

## Query Performance and Chunk Limits

The analyzer automatically optimizes query performance to prevent "too many chunks" errors from Grafana Mimir, which occur when queries try to fetch too much data.

### Automatic Step Optimization

The tool automatically calculates optimal query intervals (step size) based on your time window:

- **1 hour**: ~60-second intervals (~60 data points per series)
- **24 hours**: ~5-10 minute intervals (~150 data points per series)
- **7 days**: ~1 hour intervals (~168 data points per series)
- **30 days**: ~5 hour intervals (~150 data points per series)

**Why larger steps?** Larger step = Fewer evaluations = Fewer data points = Fewer chunks fetched = No errors!

### Manual Performance Tuning

If you need more control, use these options:

```bash
# Use specific step size (in seconds)
./cardinality-analyzer.py -w 7d --step 3600  # 1-hour intervals

# Control target data points (default: 150)
./cardinality-analyzer.py -w 24h --max-points 100  # Fewer points = larger step = faster queries
./cardinality-analyzer.py -w 24h --max-points 300  # More points = smaller step = better resolution
```

### Automatic Retry on Errors

If a query hits chunk limits, the analyzer automatically retries with larger step sizes (up to 3 attempts). You'll see warnings like:

```
Query hit chunk limit (attempt 1/3). Retrying with larger step: 600s (10min)
```

### Performance Recommendations

**For large time ranges (7+ days):**
```bash
# Let the tool auto-optimize
./cardinality-analyzer.py -w 7d

# Or manually specify larger step
./cardinality-analyzer.py -w 7d --step 3600  # 1-hour intervals

# Focus on specific metrics to reduce load
./cardinality-analyzer.py -w 7d -m my_specific_metric --top-n 10
```

**For very large tenants:**
```bash
# Use shorter windows
./cardinality-analyzer.py -w 12h --compare --compare-window 12h

# Or larger steps for longer windows
./cardinality-analyzer.py -w 7d --step 7200  # 2-hour intervals

# Or fewer metrics
./cardinality-analyzer.py -w 24h --top-n 5
```

### Understanding "Too Many Chunks" Errors

If you see HTTP 422 errors mentioning "chunks", the script will show detailed guidance:

```
NOTE: 422 error due to TOO MANY CHUNKS
This query is trying to fetch too much data and exceeded Mimir's chunk limit.

SOLUTIONS:
  1. Reduce time window: Use shorter --window duration
     Example: -w 12h instead of -w 7d

  2. Increase step: Add --step parameter with larger value
     Example: --step 1200 (currently using step=300s)

  3. Focus analysis: Use -m to analyze specific metrics instead of top-N
     Example: -m my_specific_metric
```

### Tradeoffs

- **Smaller step (more data points)**: Better temporal resolution, can detect short spikes, but may hit limits
- **Larger step (fewer data points)**: Always works, faster queries, but may miss brief cardinality spikes

For cardinality trend analysis, larger steps are usually fine since cardinality changes tend to persist.

## Output Formats

### HTML (default)
- Interactive dashboard with sortable tables
- Charts showing top cardinality contributors
- Filterable results
- Before/after comparison views
- Usage instructions included

**⚠️ Important: Viewing HTML Reports**

HTML reports must be served via HTTP (not opened directly with `file://`) for the interactive features to work properly.

**Quick Start - Serve the Reports:**
```bash
# In the same directory as your HTML files, run:
python3 -m http.server 8000

# Then open in your browser:
# http://localhost:8000/cardinality_analysis.html
```

**Why?** Browsers block JavaScript `fetch()` requests from local files for security (CORS policy). When you click "Show all values" buttons to load complete data, the browser needs to fetch the JSON file via HTTP.

**Alternative Methods:**
- Use any local web server (nginx, Apache, VS Code Live Server, etc.)
- Upload files to an internal web server
- For one-time viewing without a server, use `--top-n-embed -1` for legacy single-file mode (generates larger HTML)

**To stop the Python server:** Press `Ctrl+C` or run `lsof -ti:8000 | xargs kill`

**New: Optimized File Size with Lazy Loading**

By default, the HTML output uses a two-file approach to dramatically reduce file sizes:

- **`cardinality_analysis.html`** (~500 KB): Lightweight report with initial data
  - Contains top-N label values (default: 20) for immediate viewing
  - Includes all charts, tables, and interactive features
  - Can be easily shared via email or Slack

- **`cardinality_analysis_data.json`** (~17 MB typical): Complete dataset
  - Contains all cardinality data for all label values
  - Loaded on-demand when you click "Show all values" buttons
  - Can be compressed for archival (e.g., `gzip cardinality_analysis_data.json`)

**Important**: Keep both files in the same directory for lazy loading to work.

**File Size Comparison**:
- **Legacy mode** (single HTML file): 17+ MB
- **New mode** (HTML + JSON): 500 KB HTML + 17 MB JSON
- **Reduction**: 97% smaller HTML file for faster sharing and loading

**Customizing the Embed Size**:
```bash
# Default: embed top 20 values
./cardinality-analyzer.py -w 1h

# Embed more values for faster initial display
./cardinality-analyzer.py -w 1h --top-n-embed 50

# Legacy mode: single self-contained HTML file (large)
./cardinality-analyzer.py -w 1h --top-n-embed -1
```

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
# Incident happened at 2PM UTC yesterday (2024-01-14)
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00Z

# Compare incident with the hour before to see what changed
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00Z --compare --compare-window 1h

# Compare incident with same time on previous day
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00Z \
  --compare --compare-window 1h --compare-start-time 2024-01-13T14:00:00Z
```

### 4. Analyze a specific metric
```bash
# Focus on a single problematic metric over 4 hours
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info

# Compare how this metric changed between yesterday and today
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info \
  --compare --compare-window 4h --compare-start-time 2024-01-14T00:00:00Z
```

### 5. Analyze multiple metrics from a file
```bash
# Create a metrics file with your target metrics
cat > my_metrics.txt <<EOF
# Application metrics
http_requests_total
http_request_duration_seconds
app_errors_total

# Database metrics
db_connections_active
db_query_duration_seconds
EOF

# Analyze all metrics in the file
./cardinality-analyzer.py -w 1h -mf my_metrics.txt

# Analyze file metrics with comparison
./cardinality-analyzer.py -w 1h -mf my_metrics.txt --compare --compare-window 1h

# Combine file metrics with additional single metric
./cardinality-analyzer.py -w 1h -mf my_metrics.txt -m additional_metric
```

### 6. Export data for further analysis
```bash
# Get top 50 metrics as CSV for spreadsheet analysis
./cardinality-analyzer.py -w 30m --top-n 50 -o csv

# Generate all output formats (HTML, CSV, and CLI)
./cardinality-analyzer.py -w 1h -o all
```

### 7. Debugging deployment issues
```bash
# Your deployment happened at 10:30 AM UTC
# Compare 30 minutes before and after deployment
./cardinality-analyzer.py -w 30m -s 2024-01-15T10:30:00Z \
  --compare --compare-window 30m --compare-start-time 2024-01-15T10:00:00Z

# With AI analysis to get insights
./cardinality-analyzer.py -w 30m -s 2024-01-15T10:30:00Z \
  --compare --compare-window 30m --ai-analysis
```

### 8. Weekly pattern analysis
```bash
# Compare this Monday with last Monday (business hours)
./cardinality-analyzer.py -w 8h -s 2024-01-15T09:00:00Z \
  --compare --compare-window 8h --compare-start-time 2024-01-08T09:00:00Z
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
# ISO format with UTC timezone (recommended)
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00Z

# ISO format with explicit UTC offset
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00+00:00

# Without timezone (interpreted as UTC)
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00

# What it analyzes:
-w 1h -s 2024-01-15T14:00:00Z → Analyzes 14:00:00 to 15:00:00 UTC on Jan 15
-w 24h -s 2024-01-10T00:00:00Z → Analyzes entire day of Jan 10 UTC
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

### Metrics File (`-mf`, `--metrics-file`)
**Optional**: Analyze multiple metrics from a file (one metric name per line).

```bash
# Analyze metrics from a file
./cardinality-analyzer.py -w 1h -mf metrics.txt

# With comparison mode
./cardinality-analyzer.py -w 1h -mf metrics.txt --compare --compare-window 1h

# Combine file metrics with single metric
./cardinality-analyzer.py -w 1h -mf metrics.txt -m additional_metric

# File format (supports comments and blank lines):
# # Production metrics
# http_requests_total
# http_request_duration_seconds
#
# # Database metrics
# db_connections_active
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
# Creates: cardinality_analysis.html (~500 KB) + cardinality_analysis_data.json (~17 MB)

# CLI output - Terminal-friendly tables
./cardinality-analyzer.py -w 1h -o cli
# Displays results directly in terminal

# CSV output - For spreadsheet analysis
./cardinality-analyzer.py -w 1h -o csv
# Creates: cardinality_analysis.csv

# All formats at once
./cardinality-analyzer.py -w 1h -o all
# Creates HTML, JSON, and CSV files, plus shows CLI output
```

### HTML File Size Optimization (`--top-n-embed`)
**Optional**: Control how much data is embedded in the HTML file (default: 20).

```bash
# Default: embed top 20 label values (recommended)
./cardinality-analyzer.py -w 1h --top-n-embed 20
# HTML: ~500 KB, JSON: ~17 MB (total same as before, but HTML much smaller)

# Embed more values for richer initial view
./cardinality-analyzer.py -w 1h --top-n-embed 50
# HTML: ~1.5 MB, JSON: ~17 MB (less lazy loading needed)

# Embed only top 10 for minimal HTML size
./cardinality-analyzer.py -w 1h --top-n-embed 10
# HTML: ~300 KB, JSON: ~17 MB (smallest HTML possible)

# Legacy mode: single self-contained file (not recommended)
./cardinality-analyzer.py -w 1h --top-n-embed -1
# Single HTML: 17+ MB (no separate JSON file, everything embedded)
```

**When to use different values:**
- **Default (20)**: Best balance for most use cases - small HTML for sharing, quick initial load
- **Higher (50-100)**: When you want more data visible without clicking "Show all values" buttons
- **Lower (10)**: When HTML file size is critical (email attachments, slow connections)
- **-1 (legacy)**: Only if you need a single self-contained file and don't mind large size

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
  --compare-start-time 2024-01-14T14:00:00Z

# Compare with last week
./cardinality-analyzer.py -w 24h -s 2024-01-15T00:00:00Z --compare \
  --compare-window 24h --compare-start-time 2024-01-08T00:00:00Z

# If not specified, defaults to immediately before main window:
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00Z --compare --compare-window 1h
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
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00Z --compare --compare-window 1h
```
Analyzes the incident hour and compares with the hour before.

### "Is this metric behaving differently than yesterday?"
```bash
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info --compare \
  --compare-window 4h --compare-start-time 2024-01-14T10:00:00Z
```
Compares the metric's behavior over 4-hour windows.

### "Full investigation with all bells and whistles"
```bash
./cardinality-analyzer.py -w 1h -s 2024-01-15T14:00:00Z \
  --compare --compare-window 1h --compare-start-time 2024-01-15T12:00:00Z \
  --top-n 30 --ai-analysis -o all -v
```
Analyzes 2-3 PM, compares with 12-1 PM, includes top 30 metrics, AI insights, all output formats, and verbose logging.

### "Quick check of current state"
```bash
./cardinality-analyzer.py -w 30m --top-n 5 -o cli
```
Fast 30-minute analysis of top 5 metrics, displayed in terminal.

### "Analyze specific metric group without overwhelming the system"
```bash
# Create file with metrics to monitor
cat > critical_metrics.txt <<EOF
api_requests_total
api_response_time_seconds
database_connections
cache_hit_ratio
EOF

./cardinality-analyzer.py -w 1h -mf critical_metrics.txt --compare --compare-window 1h
```
Analyzes a specific set of metrics individually, avoiding "too many chunks" errors while still getting comprehensive analysis.

### "Export data for presentation"
```bash
./cardinality-analyzer.py -w 24h --compare --compare-window 24h \
  --ai-analysis -o html
```
Creates a comprehensive HTML report with AI insights comparing last 24 hours with the previous 24 hours.

### "Optimize HTML file size for email sharing"
```bash
# Minimal HTML file for easy sharing (default behavior)
./cardinality-analyzer.py -w 1h --top-n-embed 20
# Creates ~500 KB HTML file + separate JSON data file

# Embed more data for faster viewing (larger HTML)
./cardinality-analyzer.py -w 1h --top-n-embed 50
# Creates ~1.5 MB HTML file + separate JSON data file

# Legacy single-file mode (not recommended for large datasets)
./cardinality-analyzer.py -w 1h --top-n-embed -1
# Creates single 17+ MB HTML file with all data embedded
```
The default lazy loading approach creates smaller HTML files perfect for email attachments while preserving access to all data.

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
   OPENAI_REASONING_EFFORT="medium"  # low|medium|high
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

## Adaptive Telemetry Compatibility

All queries made by the Cardinality Analyzer include the `__ignore_usage__=""` label selector to prevent interference with [Grafana Cloud Adaptive Telemetry](https://grafana.com/docs/grafana-cloud/adaptive-telemetry/adaptive-metrics/manage-recommendations/understand-recommended-rules/#make-the-recommendations-service-ignore-a-query) recommendations.

This ensures that:
- Automated analysis queries don't count as "real" user queries
- The recommendations service won't try to preserve compatibility with these temporary queries
- Your Adaptive Metrics recommendations remain accurate based on actual dashboard and alerting usage

This feature is always enabled and requires no configuration. The `__ignore_usage__` label selector has no effect on query results—it only signals to the recommendations service to ignore these queries.
