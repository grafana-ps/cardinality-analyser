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

### 1. Set up environment

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

### 2. Configure credentials

Use the same `.env` file as dpm-finder with your Grafana Cloud credentials:

```bash
PROMETHEUS_ENDPOINT="https://prometheus-prod-13-prod-us-east-0.grafana.net"
PROMETHEUS_USERNAME="1234567"
PROMETHEUS_API_KEY="glc_key-example-..."
```

### 3. Run the analyzer

**Basic usage - analyze last hour:**
```bash
./cardinality-analyzer.py -w 1h
```

**Analyze specific metric:**
```bash
./cardinality-analyzer.py -w 1h -m my_application_requests_total
```

**Compare time windows (before/after):**
```bash
./cardinality-analyzer.py -w 1h --compare --compare-window 1h --compare-start-time 2024-01-15T10:00:00
```

**Analyze historical data:**
```bash
./cardinality-analyzer.py -w 24h -s 2024-01-10T00:00:00
```

## Command Line Options

```
Required:
-w, --window          Time window to analyze (e.g., 30m, 1h, 24h, 7d)

Optional:
-s, --start-time      Start time in ISO format (e.g., 2024-01-15T10:00:00 for local time,
                      2024-01-15T10:00:00Z for UTC). Default: current UTC time - window
-m, --metric          Specific metric to analyze. If not provided, analyzes top N metrics
--top-n               Number of top metrics to analyze when no specific metric is provided (default: 20)
-o, --output          Output format: cli, csv, html, all (default: html)
-h, --help            Show help message and exit

Comparison options:
--compare             Enable comparison mode to analyze changes between two time windows
--compare-window      Time window for comparison (required if --compare is used)
--compare-start-time  Start time for comparison window (same format as --start-time)

AI Analysis:
--ai-analysis         Generate AI-powered analysis and recommendations using OpenAI
                      (requires OPENAI_KEY env var and additional dependencies)
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

## Examples

### Investigate a spike that happened yesterday
```bash
# Analyze the spike period (2-3 PM yesterday)
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00

# Compare with normal period (same time previous day)
./cardinality-analyzer.py -w 1h -s 2024-01-14T14:00:00 \
  --compare --compare-window 1h --compare-start-time 2024-01-13T14:00:00
```

### Find problematic labels in a specific metric
```bash
./cardinality-analyzer.py -w 4h -m kubernetes_pod_info -o all
```

### Analyze top metrics with highest cardinality
```bash
./cardinality-analyzer.py -w 30m --top-n 50 -o csv
```

## AI-Powered Analysis (NEW)

The cardinality analyzer now includes optional AI-powered analysis using OpenAI's GPT models to provide insights and recommendations.

### Setting up AI Analysis

1. **Install additional dependencies**:
   ```bash
   pip install -r requirements-cardinalityanalysis.txt
   ```
   This installs the OpenAI SDK along with the standard dependencies.

2. **Configure OpenAI credentials**:
   Add these to your `.env` file:
   ```bash
   OPENAI_KEY="sk-..."  # Your OpenAI API key
   OPENAI_MODEL="gpt-4.1"  # Optional: defaults to gpt-4.1 for good balance between price and performance
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
1. Check that `OPENAI_KEY` is set correctly in `.env`
2. Verify the API key has sufficient credits
3. Ensure you've installed the extra dependencies: `pip install -r requirements-cardinalityanalysis.txt`
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
