#!/usr/bin/env python3

"""
Cardinality Analyzer - A tool to investigate metric spike issues in Grafana Cloud Mimir
by analyzing cardinality changes across time windows.
"""

import os
import sys
import time
import argparse
import requests
import json
import csv
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urljoin
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Set up logging (will be configured in main())
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def is_regex_pattern(pattern: str) -> bool:
    """Check if pattern contains regex special characters.

    Used to auto-detect if -m argument is a regex pattern vs exact metric name.
    """
    regex_chars = '.*+?^${}[]|()\\'
    return any(c in pattern for c in regex_chars)


class CardinalityAnalyzer:
    """Main class for analyzing metric cardinality in Mimir"""
    
    def __init__(self, endpoint: str, username: str, api_key: str):
        self.endpoint = endpoint.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_key)
        self.session = requests.Session()
        self.session.auth = self.auth
        
    def parse_time_window(self, window: str, start_time: Optional[str] = None) -> Tuple[int, int]:
        """Parse time window and return start/end timestamps

        All timestamps are processed in UTC to prevent timezone-related errors.
        If a start_time is provided without an explicit timezone, it will be treated as UTC
        and a warning will be logged.
        """
        # Parse duration units
        duration_map = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }

        # Extract number and unit from window
        import re
        match = re.match(r'^(\d+)([smhdw])$', window)
        if not match:
            raise ValueError(f"Invalid time window format: {window}. Use format like '30m', '1h', '7d'")

        value = int(match.group(1))
        unit = match.group(2)
        window_seconds = value * duration_map[unit]

        # Calculate timestamps
        if start_time:
            # Parse start time if provided - ALWAYS use UTC
            try:
                # Check if timezone is specified
                has_timezone = start_time.endswith('Z') or '+' in start_time or (start_time.count('-') > 2)

                # Parse the datetime
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))

                # If no timezone was specified, assume UTC and warn user
                if not has_timezone:
                    logger.warning(f"Start time '{start_time}' has no timezone specified. Treating as UTC. "
                                 f"Please use UTC format (e.g., '{start_time}Z' or '{start_time}+00:00') to avoid ambiguity.")
                    # Force UTC timezone if naive datetime
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)

                # Ensure timezone-aware datetime is in UTC
                start_dt = start_dt.astimezone(timezone.utc)

            except ValueError as e:
                raise ValueError(
                    f"Invalid start time format: {start_time}. "
                    f"Use ISO format in UTC: 'YYYY-MM-DDTHH:MM:SSZ' or 'YYYY-MM-DDTHH:MM:SS+00:00'. "
                    f"Error: {e}"
                )

            end_dt = start_dt + timedelta(seconds=window_seconds)
        else:
            # Use current time if no start time provided (in UTC)
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(seconds=window_seconds)

        # Validate timestamps are not in the future
        now_utc = datetime.now(timezone.utc)
        if start_dt > now_utc:
            raise ValueError(
                f"Start time {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} is in the future. "
                f"Current UTC time is {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
                f"Please provide a start time in the past using UTC timezone."
            )
        if end_dt > now_utc:
            # Allow a small buffer (10 seconds) for clock skew
            if (end_dt - now_utc).total_seconds() > 10:
                raise ValueError(
                    f"End time {end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} is in the future. "
                    f"Current UTC time is {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
                    f"Please adjust your start time or window duration to query past data only."
                )

        # Validate and warn about large time ranges
        duration_seconds = int(end_dt.timestamp()) - int(start_dt.timestamp())
        duration_days = duration_seconds / 86400

        MAX_RECOMMENDED_DAYS = 7
        if duration_days > MAX_RECOMMENDED_DAYS:
            logger.warning("="*60)
            logger.warning(f"LARGE TIME RANGE WARNING: {duration_days:.1f} days")
            logger.warning("="*60)
            logger.warning(f"Analyzing long time ranges may hit Mimir query limits.")
            logger.warning(f"The script will automatically use larger step sizes, but:")
            logger.warning(f"  - Temporal resolution will be reduced")
            logger.warning(f"  - Short-lived cardinality spikes may be missed")
            logger.warning(f"  - Queries may still fail on very large tenants")
            logger.warning("")
            logger.warning("Recommendations:")
            logger.warning(f"  1. Consider shorter windows (e.g., -w 24h or -w 3d)")
            logger.warning(f"  2. Use --step parameter to manually control resolution")
            logger.warning(f"  3. Focus on specific metrics with -m instead of top-N")
            logger.warning("="*60)

        return int(start_dt.timestamp()), int(end_dt.timestamp())

    def calculate_optimal_step(self, start: int, end: int, target_points: int = 150, min_step: int = 60) -> int:
        """
        Calculate optimal step size to target a specific number of data points.

        This helps prevent "too many chunks" errors in Mimir by reducing the number
        of data points fetched per query. Larger step = fewer evaluations = fewer chunks.

        Args:
            start: Start timestamp
            end: End timestamp
            target_points: Desired number of data points (default: 150)
            min_step: Minimum step in seconds (default: 60)

        Returns:
            Optimal step size in seconds
        """
        duration = end - start

        # Calculate step to achieve target points
        calculated_step = duration // target_points

        # Enforce minimum step
        step = max(min_step, calculated_step)

        # Round to nearest minute for cleaner values
        step = (step // 60) * 60

        # Ensure at least 60 seconds
        step = max(60, step)

        logger.debug(f"Calculated optimal step: {step}s for {duration}s duration (target: {target_points} points)")

        return step

    def query_prometheus_with_retry(self, query: str, start: int, end: int, step: int = 60, max_retries: int = 3) -> Dict[str, Any]:
        """
        Query Prometheus/Mimir with automatic retry on chunk limit errors.

        If a query fails with a "too many chunks" error (HTTP 422), automatically
        retry with a larger step size (doubled on each attempt).

        Args:
            query: PromQL query string
            start: Start timestamp
            end: End timestamp
            step: Initial step size in seconds
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Query result data

        Raises:
            Exception: If query fails after all retries
        """
        current_step = step
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.query_prometheus(query, start, end, current_step)
            except requests.exceptions.HTTPError as e:
                last_error = e
                if hasattr(e, 'response') and e.response.status_code == 422:
                    # Check if this is a chunk-related error
                    error_body = ""
                    try:
                        error_data = e.response.json()
                        error_body = str(error_data.get('error', '')).lower()
                    except:
                        error_body = e.response.text.lower()

                    # If it's a chunk error, retry with larger step
                    if ('chunk' in error_body or 'too many' in error_body) and attempt < max_retries - 1:
                        # Double the step and retry
                        current_step *= 2
                        logger.warning(
                            f"Query hit chunk limit (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying with larger step: {current_step}s ({current_step//60}min)"
                        )
                        continue
                # If not a chunk error or last attempt, re-raise
                raise
            except Exception as e:
                # For non-HTTP errors, don't retry
                raise

        # If we get here, all retries failed
        raise last_error if last_error else Exception(f"Failed to execute query after {max_retries} attempts")

    def query_prometheus(self, query: str, start: int, end: int, step: int = 60) -> Dict[str, Any]:
        """Execute a Prometheus query over a time range"""
        url = urljoin(self.endpoint, '/api/prom/api/v1/query_range')
        params = {
            'query': query,
            'start': start,
            'end': end,
            'step': step
        }

        # Format timestamps for human readability
        start_human = datetime.fromtimestamp(start, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        end_human = datetime.fromtimestamp(end, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        try:
            logger.debug(f"Executing query: {query}")
            logger.debug(f"Time range: {start_human} to {end_human}")
            logger.debug(f"Parameters: start={start}, end={end}, step={step}")

            response = self.session.get(url, params=params, timeout=300)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'success':
                raise Exception(f"Query failed: {data.get('error', 'Unknown error')}")

            # Log if no results returned
            if not data['data'].get('result'):
                logger.warning(f"Query returned no results: {query}")

            return data['data']
        except requests.exceptions.HTTPError as e:
            # Enhanced error diagnostics for HTTP errors
            status_code = e.response.status_code if hasattr(e, 'response') else 'Unknown'

            # Log detailed error information
            logger.error("="*80)
            logger.error(f"HTTP ERROR {status_code}: Failed to query Prometheus")
            logger.error("="*80)
            logger.error(f"Query: {query}")
            logger.error(f"Time range: {start_human} to {end_human}")
            logger.error(f"Unix timestamps: start={start}, end={end}, step={step}")
            logger.error(f"URL: {response.url if hasattr(e, 'response') else url}")

            # Try to extract error details from response body
            if hasattr(e, 'response'):
                try:
                    error_data = e.response.json()
                    if 'error' in error_data:
                        logger.error(f"Mimir error message: {error_data['error']}")
                    if 'errorType' in error_data:
                        logger.error(f"Error type: {error_data['errorType']}")
                except:
                    # If response is not JSON, log raw text
                    logger.error(f"Response body: {e.response.text[:500]}")

            # Provide specific guidance for common errors
            if status_code == 422:
                # Check if this is a "too many chunks" error
                error_body = ""
                if hasattr(e, 'response'):
                    try:
                        error_data = e.response.json()
                        error_body = str(error_data.get('error', '')).lower()
                    except:
                        error_body = e.response.text.lower()

                # Detect chunk-related errors
                if 'chunk' in error_body or 'too many' in error_body:
                    logger.error("="*80)
                    logger.error("NOTE: 422 error due to TOO MANY CHUNKS")
                    logger.error("="*80)
                    logger.error("This query is trying to fetch too much data and exceeded Mimir's chunk limit.")
                    logger.error("")
                    logger.error("SOLUTIONS:")
                    logger.error("  1. Reduce time window: Use shorter --window duration")
                    logger.error("     Example: -w 12h instead of -w 7d")
                    logger.error("")
                    logger.error("  2. Increase step: Add --step parameter with larger value")
                    logger.error(f"     Example: --step {step * 2} (currently using step={step}s)")
                    logger.error("")
                    logger.error("  3. Focus analysis: Use -m to analyze specific metrics instead of top-N")
                    logger.error("     Example: -m my_specific_metric")
                    logger.error("")
                    logger.error(f"Current query details: {(end-start)/3600:.1f}h window, step={step}s")
                    logger.error(f"  = {(end-start)//step} data points per series")
                    logger.error("="*80)
                else:
                    # Original 422 error handling for timezone issues
                    logger.error("NOTE: 422 errors often indicate timestamps in the future due to clock drift or timezone issues.")
                    logger.error("Ensure system clock is accurate and timestamps are in UTC.")
            elif status_code == 400:
                logger.error("NOTE: 400 errors typically indicate invalid query syntax or parameters.")
            elif status_code == 429:
                logger.error("NOTE: 429 errors indicate rate limiting. Too many requests sent to Mimir.")
            elif status_code >= 500:
                logger.error("NOTE: 5xx errors indicate a Mimir/Prometheus server-side issue.")

            logger.error("="*80)
            raise
        except requests.exceptions.RequestException as e:
            # Handle other request exceptions (timeouts, connection errors, etc.)
            logger.error("="*80)
            logger.error(f"REQUEST ERROR: Failed to query Prometheus")
            logger.error("="*80)
            logger.error(f"Query: {query}")
            logger.error(f"Time range: {start_human} to {end_human}")
            logger.error(f"Unix timestamps: start={start}, end={end}, step={step}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {e}")
            logger.error("="*80)
            raise
    
    def get_top_metrics(self, start: int, end: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top N metrics by cardinality with their cardinality values"""
        # Include __ignore_usage__ to prevent interference with Grafana Cloud Adaptive Telemetry
        query = f'topk({top_n}, count by (__name__)({{__name__=~".+", __ignore_usage__=""}}))'

        logger.info(f"Fetching top {top_n} metrics by cardinality...")
        data = self.query_prometheus_with_retry(query, start, end, step=end-start)
        
        metrics = []
        for result in data.get('result', []):
            metric_name = result['metric'].get('__name__', '')
            if metric_name and result.get('values'):
                # Get the cardinality value from the last data point
                cardinality = float(result['values'][-1][1])
                metrics.append((metric_name, cardinality))
        
        # Sort by cardinality (highest to lowest) to ensure proper ordering
        metrics.sort(key=lambda x: x[1], reverse=True)

        return metrics

    def get_metrics_by_pattern(self, start: int, end: int, pattern: str) -> List[Tuple[str, float]]:
        """Get all metrics matching a regex pattern with their cardinality values.

        Uses PromQL's native regex support for efficient server-side filtering.

        Args:
            start: Start timestamp
            end: End timestamp
            pattern: Regex pattern to match metric names (PromQL regex syntax)

        Returns:
            List of tuples: [(metric_name, cardinality_value), ...] sorted by cardinality descending
        """
        # Include __ignore_usage__ to prevent interference with Grafana Cloud Adaptive Telemetry
        query = f'count by (__name__)({{__name__=~"{pattern}", __ignore_usage__=""}})'

        logger.info(f"Fetching metrics matching pattern: {pattern}")
        data = self.query_prometheus_with_retry(query, start, end, step=end-start)

        metrics = []
        for result in data.get('result', []):
            metric_name = result['metric'].get('__name__', '')
            if metric_name and result.get('values'):
                # Get the cardinality value from the last data point
                cardinality = float(result['values'][-1][1])
                metrics.append((metric_name, cardinality))

        # Sort by cardinality (highest to lowest)
        metrics.sort(key=lambda x: x[1], reverse=True)

        return metrics

    def analyze_metric_cardinality(self, metric_name: str, start: int, end: int,
                                   custom_step: Optional[int] = None, target_points: int = 150) -> Dict[str, Dict[str, Any]]:
        """Analyze cardinality for a specific metric by label

        Args:
            metric_name: Name of the metric to analyze
            start: Start timestamp
            end: End timestamp
            custom_step: Optional custom step size in seconds. If None, automatically calculated.
            target_points: Target number of data points for auto-calculation (default: 150)
        """
        results = {}

        # Calculate optimal step if not provided
        if custom_step is None:
            # Use adaptive step based on time range
            step = self.calculate_optimal_step(start, end, target_points=target_points)
        else:
            step = custom_step

        duration_hours = (end - start) / 3600
        data_points = (end - start) // step
        logger.info(f"Using step={step}s ({step//60}min) for {duration_hours:.1f}h analysis window (~{data_points} data points per series)")

        # Warn about potential issues with very long ranges
        if data_points > 500:
            logger.warning(f"Query will fetch ~{data_points} data points per series. This may be slow or hit Mimir limits.")
            logger.warning(f"Consider using --step {step * 2} to reduce data points, or shorter time windows.")
        
        # Get all label names for this metric
        # Include __ignore_usage__ to prevent interference with Grafana Cloud Adaptive Telemetry
        url = urljoin(self.endpoint, f'/api/prom/api/v1/labels')
        params = {
            'match[]': f'{metric_name}{{__ignore_usage__=""}}',
            'start': start,
            'end': end
        }
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            label_names = response.json().get('data', [])
        except Exception as e:
            logger.warning(f"Could not fetch labels for {metric_name}: {e}")
            label_names = []
        
        # Analyze cardinality for each label
        for label in label_names:
            if label.startswith('__'):  # Skip internal labels
                continue

            # Include __ignore_usage__ to prevent interference with Grafana Cloud Adaptive Telemetry
            query = f'count by ({label}) ({metric_name}{{__ignore_usage__=""}})'
            try:
                # Use calculated step to capture cardinality variation while avoiding chunk limits
                # Larger time ranges automatically use larger steps to stay within Mimir limits
                # Retry wrapper will automatically increase step if chunk limit is hit
                data = self.query_prometheus_with_retry(query, start, end, step=step)

                # Calculate max cardinality over the time window to capture bursts
                label_cardinalities = defaultdict(list)
                for result in data.get('result', []):
                    label_value = result['metric'].get(label, 'unknown')
                    values = [float(v[1]) for v in result['values'] if v[1] != 'NaN']
                    if values:
                        # Take max to capture peak cardinality during bursts
                        label_cardinalities[label_value].append(max(values))
                
                # Calculate total unique label values
                total_cardinality = len(label_cardinalities)
                
                # Get top label values by cardinality
                top_values = sorted(
                    [(k, max(v)) for k, v in label_cardinalities.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                results[label] = {
                    'total_cardinality': total_cardinality,
                    'top_values': top_values,
                    'all_values': dict(label_cardinalities)
                }
                
            except Exception as e:
                logger.warning(f"Error analyzing label {label} for metric {metric_name}: {e}")
        
        # Also get overall cardinality
        # Include __ignore_usage__ to prevent interference with Grafana Cloud Adaptive Telemetry
        if label_names:
            label_list = ", ".join(label_names)
            query = f'count(count by (__name__, {label_list}) ({metric_name}{{__ignore_usage__=""}}))'
        else:
            query = f'count(count by (__name__) ({metric_name}{{__ignore_usage__=""}}))'
        try:
            data = self.query_prometheus_with_retry(query, start, end, step=step)
            if data.get('result'):
                values = [float(v[1]) for v in data['result'][0]['values'] if v[1] != 'NaN']
                if values:
                    results['__total__'] = {
                        'max_cardinality': max(values),
                        'avg_cardinality': sum(values) / len(values),
                        'min_cardinality': min(values)
                    }
        except Exception as e:
            logger.warning(f"Error getting total cardinality for {metric_name}: {e}")
        
        return results
    
    def compare_time_windows(self, metric_name: str, window1: Dict, window2: Dict) -> Dict[str, Any]:
        """Compare cardinality between two time windows"""
        comparison = {
            'metric': metric_name,
            'window1': window1,
            'window2': window2,
            'changes': {},
            'label_value_diffs': {}  # New: track which label values appeared/disappeared
        }

        # Get all labels from both windows
        all_labels = set(window1.keys()) | set(window2.keys())

        for label in all_labels:
            if label == '__total__':
                continue

            w1_data = window1.get(label, {})
            w2_data = window2.get(label, {})

            w1_card = w1_data.get('total_cardinality', 0)
            w2_card = w2_data.get('total_cardinality', 0)

            if w1_card > 0:
                change_pct = ((w2_card - w1_card) / w1_card) * 100
            else:
                change_pct = 100 if w2_card > 0 else 0

            comparison['changes'][label] = {
                'before': w1_card,
                'after': w2_card,
                'change': w2_card - w1_card,
                'change_pct': change_pct
            }

            # Compute label value differences
            w1_values = set(w1_data.get('all_values', {}).keys())
            w2_values = set(w2_data.get('all_values', {}).keys())

            new_values = w2_values - w1_values  # Appeared in window2
            removed_values = w1_values - w2_values  # Disappeared from window2
            common_values = w1_values & w2_values  # Existed in both

            # For common values, check which ones changed significantly
            changed_values = []
            for val in common_values:
                w1_count = w1_data['all_values'].get(val, [0])
                w2_count = w2_data['all_values'].get(val, [0])
                # Get max value from array if it's an array
                w1_max = max(w1_count) if isinstance(w1_count, list) else w1_count
                w2_max = max(w2_count) if isinstance(w2_count, list) else w2_count

                # Consider it changed if difference > 10% and absolute change > 5
                if w1_max > 0:
                    change_pct_val = abs(w2_max - w1_max) / w1_max * 100
                    if change_pct_val > 10 and abs(w2_max - w1_max) > 5:
                        changed_values.append({
                            'value': val,
                            'before': w1_max,
                            'after': w2_max,
                            'change': w2_max - w1_max
                        })

            # Store top 20 of each category sorted by impact
            comparison['label_value_diffs'][label] = {
                'new': sorted(
                    [{'value': v, 'cardinality': max(w2_data['all_values'][v]) if isinstance(w2_data['all_values'][v], list) else w2_data['all_values'][v]}
                     for v in new_values],
                    key=lambda x: x['cardinality'],
                    reverse=True
                )[:20],
                'removed': sorted(
                    [{'value': v, 'cardinality': max(w1_data['all_values'][v]) if isinstance(w1_data['all_values'][v], list) else w1_data['all_values'][v]}
                     for v in removed_values],
                    key=lambda x: x['cardinality'],
                    reverse=True
                )[:20],
                'changed': sorted(changed_values, key=lambda x: abs(x['change']), reverse=True)[:20],
                'new_count': len(new_values),
                'removed_count': len(removed_values),
                'changed_count': len(changed_values)
            }

        # Sort by absolute change
        comparison['sorted_changes'] = sorted(
            comparison['changes'].items(),
            key=lambda x: abs(x[1]['change']),
            reverse=True
        )

        return comparison

def generate_json_data_file(analyses: List[Dict], comparisons: Optional[List[Dict]] = None,
                           filename: str = "cardinality_analysis_data.json") -> str:
    """Generate separate JSON file with complete cardinality data for lazy loading

    Args:
        analyses: List of analysis results
        comparisons: Optional list of comparison results
        filename: Output filename for JSON data

    Returns:
        Path to generated JSON file
    """
    data = {
        'analyses': analyses,
        'comparisons': comparisons or []
    }

    with open(filename, 'w') as f:
        json.dump(data, f, default=str)

    logger.info(f"Complete data written to: {filename}")
    return filename

def generate_html_output(analyses: List[Dict], comparisons: Optional[List[Dict]] = None,
                        window: str = "", start_time: str = "", ai_analysis: Optional[str] = None,
                        compare_window: str = "", compare_start_time: str = "",
                        command_line: str = "", top_n_embed: int = 20) -> str:
    """Generate interactive HTML report

    Args:
        analyses: List of analysis results
        comparisons: Optional list of comparison results
        window: Analysis window duration
        start_time: Analysis start time
        ai_analysis: Optional AI analysis text
        compare_window: Comparison window duration
        compare_start_time: Comparison start time
        command_line: Command line used to generate report
        top_n_embed: Number of top label values to embed (default: 20).
                     Set to -1 to embed all data (legacy behavior)

    Returns:
        HTML report string
    """

    # Prepare data for JavaScript
    # If top_n_embed is -1, embed all data (legacy behavior)
    if top_n_embed == -1:
        analyses_json = json.dumps(analyses, default=str)
        comparisons_json = json.dumps(comparisons or [], default=str)
        has_external_data = False
    else:
        # Create trimmed version with only top-N values for embedding
        analyses_trimmed = []
        for analysis in analyses:
            trimmed_analysis = {
                'metric': analysis['metric'],
                'window': analysis.get('window', {}),
                'data': {}
            }

            for label, info in analysis['data'].items():
                if label == '__total__':
                    # Always include total stats
                    trimmed_analysis['data'][label] = info
                else:
                    # Only include top-N values, mark if more data available
                    all_values = info.get('all_values', {})
                    total_values_count = len(all_values)

                    trimmed_info = {
                        'total_cardinality': info.get('total_cardinality', 0),
                        'top_values': info.get('top_values', [])[:top_n_embed],
                        'has_more': total_values_count > top_n_embed,
                        'total_values_count': total_values_count
                    }
                    trimmed_analysis['data'][label] = trimmed_info

            analyses_trimmed.append(trimmed_analysis)

        # Trim comparisons similarly
        comparisons_trimmed = []
        if comparisons:
            for comp in comparisons:
                trimmed_comp = {
                    'metric': comp['metric'],
                    'changes': comp.get('changes', {}),
                    'sorted_changes': comp.get('sorted_changes', []),
                    'label_value_diffs': comp.get('label_value_diffs', {})  # Include diff data
                }
                comparisons_trimmed.append(trimmed_comp)

        analyses_json = json.dumps(analyses_trimmed, default=str)
        comparisons_json = json.dumps(comparisons_trimmed, default=str)
        has_external_data = True
    
    # Format values for the template
    comparison_tab = '<div class="tab" onclick="switchTab(\'comparison\')">Comparison</div>' if comparisons else ''
    start_time_str = f"(starting {start_time})" if start_time else ""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    # Add lazy loading note if using external data
    if has_external_data:
        lazy_loading_note = '<li><strong>Lazy Loading:</strong> Click "Show all values" buttons to load complete data from the external JSON file</li>'
    else:
        lazy_loading_note = ''
    
    # Build analysis details
    analysis_details = []
    # Always show the starting time for consistency
    analysis_details.append(f'<strong>Analysis Window:</strong> {window} (starting {start_time})')
    if comparisons and compare_window:
        analysis_details.append(f'<strong>Comparison Window:</strong> {compare_window} (starting {compare_start_time})')
    
    # Create display strings for comparison clarity
    analysis_window_display = f'{window} (starting {start_time})'
    compare_window_display = f'{compare_window} (starting {compare_start_time})' if compare_window else ''
    analysis_details.append(f'<strong>Generated:</strong> {timestamp}')
    if command_line:
        # Escape the command line for HTML
        import html
        escaped_command = html.escape(command_line)
        analysis_details.append(f'<strong>Command Used:</strong> <code style="background: #f5f5f5; padding: 4px 8px; border-radius: 4px; font-family: monospace;">{escaped_command}</code>')
    
    analysis_details_html = '<br>'.join(analysis_details)
    
    # Prepare AI analysis section if available
    ai_section = ""
    if ai_analysis:
        try:
            from cardinality_analyzer_ai_analysis import generate_ai_report_section
            ai_section = generate_ai_report_section(ai_analysis)
        except ImportError as e:
            logger.error(f"Failed to import AI report section generator: {e}")
            ai_section = f'<div class="info-box" style="background: #ffebee; border-color: #f44336;">AI analysis was requested but module not available. Install with: pip install -r requirements.txt</div>'
        except Exception as e:
            logger.error(f"Failed to generate AI report section: {e}")
            # Fallback to simple text display
            import html
            ai_section = f'''<div class="ai-analysis-section">
                <h2>AI Analysis</h2>
                <div class="ai-content" style="white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    {html.escape(ai_analysis)}
                </div>
            </div>'''
    
    # Use string replacement instead of format to avoid conflicts with JavaScript template literals
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metric Cardinality Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .metric-section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }
        th:hover {
            background: #e0e0e0;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        .filter-container {
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .filter-container input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 300px;
            font-size: 14px;
        }
        .positive-change {
            color: #d32f2f;
            font-weight: 600;
        }
        .negative-change {
            color: #388e3c;
            font-weight: 600;
        }
        .tabs {
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab:hover {
            background: #f5f5f5;
        }
        .tab.active {
            border-bottom-color: #2196f3;
            color: #2196f3;
            font-weight: 600;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            font-weight: normal;
        }
        .summary-card .value {
            font-size: 32px;
            font-weight: 600;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Metric Cardinality Analysis Report</h1>
        <div style="margin: 20px 0; line-height: 1.8;">
            {analysis_details_html}
        </div>
        
        <div class="info-box">
            <h3>How to Use This Report</h3>
            <ul>
                <li><strong>Cardinality</strong> refers to the number of unique time series for a metric</li>
                <li><strong>High cardinality labels</strong> are those with many unique values, which can cause metric spikes</li>
                <li><strong>Metrics are sorted</strong> from highest to lowest cardinality (using Prometheus topk function)</li>
                <li>Click on table headers to sort by that column</li>
                <li>Use the filter box to search for specific metrics or labels</li>
                <li>Charts show the top contributors to cardinality for each metric</li>
                <li>In comparison mode, positive changes (red) indicate increased cardinality</li>
                {lazy_loading_note}
            </ul>
        </div>
        
        {ai_section}
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('analysis')">Cardinality Analysis</div>
            {comparison_tab}
        </div>
        
        <div id="analysis-tab" class="tab-content active">
            <!-- Metric Visibility Controls -->
            <div id="metric-controls" style="margin-bottom: 20px; padding: 15px; background: #f9f9f9; border-radius: 4px; border: 1px solid #e0e0e0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong style="font-size: 14px;">Metric Visibility</strong>
                    <div>
                        <button onclick="toggleAllMetrics(true)" style="padding: 4px 8px; margin-right: 5px; background: #2196f3; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;">Select All</button>
                        <button onclick="toggleAllMetrics(false)" style="padding: 4px 8px; margin-right: 5px; background: #757575; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;">Deselect All</button>
                        <button onclick="showTopN(5)" style="padding: 4px 8px; margin-right: 5px; background: #ff9800; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;">Top 5</button>
                        <button onclick="showTopN(10)" style="padding: 4px 8px; background: #ff9800; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;">Top 10</button>
                    </div>
                </div>
                <div id="metric-checkbox-list" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; max-height: 150px; overflow-y: auto;"></div>
            </div>

            <div class="filter-container">
                <input type="text" id="filter" placeholder="Filter metrics or labels..." onkeyup="filterResults()">
                <button onclick="exportFilteredData()" style="margin-left: 15px; padding: 8px 16px; background: #4caf50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                    Export Filtered Data to CSV
                </button>
            </div>

            <div id="analysis-content"></div>
        </div>
        
        <div id="comparison-tab" class="tab-content">
            <div id="comparison-content"></div>
        </div>
    </div>
    
    <script>
        const analysisData = {analyses_json};
        const comparisonData = {comparisons_json};
        const hasExternalData = {has_external_data_js};
        const dataFileName = 'cardinality_analysis_data.json';
        let charts = [];
        let fullDataCache = null; // Cache for loaded external data
        let loadingData = false;

        // Helper function to escape HTML special characters
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Lazy load complete data from external JSON file
        async function loadCompleteData() {
            if (fullDataCache) {
                return fullDataCache;
            }

            if (loadingData) {
                // Wait for ongoing load
                while (loadingData) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
                return fullDataCache;
            }

            if (!hasExternalData) {
                console.warn('No external data file available');
                return null;
            }

            loadingData = true;
            try {
                const response = await fetch(dataFileName);
                if (!response.ok) {
                    throw new Error(`Failed to load ${dataFileName}: ${response.status} ${response.statusText}`);
                }
                fullDataCache = await response.json();
                console.log('Complete data loaded successfully');
                return fullDataCache;
            } catch (error) {
                console.error('Error loading complete data:', error);
                let errorMsg = `Failed to load complete data: ${error.message}.\\n\\n`;

                // Check if this is likely a CORS/file:// protocol issue
                if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                    errorMsg += `This is likely because you're opening the HTML file directly (file:// protocol).\\n\\n`;
                    errorMsg += `To fix this, serve the files via HTTP:\\n`;
                    errorMsg += `1. Open terminal in the same directory\\n`;
                    errorMsg += `2. Run: python3 -m http.server 8000\\n`;
                    errorMsg += `3. Open: http://localhost:8000/cardinality_analysis.html\\n\\n`;
                    errorMsg += `See README.md for more details.`;
                } else {
                    errorMsg += `Ensure ${dataFileName} is in the same directory as this HTML file.`;
                }

                alert(errorMsg);
                return null;
            } finally {
                loadingData = false;
            }
        }

        // Load all values for a specific metric and label
        async function loadAllValues(event) {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Loading...';

            // Read data from button attributes
            const metricIdx = parseInt(btn.dataset.metricIdx, 10);
            const labelName = btn.dataset.label;
            const tableIdx = parseInt(btn.dataset.tableIdx, 10);

            // Look up the metric name from analysisData
            const analysis = analysisData[metricIdx];
            if (!analysis) {
                console.error(`Invalid metric index: ${metricIdx}`);
                btn.textContent = 'Error';
                btn.disabled = false;
                return;
            }
            const metricName = analysis.metric;

            try {
                const data = await loadCompleteData();
                if (!data) {
                    btn.textContent = 'Load Failed';
                    return;
                }

                // Find the metric in the complete data
                const fullAnalysis = data.analyses.find(a => a.metric === metricName);
                if (!fullAnalysis || !fullAnalysis.data[labelName]) {
                    throw new Error(`Data not found for ${metricName}.${labelName}`);
                }

                const labelInfo = fullAnalysis.data[labelName];
                const allValues = labelInfo.all_values || {};

                // Find the row and update it with all values
                const table = document.getElementById(`table-${tableIdx}`);
                const rows = table.querySelectorAll('tbody tr');

                for (const row of rows) {
                    if (row.dataset.label === labelName) {
                        // Create expanded values display
                        const sortedValues = Object.entries(allValues)
                            .map(([val, counts]) => [val, Array.isArray(counts) ? Math.max(...counts) : counts])
                            .sort((a, b) => b[1] - a[1]);

                        const valuesStr = sortedValues.slice(0, 50)
                            .map(([val, count]) => `${escapeHtml(val)} (${count})`)
                            .join(', ');

                        const remaining = sortedValues.length - 50;
                        const finalStr = remaining > 0
                            ? `${valuesStr} ... and ${remaining} more`
                            : valuesStr;

                        row.cells[2].innerHTML = finalStr;
                        btn.remove(); // Remove the button after loading
                        break;
                    }
                }
            } catch (error) {
                console.error('Error loading all values:', error);
                btn.textContent = 'Error';
                btn.disabled = false;
            }
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (tab === 'analysis') {
                document.querySelector('.tab:nth-child(1)').classList.add('active');
                document.getElementById('analysis-tab').classList.add('active');
            } else {
                document.querySelector('.tab:nth-child(2)').classList.add('active');
                document.getElementById('comparison-tab').classList.add('active');
            }
        }
        
        function renderAnalysis() {
            const container = document.getElementById('analysis-content');
            console.log('renderAnalysis called, container:', container);
            let html = '';

            // Add note about metric count and sorting
            html += '<div style="margin-bottom: 20px; padding: 10px; background: #e3f2fd; border-radius: 4px;">';
            html += '<strong>Showing ' + analysisData.length + ' metrics</strong> sorted by highest to lowest cardinality';
            html += '</div>';
            console.log('Generated sorting note HTML');

            // Populate metric visibility checkboxes
            populateMetricCheckboxes();
            
            analysisData.forEach((analysis, idx) => {
                const metricName = analysis.metric;
                const metricNameEscaped = escapeHtml(metricName);
                const data = analysis.data;
                const totalInfo = data.__total__ || {};

                html += `
                    <div class="metric-section" data-metric="${metricNameEscaped}">
                        <h2>${metricNameEscaped}</h2>
                        <div class="summary-cards">
                            <div class="summary-card">
                                <h3>Max Cardinality</h3>
                                <div class="value">${totalInfo.max_cardinality || 'N/A'}</div>
                            </div>
                            <div class="summary-card">
                                <h3>Avg Cardinality</h3>
                                <div class="value">${totalInfo.avg_cardinality ? totalInfo.avg_cardinality.toFixed(0) : 'N/A'}</div>
                            </div>
                            <div class="summary-card">
                                <h3>Labels Analyzed</h3>
                                <div class="value">${Object.keys(data).filter(k => k !== '__total__').length}</div>
                            </div>
                        </div>
                        
                        <div class="chart-container">
                            <canvas id="chart-${idx}"></canvas>
                        </div>
                        
                        <h3>Cardinality by Label</h3>
                        <table id="table-${idx}">
                            <thead>
                                <tr>
                                    <th onclick="sortTable(${idx}, 0)">Label Name ⬍</th>
                                    <th onclick="sortTable(${idx}, 1)">Total Cardinality ⬍</th>
                                    <th onclick="sortTable(${idx}, 2)">Top Values ⬍</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                // Get all labels except __total__ and sort by cardinality
                const labelsArray = Object.entries(data)
                    .filter(([label, info]) => label !== '__total__')
                    .sort((a, b) => {
                        const cardA = a[1].total_cardinality || 0;
                        const cardB = b[1].total_cardinality || 0;
                        return cardB - cardA; // Sort highest to lowest
                    });
                
                // Render the sorted labels
                labelsArray.forEach(([label, info]) => {
                    const labelEscaped = escapeHtml(label);
                    const topValues = info.top_values || [];
                    const displayCount = Math.min(5, topValues.length);
                    const topValuesStr = topValues.slice(0, displayCount)
                        .map(([val, count]) => `${escapeHtml(val)} (${count})`)
                        .join(', ');

                    // Add "Show all" button if there are more values available
                    let showAllButton = '';
                    if (info.has_more && hasExternalData) {
                        const totalCount = info.total_values_count || info.total_cardinality || 0;
                        showAllButton = `<br><button class="load-all-values-btn"
                            data-metric-idx="${idx}"
                            data-label="${labelEscaped}"
                            data-table-idx="${idx}"
                            style="margin-top: 8px; padding: 4px 12px; background: #2196f3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
                            Show all ${totalCount} values
                        </button>`;
                    }

                    html += `
                        <tr data-label="${labelEscaped}">
                            <td>${labelEscaped}</td>
                            <td>${info.total_cardinality || 0}</td>
                            <td>${topValuesStr}${showAllButton}</td>
                        </tr>
                    `;
                });
                
                html += `
                            </tbody>
                        </table>
                    </div>
                `;
            });
            
            container.innerHTML = html;
            
            // Render charts
            analysisData.forEach((analysis, idx) => {
                renderChart(analysis, idx);
            });
        }
        
        function renderChart(analysis, idx) {
            const ctx = document.getElementById(`chart-${idx}`).getContext('2d');
            const data = analysis.data;
            
            // Prepare chart data
            const labels = [];
            const values = [];
            
            Object.entries(data).forEach(([label, info]) => {
                if (label !== '__total__' && info.total_cardinality) {
                    labels.push(label);
                    values.push(info.total_cardinality);
                }
            });
            
            // Sort and take top 10
            const sorted = labels.map((label, i) => ({label, value: values[i]}))
                .sort((a, b) => b.value - a.value)
                .slice(0, 10);
            
            const chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: sorted.map(item => item.label),
                    datasets: [{
                        label: 'Cardinality',
                        data: sorted.map(item => item.value),
                        backgroundColor: 'rgba(33, 150, 243, 0.6)',
                        borderColor: 'rgba(33, 150, 243, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `Top Labels by Cardinality - ${analysis.metric}`
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Unique Values'
                            }
                        }
                    }
                }
            });
            
            charts.push(chart);
        }
        
        function renderComparison() {
            const container = document.getElementById('comparison-content');
            if (!comparisonData || comparisonData.length === 0) {
                container.innerHTML = '<p>No comparison data available</p>';
                return;
            }
            
            let html = '<h2>Cardinality Changes Between Time Windows</h2>';
            html += '<div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px;">';
            html += '<strong>Before:</strong> {compare_window_display}<br>';
            html += '<strong>After:</strong> {analysis_window_display}';
            html += '</div>';
            
            // Add note about metric count and sorting
            html += '<div style="margin-bottom: 20px; padding: 10px; background: #e3f2fd; border-radius: 4px;">';
            html += '<strong>Showing ' + comparisonData.length + ' metrics</strong> sorted by highest to lowest cardinality';
            html += '</div>';
            
            comparisonData.forEach((comp, idx) => {
                const compMetricEscaped = escapeHtml(comp.metric);

                html += `
                    <div class="metric-section">
                        <h3>${compMetricEscaped}</h3>

                        <!-- Comparison Chart -->
                        <div class="chart-container">
                            <canvas id="comparison-chart-${idx}"></canvas>
                        </div>

                        <!-- Label Value Diff View -->
                        <div id="diff-view-${idx}" style="margin-top: 30px;"></div>

                        <h4 style="margin-top: 30px;">Detailed Changes by Label</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th>Label</th>
                                    <th>Before</th>
                                    <th>After</th>
                                    <th>Change</th>
                                    <th>Change %</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                // Sort changes by 'after' cardinality (highest to lowest)
                const changesArray = [...comp.sorted_changes].sort((a, b) => {
                    const afterA = a[1].after || 0;
                    const afterB = b[1].after || 0;
                    return afterB - afterA;
                });

                changesArray.forEach(([label, change]) => {
                    const labelEscaped = escapeHtml(label);
                    const changeClass = change.change > 0 ? 'positive-change' :
                                      change.change < 0 ? 'negative-change' : '';

                    html += `
                        <tr>
                            <td>${labelEscaped}</td>
                            <td>${change.before}</td>
                            <td>${change.after}</td>
                            <td class="${changeClass}">${change.change > 0 ? '+' : ''}${change.change}</td>
                            <td class="${changeClass}">${change.change_pct > 0 ? '+' : ''}${change.change_pct.toFixed(1)}%</td>
                        </tr>
                    `;
                });
                
                html += `
                            </tbody>
                        </table>
                    </div>
                `;
            });

            container.innerHTML = html;

            // Render comparison charts and diff views after HTML is in place
            renderComparisonCharts();
            renderDiffViews();
        }

        function renderComparisonCharts() {
            if (!comparisonData || comparisonData.length === 0) {
                return;
            }

            comparisonData.forEach((comp, idx) => {
                const ctx = document.getElementById(`comparison-chart-${idx}`);
                if (!ctx) return;

                // Prepare data for before/after comparison
                const labels = [];
                const beforeData = [];
                const afterData = [];

                // Sort by 'after' cardinality and take top 10 labels
                const sortedChanges = [...(comp.sorted_changes || [])].sort((a, b) => {
                    const afterA = a[1].after || 0;
                    const afterB = b[1].after || 0;
                    return afterB - afterA;
                }).slice(0, 10);

                sortedChanges.forEach(([label, change]) => {
                    labels.push(label);
                    beforeData.push(change.before || 0);
                    afterData.push(change.after || 0);
                });

                // Create grouped bar chart showing before vs after
                const chart = new Chart(ctx.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Before',
                                data: beforeData,
                                backgroundColor: 'rgba(96, 125, 139, 0.6)',
                                borderColor: 'rgba(96, 125, 139, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'After',
                                data: afterData,
                                backgroundColor: 'rgba(244, 67, 54, 0.6)',
                                borderColor: 'rgba(244, 67, 54, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: `Comparison: Before vs After - ${comp.metric}`,
                                font: {
                                    size: 16
                                }
                            },
                            legend: {
                                display: true,
                                position: 'top'
                            },
                            tooltip: {
                                callbacks: {
                                    afterLabel: function(context) {
                                        const labelIdx = context.dataIndex;
                                                        const before = beforeData[labelIdx];
                                        const after = afterData[labelIdx];
                                        const change = after - before;
                                        const changePct = before > 0 ? ((change / before) * 100).toFixed(1) : '∞';
                                        return `Change: ${change > 0 ? '+' : ''}${change} (${change > 0 ? '+' : ''}${changePct}%)`;
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Cardinality'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Labels'
                                }
                            }
                        }
                    }
                });

                charts.push(chart);
            });
        }

        function renderDiffViews() {
            if (!comparisonData || comparisonData.length === 0) {
                return;
            }

            comparisonData.forEach((comp, idx) => {
                const container = document.getElementById(`diff-view-${idx}`);
                if (!container) return;

                const diffs = comp.label_value_diffs || {};
                let html = '<h4>Label Value Changes</h4>';
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">';

                // For each label, show new/removed/changed values
                Object.entries(diffs).forEach(([label, diff]) => {
                    const hasChanges = (diff.new_count || 0) > 0 || (diff.removed_count || 0) > 0 || (diff.changed_count || 0) > 0;

                    if (!hasChanges) return;  // Skip labels with no changes

                    const labelEscaped = escapeHtml(label);
                    html += `<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background: #fafafa;">`;
                    html += `<h5 style="margin-top: 0; margin-bottom: 10px; color: #333;">${labelEscaped}</h5>`;

                    // New values (green)
                    if (diff.new_count > 0) {
                        html += `<div style="margin-bottom: 10px;">`;
                        html += `<strong style="color: #388e3c;">+ ${diff.new_count} New Value${diff.new_count !== 1 ? 's' : ''}</strong>`;
                        if (diff.new && diff.new.length > 0) {
                            html += `<ul style="margin: 5px 0; padding-left: 20px; font-size: 12px; max-height: 150px; overflow-y: auto;">`;
                            diff.new.slice(0, 5).forEach(item => {
                                html += `<li style="color: #2e7d32;">${escapeHtml(item.value)} (${item.cardinality})</li>`;
                            });
                            if (diff.new.length > 5) {
                                html += `<li style="color: #666; font-style: italic;">... and ${diff.new.length - 5} more</li>`;
                            }
                            html += `</ul>`;
                        }
                        html += `</div>`;
                    }

                    // Removed values (red)
                    if (diff.removed_count > 0) {
                        html += `<div style="margin-bottom: 10px;">`;
                        html += `<strong style="color: #d32f2f;">- ${diff.removed_count} Removed Value${diff.removed_count !== 1 ? 's' : ''}</strong>`;
                        if (diff.removed && diff.removed.length > 0) {
                            html += `<ul style="margin: 5px 0; padding-left: 20px; font-size: 12px; max-height: 150px; overflow-y: auto;">`;
                            diff.removed.slice(0, 5).forEach(item => {
                                html += `<li style="color: #c62828;">${escapeHtml(item.value)} (${item.cardinality})</li>`;
                            });
                            if (diff.removed.length > 5) {
                                html += `<li style="color: #666; font-style: italic;">... and ${diff.removed.length - 5} more</li>`;
                            }
                            html += `</ul>`;
                        }
                        html += `</div>`;
                    }

                    // Changed values (yellow/orange)
                    if (diff.changed_count > 0) {
                        html += `<div style="margin-bottom: 10px;">`;
                        html += `<strong style="color: #f57c00;">~ ${diff.changed_count} Changed Value${diff.changed_count !== 1 ? 's' : ''}</strong>`;
                        if (diff.changed && diff.changed.length > 0) {
                            html += `<ul style="margin: 5px 0; padding-left: 20px; font-size: 12px; max-height: 150px; overflow-y: auto;">`;
                            diff.changed.slice(0, 5).forEach(item => {
                                const changeText = item.change > 0 ? `+${item.change}` : `${item.change}`;
                                html += `<li style="color: #e65100;">${escapeHtml(item.value)}: ${item.before} → ${item.after} (${changeText})</li>`;
                            });
                            if (diff.changed.length > 5) {
                                html += `<li style="color: #666; font-style: italic;">... and ${diff.changed.length - 5} more</li>`;
                            }
                            html += `</ul>`;
                        }
                        html += `</div>`;
                    }

                    html += `</div>`;
                });

                html += '</div>';

                // Add message if no changes detected
                const totalChanges = Object.values(diffs).reduce((sum, diff) =>
                    sum + (diff.new_count || 0) + (diff.removed_count || 0) + (diff.changed_count || 0), 0);

                if (totalChanges === 0) {
                    html = '<div style="padding: 15px; background: #e8f5e9; border-radius: 4px; color: #2e7d32; margin-top: 15px;">';
                    html += '<strong>✓ No significant label value changes detected</strong><br>';
                    html += '<span style="font-size: 14px;">All labels have the same values in both windows.</span>';
                    html += '</div>';
                }

                container.innerHTML = html;
            });
        }

        function sortTable(tableIdx, columnIdx) {
            const table = document.getElementById(`table-${tableIdx}`);
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Determine sort direction
            const th = table.querySelectorAll('th')[columnIdx];
            const isAsc = th.textContent.includes('⬍');
            
            // Update header arrows
            table.querySelectorAll('th').forEach(header => {
                header.textContent = header.textContent.replace(/[⬍⬆]/g, '⬍');
            });
            th.textContent = th.textContent.replace('⬍', isAsc ? '⬆' : '⬍');
            
            // Sort rows
            rows.sort((a, b) => {
                const aVal = a.cells[columnIdx].textContent;
                const bVal = b.cells[columnIdx].textContent;
                
                if (columnIdx === 1) { // Numeric column
                    return isAsc ? 
                        parseInt(aVal) - parseInt(bVal) : 
                        parseInt(bVal) - parseInt(aVal);
                } else { // Text columns
                    return isAsc ? 
                        aVal.localeCompare(bVal) : 
                        bVal.localeCompare(aVal);
                }
            });
            
            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
        }
        
        function filterResults() {
            const filterValue = document.getElementById('filter').value.toLowerCase();

            // Filter metric sections
            document.querySelectorAll('.metric-section').forEach(section => {
                const metricName = section.dataset.metric;
                const showSection = !filterValue || metricName.toLowerCase().includes(filterValue);

                if (showSection) {
                    section.style.display = 'block';

                    // Filter rows within the section
                    section.querySelectorAll('tbody tr').forEach(row => {
                        const label = row.dataset.label;
                        const showRow = !filterValue ||
                                      metricName.toLowerCase().includes(filterValue) ||
                                      (label && label.toLowerCase().includes(filterValue));
                        row.style.display = showRow ? '' : 'none';
                    });
                } else {
                    section.style.display = 'none';
                }
            });
        }

        function exportFilteredData() {
            const csvRows = [];
            csvRows.push(['Metric', 'Label', 'Total Cardinality', 'Top Values'].join(','));

            // Get all visible metric sections
            document.querySelectorAll('.metric-section').forEach(section => {
                if (section.style.display === 'none') return;  // Skip hidden sections

                const metricName = section.dataset.metric;
                if (!metricName) return;  // Skip sections without metric name

                // Get all visible rows in this section
                section.querySelectorAll('tbody tr').forEach(row => {
                    if (row.style.display === 'none') return;  // Skip hidden rows

                    const cells = row.querySelectorAll('td');
                    if (cells.length < 3) return;

                    const label = cells[0].textContent.trim();
                    const cardinality = cells[1].textContent.trim();
                    const topValues = cells[2].textContent.trim()
                        .replace(/Show all \\d+ values/g, '')  // Remove "Show all" button text
                        .trim();

                    // Escape values for CSV (handle commas and quotes)
                    const escapeCsv = (str) => {
                        if (str.includes(',') || str.includes('"') || str.includes('\\n')) {
                            return `"${str.replace(/"/g, '""')}"`;
                        }
                        return str;
                    };

                    csvRows.push([
                        escapeCsv(metricName),
                        escapeCsv(label),
                        cardinality,
                        escapeCsv(topValues)
                    ].join(','));
                });
            });

            // Create CSV content
            const csvContent = csvRows.join('\\n');

            // Create download link
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);

            // Generate filename with timestamp
            const now = new Date();
            const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
            const filename = `cardinality_filtered_${timestamp}.csv`;

            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            console.log(`Exported ${csvRows.length - 1} rows to ${filename}`);
        }

        function populateMetricCheckboxes() {
            const checkboxContainer = document.getElementById('metric-checkbox-list');
            if (!checkboxContainer) return;

            let html = '';
            analysisData.forEach((analysis, idx) => {
                const metricName = analysis.metric;
                const metricNameEscaped = escapeHtml(metricName);
                const isChecked = localStorage.getItem(`metric_visible_${metricName}`) !== 'false';

                html += `
                    <label style="display: flex; align-items: center; cursor: pointer; user-select: none;">
                        <input type="checkbox"
                               id="metric-checkbox-${idx}"
                               data-metric-idx="${idx}"
                               ${isChecked ? 'checked' : ''}
                               onchange="toggleMetricVisibilityByIndex(${idx})"
                               style="margin-right: 6px; cursor: pointer;">
                        <span style="font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${metricNameEscaped}">
                            ${metricNameEscaped}
                        </span>
                    </label>
                `;
            });

            checkboxContainer.innerHTML = html;

            // Apply initial visibility from localStorage
            analysisData.forEach((analysis, idx) => {
                const metricName = analysis.metric;
                const isVisible = localStorage.getItem(`metric_visible_${metricName}`) !== 'false';
                if (!isVisible) {
                    toggleMetricVisibilityByIndex(idx, false, false);  // Don't save again
                }
            });
        }

        function toggleMetricVisibilityByIndex(idx, forceState = null, saveToStorage = true) {
            const analysis = analysisData[idx];
            if (!analysis) return;

            const metricName = analysis.metric;
            toggleMetricVisibility(metricName, forceState, saveToStorage);
        }

        function toggleMetricVisibility(metricName, forceState = null, saveToStorage = true) {
            const sections = document.querySelectorAll('.metric-section');
            const checkbox = document.querySelector(`input[data-metric-idx]`);

            // Find sections matching this metric
            const matchingSections = Array.from(sections).filter(section =>
                section.dataset.metric === metricName
            );

            // Find checkbox by looking through all checkboxes
            let matchingCheckbox = null;
            analysisData.forEach((analysis, idx) => {
                if (analysis.metric === metricName) {
                    matchingCheckbox = document.getElementById(`metric-checkbox-${idx}`);
                }
            });

            const newState = forceState !== null ? forceState : (matchingCheckbox ? matchingCheckbox.checked : true);

            matchingSections.forEach(section => {
                section.style.display = newState ? 'block' : 'none';
            });

            if (matchingCheckbox && forceState !== null) {
                matchingCheckbox.checked = newState;
            }

            // Save to localStorage
            if (saveToStorage) {
                localStorage.setItem(`metric_visible_${metricName}`, newState.toString());
            }
        }

        function toggleAllMetrics(show) {
            analysisData.forEach(analysis => {
                toggleMetricVisibility(analysis.metric, show, true);
            });
        }

        function showTopN(n) {
            analysisData.forEach((analysis, idx) => {
                const shouldShow = idx < n;
                toggleMetricVisibility(analysis.metric, shouldShow, true);
            });
        }

        // Initialize
        console.log('Initializing with', analysisData.length, 'analyses and', comparisonData.length, 'comparisons');

        // Set up event delegation for "Show all values" buttons
        document.addEventListener('click', function(event) {
            if (event.target && event.target.classList.contains('load-all-values-btn')) {
                loadAllValues(event);
            }
        });

        renderAnalysis();
        renderComparison();
    </script>
</body>
</html>
"""
    
    # Replace placeholders without using format() to avoid conflicts with JavaScript ${} syntax
    html_output = html_template.replace('{analysis_details_html}', analysis_details_html)
    html_output = html_output.replace('{analyses_json}', analyses_json)
    html_output = html_output.replace('{comparisons_json}', comparisons_json)
    html_output = html_output.replace('{comparison_tab}', comparison_tab)
    html_output = html_output.replace('{ai_section}', ai_section)
    html_output = html_output.replace('{analysis_window_display}', analysis_window_display)
    html_output = html_output.replace('{compare_window_display}', compare_window_display)
    html_output = html_output.replace('{has_external_data_js}', 'true' if has_external_data else 'false')
    html_output = html_output.replace('{lazy_loading_note}', lazy_loading_note)

    return html_output

def generate_csv_output(analyses: List[Dict], comparisons: Optional[List[Dict]] = None, 
                        filename: str = "cardinality_analysis.csv"):
    """Generate CSV output with optional comparison data"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Determine headers based on whether we have comparison data
        if comparisons:
            writer.writerow(['Metric', 'Label', 'Cardinality_Before', 'Cardinality_After', 
                           'Change', 'Change_Percent', 'Top_Values_After'])
        else:
            writer.writerow(['Metric', 'Label', 'Cardinality', 'Top Values'])
        
        if comparisons:
            # Output comparison data
            for comp in comparisons:
                metric_name = comp['metric']
                
                # Get the corresponding analysis data for top values
                analysis_data = next((a['data'] for a in analyses if a['metric'] == metric_name), {})
                
                for label, change in comp['changes'].items():
                    # Get top values from the analysis (after) data
                    label_info = analysis_data.get(label, {})
                    top_values = label_info.get('top_values', [])
                    top_values_str = '; '.join([f"{val}:{count}" for val, count in top_values[:5]])
                    
                    writer.writerow([
                        metric_name,
                        label,
                        change['before'],
                        change['after'],
                        change['change'],
                        f"{change['change_pct']:.1f}",
                        top_values_str
                    ])
        else:
            # Output regular analysis data
            for analysis in analyses:
                metric_name = analysis['metric']
                data = analysis['data']
                
                for label, info in data.items():
                    if label == '__total__':
                        continue
                        
                    cardinality = info.get('total_cardinality', 0)
                    top_values = info.get('top_values', [])
                    top_values_str = '; '.join([f"{val}:{count}" for val, count in top_values[:5]])
                    
                    writer.writerow([metric_name, label, cardinality, top_values_str])
    
    return filename

def generate_cli_output(analyses: List[Dict], comparisons: Optional[List[Dict]] = None):
    """Generate CLI output"""
    print("\n" + "="*80)
    print("METRIC CARDINALITY ANALYSIS")
    print("="*80)
    
    for analysis in analyses:
        metric_name = analysis['metric']
        data = analysis['data']
        total_info = data.get('__total__', {})
        
        print(f"\nMetric: {metric_name}")
        print("-" * 40)
        
        if total_info:
            print(f"  Max Cardinality: {total_info.get('max_cardinality', 'N/A')}")
            print(f"  Avg Cardinality: {total_info.get('avg_cardinality', 'N/A'):.0f}")
        
        print("\n  Top Labels by Cardinality:")
        
        # Sort labels by cardinality
        sorted_labels = sorted(
            [(k, v) for k, v in data.items() if k != '__total__'],
            key=lambda x: x[1].get('total_cardinality', 0),
            reverse=True
        )[:10]
        
        for label, info in sorted_labels:
            cardinality = info.get('total_cardinality', 0)
            print(f"    {label}: {cardinality} unique values")
            
            # Show top 3 values
            top_values = info.get('top_values', [])[:3]
            if top_values:
                values_str = ', '.join([f"{val} ({count})" for val, count in top_values])
                print(f"      Top values: {values_str}")
    
    if comparisons:
        print("\n" + "="*80)
        print("COMPARISON BETWEEN TIME WINDOWS")
        print("="*80)
        
        for comp in comparisons:
            print(f"\nMetric: {comp['metric']}")
            print("-" * 40)
            
            # Show top changes
            for label, change in comp['sorted_changes'][:10]:
                if abs(change['change']) == 0:
                    continue
                    
                sign = '+' if change['change'] > 0 else ''
                print(f"  {label}: {change['before']} → {change['after']} "
                      f"({sign}{change['change']}, {sign}{change['change_pct']:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze metric cardinality in Grafana Cloud Mimir to investigate spike issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Analyze last hour
  %(prog)s -w 1h

  # Compare last hour with the hour before
  %(prog)s -w 1h --compare --compare-window 1h

  # Analyze specific time period (ALWAYS use UTC with 'Z' suffix)
  %(prog)s -w 1h -s 2024-01-15T14:00:00Z

  # Focus on specific metric with AI analysis
  %(prog)s -w 1h -m kubernetes_pod_info --ai-analysis

  # Compare specific time windows (both in UTC)
  %(prog)s -w 1h -s 2024-01-15T15:00:00Z --compare --compare-window 1h --compare-start-time 2024-01-15T14:00:00Z

IMPORTANT: All timestamps must be in UTC timezone. Use 'Z' suffix or '+00:00' to specify UTC explicitly.
"""
    )
    
    parser.add_argument('-w', '--window', '--duration', required=True,
                       help='Time window duration (e.g., 30m, 1h, 24h, 7d)')
    parser.add_argument('-s', '--start-time', '--from',
                       help='Start time in ISO format using UTC timezone. '
                            'IMPORTANT: Always use UTC! Examples: '
                            '2024-01-15T10:00:00Z or 2024-01-15T10:00:00+00:00. '
                            'If timezone is omitted, UTC is assumed with a warning. '
                            'If not provided, uses current UTC time minus window duration. '
                            'This sets the beginning of the analysis window.')
    parser.add_argument('-m', '--metric',
                       help='Specific metric name or regex pattern to analyze (e.g., "foo_.*" for metrics starting with foo_). If not provided, analyzes top metrics')
    parser.add_argument('-mf', '--metrics-file',
                       help='Path to file containing metric names (one per line). Supports # comments and blank lines. Can be combined with -m option')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top metrics to analyze when no specific metric is provided (default: 20)')
    parser.add_argument('-o', '--output', default='html',
                       choices=['cli', 'csv', 'html', 'all'],
                       help='Output format (default: html)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')

    # Performance tuning options
    parser.add_argument('--step', type=int,
                       help='Query step interval in seconds. Larger values reduce data points but lower temporal resolution. '
                            'If not specified, automatically calculated based on time window to avoid chunk limits. '
                            'Examples: 60 (1min), 300 (5min), 3600 (1hour). '
                            'Use larger steps for longer time ranges to prevent "too many chunks" errors.')
    parser.add_argument('--max-points', type=int, default=150,
                       help='Target number of data points per series when auto-calculating step (default: 150). '
                            'Lower values use larger steps and fetch less data, reducing chunk limit errors. '
                            'Higher values provide better temporal resolution but may hit Mimir limits.')

    # Comparison options
    parser.add_argument('--compare', action='store_true',
                       help='Enable comparison mode to analyze changes between two time periods')
    parser.add_argument('--compare-window', '--compare-duration',
                       help='Duration of the comparison window (e.g., 1h, same format as --window). '
                            'Required if --compare is used')
    parser.add_argument('--compare-start-time', '--compare-from',
                       help='Start time for the comparison window in UTC (same format as --start-time). '
                            'Examples: 2024-01-15T10:00:00Z. '
                            'If not provided with --compare, defaults to immediately before the main analysis window')
    
    # AI analysis option
    parser.add_argument('--ai-analysis', action='store_true',
                       help='Generate AI-powered analysis using OpenAI Responses API (requires OPENAI_API_KEY env var)')

    # HTML optimization options
    parser.add_argument('--top-n-embed', type=int, default=20,
                       help='Number of top label values to embed in HTML file for initial rendering (default: 20). '
                            'Remaining data is stored in a separate JSON file and loaded on-demand. '
                            'Higher values increase HTML file size but show more data without lazy loading. '
                            'Set to -1 to embed all data (legacy behavior, not recommended for large datasets).')

    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Build command line string for reproducibility
    command_parts = [sys.argv[0]]
    command_parts.extend(sys.argv[1:])
    command_line = ' '.join(command_parts)
    
    # Validate environment variables
    endpoint = os.getenv('PROMETHEUS_ENDPOINT')
    username = os.getenv('PROMETHEUS_USERNAME')
    api_key = os.getenv('PROMETHEUS_API_KEY')
    
    if not all([endpoint, username, api_key]):
        logger.error("Missing required environment variables. Please ensure PROMETHEUS_ENDPOINT, "
                    "PROMETHEUS_USERNAME, and PROMETHEUS_API_KEY are set in your .env file")
        sys.exit(1)
    
    # Validate comparison arguments
    if args.compare and not args.compare_window:
        logger.error("--compare-window is required when using --compare")
        parser.print_help()
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = CardinalityAnalyzer(endpoint, username, api_key)
    
    try:
        # Parse time windows
        start_ts, end_ts = analyzer.parse_time_window(args.window, args.start_time)
        # Format the actual start time for display
        actual_start_time = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        actual_end_time = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Show a clear summary of what we're analyzing
        logger.info("="*60)
        logger.info(f"MAIN ANALYSIS WINDOW:")
        logger.info(f"  From: {actual_start_time}")
        logger.info(f"  To:   {actual_end_time}")
        logger.info(f"  Duration: {args.window}")
        logger.info("="*60)
        
        # Determine which metrics to analyze
        metrics_to_analyze = []

        # Add single metric or regex pattern if provided
        if args.metric:
            if is_regex_pattern(args.metric):
                logger.info(f"Regex pattern detected: {args.metric}")
                matched_metrics = analyzer.get_metrics_by_pattern(start_ts, end_ts, args.metric)
                if not matched_metrics:
                    logger.error(f"No metrics found matching pattern: {args.metric}")
                    sys.exit(1)
                logger.info(f"Found {len(matched_metrics)} metrics matching pattern:")
                for metric, cardinality in matched_metrics:
                    logger.info(f"  {metric}: {cardinality:.0f}")
                metrics_to_analyze = [m for m, _ in matched_metrics]
            else:
                metrics_to_analyze.append(args.metric)
                logger.info(f"Single metric specified: {args.metric}")

        # Add metrics from file if provided
        if args.metrics_file:
            logger.info(f"Reading metrics from file: {args.metrics_file}")
            try:
                with open(args.metrics_file, 'r') as f:
                    file_metrics = []
                    for line_num, line in enumerate(f, 1):
                        # Strip whitespace
                        line = line.strip()
                        # Skip blank lines and comments
                        if not line or line.startswith('#'):
                            continue
                        file_metrics.append(line)

                    if not file_metrics:
                        logger.error(f"No valid metrics found in {args.metrics_file}")
                        sys.exit(1)

                    logger.info(f"Loaded {len(file_metrics)} metric(s) from file:")
                    for metric in file_metrics:
                        logger.info(f"  - {metric}")

                    metrics_to_analyze.extend(file_metrics)

            except FileNotFoundError:
                logger.error(f"Metrics file not found: {args.metrics_file}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to read metrics file: {e}")
                sys.exit(1)

        # If no metrics specified, get top metrics by cardinality
        if not metrics_to_analyze:
            # Get top metrics by cardinality
            top_metrics = analyzer.get_top_metrics(start_ts, end_ts, args.top_n)
            if not top_metrics:
                logger.error("No metrics found to analyze")
                sys.exit(1)
            # Log the metrics and their cardinality values
            logger.info(f"Top {len(top_metrics)} metrics by cardinality:")
            for metric, cardinality in top_metrics[:args.top_n]:
                logger.info(f"  {metric}: {cardinality:.0f}")

            metrics_to_analyze = [metric for metric, cardinality in top_metrics]
        
        # Analyze each metric
        analyses = []
        total_metrics = len(metrics_to_analyze)
        for idx, metric in enumerate(metrics_to_analyze, 1):
            if total_metrics > 1:
                logger.info(f"Analyzing metric {idx}/{total_metrics}: {metric}")
            else:
                logger.info(f"Analyzing cardinality for metric: {metric}")
            try:
                cardinality_data = analyzer.analyze_metric_cardinality(
                    metric, start_ts, end_ts,
                    custom_step=args.step,  # Pass user-specified step (or None for auto-calc)
                    target_points=args.max_points  # Pass target points for auto-calculation
                )
                analyses.append({
                    'metric': metric,
                    'data': cardinality_data,
                    'window': {'start': start_ts, 'end': end_ts}
                })
            except Exception as e:
                logger.warning(f"Failed to analyze {metric}: {e}")
        
        if not analyses:
            logger.error("No successful analyses completed. This could mean:")
            logger.error("  1. No metrics found in the specified time window")
            logger.error("  2. Connection issues with Mimir/Prometheus")
            logger.error("  3. Invalid metric name if using -m option")
            logger.error("Try running with -v for verbose output to see detailed errors")
            sys.exit(1)
        
        # Handle comparison if requested
        comparisons = None
        actual_compare_start_time = None
        if args.compare:
            # If no compare start time specified, default to immediately before the main window
            if not args.compare_start_time:
                # Calculate comparison start to be immediately before main window
                import re
                match = re.match(r'^(\d+)([smhdw])$', args.compare_window)
                if match:
                    duration_map = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
                    compare_duration_seconds = int(match.group(1)) * duration_map[match.group(2)]
                    # Set comparison to end where main window starts
                    compare_end_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
                    compare_start_dt = compare_end_dt - timedelta(seconds=compare_duration_seconds)
                    default_compare_start = compare_start_dt.isoformat()
                    logger.info(f"No --compare-start-time specified, defaulting to {compare_start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} (immediately before main window)")
                    comp_start, comp_end = analyzer.parse_time_window(args.compare_window, default_compare_start)
                else:
                    comp_start, comp_end = analyzer.parse_time_window(args.compare_window, args.compare_start_time)
            else:
                comp_start, comp_end = analyzer.parse_time_window(args.compare_window, args.compare_start_time)
            
            actual_compare_start_time = datetime.fromtimestamp(comp_start, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            actual_compare_end_time = datetime.fromtimestamp(comp_end, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            
            logger.info("="*60)
            logger.info(f"COMPARISON WINDOW (baseline):")
            logger.info(f"  From: {actual_compare_start_time}")
            logger.info(f"  To:   {actual_compare_end_time}")
            logger.info(f"  Duration: {args.compare_window}")
            logger.info("="*60)
            
            comparisons = []
            for metric in metrics_to_analyze:
                try:
                    logger.info(f"Analyzing comparison window for metric: {metric}")
                    comp_data = analyzer.analyze_metric_cardinality(
                        metric, comp_start, comp_end,
                        custom_step=args.step,  # Use same step settings for consistency
                        target_points=args.max_points
                    )
                    
                    # Log if no data found
                    if not comp_data or all(not v for k, v in comp_data.items() if k != '__total__'):
                        logger.warning(f"No cardinality data found for {metric} in comparison window")
                    
                    # Find the original analysis
                    orig_analysis = next((a for a in analyses if a['metric'] == metric), None)
                    if orig_analysis:
                        comparison = analyzer.compare_time_windows(
                            metric,
                            comp_data,  # comparison window as "before"
                            orig_analysis['data']  # analysis window as "after"
                        )
                        comparisons.append(comparison)
                except Exception as e:
                    logger.warning(f"Failed to compare {metric}: {e}")
        
        # Generate AI analysis if requested
        ai_analysis_text = None
        if args.ai_analysis:
            logger.info("Generating AI analysis...")
            try:
                from cardinality_analyzer_ai_analysis import get_ai_analysis
                ai_analysis_text = get_ai_analysis(analyses, comparisons, args.window, args.start_time)
                logger.info("AI analysis completed successfully")
            except ImportError:
                logger.error("AI analysis module not available. Install with: pip install -r requirements-cardinalityanalysis.txt")
            except Exception as e:
                logger.error(f"Failed to generate AI analysis: {e}")
        
        # Generate outputs
        if args.output in ['cli', 'all']:
            generate_cli_output(analyses, comparisons)
            if ai_analysis_text and args.ai_analysis:
                print("\n" + "="*80)
                print("AI ANALYSIS AND RECOMMENDATIONS")
                print("="*80)
                print(ai_analysis_text)
        
        if args.output in ['csv', 'all']:
            csv_file = generate_csv_output(analyses, comparisons)
            logger.info(f"CSV output written to: {csv_file}")
        
        if args.output in ['html', 'all']:
            # Generate separate JSON data file if using lazy loading
            if args.top_n_embed != -1:
                json_file = generate_json_data_file(analyses, comparisons)
                logger.info(f"Complete data file size: {os.path.getsize(json_file) / (1024*1024):.2f} MB")

            html_content = generate_html_output(
                analyses, comparisons,
                args.window,
                actual_start_time,  # Always show the calculated start time
                ai_analysis_text,
                args.compare_window if args.compare else "",
                actual_compare_start_time if args.compare else "",  # Show calculated compare start time
                command_line,
                args.top_n_embed  # Pass the top-n-embed parameter
            )
            html_file = "cardinality_analysis.html"
            with open(html_file, 'w') as f:
                f.write(html_content)

            html_size_mb = os.path.getsize(html_file) / (1024*1024)
            logger.info(f"HTML report written to: {html_file} ({html_size_mb:.2f} MB)")

            if args.output == 'html':
                print(f"\n✅ Analysis complete! Open {html_file} in your browser to view the interactive report")
                print(f"   HTML File: {os.path.abspath(html_file)} ({html_size_mb:.2f} MB)")
                if args.top_n_embed != -1:
                    json_size_mb = os.path.getsize(json_file) / (1024*1024)
                    print(f"   Data File: {os.path.abspath(json_file)} ({json_size_mb:.2f} MB)")
                    print(f"   Note: Keep both files in the same directory for lazy loading to work")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()