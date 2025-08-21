#!/usr/bin/env python3

"""
AI Analysis module for Cardinality Analyzer
Provides LLM-based insights and recommendations for cardinality analysis results
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def get_ai_analysis(analyses: List[Dict], comparisons: Optional[List[Dict]], 
                   window: str, start_time: Optional[str] = None) -> str:
    """
    Send cardinality analysis data to OpenAI and get insights
    
    Args:
        analyses: List of metric analysis results
        comparisons: Optional list of comparison results
        window: Time window analyzed
        start_time: Optional start time for the analysis
        
    Returns:
        AI-generated analysis and recommendations
    """
    try:
        # Import OpenAI only when needed
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI SDK not installed. Please install with: pip install -r requirements.txt")
        return "Error: OpenAI SDK not available"
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not set. Please set OPENAI_API_KEY")
        return "Error: OPENAI_API_KEY not configured"
    
    # Get model from env or use default (gpt-5-mini)
    model = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
    # Reasoning effort control (low | medium | high)
    reasoning_effort = os.getenv('OPENAI_REASONING_EFFORT', 'high').lower()
    if reasoning_effort not in {'low', 'medium', 'high'}:
        reasoning_effort = 'medium'
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Prepare the data summary for the AI
    data_summary = prepare_data_summary(analyses, comparisons, window, start_time)
    
    # System prompt explaining cardinality analysis
    system_prompt = """You are an expert in Prometheus/Mimir metrics and cardinality analysis. Your task is to analyze metric cardinality data and explain the data clearly, focusing on describing the current state and any changes between time windows.
Key concepts:
- Cardinality: The number of unique time series for a metric (unique combinations of label values)
- Labels: Key-value pairs that create metric dimensions (e.g., instance="server1", job="api")
- High cardinality labels: Labels with many unique values, consuming more resources
Instructions:
1. Describe the current cardinality state for each metric and label
2. For comparisons: Explain what changed between the two time windows, with specific details
3. Highlight which labels have the highest cardinality and list relevant values
4. Provide detailed breakdowns of differences; do not give recommendations
5. Stick to factual observations only - NO solutions or advice
IMPORTANT: Do NOT provide recommendations, solutions, or next steps. Differences you observe are expected and intentional. Your role is to explain the data in detail.
Format your response using:
- **Key Findings**: Bullet points summarizing observations
- **Detailed Analysis**: In-depth breakdown by metric and label
- For comparisons: **Changes Between Windows** (specific differences with numbers)
Format with Markdown: use **bold** for emphasis and bullet points for clarity."""
    
    # User prompt with the actual data
    user_prompt = f"""Please analyze this Prometheus/Mimir cardinality data and describe what it shows:
Analysis Window: {window} {f'(starting {start_time})' if start_time else '(recent)'}
Analysis Type: {'Comparison between two time windows' if comparisons else 'Single time window analysis'}
{data_summary}
Please explain:
1. Current cardinality levels for each metric and label
2. Which labels have the highest unique values and what those values are
3. For comparisons: What changed between the windows (with numbers and percentages)
4. Distribution of cardinality across labels
5. Any significant patterns or observations
Only describe and explain the data - do not give recommendations or solutions"""
    
    try:
        # Always use Responses API with gpt-5-mini
        logger.info(f"Sending analysis to OpenAI using model: {model}")
        
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt}
                    ]
                }
            ],
            max_output_tokens=65536,  # Maximum for gpt-5-mini (low cost model)
            reasoning={"effort": reasoning_effort}
        )

        # Extract the response text - try multiple approaches for different SDK versions
        ai_analysis = None
        
        # Method 1: output_text attribute (Responses API - simplest when it works)
        try:
            ai_analysis = getattr(response, 'output_text', None)
            if ai_analysis:
                logger.debug(f"Extracted AI analysis using 'output_text' attribute: {len(ai_analysis)} chars")
            else:
                logger.debug("output_text attribute is empty or None")
        except Exception as e:
            logger.debug(f"Failed to get output_text: {e}")
        
        # Method 2: Direct content attribute (for some SDK versions)
        if not ai_analysis:
            try:
                ai_analysis = getattr(response, 'content', None)
                if ai_analysis:
                    logger.debug("Extracted AI analysis using 'content' attribute")
            except Exception:
                pass
        
        # Method 3: choices[0].message.content (standard chat completions)
        if not ai_analysis:
            try:
                choices = getattr(response, 'choices', [])
                if choices and len(choices) > 0:
                    message = getattr(choices[0], 'message', None)
                    if message:
                        ai_analysis = getattr(message, 'content', None)
                        if ai_analysis:
                            logger.debug("Extracted AI analysis using 'choices[0].message.content'")
            except Exception:
                pass
        
        # Method 4: Parse output array structure (for Responses API)
        if not ai_analysis:
            try:
                texts = []
                output_items = getattr(response, 'output', []) or []
                for item in output_items:
                    # Get item type
                    item_type = getattr(item, 'type', None)
                    
                    # Look for message items
                    if item_type == 'message':
                        # Get content from the message
                        content_list = getattr(item, 'content', [])
                        if content_list:
                            for content in content_list:
                                # Check content type
                                content_type = getattr(content, 'type', None)
                                if content_type in {'output_text', 'text'}:
                                    # Extract the text
                                    text = getattr(content, 'text', '')
                                    if text:
                                        texts.append(text)
                
                ai_analysis = '\n'.join(texts).strip()
                if ai_analysis:
                    logger.debug("Extracted AI analysis using 'output' array structure")
            except Exception as e:
                logger.debug(f"Failed to parse output array: {e}")
        
        if not ai_analysis:
            logger.warning("Could not extract AI analysis from response - all methods failed")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")
            # Debug output structure
            try:
                logger.debug(f"Output items count: {len(response.output)}")
                for i, item in enumerate(response.output):
                    logger.debug(f"Output[{i}]: type={getattr(item, 'type', 'unknown')}")
                    if hasattr(item, 'content') and item.content:
                        logger.debug(f"  Content items: {len(item.content)}")
                        for j, c in enumerate(item.content[:2]):  # Log first 2 content items
                            logger.debug(f"    [{j}] type={getattr(c, 'type', 'unknown')}, text_len={len(getattr(c, 'text', ''))}")
            except Exception as e:
                logger.debug(f"Failed to debug output structure: {e}")

        # Log reasoning items (if present) at DEBUG
        try:
            output_items = getattr(response, 'output', []) or []
            reasoning_chunks = []
            for item in output_items:
                item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
                if item_type == 'message':
                    content_list = item.get('content') if isinstance(item, dict) else getattr(item, 'content', [])
                    for content in content_list or []:
                        ctype = content.get('type') if isinstance(content, dict) else getattr(content, 'type', None)
                        if ctype and 'reasoning' in str(ctype):
                            reasoning_text = content.get('text') if isinstance(content, dict) else getattr(content, 'text', '')
                            if reasoning_text:
                                reasoning_chunks.append(reasoning_text)
            if reasoning_chunks:
                logger.debug("Reasoning content from model (truncated): %s", ("\n".join(reasoning_chunks))[:2000])
        except Exception:
            pass

        # Log token usage for monitoring (Responses API fields)
        try:
            usage = getattr(response, 'usage', None)
            if usage:
                input_tokens = getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', None)
                output_tokens = getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', None)

                # Try multiple locations for reasoning token counts
                reasoning_tokens = getattr(usage, 'reasoning_tokens', None)
                if reasoning_tokens is None:
                    output_token_details = getattr(usage, 'output_token_details', None)
                    if output_token_details is not None:
                        # Handle object or dict shape
                        try:
                            reasoning_tokens = getattr(output_token_details, 'reasoning_tokens', None)
                        except Exception:
                            reasoning_tokens = None
                        if reasoning_tokens is None and isinstance(output_token_details, dict):
                            reasoning_tokens = output_token_details.get('reasoning_tokens')

                # Friendly fallback label
                reasoning_display = reasoning_tokens if reasoning_tokens is not None else 'n/a'
                logger.info("OpenAI usage - Input: %s, Output: %s, Reasoning: %s", input_tokens, output_tokens, reasoning_display)
        except Exception:
            pass

        return ai_analysis or ""

    except Exception as e:
        logger.error(f"Error calling OpenAI Responses API: {e}")
        return f"Error generating AI analysis: {str(e)}"

def prepare_data_summary(analyses: List[Dict], comparisons: Optional[List[Dict]], 
                        window: str, start_time: Optional[str]) -> str:
    """
    Prepare a structured summary of the cardinality data for AI analysis
    """
    summary_parts = []
    
    # Summary of analyzed metrics
    summary_parts.append(f"=== METRICS ANALYZED: {len(analyses)} ===\n")
    
    # For each metric, provide key cardinality information
    for analysis in analyses:
        metric_name = analysis['metric']
        data = analysis['data']
        total_info = data.get('__total__', {})
        
        # Get label cardinalities sorted by value
        label_cardinalities = []
        for label, info in data.items():
            if label == '__total__':
                continue
            if 'total_cardinality' in info:
                label_cardinalities.append((label, info['total_cardinality'], info.get('top_values', [])))
        
        label_cardinalities.sort(key=lambda x: x[1], reverse=True)
        
        summary_parts.append(f"\nMetric: {metric_name}")
        if total_info:
            summary_parts.append(f"  Total Cardinality: max={total_info.get('max_cardinality', 'N/A')}, "
                               f"avg={total_info.get('avg_cardinality', 'N/A'):.0f}")
        
        summary_parts.append("  Top Labels by Cardinality:")
        for label, cardinality, top_values in label_cardinalities[:5]:
            summary_parts.append(f"    - {label}: {cardinality} unique values")
            if top_values:
                top_3 = ', '.join([f"{val[0]} ({val[1]:.0f} series)" for val in top_values[:3]])
                summary_parts.append(f"      Top values: {top_3}")
    
    # Add comparison data if available
    if comparisons:
        summary_parts.append("\n\n=== COMPARISON ANALYSIS ===")
        summary_parts.append("Changes between time windows:\n")
        
        for comp in comparisons:
            metric_name = comp['metric']
            summary_parts.append(f"\nMetric: {metric_name}")
            summary_parts.append("  Significant changes:")
            
            # Get top 5 changes by absolute value
            changes = comp.get('sorted_changes', [])[:5]
            for label, change_data in changes:
                change = change_data['change']
                change_pct = change_data['change_pct']
                if abs(change) > 0:
                    direction = "increased" if change > 0 else "decreased"
                    summary_parts.append(f"    - {label}: {direction} by {abs(change)} "
                                       f"({abs(change_pct):.1f}%) "
                                       f"[{change_data['before']} → {change_data['after']}]")
    
    # Add observation notes
    summary_parts.append("\n\n=== NOTABLE OBSERVATIONS ===")
    summary_parts.append("Key patterns in the data:")
    summary_parts.append("- Labels with extremely high unique value counts (>1000)")
    summary_parts.append("- Significant changes in cardinality between time windows")
    summary_parts.append("- Distribution of cardinality across different label types")
    summary_parts.append("- Metrics with the most diverse label combinations")
    
    return '\n'.join(summary_parts)

def convert_markdown_to_html(text: str) -> str:
    """
    Convert basic markdown to HTML for better formatting
    """
    import html
    import re
    
    # First escape HTML to prevent injection
    text = html.escape(text)
    
    # Convert markdown formatting
    # Headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Bold text
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # Italic text
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    # Bullet points
    lines = text.split('\n')
    in_list = False
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list and not stripped.startswith('  '):
                formatted_lines.append('</ul>')
                in_list = False
            if stripped:
                formatted_lines.append(f'<p>{line}</p>')
            else:
                formatted_lines.append('<br>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return '\n'.join(formatted_lines)

def generate_ai_report_section(ai_analysis: str) -> str:
    """
    Generate an HTML section for the AI analysis to be inserted into the main report
    """
    # Convert markdown to HTML
    formatted_analysis = convert_markdown_to_html(ai_analysis)
    
    html_section = f"""
    <div class="ai-analysis-section">
        <h2>AI Analysis</h2>
        <div class="ai-disclaimer">
            <strong>Note:</strong> This analysis was generated by {os.getenv('OPENAI_MODEL', 'gpt-5-mini')} 
            based on the cardinality data.
        </div>
        <div class="ai-content">
            {formatted_analysis}
        </div>
    </div>
    
    <style>
        .ai-analysis-section {{
            margin: 30px 0;
            padding: 25px;
            background: #f0f7ff;
            border: 1px solid #2196f3;
            border-radius: 8px;
        }}
        .ai-analysis-section h2 {{
            color: #1976d2;
            margin-bottom: 15px;
        }}
        .ai-disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .ai-content {{
            background: white;
            padding: 20px;
            border-radius: 4px;
            line-height: 1.8;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .ai-content h1, .ai-content h2, .ai-content h3 {{
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .ai-content h1 {{ font-size: 1.8em; }}
        .ai-content h2 {{ font-size: 1.5em; }}
        .ai-content h3 {{ font-size: 1.2em; }}
        .ai-content p {{
            margin: 10px 0;
        }}
        .ai-content ul {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        .ai-content li {{
            margin: 5px 0;
        }}
        .ai-content strong {{
            color: #1976d2;
            font-weight: 600;
        }}
        .ai-content em {{
            font-style: italic;
            color: #666;
        }}
    </style>
    """
    
    return html_section