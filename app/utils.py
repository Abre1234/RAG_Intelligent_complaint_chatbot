# app/utils.py
import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import html

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later",
    "Savings account",
    "Money transfer"
]

def clean_response_text(text: str) -> str:
    """Enhanced response cleaning with more patterns"""
    if not text:
        return ""
    
    # Remove everything after common cutoff phrases
    cutoff_phrases = [
        "I am an AI",
        "as a language model",
        "Note:",
        "Disclaimer:",
        "However, please note",
        "Keep in mind that"
    ]
    for phrase in cutoff_phrases:
        if phrase.lower() in text.lower():
            text = text[:text.lower().find(phrase.lower())]
    
    # Remove incomplete sentences at end
    text = re.sub(r'[^.!?]*$', '', text.strip())
    
    # Remove multiple newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove markdown formatting if present
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # italics
    
    # Fix common formatting issues
    text = text.replace(' .', '.').replace(' ,', ',')
    
    # Capitalize properly and ensure punctuation
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith(('.', '!', '?')):
            text += '.'
    
    return text

def format_context_sources(context_chunks: List[Dict], max_length: int = 250) -> str:
    """
    Enhanced source formatting with HTML for better display
    """
    if not context_chunks:
        return "<p>No sources were used for this answer.</p>"
    
    html_content = [
        "<div style='font-family: Arial, sans-serif; margin-top: 10px;'>",
        "<h4 style='margin-bottom: 5px;'>Sources used:</h4>",
        "<div style='max-height: 300px; overflow-y: auto; padding: 5px; border: 1px solid #ddd; border-radius: 5px;'>"
    ]
    
    for i, chunk in enumerate(context_chunks, 1):
        product = html.escape(chunk.get('product', 'Unknown product'))
        score = chunk.get('score', 0)
        chunk_id = chunk.get('id', 'N/A')
        text = html.escape(chunk.get('text', '')[:max_length])
        
        html_content.append(
            f"<div style='margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee;'>"
            f"<b>Source {i}: {product} (ID: {chunk_id}, relevance: {score:.2f})</b><br>"
            f"<div style='margin-left: 10px; margin-top: 5px; color: #555;'>{text}"
            f"{'...' if len(chunk.get('text', '')) > max_length else ''}</div>"
            f"</div>"
        )
    
    html_content.extend(["</div>", "</div>"])
    return "".join(html_content)

def validate_product_filter(product: str) -> bool:
    """Case-insensitive product validation"""
    return product.lower() in [p.lower() for p in SUPPORTED_PRODUCTS]

def parse_date_range(date_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """More robust date range parsing"""
    if not date_str:
        return None, None
    
    try:
        # Handle "last N days/weeks/months"
        if date_str.lower().startswith('last '):
            parts = date_str[5:].split()
            num = int(parts[0])
            unit = parts[1].lower() if len(parts) > 1 else 'days'
            
            end_date = datetime.now()
            if 'day' in unit:
                start_date = end_date - timedelta(days=num)
            elif 'week' in unit:
                start_date = end_date - timedelta(weeks=num)
            elif 'month' in unit:
                start_date = end_date - timedelta(days=num*30)
            else:
                start_date = end_date - timedelta(days=num)
            
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        # Handle explicit date ranges
        if ' to ' in date_str:
            start, end = date_str.split(' to ', 1)
            return start.strip(), end.strip()
        
        # Handle single date
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str, date_str
    
    except Exception as e:
        logger.warning(f"Failed to parse date range '{date_str}': {e}")
    
    return None, None

def analyze_query_intent(query: str) -> Dict:
    """
    Basic query intent analysis
    Returns:
        {
            "products": list of mentioned products,
            "is_comparison": bool,
            "time_period": optional time period,
            "question_type": "trend", "specific", "general", etc.
        }
    """
    query_lower = query.lower()
    result = {
        "products": [],
        "is_comparison": False,
        "time_period": None,
        "question_type": "general"
    }
    
    # Detect products
    for product in SUPPORTED_PRODUCTS:
        if product.lower() in query_lower:
            result["products"].append(product)
    
    # Detect comparisons
    result["is_comparison"] = any(
        word in query_lower for word in ['vs', 'versus', 'compared to', 'difference between']
    )
    
    # Detect time periods
    time_terms = {
        'recent': 'last 30 days',
        'last week': 'last 7 days',
        'last month': 'last 30 days',
        'this year': f'{datetime.now().year}-01-01 to {datetime.now().strftime("%Y-%m-%d")}'
    }
    for term, period in time_terms.items():
        if term in query_lower:
            result["time_period"] = period
            break
    
    # Detect question type
    if any(word in query_lower for word in ['trend', 'increase', 'decrease', 'more common']):
        result["question_type"] = "trend"
    elif any(word in query_lower for word in ['specific', 'particular', 'example']):
        result["question_type"] = "specific"
    elif 'common' in query_lower:
        result["question_type"] = "common_issues"
    
    return result