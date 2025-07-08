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

def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file at {config_path}")
        return {}

def clean_response_text(text: str) -> str:
    """Clean and format the response text from the LLM"""
    if not text:
        return ""
    
    # Remove any incomplete sentences at the end
    text = re.sub(r'[^.!?]*$', '', text.strip())
    
    # Remove common LLM boilerplate phrases
    patterns_to_remove = [
        r'^As an AI(?: language)? model,?',
        r'^I(?: am an AI that| do not| can\'t)',
        r'^(?:Based on|According to) the (?:provided|given) context',
        r'^I don\'t have (?:access|enough information)',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Capitalize first letter and ensure proper punctuation
    text = text.strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def validate_product_filter(product: str) -> bool:
    """Validate if the product filter is one of our supported products"""
    return product.lower() in [p.lower() for p in SUPPORTED_PRODUCTS]

def get_available_products(data_dir: str = "data") -> List[str]:
    """
    Get list of available products from data directory
    
    Args:
        data_dir: Directory containing complaint data files
    
    Returns:
        List of available product names
    """
    # First check our standard supported products
    available = set(SUPPORTED_PRODUCTS)
    
    # Then scan data files for additional products
    try:
        for file in Path(data_dir).glob("*.csv"):
            try:
                df = pd.read_csv(file, nrows=1)  # Just read header
                if 'Product' in df.columns:
                    available.update(df['Product'].unique())
            except Exception as e:
                logger.warning(f"Could not read {file}: {e}")
    except FileNotFoundError:
        logger.warning(f"Data directory {data_dir} not found")
    
    return sorted(available)

def format_context_sources(context_chunks: List[Dict], max_length: int = 200) -> str:
    """
    Format retrieved context chunks for display in UI with source information
    
    Args:
        context_chunks: List of dicts with 'text', 'product', 'score', 'id'
        max_length: Maximum length of each chunk to display
    
    Returns:
        Formatted string with sources
    """
    if not context_chunks:
        return "No sources found"
    
    formatted = ["<b>Sources used in this answer:</b>"]
    for i, chunk in enumerate(context_chunks, 1):
        product = chunk.get('product', 'Unknown product')
        score = chunk.get('score', 0)
        chunk_id = chunk.get('id', 'N/A')
        text = chunk.get('text', '')[:max_length]
        
        formatted.append(
            f"\n<b>[{i}] {product} (ID: {chunk_id}, relevance: {score:.2f})</b>\n"
            f"{text}{'...' if len(chunk.get('text', '')) > max_length else ''}"
        )
    
    return "\n".join(formatted)

def parse_date_range(date_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse date range string into start and end dates
    
    Args:
        date_str: String in format "YYYY-MM-DD to YYYY-MM-DD" or "last N days"
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    if not date_str:
        return None, None
    
    try:
        if " to " in date_str:
            start, end = date_str.split(" to ")
            return start.strip(), end.strip()
        elif date_str.startswith("last "):
            days = int(date_str[5:].split()[0])
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            return start_date, end_date
    except Exception as e:
        logger.warning(f"Failed to parse date range '{date_str}': {e}")
    
    return None, None

def save_user_query_history(user_id: str, query: str, response: str, 
                          storage_path: str = "user_history"):
    """
    Save user query and response to history file
    
    Args:
        user_id: Unique identifier for user
        query: User's question
        response: Generated answer
        storage_path: Directory to store history files
    """
    try:
        Path(storage_path).mkdir(exist_ok=True)
        history_file = Path(storage_path) / f"{user_id}.jsonl"
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        
        with open(history_file, 'a') as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to save user history: {e}")

def rate_limit_check(user_id: str, max_requests: int = 30, 
                    window_minutes: int = 60) -> bool:
    """
    Simple rate limiting check for a user
    
    Args:
        user_id: Unique identifier for user
        max_requests: Maximum allowed requests in time window
        window_minutes: Time window in minutes
    
    Returns:
        True if request should be allowed, False if rate limited
    """
    # In production, you'd want to use Redis or similar for this
    # This is a simplified in-memory version for demonstration
    if not hasattr(rate_limit_check, "request_counts"):
        rate_limit_check.request_counts = defaultdict(int)
        rate_limit_check.last_reset = datetime.now()
    
    # Reset counts if window has passed
    if (datetime.now() - rate_limit_check.last_reset).seconds > window_minutes * 60:
        rate_limit_check.request_counts.clear()
        rate_limit_check.last_reset = datetime.now()
    
    # Increment count for this user
    rate_limit_check.request_counts[user_id] += 1
    
    return rate_limit_check.request_counts[user_id] <= max_requests

def load_user_history(user_id: str, storage_path: str = "user_history", 
                     limit: int = 50) -> List[Dict]:
    """
    Load user's query history
    
    Args:
        user_id: Unique identifier for user
        storage_path: Directory where history files are stored
        limit: Maximum number of history items to return
    
    Returns:
        List of history records sorted by timestamp (newest first)
    """
    try:
        history_file = Path(storage_path) / f"{user_id}.jsonl"
        if not history_file.exists():
            return []
        
        with open(history_file, 'r') as f:
            lines = f.readlines()[-limit:]  # Get most recent entries
            history = [json.loads(line) for line in lines if line.strip()]
        
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)
    except Exception as e:
        logger.error(f"Failed to load user history: {e}")
        return []

if __name__ == "__main__":
    # Test utility functions
    print("=== Testing get_available_products() ===")
    print("Available products:", get_available_products())
    
    print("\n=== Testing clean_response_text() ===")
    test_text = "   as an AI model, I can't answer that. Based on the context, users are unhappy.   "
    print(f"Before: '{test_text}'")
    print(f"After: '{clean_response_text(test_text)}'")
    
    print("\n=== Testing format_context_sources() ===")
    test_chunks = [
        {"text": "Customers report issues with late fees on BNPL services", 
         "product": "BNPL", "score": 0.92, "id": "123"},
        {"text": "Some users can't access their payment history", 
         "product": "Credit Card", "score": 0.87, "id": "456"}
    ]
    print(format_context_sources(test_chunks))