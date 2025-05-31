"""
Validation utilities for the dataset agent.
"""

import re
import logging
import requests
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse

# Set up logger
logger = logging.getLogger(__name__)


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    if not url:
        return False
        
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.warning(f"Error validating URL {url}: {str(e)}")
        return False


def validate_url(url: str, timeout: int = 10) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a URL by sending a request and checking the response.
    
    Args:
        url: URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Validation result and response details
    """
    if not is_valid_url(url):
        return False, {"error": "Invalid URL format"}
        
    try:
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, timeout=timeout, headers=headers)
        status_code = response.status_code
        
        # Format the response based on status code
        valid = status_code == 200
        
        # Get the content type and length
        content_type = response.headers.get('Content-Type', 'unknown')
        content_length = len(response.text)
        
        # Get a preview of the content
        content_preview = response.text[:500] if valid else ""
        
        return valid, {
            "url": url,
            "status_code": status_code,
            "content_type": content_type,
            "content_length": content_length,
            "content_preview": content_preview
        }
    except Exception as e:
        logger.warning(f"Error validating URL {url}: {str(e)}")
        return False, {"url": url, "error": str(e)} 