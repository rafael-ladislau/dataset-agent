"""
Authentication module for API key validation.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(self, api_keys: Optional[str] = None):
        """
        Initialize API key authentication.
        
        Args:
            api_keys: Comma-separated list of API keys, defaults to API_KEYS environment variable
        """
        if api_keys is None:
            api_keys = os.environ.get('API_KEYS', '')
        
        self.api_keys = [key.strip() for key in api_keys.split(',') if key.strip()]
        if not self.api_keys:
            logger.warning("No API keys configured. API endpoints will not be secured.")
        else:
            logger.info(f"Initialized with {len(self.api_keys)} API key(s)")
    
    def is_valid_key(self, api_key: str) -> bool:
        """
        Check if the provided API key is valid.
        
        Args:
            api_key: API key to validate
            
        Returns:
            bool: True if the key is valid, False otherwise
        """
        # If no keys are configured, allow all access (for development only)
        if not self.api_keys:
            logger.warning("No API keys configured but access was attempted. Allowing access by default.")
            return True
            
        is_valid = api_key in self.api_keys
        if not is_valid:
            logger.warning(f"Invalid API key attempt: {api_key[:5]}...")
        
        return is_valid
    
    def get_client_id(self, api_key: str) -> str:
        """
        Get a client identifier from the API key.
        
        This can be used for attribution in logs and database records.
        We use a simple index-based approach here, but this could be
        enhanced to extract client info from JWT tokens or similar.
        
        Args:
            api_key: The API key used for the request
            
        Returns:
            str: Client identifier
        """
        # If keys aren't configured, return a default
        if not self.api_keys:
            return 'anonymous'
            
        # Find the index of the API key in our list
        try:
            idx = self.api_keys.index(api_key)
            return f"client_{idx + 1}"
        except ValueError:
            # Should not happen if is_valid_key is called first
            return 'unknown' 