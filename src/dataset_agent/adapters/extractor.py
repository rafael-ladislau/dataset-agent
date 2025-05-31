"""
Extractor implementations for processing text responses from LLMs.
"""

import re
import ast
from typing import List, Optional

from ..domain.usecases import TextExtractor


class LLMOutputExtractor(TextExtractor):
    """Implementation of TextExtractor for LLM outputs."""
    
    def extract_list(self, text: str) -> List[str]:
        """
        Extract a Python list from a text response using multiple approaches.
        
        Args:
            text: Text response containing a Python list
            
        Returns:
            List[str]: Extracted list or empty list if not found
        """
        # Remove any thinking sections from the response
        if "<think>" in text and "</think>" in text:
            # Process only the text after the last thinking section
            parts = text.split("</think>")
            text = parts[-1].strip()
        
        # Remove common citation format headers and metadata fragments
        citation_markers = [
            "Citation", "Example:", "Component", "Details:", "Source:", "Author:", "Title:", 
            "URL:", "Retrieval Date:", "Format:", "Guidelines:", "For HTML", "MLA", "APA", 
            "Chicago", "Standard Citation", "Citation Request", "URL**", "Author**", 
            "Title**", "Retrieval Date**", "Components**", "Details**", "Example**", 
            "Check the Document", "Census Bureau Guidelines"
        ]
        
        # Clean up the response_text to make it easier to parse
        for marker in citation_markers:
            text = re.sub(rf'(?i){re.escape(marker)}.*?\n', '\n', text)
            
        # Remove URL fragments and links
        text = re.sub(r'https?://[^\s\]\"]+', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        
        try:
            # First attempt: Try to find a Python list using regex
            list_pattern = r'\[(.*?)\]'
            match = re.search(list_pattern, text, re.DOTALL)
            
            if match:
                # Get content inside brackets and try to parse
                items_text = match.group(1)
                list_str = f"[{items_text}]"
                
                try:
                    # Parse as Python literal
                    items = ast.literal_eval(list_str)
                    if isinstance(items, list):
                        # Filter out items that are likely not aliases
                        items = [item for item in items if isinstance(item, str) and 
                                len(item.strip()) > 0 and
                                not item.startswith("http") and
                                ":" not in item and
                                "**" not in item and
                                "(" not in item and ")" not in item and
                                "[" not in item and "]" not in item]
                        return items
                except (SyntaxError, ValueError):
                    # If can't parse as literal, try another approach
                    pass
            
            # Second attempt: Look for lines that look like list items
            lines = text.split('\n')
            items = []
            for line in lines:
                line = line.strip()
                # Match patterns like "1. Item" or "- Item" or "* Item" or "• Item"
                match = re.match(r'^(?:\d+\.|\-|\*|•|\>)\s*(.*?)$', line)
                if match and len(match.group(1).strip()) > 0:
                    item = match.group(1).strip()
                    if len(item) > 0 and not any(m in item.lower() for m in citation_markers):
                        # Remove quotes if present
                        item = item.strip('"\'')
                        # Filter out citation format descriptors and URLs
                        if (item and 
                            ":" not in item and 
                            not item.startswith("http") and
                            "**" not in item and
                            "(" not in item and ")" not in item and
                            "[" not in item and "]" not in item):
                            items.append(item)
            
            if items:
                return items
            
            # Third attempt: Try extracting quoted strings that might be aliases
            quoted_items = re.findall(r'["\']([^"\']+)["\']', text)
            valid_items = []
            for item in quoted_items:
                # Filter items similar to above
                if (item and len(item.strip()) > 0 and 
                    ":" not in item and 
                    not item.startswith("http") and
                    "**" not in item and
                    "(" not in item and ")" not in item and
                    "[" not in item and "]" not in item):
                    valid_items.append(item.strip())
            
            if valid_items:
                return valid_items
                
        except Exception as e:
            # If there's an error, log it and return an empty list
            print(f"Error extracting list from response: {str(e)}")
        
        # If all attempts fail, return an empty list
        return []
    
    def extract_url(self, text: str) -> Optional[str]:
        """
        Extract a URL from a text response using multiple approaches.
        
        Args:
            text: Text response containing a URL
            
        Returns:
            Optional[str]: Extracted URL or None if not found
        """
        try:
            # Remove thinking part if present
            if "<think>" in text and "</think>" in text:
                # Get the text after the last </think> tag
                text = text.split("</think>")[-1].strip()
            
            # First attempt: Check if the entire response is a URL
            url = text.strip()
            if url.startswith(('http://', 'https://')):
                return url
            
            # Second attempt: Extract URLs using regex
            url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
            matches = re.findall(url_pattern, text)
            
            if matches:
                # Return the first URL found
                url = matches[0]
                if url.startswith('www.'):
                    url = 'https://' + url
                return url
                
            # Third attempt: Check for common patterns
            if text.strip().startswith('www.'):
                return 'https://' + text.strip()
                
            # If no URL found, return None
            return None
                
        except Exception as e:
            # Log error and return None
            print(f"Error extracting URL from response: {str(e)}")
            return None 