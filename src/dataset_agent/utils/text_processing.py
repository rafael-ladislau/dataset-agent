"""
Text processing utilities for the dataset agent.
"""

import re
from typing import List, Optional


def clean_description(text: str) -> str:
    """
    Clean a description by removing thinking sections, markdown formatting, and other artifacts.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove thinking sections enclosed in <think> tags
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        text = parts[-1].strip()  # Take only the part after the last </think> tag
    
    # Remove any other thinking indicators
    text = re.sub(r'[\n\r]*Thinking:[\s\S]*?(?=\n\n|\Z)', '', text)
    
    # Remove markdown formatting
    # Remove bold/italic formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'__(.*?)__', r'\1', text)      # Remove __bold__
    text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _italic_
    
    # Remove markdown links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Replace [text](url) with just text
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)  # Remove inline code
    
    # Handle escape sequences
    # Replace literal \n with actual space
    text = text.replace('\\n', ' ')
    # Replace literal \t with actual space
    text = text.replace('\\t', ' ')
    
    # Replace common Unicode escape sequences
    text = text.replace('\\u2019', "'")  # Right single quotation mark
    text = text.replace('\\u201c', '"')  # Left double quotation mark
    text = text.replace('\\u201d', '"')  # Right double quotation mark
    text = text.replace('\\u2018', "'")  # Left single quotation mark
    
    # Remove word count metadata
    text = re.sub(r'\s*\(\d+\s+words?\)\s*$', '', text)
    
    # Clean up any remaining artifacts
    text = text.strip()
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    return text


def sanitize_filename(name: str) -> str:
    """
    Create a safe filename from a string.
    
    Args:
        name: String to sanitize
        
    Returns:
        str: Sanitized filename
    """
    # Remove special characters
    filename = re.sub(r'[^\w\s-]', '', name.lower())
    # Replace whitespace with underscores
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename


def normalize_aliases(aliases: List[str]) -> List[str]:
    """
    Normalize a list of aliases by removing duplicates and standardizing format.
    
    Args:
        aliases: List of aliases to normalize
        
    Returns:
        List[str]: Normalized aliases
    """
    # Initialize result
    result = []
    
    # Set for case-insensitive deduplication
    seen = set()
    
    for alias in aliases:
        # Skip empty aliases
        if not alias or not alias.strip():
            continue
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', alias.strip())
        
        # Skip if already processed (case-insensitive)
        if normalized.lower() in seen:
            continue
            
        # Add to results
        result.append(normalized)
        seen.add(normalized.lower())
    
    return result


def filter_aliases_by_substrings(aliases: List[str]) -> List[str]:
    """
    Filter aliases by removing longer aliases when a shorter alias is contained
    as a whole word/phrase within them.
    
    Rules:
    - If alias A is a substring of alias B, remove B (the longer one)
    - For A to be considered a substring of B, it must:
      * Match at the start of B, or
      * Match at the end of B, or
      * Be surrounded by spaces in B
    - Does not remove if the match is part of another word (like in a URL)
    
    Args:
        aliases: List of aliases to filter
        
    Returns:
        List[str]: Filtered aliases
    """
    if not aliases:
        return []
    
    # First normalize all aliases
    normalized_aliases = normalize_aliases(aliases)
    
    # Sort aliases by length (shortest first)
    normalized_aliases.sort(key=len)
    
    # Track which aliases to keep
    keep_alias = [True] * len(normalized_aliases)
    
    # Compare each alias with all longer aliases
    for i, shorter in enumerate(normalized_aliases):
        if not keep_alias[i]:
            continue  # Skip if this alias was already marked for removal
            
        shorter_lower = shorter.lower()
        
        for j in range(i + 1, len(normalized_aliases)):
            if not keep_alias[j]:
                continue  # Skip if longer alias was already marked for removal
                
            longer = normalized_aliases[j]
            longer_lower = longer.lower()
            
            # Check if shorter is a substring of longer with word boundaries
            # 1. Check if shorter is at the start of longer
            # 2. Check if shorter is at the end of longer
            # 3. Check if shorter is surrounded by spaces in longer
            if (longer_lower.startswith(shorter_lower + " ") or
                longer_lower.endswith(" " + shorter_lower) or
                f" {shorter_lower} " in f" {longer_lower} "):
                
                # Mark the longer alias for removal
                keep_alias[j] = False
    
    # Return only the aliases marked to keep
    result = [alias for i, alias in enumerate(normalized_aliases) if keep_alias[i]]
    return result


# Tests for filter_aliases_by_substrings
if __name__ == "__main__":
    # Test case 1: Example from requirements
    test1 = ["Ag Census", "Usda Ag Census"]
    result1 = filter_aliases_by_substrings(test1)
    expected1 = ["Ag Census"]
    assert result1 == expected1, f"Test 1 failed. Expected {expected1}, got {result1}"
    
    # Test case 2: Example from requirements
    test2 = ["AgCensus", "USDA AgCensus"]
    result2 = filter_aliases_by_substrings(test2)
    expected2 = ["AgCensus"]
    assert result2 == expected2, f"Test 2 failed. Expected {expected2}, got {result2}"
    
    # Test case 3: Example from requirements (URL case)
    test3 = ["AgCensus", "http://nass.usda.gov.br/AgCensus"]
    result3 = filter_aliases_by_substrings(test3)
    expected3 = ["AgCensus", "http://nass.usda.gov.br/AgCensus"]
    assert result3 == expected3, f"Test 3 failed. Expected {expected3}, got {result3}"
    
    # Test case 4: Multiple overlapping aliases
    test4 = ["Census", "Ag Census", "USDA Ag Census", "NASS USDA Ag Census"]
    result4 = filter_aliases_by_substrings(test4)
    expected4 = ["Census", "Ag Census"]
    assert result4 == expected4, f"Test 4 failed. Expected {expected4}, got {result4}"
    
    # Test case 5: Case insensitivity
    test5 = ["ag census", "AG CENSUS", "Ag Census"]
    result5 = filter_aliases_by_substrings(test5)
    expected5 = ["ag census"]  # Should only keep the first one due to normalization
    assert len(result5) == 1, f"Test 5 failed. Expected one result, got {result5}"
    
    # Test case 6: Empty input
    test6 = []
    result6 = filter_aliases_by_substrings(test6)
    expected6 = []
    assert result6 == expected6, f"Test 6 failed. Expected {expected6}, got {result6}"
    
    # Test case 7: No overlaps
    test7 = ["Census", "NASS", "USDA", "Agriculture"]
    result7 = filter_aliases_by_substrings(test7)
    expected7 = ["Census", "NASS", "USDA", "Agriculture"]
    assert result7 == expected7, f"Test 7 failed. Expected {expected7}, got {result7}"
    
    print("All tests passed!") 