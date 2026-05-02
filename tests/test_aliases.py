#!/usr/bin/env python3
"""
Test script for the alias filtering functionality.
"""

from src.dataset_agent.utils.text_processing import filter_aliases_by_substrings

def main():
    """Run the tests for alias filtering."""
    # Test case 1: Example from requirements
    test1 = ["Ag Census", "Usda Ag Census"]
    result1 = filter_aliases_by_substrings(test1)
    print(f"Test 1: {test1} -> {result1}")
    
    # Test case 2: Example from requirements
    test2 = ["AgCensus", "USDA AgCensus"]
    result2 = filter_aliases_by_substrings(test2)
    print(f"Test 2: {test2} -> {result2}")
    
    # Test case 3: Example from requirements (URL case)
    test3 = ["AgCensus", "http://nass.usda.gov.br/AgCensus"]
    result3 = filter_aliases_by_substrings(test3)
    print(f"Test 3: {test3} -> {result3}")
    
    # Test case 4: Multiple overlapping aliases
    test4 = ["Census", "Ag Census", "USDA Ag Census", "NASS USDA Ag Census"]
    result4 = filter_aliases_by_substrings(test4)
    print(f"Test 4: {test4} -> {result4}")
    
    # Test case 5: Case insensitivity
    test5 = ["ag census", "AG CENSUS", "Ag Census"]
    result5 = filter_aliases_by_substrings(test5)
    print(f"Test 5: {test5} -> {result5}")
    
    # Test case 6: Real world example from Census of Agriculture
    test6 = [
        "Ag Census",
        "Usda Ag Census",
        "Nass Ag Census",
        "Census of Agriculture",
        "Usda Census of Agriculture",
        "Nass Census of Agriculture",
        "Census of Agriculture - Nass",
        "Census of Agriculture - Usda Nass",
        "Census of Agriculture Report Forms",
        "National Agricultural Statistics Service Census",
        "Usda National Agricultural Statistics Service Census of Agriculture"
    ]
    result6 = filter_aliases_by_substrings(test6)
    print(f"Test 6 (Real world example):")
    print(f"  Original: {len(test6)} aliases")
    print(f"  Filtered: {len(result6)} aliases")
    print(f"  Kept aliases: {result6}")
    
if __name__ == "__main__":
    main() 