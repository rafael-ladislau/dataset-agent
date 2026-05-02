#!/usr/bin/env python3
"""
Test script for the description cleaning functionality.
"""

from src.dataset_agent.utils.text_processing import clean_description

def main():
    """Test the description cleaning function with a problematic description."""
    # The problematic description with markdown and escape sequences
    original_desc = """The **Census of Agriculture** is a comprehensive dataset created by the U.S. Department of Agriculture's National Agricultural Statistics Service (NASS). Conducted every five years, it provides a complete count of U.S. farms and ranches, including operations that generate $1,000 or more in agricultural products. The dataset captures detailed information on land use, crop and livestock production, farm demographics (e.g., operator age, gender, ethnicity), economic performance, and emerging topics like precision agriculture and hemp production. \n\nIts primary purpose is to inform agricultural policy, research, and resource allocation by offering a snapshot of the industry\u2019s structure and trends. Use cases include academic research, business planning, grant applications, and federal funding decisions. Key features include its exhaustive scope (covering both traditional and specialty crops, as well as urban agriculture), granular geographic breakdowns (national, state, county levels), and longitudinal data spanning decades, enabling analysis of long-term shifts in farming practices and demographics. Data is accessible via tools like Quick Stats and the Census Data Query Tool, ensuring usability for diverse stakeholders."""
    
    # Clean the description
    cleaned_desc = clean_description(original_desc)
    
    # Print the results
    print("Original Description:")
    print("-" * 80)
    print(original_desc)
    print("\nCleaned Description:")
    print("-" * 80)
    print(cleaned_desc)
    
    # Check if the problematic elements were fixed
    problems_fixed = (
        "**" not in cleaned_desc and  # Check if markdown bold is removed
        "\\n" not in cleaned_desc and  # Check if newline escapes are removed
        "\\u" not in cleaned_desc       # Check if Unicode escapes are removed
    )
    
    print(f"\nAll problems fixed: {problems_fixed}")
    
    # Print specific checks
    print(f"- Markdown bold removed: {'**' not in cleaned_desc}")
    print(f"- Newline escapes removed: {'\\n' not in cleaned_desc}")
    print(f"- Unicode escapes removed: {'\\u' not in cleaned_desc}")
    
if __name__ == "__main__":
    main() 