#!/usr/bin/env python3
"""
Test script for the dataset agent.

This script runs a simple test of the dataset research agent with a sample dataset.
"""

import json
import time
import sys
from src.dataset_agent.config import Config
from src.dataset_agent.main import run_research


def main():
    """Run a test of the dataset research agent."""
    # Sample dataset for testing
    dataset_name = "NASS Census of Agriculture"
    dataset_url = "https://www.nass.usda.gov/AgCensus/"
    
    print(f"Testing dataset research agent with {dataset_name}...")
    
    # Create configuration
    config = Config(log_level="INFO")
    
    # Time the research
    start_time = time.time()
    
    try:
        # Run research
        result = run_research(dataset_name, dataset_url, config)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Print results
        print("\nTest Results:")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Description length: {len(result.description)} chars")
        print(f"Aliases: {', '.join(result.aliases[:5])}...")
        print(f"Organizations: {', '.join(result.organizations[:5])}...")
        print(f"Access type: {result.access_type}")
        print(f"Data URL: {result.data_url}")
        print(f"Schema URL: {result.schema_url}")
        print(f"Documentation URL: {result.documentation_url}")
        
        print("\nTest completed successfully!")
        return 0
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())