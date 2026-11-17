#!/usr/bin/env python3
"""
Test script for the dataset agent.

This script runs a simple test of the dataset research agent with a sample dataset.
"""

import asyncio
import time
import sys

from dataset_agent.bootstrap import run_full_pipeline
from dataset_agent.domain.models import ResearchRequest
from dataset_agent.settings import Settings


def main():
    """Run a test of the dataset research agent."""
    # Sample dataset for testing
    dataset_name = "NASS Census of Agriculture"
    dataset_url = "https://www.nass.usda.gov/AgCensus/"
    
    print(f"Testing dataset research agent with {dataset_name}...")
    
    # Configuration comes from Settings (environment / .env)
    settings = Settings()
    
    # Time the research
    start_time = time.time()
    
    try:
        # Run research
        request = ResearchRequest(dataset_name=dataset_name, dataset_url=dataset_url)
        result, _path = asyncio.run(run_full_pipeline(request, settings))
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Print results
        print("\nTest Results:")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Description length: {len(result.description)} chars")
        print(f"Aliases: {', '.join(result.dataset_names[:5])}...")
        print(f"Flag terms: {', '.join(result.flag_terms[:5])}...")
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