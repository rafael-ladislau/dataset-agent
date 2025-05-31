"""
Main entry point for the dataset agent application.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .config import Config, setup_dependencies
from .domain.models import DatasetInfo

# Set up logger
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Research a dataset using LLM and web search")
    parser.add_argument("dataset_name", help="Name of the dataset to research")
    parser.add_argument("--url", help="Optional URL for the dataset", default=None)
    parser.add_argument("--output-dir", help="Directory to store output files", default=None)
    parser.add_argument("--log-level", help="Logging level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-file", help="Path to log file", default="dataset_agent.log")
    parser.add_argument("--llm-provider", help="LLM provider to use", default="ollama",
                        choices=["ollama", "openrouter"])
    parser.add_argument("--llm-model", help="LLM model to use", default="llama3")
    parser.add_argument("--web-search-provider", help="Web search provider to use", default="duckduckgo",
                        choices=["duckduckgo", "tavily"])
    parser.add_argument("--env-file", help="Path to .env file with API credentials", default=".env")
    
    return parser.parse_args()


def run_research(dataset_name: str, dataset_url: Optional[str] = None, config: Optional[Config] = None) -> DatasetInfo:
    """
    Run dataset research process.
    
    Args:
        dataset_name: Name of the dataset to research
        dataset_url: Optional URL for the dataset
        config: Configuration for the application
        
    Returns:
        DatasetInfo: Research results
    """
    # Set up configuration if not provided
    if config is None:
        config = Config()
    
    # Set up dependencies
    dependencies = setup_dependencies(config)
    
    # Get the use case
    use_case = dependencies["use_case"]
    
    # Run the research
    logger.info(f"Starting research for dataset: {dataset_name}")
    start_time = time.time()
    
    try:
        # Execute the research
        result = use_case.execute(dataset_name, dataset_url)
        
        # Log the results
        execution_time = time.time() - start_time
        logger.info(f"Research completed in {execution_time:.2f} seconds")
        logger.info(f"Description length: {len(result.description)} chars")
        logger.info(f"Found {len(result.aliases)} aliases")
        logger.info(f"Found {len(result.organizations)} organizations")
        logger.info(f"Access type: {result.access_type}")
        
        return result
    except Exception as e:
        logger.error(f"Research failed: {str(e)}", exc_info=True)
        raise


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load environment variables if env file exists
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        logger.info(f"Loaded environment variables from {args.env_file}")
    
    # Log the web search provider being used
    logger.info(f"Using web search provider: {args.web_search_provider}")
    
    # Create configuration
    config = Config(
        log_level=args.log_level,
        log_file=args.log_file,
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        web_search_provider=args.web_search_provider
    )
    
    try:
        # Run research
        result = run_research(args.dataset_name, args.url, config)
        
        # Print results to console
        print("\nDataset Research Results:")
        print(json.dumps(result.to_dict(), indent=2))
        
        return 0
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 