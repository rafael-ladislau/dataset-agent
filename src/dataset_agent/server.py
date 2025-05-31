"""
Server entry point for the dataset research API.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

import uvicorn
from dotenv import load_dotenv

# Add parent directory to sys.path for running from project root
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from .api import app

def setup_logging(log_path: str, log_level: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_path: Directory to store log files
        log_level: Logging level (DEBUG, INFO, etc.)
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    verbose_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(verbose_formatter)
    root_logger.addHandler(console_handler)
    
    # Create rotating file handler for general logs
    general_log_path = log_dir / 'dataset_api.log'
    general_handler = RotatingFileHandler(
        general_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    general_handler.setFormatter(verbose_formatter)
    root_logger.addHandler(general_handler)
    
    # Create rotating file handler for errors
    error_log_path = log_dir / 'dataset_api_errors.log'
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(verbose_formatter)
    root_logger.addHandler(error_handler)
    
    # Create access log handler
    access_log_path = log_dir / 'dataset_api_access.log'
    access_handler = RotatingFileHandler(
        access_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    access_formatter = logging.Formatter('%(asctime)s - %(message)s')
    access_handler.setFormatter(access_formatter)
    
    # Set access log handler for uvicorn
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = [access_handler]
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized. Log files will be stored in {log_dir}")

def load_environment() -> Dict[str, Any]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dict[str, Any]: Environment configuration
    """
    # Load .env file
    env_file = os.environ.get('ENV_FILE', '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded environment variables from {env_file}")
    
    # Extract configuration from environment variables
    return {
        'host': os.environ.get('API_HOST', '0.0.0.0'),
        'port': int(os.environ.get('API_PORT', '8000')),
        'log_path': os.environ.get('LOG_PATH', './logs'),
        'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
        'sqlite_db_path': os.environ.get('SQLITE_DB_PATH', './data/research.db'),
        'api_keys': os.environ.get('API_KEYS', ''),
        'ollama_host': os.environ.get('OLLAMA_HOST', 'http://localhost:11434'),
        'openrouter_api_key': os.environ.get('OPENROUTER_API_KEY', ''),
        'tavily_api_key': os.environ.get('TAVILY_API_KEY', ''),
    }

def main() -> None:
    """Main entry point for the server."""
    # Load environment variables
    config = load_environment()
    
    # Setup logging
    setup_logging(config['log_path'], config['log_level'])
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Dataset Research API server on {config['host']}:{config['port']}")
    
    # Start the server
    uvicorn.run(
        "dataset_agent.api:app",
        host=config['host'],
        port=config['port'],
        log_level=config['log_level'].lower(),
        reload=False
    )

# For running from project root with: python -m src.dataset_agent.server
if __name__ == "__main__":
    main() 