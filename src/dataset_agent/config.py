"""
Configuration settings for the dataset agent.
"""

import os
import logging
from typing import Dict, Any, Optional


class Config:
    """Configuration settings for the dataset agent."""
    
    def __init__(self, log_level: str = "INFO", log_file: str = "dataset_agent.log", 
                 output_dir: str = None, llm_provider: str = "ollama", llm_model: str = "qwen3:32b",
                 web_search_provider: str = "tavily"):
        """
        Initialize configuration.
        
        Args:
            log_level: Logging level
            log_file: Path to log file
            output_dir: Directory to store output files
            llm_provider: LLM provider (ollama or openrouter)
            llm_model: LLM model to use
            web_search_provider: Web search provider (duckduckgo or tavily)
        """
        self.log_level = self._parse_log_level(log_level)
        self.log_file = log_file
        self.output_dir = output_dir or os.getcwd()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.web_search_provider = web_search_provider
        
        # Set web search provider in environment immediately
        os.environ["WEB_SEARCH_PROVIDER"] = self.web_search_provider
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        self._configure_logging()
    
    def _parse_log_level(self, log_level: str) -> int:
        """
        Parse log level string to logging level.
        
        Args:
            log_level: Log level as string
            
        Returns:
            int: Logging level
        """
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return levels.get(log_level.upper(), logging.INFO)
    
    def _configure_logging(self):
        """Configure logging for the application."""
        # Create logger
        logger = logging.getLogger("dataset_agent")
        logger.setLevel(self.log_level)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Logging configured")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables.
        
        Returns:
            Config: Configuration instance
        """
        log_level = os.environ.get("DATASET_AGENT_LOG_LEVEL", "INFO")
        log_file = os.environ.get("DATASET_AGENT_LOG_FILE", "dataset_agent.log")
        output_dir = os.environ.get("DATASET_AGENT_OUTPUT_DIR", None)
        llm_provider = os.environ.get("DATASET_AGENT_LLM_PROVIDER", "ollama")
        llm_model = os.environ.get("DATASET_AGENT_LLM_MODEL", "llama3")
        web_search_provider = os.environ.get("WEB_SEARCH_PROVIDER", "duckduckgo")
        
        return cls(
            log_level=log_level, 
            log_file=log_file, 
            output_dir=output_dir,
            llm_provider=llm_provider,
            llm_model=llm_model,
            web_search_provider=web_search_provider
        )


def setup_dependencies(config: Config) -> Dict[str, Any]:
    """
    Set up dependencies for the application.
    
    Args:
        config: Configuration settings
        
    Returns:
        Dict[str, Any]: Dictionary of dependencies
    """
    from .adapters.agent import LangChainAgent
    from .adapters.extractor import LLMOutputExtractor
    from .adapters.storage import JSONFileRepository
    from .domain.usecases import DatasetResearchUseCase
    
    # Create agent based on provider configuration
    if config.llm_provider == "ollama":
        agent = LangChainAgent(model_name=config.llm_model)
    elif config.llm_provider == "openrouter":
        # Verify that OpenRouter API key exists
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter provider")
        
        # Create agent with OpenRouter configuration
        agent = LangChainAgent(
            model_name=config.llm_model,
            provider="openrouter",
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    # Create other dependencies
    extractor = LLMOutputExtractor()
    repository = JSONFileRepository(output_dir=config.output_dir)
    
    # Create use case
    use_case = DatasetResearchUseCase(
        agent=agent,
        extractor=extractor,
        repository=repository
    )
    
    return {
        "agent": agent,
        "extractor": extractor,
        "repository": repository,
        "use_case": use_case
    } 