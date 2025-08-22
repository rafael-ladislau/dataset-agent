"""
Configuration settings for the dataset agent.
"""

import os
import logging
from typing import Dict, Any, Optional


class Config:
    """Configuration settings for the dataset agent."""
    
    def __init__(self, log_level: str = None, log_file: str = None, 
                 output_dir: str = None, llm_provider: str = None, llm_model: str = None,
                 web_search_provider: str = None, temperature: float = None,
                 top_k: int = None, top_p: float = None):
        """
        Initialize configuration.
        
        Automatically loads environment variables from .env file if available.
        Uses provided parameters or falls back to environment variables, then to defaults.
        
        Args:
            log_level: Logging level (default: "INFO")
            log_file: Path to log file (default: "dataset_agent.log")
            output_dir: Directory to store output files (default: current working directory)
            llm_provider: LLM provider (default: "ollama")
            llm_model: LLM model to use (default: "qwen3:32b")
            web_search_provider: Web search provider (default: "tavily")
        """
        # Load environment variables from .env file if it exists
        self._load_env_file()
        
        # Set configuration values with priority: parameter > env var > default
        self.log_level = self._parse_log_level(
            log_level or os.environ.get("DATASET_AGENT_LOG_LEVEL", "INFO")
        )
        self.log_file = log_file or os.environ.get("DATASET_AGENT_LOG_FILE", "dataset_agent.log")
        self.output_dir = (
            output_dir or 
            os.environ.get("DATASET_AGENT_OUTPUT_DIR") or 
            os.getcwd()
        )
        self.llm_provider = llm_provider or os.environ.get("DATASET_AGENT_LLM_PROVIDER", "ollama")
        self.llm_model = llm_model or os.environ.get("DATASET_AGENT_LLM_MODEL", "qwen3:32b")
        self.web_search_provider = (
            web_search_provider or 
            os.environ.get("WEB_SEARCH_PROVIDER", "tavily")
        )
        # Sampling parameters (only applied for providers that support them, e.g., Ollama)
        self.temperature = float(temperature) if temperature is not None else float(os.environ.get("DATASET_AGENT_TEMPERATURE", 0.85))
        self.top_k = int(top_k) if top_k is not None else int(os.environ.get("DATASET_AGENT_TOP_K", 40))
        try:
            self.top_p = float(top_p) if top_p is not None else float(os.environ.get("DATASET_AGENT_TOP_P", 0.95))
        except ValueError:
            self.top_p = 0.95
        
        # Set web search provider in environment for other components
        os.environ["WEB_SEARCH_PROVIDER"] = self.web_search_provider
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        self._configure_logging()
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file if it exists."""
        try:
            from dotenv import load_dotenv, find_dotenv
            env_file = find_dotenv()
            if env_file:
                load_dotenv(env_file)
        except ImportError:
            # dotenv not available, skip loading
            pass
    
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
    
    def _configure_logging(self) -> None:
        """Configure logging for the application."""
        # Create logger
        logger = logging.getLogger("dataset_agent")
        logger.setLevel(self.log_level)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
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
    
    def __repr__(self) -> str:
        """Return string representation of the configuration."""
        return (
            f"Config(log_level={logging.getLevelName(self.log_level)}, "
            f"log_file='{self.log_file}', "
            f"output_dir='{self.output_dir}', "
            f"llm_provider='{self.llm_provider}', "
            f"llm_model='{self.llm_model}', "
            f"web_search_provider='{self.web_search_provider}', "
            f"temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            "log_level": logging.getLevelName(self.log_level),
            "log_file": self.log_file,
            "output_dir": self.output_dir,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "web_search_provider": self.web_search_provider,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p
        }


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
    
    print(f"Config: {config}")
    # Create agent based on provider configuration
    if config.llm_provider == "ollama":
        agent = LangChainAgent(
            model_name=config.llm_model,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
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