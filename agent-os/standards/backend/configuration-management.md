## Configuration Management Standards

### Configuration Layer Design

- **Centralized Config Class**: Use a single `Config` class to manage all application configuration with clear defaults
- **Environment Variable Precedence**: Constructor parameters override environment variables, which override defaults
- **Validation at Initialization**: Validate configuration values when the Config object is created, not at usage time
- **Immutable Configuration**: Configuration should be set once at initialization and not modified during runtime
- **Type Safety**: Use appropriate types for configuration values (str, int, float, bool) with type hints

### Environment Variable Naming

- **Consistent Prefix**: Use a consistent prefix for all environment variables (e.g., `DATASET_AGENT_`, `LMSTUDIO_`, `OPENROUTER_`)
- **Hierarchical Organization**: Group related settings with common prefixes
- **Uppercase with Underscores**: Follow standard environment variable naming convention
- **Clear Semantics**: Variable names should clearly indicate their purpose and scope

### Configuration Pattern

```python
# ✅ Good: Centralized configuration with validation
class Config:
    def __init__(
        self,
        log_level: str = None,
        llm_provider: str = None,
        llm_model: str = None,
        lmstudio_base_url: str = None,
        temperature: float = None
    ):
        # Environment variables with fallbacks
        self.log_level = self._validate_log_level(
            log_level or os.environ.get("DATASET_AGENT_LOG_LEVEL", "INFO")
        )
        
        self.llm_provider = (
            llm_provider or 
            os.environ.get("DATASET_AGENT_LLM_PROVIDER", "ollama")
        )
        
        self.llm_model = (
            llm_model or 
            os.environ.get("DATASET_AGENT_LLM_MODEL", "qwen3:32b")
        )
        
        self.lmstudio_base_url = (
            lmstudio_base_url or 
            os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        )
        
        self.temperature = float(
            temperature if temperature is not None else 
            os.environ.get("DATASET_AGENT_TEMPERATURE", "0.85")
        )
        
        # Validate provider
        if self.llm_provider not in ["ollama", "openrouter", "lmstudio"]:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")
    
    def _validate_log_level(self, level: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {level}")
        return level.upper()
    
    def __repr__(self) -> str:
        return (
            f"Config(llm_provider='{self.llm_provider}', "
            f"llm_model='{self.llm_model}', "
            f"temperature={self.temperature}, ...)"
        )
```

### Setup Dependencies Pattern

- **Factory Function**: Use a factory function (`setup_dependencies`) to create and wire dependencies based on configuration
- **Provider-Specific Initialization**: Branch on provider type to create appropriate implementations
- **Dependency Injection**: Pass configuration to constructors rather than using global state
- **Clear Error Messages**: Raise clear errors for unsupported or misconfigured providers

```python
# ✅ Good: Factory function with provider branching
def setup_dependencies(config: Config) -> Dict[str, Any]:
    # Create agent based on provider configuration
    if config.llm_provider == "ollama":
        agent = LangChainAgent(
            model_name=config.llm_model,
            temperature=config.temperature,
            provider="ollama"
        )
    elif config.llm_provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for OpenRouter provider")
        
        agent = LangChainAgent(
            model_name=config.llm_model,
            provider="openrouter",
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL")
        )
    elif config.llm_provider == "lmstudio":
        agent = LangChainAgent(
            model_name=config.llm_model,
            provider="lmstudio",
            api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=config.lmstudio_base_url,
            temperature=config.temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    # Create other dependencies
    extractor = LLMOutputExtractor()
    repository = JSONFileRepository(config.output_dir)
    use_case = DatasetResearchUseCase(agent, extractor, repository)
    
    return {
        "agent": agent,
        "extractor": extractor,
        "repository": repository,
        "use_case": use_case,
        "config": config
    }
```

### Configuration Documentation

- **Example Files**: Provide `.env.example` files with all configuration options documented
- **Inline Comments**: Include comments explaining each configuration option's purpose and valid values
- **Default Values**: Clearly document default values for optional configuration
- **Required vs Optional**: Distinguish between required and optional configuration settings

### Provider-Specific Configuration

- **Namespace Isolation**: Keep provider-specific settings in their own namespace
- **Optional Configuration**: Provider-specific settings should only be required when that provider is selected
- **Validation Context**: Validate provider-specific settings only when the provider is active

### Command-Line Interface

- **Argument Override**: Command-line arguments should override environment variables
- **Consistent Naming**: Use consistent naming between CLI args and env variables (with appropriate format conversion)
- **Help Text**: Provide clear help text documenting all CLI options
- **Type Conversion**: Handle type conversion (string to int/float/bool) properly for CLI arguments
