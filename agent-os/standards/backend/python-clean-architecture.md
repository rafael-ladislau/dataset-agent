## Python Clean Architecture Standards

### Project Structure

Follow clean architecture with clear layer separation:

```
src/dataset_agent/
├── __init__.py
├── main.py              # Application entry point
├── config.py            # Configuration handling
├── domain/              # Core business logic (no external dependencies)
│   ├── __init__.py
│   ├── models.py        # Domain models (dataclasses)
│   └── usecases.py      # Business logic interfaces (ABC)
├── adapters/            # Implementation adapters
│   ├── __init__.py
│   ├── agent.py         # LLM agent implementation
│   ├── extractor.py     # Text extraction implementation
│   ├── storage.py       # Data storage implementation
│   └── tools.py         # Tool implementations
└── utils/               # Utility functions
    └── __init__.py
```

### Dependency Direction

- **Domain Layer**: Contains business logic, no external dependencies
- **Adapters Layer**: Implements interfaces defined in domain, depends on external libraries
- **Application Layer**: Wires dependencies together, orchestrates execution

### Interface-Based Design with ABC

Use Python's Abstract Base Classes for defining interfaces:

```python
# ✅ Good: Clear interface definition
from abc import ABC, abstractmethod
from typing import Optional

class AgentInterface(ABC):
    """Interface for LLM agent implementations."""
    
    @abstractmethod
    def query(self, prompt: str, tools: Optional[list] = None) -> str:
        """Execute a query with optional tools."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state."""
        pass

class LangChainAgent(AgentInterface):
    """LangChain implementation of AgentInterface."""
    
    def query(self, prompt: str, tools: Optional[list] = None) -> str:
        # Implementation using LangChain
        pass
    
    def reset(self) -> None:
        # Implementation-specific reset logic
        pass
```

### Dependency Injection Pattern

- **Constructor Injection**: Pass dependencies through constructors, not global state
- **Interface Parameters**: Accept interfaces, not concrete implementations
- **Factory Functions**: Use factory functions to wire dependencies based on configuration

```python
# ✅ Good: Dependency injection via constructor
class DatasetResearchUseCase:
    def __init__(
        self,
        agent: AgentInterface,
        extractor: ExtractorInterface,
        repository: RepositoryInterface
    ):
        self.agent = agent
        self.extractor = extractor
        self.repository = repository
    
    def execute(self, dataset_name: str) -> DatasetInfo:
        # Use injected dependencies
        description = self.agent.query(f"Describe {dataset_name}")
        extracted = self.extractor.extract(description)
        self.repository.save(extracted)
        return extracted
```

### Type Hints and Type Safety

- **Complete Annotations**: Use type hints for all function parameters and return values
- **Optional Types**: Use `Optional[Type]` for nullable values
- **Generic Types**: Use `List[Type]`, `Dict[str, Type]` for collections
- **Type Aliases**: Create type aliases for complex types to improve readability

```python
# ✅ Good: Complete type annotations
from typing import List, Dict, Optional, Any

def process_datasets(
    dataset_names: List[str],
    config: Config,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Process multiple datasets and return results."""
    results: Dict[str, DatasetInfo] = {}
    
    for name in dataset_names:
        info = research_dataset(name, config)
        results[name] = info
    
    return results
```

### Dataclass Usage

- **Domain Models**: Use `@dataclass` for domain models and data transfer objects
- **Immutability**: Consider `frozen=True` for immutable data structures
- **Default Factories**: Use `field(default_factory=list)` for mutable defaults
- **Serialization**: Include `to_dict()` and `from_dict()` methods

```python
# ✅ Good: Dataclass for domain model
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DatasetInfo:
    dataset_name: str
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "description": self.description,
            "aliases": self.aliases,
            "organizations": self.organizations
        }
```

### Error Handling

- **Specific Exceptions**: Define custom exception types for domain errors
- **Exception Hierarchy**: Create exception hierarchies for related errors
- **Context Preservation**: Include context in exception messages
- **Catch Specific**: Catch specific exceptions, not bare `except:`

```python
# ✅ Good: Custom exception hierarchy
class DatasetAgentError(Exception):
    """Base exception for dataset agent errors."""
    pass

class ProviderNotAvailableError(DatasetAgentError):
    """Raised when LLM provider is not available."""
    pass

class ConfigurationError(DatasetAgentError):
    """Raised when configuration is invalid."""
    pass

# Usage
def initialize_provider(config: Config) -> AgentInterface:
    if config.llm_provider not in ["ollama", "openrouter", "lmstudio"]:
        raise ConfigurationError(
            f"Invalid provider: {config.llm_provider}. "
            f"Must be one of: ollama, openrouter, lmstudio"
        )
```

### Logging Best Practices

- **Logger per Module**: Use `logger = logging.getLogger(__name__)`
- **Appropriate Levels**: DEBUG for detailed info, INFO for progress, WARNING for issues, ERROR for failures
- **Structured Context**: Include relevant context in log messages
- **No Print Statements**: Use logging instead of print() for production code

```python
# ✅ Good: Proper logging usage
import logging

logger = logging.getLogger(__name__)

class LangChainAgent(AgentInterface):
    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider
        logger.info(
            f"Initializing agent with provider={provider}, model={model_name}"
        )
        self._initialize()
    
    def query(self, prompt: str) -> str:
        logger.debug(f"Executing query with prompt length: {len(prompt)}")
        try:
            result = self._execute_query(prompt)
            logger.info("Query executed successfully")
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            raise
```

### Testing Patterns

- **Test Isolation**: Each test should be independent and not rely on execution order
- **Mock External Dependencies**: Use mocks for external services (LLMs, APIs)
- **Test Interfaces**: Test against interfaces, not implementations
- **Fixture Organization**: Use pytest fixtures for common test setup

```python
# ✅ Good: Testing with mocks and fixtures
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_agent():
    agent = Mock(spec=AgentInterface)
    agent.query.return_value = "Test description"
    return agent

@pytest.fixture
def mock_repository():
    repository = Mock(spec=RepositoryInterface)
    return repository

def test_dataset_research_use_case(mock_agent, mock_repository):
    # Arrange
    extractor = LLMOutputExtractor()
    use_case = DatasetResearchUseCase(mock_agent, extractor, mock_repository)
    
    # Act
    result = use_case.execute("Test Dataset")
    
    # Assert
    assert result.dataset_name == "Test Dataset"
    mock_agent.query.assert_called()
    mock_repository.save.assert_called_once()
```

### Resource Management

- **Context Managers**: Use context managers for file handles, connections
- **Explicit Cleanup**: Implement cleanup in finally blocks or context manager __exit__
- **Path Objects**: Use `pathlib.Path` instead of string manipulation for file paths

```python
# ✅ Good: Resource management with context managers
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def managed_file_write(filepath: Path):
    """Context manager for safe file writing."""
    temp_path = filepath.with_suffix('.tmp')
    
    try:
        f = open(temp_path, 'w')
        yield f
        f.close()
        # Atomic rename
        temp_path.rename(filepath)
    except Exception:
        f.close()
        if temp_path.exists():
            temp_path.unlink()
        raise

# Usage
def save_dataset(data: DatasetInfo, output_dir: Path):
    filepath = output_dir / f"{data.dataset_name}.json"
    
    with managed_file_write(filepath) as f:
        json.dump(data.to_dict(), f, indent=2)
```

### Environment Variable Handling

- **os.environ.get() with Defaults**: Always provide sensible defaults
- **Type Conversion**: Convert environment variables to appropriate types
- **Validation**: Validate environment variable values at startup
- **Documentation**: Document all environment variables in README and .env.example

```python
# ✅ Good: Environment variable handling
import os
from typing import Optional

class Config:
    def __init__(self, llm_provider: Optional[str] = None):
        # Get from parameter, env var, or default
        self.llm_provider = (
            llm_provider or 
            os.environ.get("DATASET_AGENT_LLM_PROVIDER", "ollama")
        )
        
        # Type conversion with validation
        self.temperature = float(
            os.environ.get("DATASET_AGENT_TEMPERATURE", "0.85")
        )
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
```
