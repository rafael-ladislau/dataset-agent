## Domain Model Patterns

### Dataclass Usage for Domain Models

- **Immutable Data Structures**: Use Python `@dataclass` with `frozen=True` for domain models representing data entities
- **Type Annotations**: Always include complete type annotations for all fields
- **Optional Fields**: Use `Optional[Type]` for fields that may be None, with clear defaults
- **Serialization Methods**: Include `to_dict()` and `from_dict()` methods for JSON serialization/deserialization
- **Metadata Fields**: Separate business data from metadata (e.g., `_metadata` field for timing, status info)

### DatasetInfo Model Pattern

```python
# ✅ Good: Domain model with clear structure
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DatasetInfo:
    """Domain model representing dataset research results."""
    
    # Required core fields
    dataset_name: str
    home_url: Optional[str] = None
    
    # Research results
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    
    # Official name detection (new feature)
    official_name: Optional[str] = None
    relationship_type: Optional[str] = None  # official_name, subset_of, table_within, component_of
    official_name_reasoning: Optional[str] = None
    
    # Access information
    access_type: Optional[str] = None
    data_url: Optional[str] = None
    schema_url: Optional[str] = None
    documentation_url: Optional[str] = None
    
    # Metadata (separate from business data)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "home_url": self.home_url,
            "description": self.description,
            "aliases": self.aliases,
            "organizations": self.organizations,
            "official_name": self.official_name,
            "relationship_type": self.relationship_type,
            "official_name_reasoning": self.official_name_reasoning,
            "access_type": self.access_type,
            "data_url": self.data_url,
            "schema_url": self.schema_url,
            "documentation_url": self.documentation_url,
            "_metadata": self._metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """Create instance from dictionary."""
        return cls(
            dataset_name=data.get("dataset_name", ""),
            home_url=data.get("home_url"),
            description=data.get("description"),
            aliases=data.get("aliases", []),
            organizations=data.get("organizations", []),
            official_name=data.get("official_name"),
            relationship_type=data.get("relationship_type"),
            official_name_reasoning=data.get("official_name_reasoning"),
            access_type=data.get("access_type"),
            data_url=data.get("data_url"),
            schema_url=data.get("schema_url"),
            documentation_url=data.get("documentation_url"),
            _metadata=data.get("_metadata", {})
        )
```

### Use Case Interface Pattern

- **Clear Abstraction**: Define interfaces (abstract base classes) for use cases separate from implementations
- **Single Responsibility**: Each use case should handle one business operation
- **Dependency Injection**: Use cases receive dependencies through constructor injection
- **Return Domain Models**: Use cases return domain model instances, not dictionaries or primitive types

```python
# ✅ Good: Use case with clear interface
from abc import ABC, abstractmethod

class DatasetResearchUseCaseInterface(ABC):
    """Interface for dataset research use case."""
    
    @abstractmethod
    def execute(self, dataset_name: str, dataset_url: Optional[str] = None) -> DatasetInfo:
        """Execute dataset research and return structured information."""
        pass

class DatasetResearchUseCase(DatasetResearchUseCaseInterface):
    """Implementation of dataset research use case."""
    
    def __init__(
        self,
        agent: AgentInterface,
        extractor: ExtractorInterface,
        repository: RepositoryInterface
    ):
        self.agent = agent
        self.extractor = extractor
        self.repository = repository
    
    def execute(self, dataset_name: str, dataset_url: Optional[str] = None) -> DatasetInfo:
        # Use case orchestration logic
        description = self._get_description(dataset_name, dataset_url)
        organizations = self._get_organizations(dataset_name, description)
        official_name_info = self._get_official_name_info(dataset_name, description)
        
        # Build and return domain model
        dataset_info = DatasetInfo(
            dataset_name=dataset_name,
            home_url=dataset_url,
            description=description,
            organizations=organizations,
            official_name=official_name_info['official_name'],
            relationship_type=official_name_info['relationship_type'],
            official_name_reasoning=official_name_info['reasoning']
        )
        
        # Persist via repository
        self.repository.save(dataset_info)
        
        return dataset_info
```

### Adapter Pattern for External Services

- **Interface Definition**: Define interfaces for external service adapters (LLM, storage, search)
- **Implementation Isolation**: Hide external library details behind adapter interface
- **Testability**: Adapters should be easily mockable for testing use cases
- **Error Translation**: Convert external service errors to domain-specific exceptions

```python
# ✅ Good: Storage adapter with clean interface
class RepositoryInterface(ABC):
    """Interface for dataset information storage."""
    
    @abstractmethod
    def save(self, dataset_info: DatasetInfo) -> None:
        """Save dataset information."""
        pass
    
    @abstractmethod
    def find_by_name(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Retrieve dataset information by name."""
        pass

class JSONFileRepository(RepositoryInterface):
    """JSON file-based implementation of repository."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, dataset_info: DatasetInfo) -> None:
        filename = self._sanitize_filename(dataset_info.dataset_name)
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump(dataset_info.to_dict(), f, indent=2)
    
    def find_by_name(self, dataset_name: str) -> Optional[DatasetInfo]:
        filename = self._sanitize_filename(dataset_name)
        filepath = self.output_dir / f"{filename}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            return DatasetInfo.from_dict(data)
```

### Clean Architecture Principles

- **Domain Layer Independence**: Domain models and use case interfaces should not depend on external libraries
- **Dependency Direction**: Dependencies flow inward (adapters depend on domain, not vice versa)
- **Business Logic Isolation**: Keep business rules in use cases and domain models, not in adapters
- **Framework Agnostic**: Core business logic should be independent of FastAPI, LangChain, or other frameworks
