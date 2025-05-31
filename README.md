# Dataset Research Agent

A tool for automatically researching and gathering information about datasets using Large Language Models (LLMs) and web search.

## Overview

The Dataset Research Agent is designed to automate the process of gathering information about datasets. It leverages large language models (LLMs) via Ollama or OpenRouter and the LangChain framework to efficiently gather the following information about datasets:

- Comprehensive descriptions
- Alternative names and identifiers (aliases)
- Organizations associated with the dataset
- Access type (Open, Restricted, Unknown)
- Data download URLs
- Schema/data dictionary URLs
- Documentation URLs

This tool is particularly useful for data cataloging, metadata enrichment, and dataset discovery tasks.

## Features

- **Automated Research**: Automatically searches the web to find information about datasets
- **Comprehensive Information Gathering**: Collects descriptions, aliases, organizations, URLs, and more
- **Robust Text Processing**: Uses multiple extraction methods to process LLM responses
- **Clean Architecture**: Follows SOLID principles with a clear separation of concerns
- **Flexible Storage**: Stores research results in JSON files for easy integration with other systems

## Prerequisites

- Python 3.8+
- For Ollama (default):
  - [Ollama](https://ollama.com/) installed and running
  - LLM model pulled in Ollama (e.g., llama3)
- For OpenRouter:
  - OpenRouter API key (set in .env file)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dataset-research-agent.git
   cd dataset-research-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your LLM provider:
   
   **For Ollama (default):**
   ```
   # Install Ollama (on macOS or Linux)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull the model you want to use
   ollama pull llama3
   ```

   **For OpenRouter:**
   
   Create a `.env` file in the root directory with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

### Command Line Interface

The simplest way to use the Dataset Research Agent is through the command line:

```
python dataset_research.py "Census of Agriculture"
```

You can also provide an optional URL as a starting point:

```
python dataset_research.py "Census of Agriculture" --url "https://www.nass.usda.gov/AgCensus/"
```

Additional command line options:

```
python dataset_research.py "Dataset Name" [--url URL] [--output-dir OUTPUT_DIR] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE] [--llm-provider {ollama,openrouter}] [--llm-model MODEL_NAME] [--env-file ENV_FILE_PATH]
```

#### LLM Provider Options

- `--llm-provider`: Choose between "ollama" (default, uses locally running Ollama) or "openrouter" (uses OpenRouter API)
- `--llm-model`: Specify the model to use
  - For Ollama: model name (e.g., "llama3")
  - For OpenRouter: model identifier (e.g., "anthropic/claude-3-opus")
- `--env-file`: Path to .env file with API credentials (defaults to ".env")

Examples:

Using Ollama with a specific model:
```
python dataset_research.py "Census of Agriculture" --llm-provider ollama --llm-model llama3
```

Using OpenRouter:
```
python dataset_research.py "Census of Agriculture" --llm-provider openrouter --llm-model anthropic/claude-3-opus
```

### Programmatic Usage

You can also use the Dataset Research Agent in your own Python code:

```python
from src.dataset_agent.config import Config
from src.dataset_agent.main import run_research

# Create a configuration (optional)
config = Config(
    log_level="INFO", 
    output_dir="/path/to/output",
    llm_provider="openrouter",  # or "ollama"
    llm_model="anthropic/claude-3-opus"  # or your preferred model
)

# Run research
dataset_info = run_research("Census of Agriculture", config=config)

# Access research results
print(f"Description: {dataset_info.description}")
print(f"Aliases: {dataset_info.aliases}")
print(f"Access type: {dataset_info.access_type}")
print(f"Data URL: {dataset_info.data_url}")
```

## Project Structure

The project follows a clean architecture approach with clear separation of concerns:

```
src/
├── dataset_agent/
│   ├── __init__.py
│   ├── main.py          # Application entry point
│   ├── config.py        # Configuration handling
│   ├── domain/          # Core business logic
│   │   ├── __init__.py
│   │   ├── models.py    # Data models
│   │   └── usecases.py  # Business logic interfaces
│   ├── adapters/        # Implementation adapters
│   │   ├── __init__.py
│   │   ├── agent.py     # LLM agent implementation
│   │   ├── extractor.py # Text extraction implementation
│   │   ├── storage.py   # Data storage implementation
│   │   └── tools.py     # Tool implementations
│   └── utils/           # Utility functions
│       └── __init__.py
└── __init__.py
```

## Design Principles

This project follows SOLID principles and clean architecture:

- **Single Responsibility Principle**: Each class has a single responsibility
- **Open/Closed Principle**: The code is open for extension but closed for modification
- **Liskov Substitution Principle**: Different implementations can be substituted without affecting the core logic
- **Interface Segregation Principle**: Interfaces are specific to their clients
- **Dependency Inversion Principle**: High-level modules depend on abstractions

The project uses a use-case driven approach that:

1. Clearly defines the primary business logic in the domain layer
2. Implements adapters for external services (LLM, web search, storage)
3. Uses dependency injection to make the system flexible and testable

## Output Format

Research results are saved as JSON files with the following structure:

```json
{
  "dataset_name": "Census of Agriculture",
  "home_url": "https://www.nass.usda.gov/AgCensus/",
  "description": "...",
  "aliases": ["Ag Census", "USDA Census of Agriculture", ...],
  "organizations": ["USDA", "National Agricultural Statistics Service", ...],
  "access_type": "Open",
  "data_url": "https://www.nass.usda.gov/AgCensus/Data/",
  "schema_url": "https://www.nass.usda.gov/AgCensus/Documentation/",
  "documentation_url": "https://www.nass.usda.gov/AgCensus/Help/",
  "_metadata": {
    "timing": {
      "description": 12.5,
      "organizations": 8.3,
      "total": 120.2
    },
    "status": "success",
    "completed": true
  }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.