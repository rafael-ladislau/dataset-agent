# Dataset Research Agent

A tool for automatically researching and gathering information about datasets using Large Language Models (LLMs) and web search.

## Overview

The Dataset Research Agent is designed to automate the process of gathering information about datasets. It leverages large language models (LLMs) via Ollama, OpenRouter, or LMStudio and the LangChain framework to efficiently gather the following information about datasets:

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
- **Choose one LLM provider:**
  - **For Ollama (default):**
    - [Ollama](https://ollama.com/) installed and running
    - LLM model pulled in Ollama (e.g., llama3)
  - **For OpenRouter:**
    - OpenRouter API key (set in .env file)
  - **For LMStudio:**
    - [LMStudio](https://lmstudio.ai) installed and running
    - LMStudio server enabled on port 1234
    - A compatible model loaded in LMStudio (e.g., gpt-oss-120b)

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
   ```bash
   # Install Ollama (on macOS or Linux)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull the model you want to use
   ollama pull llama3
   ```

   **For OpenRouter:**
   
   Create a `.env` file in the root directory with your OpenRouter API key:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```

   **For LMStudio:**
   
   1. Download and install [LMStudio](https://lmstudio.ai)
   2. Open LMStudio and download a model (e.g., gpt-oss-120b)
   3. Load the model by clicking the ↔ icon
   4. Start the local server:
      - Go to **Developer** → **Local Server**
      - Click **Start Server** (default port: 1234)
      - Verify it's running: `curl http://localhost:1234/v1/models`
   5. Configure the agent to use LMStudio:
      ```bash
      # Add to .env file
      DATASET_AGENT_LLM_PROVIDER=lmstudio
      DATASET_AGENT_LLM_MODEL=gpt-oss-120b
      LMSTUDIO_BASE_URL=http://localhost:1234/v1
      ```

## Usage

### Command Line Interface

The simplest way to use the Dataset Research Agent is through the command line:

```bash
python -m src.dataset_agent.main "Census of Agriculture"
```

You can also provide an optional URL as a starting point:

```bash
python -m src.dataset_agent.main "Census of Agriculture" --url "https://www.nass.usda.gov/AgCensus/"
```

Additional command line options:

```bash
python -m src.dataset_agent.main "Dataset Name" [--url URL] [--output-dir OUTPUT_DIR] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE] [--llm-provider {ollama,openrouter,lmstudio}] [--llm-model MODEL_NAME] [--env-file ENV_FILE_PATH]
```

#### LLM Provider Options

- `--llm-provider`: Choose between:
  - `ollama` (default) - Uses locally running Ollama
  - `openrouter` - Uses OpenRouter cloud API
  - `lmstudio` - Uses locally running LMStudio with OpenAI-compatible API
- `--llm-model`: Specify the model to use
  - For Ollama: model name (e.g., "llama3", "qwen3:32b")
  - For OpenRouter: model identifier (e.g., "anthropic/claude-3-opus")
  - For LMStudio: model name shown in LMStudio (e.g., "gpt-oss-120b")
- `--env-file`: Path to .env file with API credentials (defaults to ".env")

#### Examples:

**Using Ollama with a specific model:**
```bash
python -m src.dataset_agent.main "Census of Agriculture" --llm-provider ollama --llm-model llama3
```

**Using OpenRouter:**
```bash
python -m src.dataset_agent.main "Census of Agriculture" --llm-provider openrouter --llm-model anthropic/claude-3-opus
```

**Using LMStudio:**
```bash
# Make sure LMStudio is running with a model loaded
python -m src.dataset_agent.main "Census of Agriculture" --llm-provider lmstudio --llm-model gpt-oss-120b
```

**Using LMStudio with custom port:**
```bash
LMSTUDIO_BASE_URL=http://localhost:5000/v1 python -m src.dataset_agent.main "Census of Agriculture" --llm-provider lmstudio --llm-model gpt-oss-120b
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
    llm_provider="lmstudio",  # or "ollama", "openrouter"
    llm_model="gpt-oss-120b",  # or your preferred model
    lmstudio_base_url="http://localhost:1234/v1"  # only needed for LMStudio
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
dataset-agent/
├── src/
│   └── dataset_agent/
│       ├── main.py          # CLI entry point
│       ├── config.py        # Configuration
│       ├── domain/          # Core business logic (models, usecases)
│       ├── adapters/        # LLM agent, storage, tools, extractor
│       └── utils/           # Text processing utilities
├── docs/                    # Project documentation
│   ├── agent-os-overview.md
│   ├── lmstudio-integration-guide.md
│   ├── lmstudio-provider-spec.md
│   ├── lmstudio-quick-reference.md
│   ├── official-name-detection-summary.md
│   ├── datasets-social-economic.md
│   ├── aliases-agent-summary.md
│   └── white-paper-dataset-research-agent.md
├── scripts/                 # Batch run scripts + output processing
│   ├── README.md            # Script documentation
│   ├── run_asthma_datasets.sh
│   ├── run_employment_datasets.sh
│   ├── run_health_welfare_datasets.sh
│   ├── run_pediatric_datasets.sh
│   ├── run_social_economic_datasets.sh
│   └── dataset_metadata_spreadsheet.py
├── tests/                   # Test scripts
│   ├── test_aliases.py
│   ├── test_description_cleaning.py
│   └── test_integration.py
├── results/                 # Research result JSON files
│   └── TRACKING.md          # Batch run history and metadata
├── output/                  # Employment batch results
├── workforce/               # Workforce batch results
├── data/                    # Data files and generated spreadsheets
├── agent-os/                # AI agent configuration (standards, specs)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
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

## Troubleshooting

### LMStudio Issues

**Error: "LMStudio server is not responding"**
- Ensure LMStudio is running
- Check that the server is enabled in LMStudio (**Developer** → **Local Server**)
- Verify the port is 1234 (or update `LMSTUDIO_BASE_URL` in your `.env` file)
- Test the connection: `curl http://localhost:1234/v1/models`

**Error: "No models loaded"**
- Load a model in LMStudio before running the agent
- Click the **↔** icon in LMStudio to load a model into memory
- Verify the model is loaded: `curl http://localhost:1234/v1/models`

**Slow responses with LMStudio**
- Use a smaller model for faster inference
- Reduce context length in LMStudio settings
- Enable GPU acceleration if available (check LMStudio settings)
- Consider using a quantized model (e.g., Q4 or Q8)

**Tool calling not working**
- Ensure your model supports function calling
- Check LMStudio logs for tool-related errors
- Try a different model (recommended: gpt-oss, llama-3.2, qwen)
- Verify that the model is properly loaded and responding

### General Issues

**Import Errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- If using a virtual environment, ensure it's activated

**Web Search Failures**
- Check your internet connection
- Verify your Tavily API key is set (if using Tavily)
- Try switching to DuckDuckGo: `WEB_SEARCH_PROVIDER=duckduckgo`

## Batch Processing

To run the agent against a predefined list of datasets, use the batch scripts in `scripts/`. See [`scripts/README.md`](scripts/README.md) for full documentation.

```bash
./scripts/run_health_welfare_datasets.sh
START_FROM=5 ./scripts/run_employment_datasets.sh
```

To convert results to an Excel spreadsheet:

```bash
python scripts/dataset_metadata_spreadsheet.py results/ data/output.xlsx
```

## Documentation

| Document | Description |
|---|---|
| [docs/agent-os-overview.md](docs/agent-os-overview.md) | Agent OS structure and product context |
| [docs/lmstudio-integration-guide.md](docs/lmstudio-integration-guide.md) | LMStudio setup and integration guide |
| [docs/lmstudio-provider-spec.md](docs/lmstudio-provider-spec.md) | LMStudio feature specification |
| [docs/official-name-detection-summary.md](docs/official-name-detection-summary.md) | Official name detection implementation |
| [docs/white-paper-dataset-research-agent.md](docs/white-paper-dataset-research-agent.md) | Comprehensive project white paper |
| [results/TRACKING.md](results/TRACKING.md) | History of batch runs and agent evolution |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.