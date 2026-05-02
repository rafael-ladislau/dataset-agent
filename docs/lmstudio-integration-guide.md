# LMstudio Provider - Quick Implementation Guide

This guide provides step-by-step instructions for implementing LMstudio support based on the [SPEC_LMSTUDIO_PROVIDER.md](./SPEC_LMSTUDIO_PROVIDER.md) specification.

## Prerequisites

Before starting implementation:

1. ✅ Read the full specification: `SPEC_LMSTUDIO_PROVIDER.md`
2. ✅ Install LMstudio: https://lmstudio.ai
3. ✅ Download a compatible model in LMstudio (e.g., qwen3:32b)
4. ✅ Load the model and start the server in LMstudio
5. ✅ Test the server: `curl http://localhost:1234/v1/models`

## Implementation Steps

### Phase 1: Core Implementation (2 hours)

#### Step 1: Update `src/dataset_agent/config.py` (30 min)

**1.1** Add LMstudio configuration parameter to `Config.__init__`:

```python
def __init__(self, log_level: str = None, log_file: str = None, 
             output_dir: str = None, llm_provider: str = None, llm_model: str = None,
             web_search_provider: str = None, temperature: float = None,
             top_k: int = None, top_p: float = None,
             lmstudio_base_url: str = None):  # ADD THIS LINE
```

**1.2** Add LMstudio URL configuration in the `__init__` body:

```python
# After the existing configurations, add:
self.lmstudio_base_url = (
    lmstudio_base_url or 
    os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
)
```

**1.3** Update `setup_dependencies` function to handle `lmstudio` provider:

```python
def setup_dependencies(config: Config) -> Dict[str, Any]:
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
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter provider")
        
        agent = LangChainAgent(
            model_name=config.llm_model,
            provider="openrouter",
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
    elif config.llm_provider == "lmstudio":  # ADD THIS BLOCK
        # Create agent with LMstudio configuration
        agent = LangChainAgent(
            model_name=config.llm_model,
            provider="lmstudio",
            api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=config.lmstudio_base_url,
            temperature=config.temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    # ... rest of function
```

**1.4** Update the `__repr__` method to include lmstudio_base_url:

```python
def __repr__(self) -> str:
    """Return string representation of the configuration."""
    return (
        f"Config(log_level={logging.getLevelName(self.log_level)}, "
        f"log_file='{self.log_file}', "
        f"output_dir='{self.output_dir}', "
        f"llm_provider='{self.llm_provider}', "
        f"llm_model='{self.llm_model}', "
        f"web_search_provider='{self.web_search_provider}', "
        f"temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}, "
        f"lmstudio_base_url='{self.lmstudio_base_url}')"  # ADD THIS
    )
```

#### Step 2: Update `src/dataset_agent/adapters/agent.py` (1.5 hours)

**2.1** Update the class docstring to mention LMstudio:

```python
class LangChainAgent(AgentInterface):
    """
    Implementation of AgentInterface using LangChain with Ollama, OpenRouter, or LMstudio.
    """
```

**2.2** Update `__init__` docstring to document LMstudio support:

```python
def __init__(self, model_name: str = "gpt-oss:20b", temperature: float = 0.85, 
             provider: str = "ollama", api_key: Optional[str] = None, 
             base_url: Optional[str] = None, top_k: Optional[int] = None, top_p: Optional[float] = None):
    """
    Initialize the LangChain agent.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for the model
        provider: LLM provider (ollama, openrouter, or lmstudio)  # UPDATE THIS LINE
        api_key: API key for OpenRouter or placeholder for LMstudio  # UPDATE THIS LINE
        base_url: Base URL for OpenRouter or LMstudio API  # UPDATE THIS LINE
        top_k: Top-K sampling (Ollama only, LMstudio configures via UI)  # UPDATE THIS LINE
        top_p: Top-P sampling (Ollama only, LMstudio configures via UI)  # UPDATE THIS LINE
    """
```

**2.3** Add LMstudio initialization block in `_initialize_agent` method, after the OpenRouter block:

```python
elif self.provider == "lmstudio":
    from langchain_openai import ChatOpenAI
    
    # Check if LMstudio is available
    if not self._check_lmstudio_health():
        logger.warning(
            f"LMstudio server is not responding at {self.base_url or 'http://localhost:1234/v1'}. "
            "Please ensure LMstudio is running with a loaded model. "
            "Visit https://lmstudio.ai for installation instructions."
        )
    
    # Initialize LMstudio LLM using OpenAI-compatible client
    llm = ChatOpenAI(
        model=self.model_name,
        temperature=self.temperature,
        api_key=self.api_key or "lm-studio",
        base_url=self.base_url or "http://localhost:1234/v1",
        streaming=False
    )
    logger.info(
        f"Initialized LMstudio LLM with model {self.model_name} "
        f"at {self.base_url or 'http://localhost:1234/v1'}"
    )
    
else:
    raise ValueError(f"Unsupported provider: {self.provider}")
```

**2.4** Add the `_check_lmstudio_health` method after `_check_ollama_health`:

```python
def _check_lmstudio_health(self) -> bool:
    """Check if LMstudio server is running and has models loaded."""
    import requests
    
    try:
        base = self.base_url or "http://localhost:1234/v1"
        health_url = f"{base.rstrip('/')}/models"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            
            if models:
                model_names = [m.get("id", "unknown") for m in models]
                logger.info(
                    f"LMstudio is running with {len(models)} model(s) loaded: "
                    f"{', '.join(model_names)}"
                )
                return True
            else:
                logger.warning(
                    "LMstudio server is running but no models are loaded. "
                    "Please load a model in LMstudio before using the agent."
                )
                return False
        return False
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Cannot connect to LMstudio at {base}. "
            "Ensure LMstudio is running and the server is enabled."
        )
        return False
    except Exception as e:
        logger.error(f"LMstudio server health check failed: {str(e)}")
        return False
```

### Phase 2: Testing (1.5 hours)

#### Step 3: Manual Testing

**3.1** Set up environment:
```bash
export DATASET_AGENT_LLM_PROVIDER=lmstudio
export DATASET_AGENT_LLM_MODEL=qwen3:32b
export LMSTUDIO_BASE_URL=http://localhost:1234/v1
export TAVILY_API_KEY=your_tavily_key
```

**3.2** Test basic functionality:
```bash
python -m src.dataset_agent.main "Census of Agriculture"
```

**3.3** Expected output:
- Agent should initialize successfully
- Health check should detect LMstudio
- Query should execute and produce results
- Check the log file for proper initialization messages

**3.4** Test with API:
```bash
# Start the API server
python -m src.dataset_agent.server

# In another terminal, submit a request
curl -X POST "http://localhost:8000/api/research" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"dataset_name": "Test Dataset"}'
```

#### Step 4: Error Handling Testing

**4.1** Test with LMstudio stopped:
```bash
# Stop LMstudio server
# Run the agent again
python -m src.dataset_agent.main "Test Dataset"
```

Expected: Clear error message about LMstudio not running

**4.2** Test with no model loaded:
```bash
# In LMstudio, unload all models
# Run the agent again
python -m src.dataset_agent.main "Test Dataset"
```

Expected: Warning about no models loaded

#### Step 5: Provider Switching Test

**5.1** Test switching from Ollama to LMstudio:
```bash
# Test with Ollama
export DATASET_AGENT_LLM_PROVIDER=ollama
python -m src.dataset_agent.main "Test Dataset"

# Switch to LMstudio
export DATASET_AGENT_LLM_PROVIDER=lmstudio
python -m src.dataset_agent.main "Test Dataset"
```

**5.2** Verify no interference between providers

### Phase 3: Documentation (1 hour)

#### Step 6: Update README.md

**6.1** Add LMstudio to prerequisites:
```markdown
### Prerequisites

- Python 3.8+
- For Ollama (default):
  - [Ollama](https://ollama.com/) installed and running
  - LLM model pulled in Ollama (e.g., llama3)
- For OpenRouter:
  - OpenRouter API key (set in .env file)
- For LMstudio:
  - [LMstudio](https://lmstudio.ai) installed and running
  - Compatible model downloaded and loaded
  - Local server enabled (port 1234)
```

**6.2** Add LMstudio setup section:
```markdown
### Using LMstudio

**Installation:**
1. Download LMstudio from https://lmstudio.ai
2. Install and launch the application
3. Download a compatible model (recommended: qwen3:32b, llama-3.2-3b)

**Configuration:**
1. Load a model in LMstudio (click the ↔ icon)
2. Enable the local server:
   - Go to Developer → Local Server
   - Click "Start Server"
   - Default port is 1234
3. Verify it's running: `curl http://localhost:1234/v1/models`

**Usage:**
```bash
# Set environment variables
export DATASET_AGENT_LLM_PROVIDER=lmstudio
export DATASET_AGENT_LLM_MODEL=qwen3:32b

# Run the agent
python dataset_research.py "Census of Agriculture"
```

**Or use command line arguments:**
```bash
python dataset_research.py "Census of Agriculture" \
  --llm-provider lmstudio \
  --llm-model qwen3:32b
```
```

**6.3** Add troubleshooting section (see full content in SPEC_LMSTUDIO_PROVIDER.md, section 6.1)

#### Step 7: Update src/dataset_agent/README.md

**7.1** Update provider list in API documentation to include LMstudio

**7.2** Add configuration examples for LMstudio

### Phase 4: Final Checks (30 minutes)

#### Step 8: Code Quality

```bash
# Check for linting errors
flake8 src/dataset_agent/config.py
flake8 src/dataset_agent/adapters/agent.py

# Check for type errors (if using mypy)
mypy src/dataset_agent/config.py
mypy src/dataset_agent/adapters/agent.py
```

#### Step 9: Final Testing

Run through the complete test checklist from Appendix B.2 in the specification.

#### Step 10: Documentation Review

- [ ] All code changes have docstrings
- [ ] README.md is updated
- [ ] Examples are tested and working
- [ ] Error messages are clear and helpful

## Testing Checklist

Use this checklist to verify implementation:

### Basic Functionality
- [ ] Agent initializes with LMstudio provider
- [ ] Health check detects running LMstudio
- [ ] Health check detects loaded models
- [ ] Simple query executes successfully
- [ ] Web search tool works
- [ ] Make request tool works
- [ ] Full research workflow completes

### Error Handling
- [ ] Clear error when LMstudio not running
- [ ] Clear warning when no models loaded
- [ ] Graceful failure with helpful messages
- [ ] Proper logging at all stages

### Configuration
- [ ] Environment variables work
- [ ] Command line arguments work
- [ ] Programmatic configuration works
- [ ] API configuration works
- [ ] Custom base URL works

### Provider Compatibility
- [ ] Ollama still works
- [ ] OpenRouter still works
- [ ] Can switch between providers
- [ ] No interference between providers

### Documentation
- [ ] README.md has LMstudio instructions
- [ ] Examples are accurate
- [ ] Troubleshooting guide is helpful
- [ ] Configuration is documented

## Common Issues and Solutions

### Issue: "Cannot connect to LMstudio"
**Solution:** 
1. Ensure LMstudio is running
2. Check that the server is enabled in LMstudio settings
3. Verify the port (default: 1234)
4. Test with: `curl http://localhost:1234/v1/models`

### Issue: "No models loaded"
**Solution:**
1. Open LMstudio
2. Click the ↔ icon to load a model into memory
3. Wait for the model to finish loading

### Issue: Tool calling not working
**Solution:**
1. Ensure your model supports function calling
2. Try a different model (recommended: GPT-OSS, Llama 3.2, Qwen)
3. Check LMstudio version (update if needed)

### Issue: Slow responses
**Solution:**
1. Use a smaller model
2. Reduce context length in LMstudio settings
3. Enable GPU acceleration if available

## Quick Reference: Files to Modify

1. ✏️ `src/dataset_agent/config.py`
   - Add `lmstudio_base_url` parameter
   - Add LMstudio case in `setup_dependencies()`
   - Update `__repr__()` method

2. ✏️ `src/dataset_agent/adapters/agent.py`
   - Add LMstudio initialization block
   - Add `_check_lmstudio_health()` method
   - Update docstrings

3. ✏️ `README.md`
   - Add LMstudio to prerequisites
   - Add LMstudio setup section
   - Add usage examples
   - Add troubleshooting section

4. ✏️ `src/dataset_agent/README.md`
   - Update provider documentation
   - Add configuration examples

## Estimated Time

- Phase 1 (Core Implementation): 2 hours
- Phase 2 (Testing): 1.5 hours
- Phase 3 (Documentation): 1 hour
- Phase 4 (Final Checks): 0.5 hours

**Total: ~5 hours**

## Next Steps After Implementation

1. Create a pull request with the changes
2. Request code review
3. Address feedback
4. Merge to main branch
5. Update release notes
6. Announce the new feature

## Resources

- **Specification**: [SPEC_LMSTUDIO_PROVIDER.md](./SPEC_LMSTUDIO_PROVIDER.md)
- **LMstudio Docs**: https://lmstudio.ai/docs/app/api/endpoints/openai
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
- **LangChain OpenAI**: https://python.langchain.com/docs/integrations/chat/openai

---

**Good luck with the implementation!** 🚀

If you encounter any issues not covered in this guide, refer to the full specification or consult the troubleshooting section.

