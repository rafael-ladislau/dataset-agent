# Feature Specification: LMstudio Provider Integration

## 1. Overview

### 1.1 Purpose
Add LMstudio as a third LLM provider option for the Dataset Research Agent, enabling users to run local language models through LMstudio's OpenAI-compatible API.

### 1.2 Background
Currently, the agent supports two LLM providers:
- **Ollama**: Local inference with custom API
- **OpenRouter**: Cloud-based inference with OpenAI-compatible API

LMstudio provides a local inference solution with an OpenAI-compatible API, making it an ideal middle ground that combines the privacy of local inference with the standardization of OpenAI's API format.

### 1.3 Goals
- ✅ Enable users to use LMstudio as an LLM provider
- ✅ Maintain backward compatibility with existing Ollama and OpenRouter configurations
- ✅ Follow OpenAI API best practices as documented in LMstudio documentation
- ✅ Support all agent capabilities (tool calling, streaming, structured output)
- ✅ Provide clear configuration and setup instructions

### 1.4 Non-Goals
- ❌ Modify existing Ollama or OpenRouter implementations
- ❌ Support LMstudio-specific features not compatible with OpenAI API
- ❌ Automatic LMstudio installation or model management
- ❌ LMstudio server management (start/stop)

---

## 2. Technical Specifications

### 2.1 System Requirements

**Prerequisites:**
- LMstudio application installed and running locally
- LMstudio server started on port 1234 (default)
- A compatible model loaded in LMstudio
- Python 3.8+
- `langchain-openai` package (already in requirements.txt)

**Supported LMstudio Features:**
- Chat completions endpoint (`/v1/chat/completions`)
- Streaming responses
- Tool calling (function calling)
- Model listing (`/v1/models`)
- OpenAI-compatible parameter set

### 2.2 Architecture Changes

#### 2.2.1 Configuration Layer (`config.py`)

**Changes Required:**
1. Add `lmstudio` as a valid provider option
2. Add LMstudio-specific configuration parameters
3. Add LMstudio base URL configuration with default

**New Configuration Parameters:**

```python
# Environment Variables
DATASET_AGENT_LLM_PROVIDER=lmstudio  # New option: "ollama" | "openrouter" | "lmstudio"
LMSTUDIO_BASE_URL=http://localhost:1234/v1  # Default LMstudio API URL
LMSTUDIO_API_KEY=lm-studio  # Placeholder key (LMstudio doesn't require real keys)
```

**Modified `Config` class:**
```python
class Config:
    def __init__(self, ..., lmstudio_base_url: str = None):
        # ... existing code ...
        self.lmstudio_base_url = (
            lmstudio_base_url or 
            os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        )
```

**Modified `setup_dependencies` function:**
```python
def setup_dependencies(config: Config) -> Dict[str, Any]:
    # ... existing code ...
    
    elif config.llm_provider == "lmstudio":
        # Create agent with LMstudio configuration
        agent = LangChainAgent(
            model_name=config.llm_model,
            provider="lmstudio",
            api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=config.lmstudio_base_url,
            temperature=config.temperature,
            # Note: top_k and top_p are controlled in LMstudio UI, not via API
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
```

#### 2.2.2 Agent Layer (`adapters/agent.py`)

**Changes Required:**
1. Add `lmstudio` provider option to `__init__` method
2. Implement LMstudio initialization logic using `ChatOpenAI` from `langchain-openai`
3. Add LMstudio health check (optional but recommended)

**Modified `LangChainAgent.__init__` signature:**
```python
def __init__(
    self, 
    model_name: str = "gpt-oss:20b", 
    temperature: float = 0.85, 
    provider: str = "ollama",  # Now supports "ollama" | "openrouter" | "lmstudio"
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
):
    """
    Initialize the LangChain agent.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for the model
        provider: LLM provider (ollama, openrouter, or lmstudio)
        api_key: API key (for OpenRouter) or placeholder (for LMstudio)
        base_url: Base URL for API (OpenRouter or LMstudio)
        top_k: Top-K sampling (Ollama only, LMstudio configures via UI)
        top_p: Top-P sampling (Ollama only, LMstudio configures via UI)
    """
```

**New LMstudio initialization block in `_initialize_agent`:**
```python
def _initialize_agent(self):
    """Initialize the agent with tools and LLM."""
    try:
        # ... existing imports ...
        
        if self.provider == "ollama":
            # ... existing Ollama code ...
            
        elif self.provider == "openrouter":
            # ... existing OpenRouter code ...
            
        elif self.provider == "lmstudio":
            from langchain_openai import ChatOpenAI
            
            # Check if LMstudio is available (optional but recommended)
            if not self._check_lmstudio_health():
                logger.warning(
                    "LMstudio server is not responding at {}. "
                    "Please ensure LMstudio is running with a loaded model."
                    .format(self.base_url)
                )
            
            # Initialize LMstudio LLM using OpenAI-compatible client
            # Following best practices from: https://lmstudio.ai/docs/app/api/endpoints/openai
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key or "lm-studio",  # Placeholder key
                base_url=self.base_url or "http://localhost:1234/v1",
                streaming=False,  # Can be enabled for streaming responses
                # Note: LMstudio parameters like top_k, top_p are set in the UI
            )
            logger.info(
                f"Initialized LMstudio LLM with model {self.model_name} "
                f"at {self.base_url or 'http://localhost:1234/v1'}"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # ... rest of agent initialization ...
```

**New LMstudio health check method:**
```python
def _check_lmstudio_health(self) -> bool:
    """Check if LMstudio server is running and has a model loaded."""
    import requests
    
    try:
        base = self.base_url or "http://localhost:1234/v1"
        health_url = f"{base.rstrip('/')}/models"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                logger.info(f"LMstudio is running with {len(models)} model(s) loaded")
                return True
            else:
                logger.warning("LMstudio is running but no models are loaded")
                return False
        return False
    except Exception as e:
        logger.error(f"LMstudio server health check failed: {str(e)}")
        return False
```

#### 2.2.3 API Layer (`api.py`)

**Changes Required:**
- No changes needed! The API layer is provider-agnostic and passes configuration through.
- Per-request model overrides will work automatically with LMstudio.

#### 2.2.4 Documentation Updates

**Files to Update:**
1. `README.md` - Add LMstudio setup and usage instructions
2. `src/dataset_agent/README.md` - Add LMstudio provider documentation
3. `.env.example` - Add LMstudio environment variable examples (create if doesn't exist)

---

## 3. Implementation Plan

### 3.1 Task Breakdown

#### Phase 1: Core Implementation (Required)
1. **Task 1.1**: Update `config.py`
   - Add LMstudio configuration parameters
   - Update `setup_dependencies()` to handle `lmstudio` provider
   - Estimated time: 30 minutes

2. **Task 1.2**: Update `adapters/agent.py`
   - Add LMstudio provider initialization
   - Implement health check method
   - Update docstrings
   - Estimated time: 1 hour

3. **Task 1.3**: Add LMstudio health check utility
   - Create `_check_lmstudio_health()` method
   - Add informative logging
   - Estimated time: 30 minutes

4. **Task 1.4**: Update configuration validation
   - Ensure `lmstudio` is recognized as valid provider
   - Add validation for LMstudio-specific settings
   - Estimated time: 20 minutes

#### Phase 2: Documentation (Required)
5. **Task 2.1**: Create `.env.example` file
   - Document all environment variables
   - Include examples for all three providers
   - Estimated time: 30 minutes

6. **Task 2.2**: Update main `README.md`
   - Add LMstudio setup section
   - Add usage examples
   - Add troubleshooting section
   - Estimated time: 45 minutes

7. **Task 2.3**: Update `src/dataset_agent/README.md`
   - Add LMstudio provider documentation
   - Update architecture section
   - Add configuration examples
   - Estimated time: 30 minutes

#### Phase 3: Testing (Required)
8. **Task 3.1**: Manual testing with LMstudio
   - Test basic query execution
   - Test tool calling functionality
   - Test error handling
   - Estimated time: 1 hour

9. **Task 3.2**: Integration testing
   - Test with API endpoints
   - Test configuration switching
   - Test model listing
   - Estimated time: 45 minutes

10. **Task 3.3**: Create test documentation
    - Document test cases
    - Document expected behavior
    - Estimated time: 30 minutes

#### Phase 4: Enhancement (Optional)
11. **Task 4.1**: Add streaming support
    - Enable streaming for better UX
    - Handle streaming responses
    - Estimated time: 1 hour

12. **Task 4.2**: Add model auto-detection
    - Query available models from LMstudio
    - Validate model selection
    - Estimated time: 45 minutes

**Total Estimated Time**: 7-8 hours (6 hours for required tasks, 1-2 hours for optional enhancements)

### 3.2 Dependencies

**Code Dependencies:**
- `langchain-openai>=0.0.2` (already in requirements.txt)
- `requests>=2.31.0` (already in requirements.txt)

**External Dependencies:**
- LMstudio application installed
- LMstudio server running on port 1234
- Compatible model loaded in LMstudio

### 3.3 Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LMstudio not running when agent starts | High | Medium | Add health check with clear error messages |
| Model not loaded in LMstudio | High | Medium | Check `/v1/models` endpoint and provide guidance |
| Port 1234 already in use | Medium | Low | Make port configurable via environment variable |
| Tool calling not supported by loaded model | Medium | Medium | Log warnings when tools are used, add model capability detection |
| Breaking changes to existing providers | High | Low | Maintain strict separation, add comprehensive tests |

---

## 4. Configuration Examples

### 4.1 Environment Variables (.env file)

```bash
# ============================================
# LLM Provider Configuration
# ============================================
# Options: "ollama", "openrouter", "lmstudio"
DATASET_AGENT_LLM_PROVIDER=lmstudio

# ============================================
# LMstudio Configuration (when provider=lmstudio)
# ============================================
# Base URL for LMstudio API (default: http://localhost:1234/v1)
LMSTUDIO_BASE_URL=http://localhost:1234/v1

# Model identifier (use the model name shown in LMstudio)
# Examples: "qwen3:32b", "llama-3.2-3b", "mistral-7b"
DATASET_AGENT_LLM_MODEL=qwen3:32b

# API key (placeholder, LMstudio doesn't require real keys)
LMSTUDIO_API_KEY=lm-studio

# Sampling temperature (0.0-2.0, default: 0.85)
DATASET_AGENT_TEMPERATURE=0.85

# Note: top_k and top_p are configured in LMstudio UI, not via API

# ============================================
# Ollama Configuration (when provider=ollama)
# ============================================
DATASET_AGENT_URL=http://localhost:11434
DATASET_AGENT_LLM_MODEL=qwen3:32b
DATASET_AGENT_TEMPERATURE=0.85
DATASET_AGENT_TOP_K=40
DATASET_AGENT_TOP_P=0.95

# ============================================
# OpenRouter Configuration (when provider=openrouter)
# ============================================
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DATASET_AGENT_LLM_MODEL=anthropic/claude-3-opus

# ============================================
# Web Search Configuration
# ============================================
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your_tavily_key_here

# ============================================
# API Configuration (for server mode)
# ============================================
API_KEYS=key1,key2,key3
SQLITE_DB_PATH=./data/research.db

# ============================================
# Logging Configuration
# ============================================
DATASET_AGENT_LOG_LEVEL=INFO
DATASET_AGENT_LOG_FILE=dataset_agent.log
LOG_PATH=./logs
```

### 4.2 Command Line Usage

```bash
# Using LMstudio (assumes .env is configured)
python dataset_research.py "Census of Agriculture" --llm-provider lmstudio

# Using LMstudio with explicit model selection
python dataset_research.py "Census of Agriculture" \
  --llm-provider lmstudio \
  --llm-model "qwen3:32b"

# Using LMstudio with custom port
LMSTUDIO_BASE_URL=http://localhost:5000/v1 \
  python dataset_research.py "Census of Agriculture" \
  --llm-provider lmstudio
```

### 4.3 Programmatic Usage

```python
from src.dataset_agent.config import Config
from src.dataset_agent.main import run_research

# Create configuration for LMstudio
config = Config(
    log_level="INFO",
    output_dir="/path/to/output",
    llm_provider="lmstudio",
    llm_model="qwen3:32b",
    temperature=0.85,
    lmstudio_base_url="http://localhost:1234/v1"
)

# Run research
dataset_info = run_research(
    dataset_name="Census of Agriculture",
    dataset_url="https://www.nass.usda.gov/AgCensus/",
    config=config
)

# Access results
print(f"Description: {dataset_info.description}")
```

### 4.4 API Usage

```bash
# Submit research request using LMstudio
curl -X POST "http://localhost:8000/api/research" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "dataset_name": "Census of Agriculture",
    "dataset_url": "https://www.nass.usda.gov/AgCensus/",
    "model": "qwen3:32b"
  }'
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**Test Cases:**
1. **Config Validation**
   - ✅ Test `lmstudio` is recognized as valid provider
   - ✅ Test LMstudio base URL configuration
   - ✅ Test default values

2. **Agent Initialization**
   - ✅ Test LMstudio agent initialization succeeds
   - ✅ Test with custom base URL
   - ✅ Test with custom model name
   - ✅ Test error handling when LMstudio unavailable

3. **Health Check**
   - ✅ Test health check with running LMstudio
   - ✅ Test health check with stopped LMstudio
   - ✅ Test health check with no loaded model

### 5.2 Integration Tests

**Test Cases:**
1. **Basic Query Execution**
   - ✅ Execute simple query using LMstudio
   - ✅ Verify response format
   - ✅ Verify response quality

2. **Tool Calling**
   - ✅ Execute query requiring web search
   - ✅ Execute query requiring URL validation
   - ✅ Verify tool calls work correctly

3. **Full Research Workflow**
   - ✅ Run complete dataset research
   - ✅ Verify all fields populated
   - ✅ Verify JSON output format

4. **Provider Switching**
   - ✅ Switch from Ollama to LMstudio
   - ✅ Switch from OpenRouter to LMstudio
   - ✅ Verify no interference between providers

### 5.3 Manual Testing Checklist

**Prerequisites:**
- [ ] LMstudio installed and running
- [ ] Model loaded in LMstudio
- [ ] Environment configured for LMstudio

**Tests:**
- [ ] Run simple dataset research query
- [ ] Verify web search tool is called
- [ ] Verify make_request tool is called
- [ ] Check log output for proper initialization messages
- [ ] Test with LMstudio stopped (should fail gracefully)
- [ ] Test with different models
- [ ] Test with API server mode
- [ ] Test with different temperature settings

### 5.4 Performance Testing

**Metrics to Track:**
- Response time vs. Ollama
- Response time vs. OpenRouter
- Memory usage
- Tool calling latency

---

## 6. Documentation Requirements

### 6.1 User-Facing Documentation

**README.md Updates:**
1. Add LMstudio to prerequisites section
2. Add LMstudio installation instructions with link to https://lmstudio.ai
3. Add LMstudio setup section:
   - How to install LMstudio
   - How to download a model
   - How to start the server
   - How to configure the agent
4. Add LMstudio usage examples
5. Add troubleshooting section for LMstudio

**Troubleshooting Section (to add):**
```markdown
### LMstudio Troubleshooting

**Error: "LMstudio server is not responding"**
- Ensure LMstudio is running
- Check that the server is enabled in LMstudio (Developer → Local Server)
- Verify the port is 1234 (or update LMSTUDIO_BASE_URL)
- Test the connection: `curl http://localhost:1234/v1/models`

**Error: "No models loaded"**
- Load a model in LMstudio before running the agent
- Click the ↔ icon in LMstudio to load a model to memory

**Slow responses**
- Use a smaller model for faster inference
- Reduce context length in LMstudio settings
- Enable GPU acceleration if available

**Tool calling not working**
- Ensure your model supports function calling
- Check LMstudio logs for tool-related errors
- Try a different model (recommended: GPT-OSS, Llama 3.2, Qwen)
```

### 6.2 Developer Documentation

**Architecture Documentation:**
- Update architecture diagrams to show LMstudio as third option
- Document LMstudio-specific implementation details
- Add sequence diagrams for LMstudio interactions

**Code Comments:**
- Add inline comments explaining LMstudio-specific code
- Document OpenAI API compatibility considerations
- Reference LMstudio documentation URLs

---

## 7. Deployment Considerations

### 7.1 Environment Setup

**Development:**
- LMstudio runs locally on developer machine
- Port 1234 must be available
- Model must be manually loaded

**Production:**
- Consider using Ollama for production deployments (more stable API)
- LMstudio is best suited for development and testing
- If using LMstudio in production:
  - Ensure LMstudio service is monitored
  - Implement automatic model loading
  - Set up health checks and alerts

### 7.2 Docker Considerations

**Note:** LMstudio cannot run inside Docker (requires GUI application).

**Workaround for containerized environments:**
1. Run LMstudio on host machine
2. Configure container to connect to host's port 1234
3. Use host.docker.internal (Docker Desktop) or host networking

**docker-compose.yml example:**
```yaml
services:
  dataset-agent:
    build: .
    environment:
      - DATASET_AGENT_LLM_PROVIDER=lmstudio
      - LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

---

## 8. Success Criteria

### 8.1 Functional Requirements
- ✅ LMstudio can be selected as a provider via configuration
- ✅ Agent successfully connects to LMstudio API
- ✅ Basic queries work correctly
- ✅ Tool calling (web search, URL validation) works
- ✅ All dataset research features work
- ✅ Error messages are clear and actionable
- ✅ Health checks provide useful feedback

### 8.2 Non-Functional Requirements
- ✅ Response time within 20% of Ollama performance
- ✅ No breaking changes to existing providers
- ✅ Code follows existing architecture patterns
- ✅ Documentation is comprehensive and clear
- ✅ Configuration is straightforward

### 8.3 Quality Requirements
- ✅ All tests pass
- ✅ No linting errors
- ✅ Code coverage maintained or improved
- ✅ No memory leaks
- ✅ Proper error handling

---

## 9. Future Enhancements

### 9.1 Near-term (Optional for initial release)
1. **Streaming Support**
   - Enable real-time response streaming
   - Update UI to show streaming responses

2. **Model Auto-detection**
   - Query available models from LMstudio
   - Auto-select appropriate model if not specified

3. **Advanced Configuration**
   - Support LMstudio preset configurations
   - Allow per-request parameter overrides via API

### 9.2 Long-term
1. **LMstudio Model Management**
   - Auto-load models via API (if LMstudio adds this capability)
   - Model warm-up similar to Ollama implementation

2. **Performance Optimization**
   - Connection pooling
   - Response caching
   - Batch request support

3. **Enhanced Monitoring**
   - Track LMstudio resource usage
   - Performance metrics dashboard
   - Model performance comparison

---

## 10. References

### 10.1 External Documentation
- **LMstudio OpenAI API Documentation**: https://lmstudio.ai/docs/app/api/endpoints/openai
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
- **LangChain OpenAI Integration**: https://python.langchain.com/docs/integrations/chat/openai

### 10.2 Internal Documentation
- `README.md` - Main project documentation
- `src/dataset_agent/README.md` - API documentation
- `src/dataset_agent/config.py` - Configuration management
- `src/dataset_agent/adapters/agent.py` - Agent implementation

---

## 11. Approval and Sign-off

**Created by**: Dataset Agent Development Team  
**Created on**: 2025-10-11  
**Version**: 1.0  

**Approved by**:
- [ ] Technical Lead
- [ ] Product Owner
- [ ] QA Lead

**Implementation Start Date**: TBD  
**Target Completion Date**: TBD  

---

## Appendix A: Code Samples

### A.1 Complete LMstudio Initialization Code

```python
elif self.provider == "lmstudio":
    from langchain_openai import ChatOpenAI
    
    # Set base URL with default
    lmstudio_base_url = self.base_url or "http://localhost:1234/v1"
    
    # Check if LMstudio is available
    if not self._check_lmstudio_health():
        logger.warning(
            f"LMstudio server is not responding at {lmstudio_base_url}. "
            "Please ensure LMstudio is running with a loaded model. "
            "Visit https://lmstudio.ai for installation instructions."
        )
        # Continue anyway - let the first request fail with clear error
    
    # Initialize LMstudio LLM using OpenAI-compatible API
    # Following best practices from https://lmstudio.ai/docs/app/api/endpoints/openai
    llm = ChatOpenAI(
        model=self.model_name,
        temperature=self.temperature,
        api_key=self.api_key or "lm-studio",  # Placeholder, not validated
        base_url=lmstudio_base_url,
        streaming=False,
        # Parameters supported by OpenAI API:
        # - temperature, top_p (controlled via API)
        # - max_tokens, presence_penalty, frequency_penalty, etc.
        # Note: Some parameters like top_k are model-specific in LMstudio
    )
    
    logger.info(
        f"Initialized LMstudio LLM with model '{self.model_name}' "
        f"at {lmstudio_base_url}"
    )
```

### A.2 Complete Health Check Code

```python
def _check_lmstudio_health(self) -> bool:
    """
    Check if LMstudio server is running and has models loaded.
    
    Returns:
        bool: True if LMstudio is healthy, False otherwise
    """
    import requests
    
    try:
        base = self.base_url or "http://localhost:1234/v1"
        models_url = f"{base.rstrip('/')}/models"
        
        # Try to get list of loaded models
        response = requests.get(models_url, timeout=5)
        
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
        else:
            logger.warning(
                f"LMstudio server returned unexpected status: {response.status_code}"
            )
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Cannot connect to LMstudio at {base}. "
            "Ensure LMstudio is running and the server is enabled."
        )
        return False
    except Exception as e:
        logger.error(f"LMstudio health check failed: {str(e)}")
        return False
```

### A.3 Example Usage Script

```python
#!/usr/bin/env python3
"""
Example script demonstrating LMstudio provider usage.
"""
import os
from src.dataset_agent.config import Config
from src.dataset_agent.main import run_research

def main():
    # Configure for LMstudio
    config = Config(
        llm_provider="lmstudio",
        llm_model="qwen3:32b",  # Use model name from LMstudio
        temperature=0.85,
        lmstudio_base_url="http://localhost:1234/v1",
        log_level="INFO",
        output_dir="./results"
    )
    
    print("Dataset Research Agent - LMstudio Provider")
    print("=" * 50)
    print(f"Configuration: {config}")
    print()
    
    # Run research
    dataset_name = "Census of Agriculture"
    dataset_url = "https://www.nass.usda.gov/AgCensus/"
    
    print(f"Researching: {dataset_name}")
    print(f"URL: {dataset_url}")
    print()
    
    try:
        result = run_research(dataset_name, dataset_url, config)
        
        print("Research completed successfully!")
        print()
        print(f"Description: {result.description[:200]}...")
        print(f"Aliases: {len(result.aliases)} found")
        print(f"Organizations: {len(result.organizations)} found")
        print(f"Access Type: {result.access_type}")
        print(f"Data URL: {result.data_url}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
```

---

## Appendix B: Testing Checklist

### B.1 Pre-Implementation Checklist
- [ ] LMstudio installed on test machine
- [ ] Compatible model downloaded (e.g., qwen3:32b, llama-3.2-3b)
- [ ] LMstudio server tested with curl
- [ ] Current codebase runs without errors

### B.2 Implementation Checklist
- [ ] `config.py` updated with LMstudio support
- [ ] `adapters/agent.py` updated with LMstudio initialization
- [ ] Health check method implemented
- [ ] Error messages updated with clear guidance
- [ ] Docstrings updated
- [ ] Type hints added where appropriate

### B.3 Testing Checklist
- [ ] Unit tests for config validation
- [ ] Unit tests for agent initialization
- [ ] Unit tests for health check
- [ ] Integration test: basic query
- [ ] Integration test: web search tool
- [ ] Integration test: make_request tool
- [ ] Integration test: full research workflow
- [ ] Manual test: simple dataset research
- [ ] Manual test: complex dataset research
- [ ] Manual test: provider switching
- [ ] Performance test: response time comparison

### B.4 Documentation Checklist
- [ ] `.env.example` created or updated
- [ ] Main `README.md` updated
- [ ] `src/dataset_agent/README.md` updated
- [ ] Inline code comments added
- [ ] Troubleshooting guide created
- [ ] Usage examples added

### B.5 Release Checklist
- [ ] All tests passing
- [ ] Code reviewed
- [ ] Documentation reviewed
- [ ] Performance metrics documented
- [ ] Known issues documented
- [ ] Release notes prepared

---

*End of Specification Document*

