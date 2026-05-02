# LMstudio Provider - Quick Reference

> **📄 Full Documentation**: See [SPEC_LMSTUDIO_PROVIDER.md](./SPEC_LMSTUDIO_PROVIDER.md) for complete specification  
> **🔧 Implementation Guide**: See [IMPLEMENTATION_GUIDE_LMSTUDIO.md](./IMPLEMENTATION_GUIDE_LMSTUDIO.md) for step-by-step instructions

## What is This?

This is a quick reference for adding **LMstudio** as a third LLM provider to the Dataset Research Agent, alongside the existing Ollama and OpenRouter providers.

## Why LMstudio?

| Feature | Ollama | OpenRouter | **LMstudio** |
|---------|--------|------------|--------------|
| **Location** | Local | Cloud | Local |
| **API Style** | Custom | OpenAI-compatible | OpenAI-compatible |
| **GUI** | No | No | ✅ Yes |
| **Easy Setup** | Command line | API key | GUI + click |
| **Cost** | Free | Pay-per-use | Free |

**LMstudio combines the best of both worlds**: local inference (like Ollama) with OpenAI-compatible API (like OpenRouter), plus a user-friendly GUI.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Agent                             │
│                                                               │
│  ┌─────────────┐      ┌──────────────┐                      │
│  │   config.py │─────▶│ agent.py     │                      │
│  │             │      │              │                      │
│  │ Providers:  │      │ LangChain    │                      │
│  │ • ollama    │      │ Integration  │                      │
│  │ • openrouter│      │              │                      │
│  │ • lmstudio  │◀─────│              │                      │
│  └─────────────┘      └──────────────┘                      │
│                              │                                │
└──────────────────────────────┼────────────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────┐
        │        LLM Provider (choose one)         │
        ├──────────────┬───────────────┬───────────┤
        │   Ollama     │  OpenRouter   │ LMstudio  │
        │ localhost:   │   Cloud API   │localhost: │
        │   11434      │               │   1234    │
        └──────────────┴───────────────┴───────────┘
```

## Files to Modify

### 1. `src/dataset_agent/config.py`

**Add:**
```python
# In Config.__init__():
self.lmstudio_base_url = lmstudio_base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

# In setup_dependencies():
elif config.llm_provider == "lmstudio":
    agent = LangChainAgent(
        model_name=config.llm_model,
        provider="lmstudio",
        api_key=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
        base_url=config.lmstudio_base_url,
        temperature=config.temperature
    )
```

### 2. `src/dataset_agent/adapters/agent.py`

**Add:**
```python
# In _initialize_agent():
elif self.provider == "lmstudio":
    from langchain_openai import ChatOpenAI
    
    if not self._check_lmstudio_health():
        logger.warning("LMstudio not responding...")
    
    llm = ChatOpenAI(
        model=self.model_name,
        temperature=self.temperature,
        api_key=self.api_key or "lm-studio",
        base_url=self.base_url or "http://localhost:1234/v1",
        streaming=False
    )

# Add new method:
def _check_lmstudio_health(self) -> bool:
    """Check if LMstudio is running with models loaded."""
    # Implementation in spec
```

### 3. `README.md`

**Add sections:**
- LMstudio prerequisites
- LMstudio setup instructions
- LMstudio usage examples
- Troubleshooting guide

## Configuration

### Environment Variables

```bash
# Choose provider
DATASET_AGENT_LLM_PROVIDER=lmstudio

# LMstudio settings
LMSTUDIO_BASE_URL=http://localhost:1234/v1
DATASET_AGENT_LLM_MODEL=qwen3:32b
DATASET_AGENT_TEMPERATURE=0.85
```

### Command Line

```bash
python dataset_research.py "Dataset Name" \
  --llm-provider lmstudio \
  --llm-model qwen3:32b
```

### Python Code

```python
from src.dataset_agent.config import Config
from src.dataset_agent.main import run_research

config = Config(
    llm_provider="lmstudio",
    llm_model="qwen3:32b",
    lmstudio_base_url="http://localhost:1234/v1"
)

result = run_research("Dataset Name", config=config)
```

## Setup for Users

1. **Install LMstudio**: Download from https://lmstudio.ai
2. **Download a model**: In LMstudio, search and download a model (e.g., qwen3:32b)
3. **Load the model**: Click the ↔ icon to load into memory
4. **Start server**: Developer → Local Server → Start Server
5. **Verify**: `curl http://localhost:1234/v1/models`
6. **Configure agent**: Set `DATASET_AGENT_LLM_PROVIDER=lmstudio`
7. **Run**: Use the agent as normal

## Key Differences from Other Providers

### vs. Ollama
- ✅ Has GUI (easier for non-technical users)
- ✅ OpenAI-compatible API (standard)
- ❌ No CLI management (must use GUI)
- ❌ No advanced features like keep_alive

### vs. OpenRouter
- ✅ Runs locally (privacy, no cost)
- ✅ Works offline
- ❌ Need to manage local resources
- ❌ Limited to models you download

## Implementation Checklist

Quick checklist for implementers:

```
Phase 1: Core Implementation
☐ Update config.py with lmstudio_base_url
☐ Add lmstudio case in setup_dependencies()
☐ Add lmstudio initialization in agent.py
☐ Add _check_lmstudio_health() method
☐ Update docstrings

Phase 2: Testing
☐ Test basic query with LMstudio
☐ Test tool calling (web search)
☐ Test error handling (server down)
☐ Test provider switching
☐ Test API endpoint

Phase 3: Documentation
☐ Update README.md
☐ Update src/dataset_agent/README.md
☐ Add troubleshooting guide
☐ Add usage examples

Phase 4: Final
☐ Run linting
☐ Code review
☐ Manual testing
☐ Update release notes
```

## Code Changes Summary

| File | Lines Added | Lines Modified | Complexity |
|------|-------------|----------------|------------|
| `config.py` | ~15 | ~5 | Low |
| `agent.py` | ~50 | ~10 | Medium |
| `README.md` | ~100 | ~20 | Low |
| Total | ~165 | ~35 | Low-Medium |

**Estimated Time**: 5 hours

## Testing Commands

```bash
# 1. Setup environment
export DATASET_AGENT_LLM_PROVIDER=lmstudio
export DATASET_AGENT_LLM_MODEL=qwen3:32b
export TAVILY_API_KEY=your_key

# 2. Test basic functionality
python -m src.dataset_agent.main "Test Dataset"

# 3. Test API mode
python -m src.dataset_agent.server &
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"dataset_name": "Test Dataset"}'

# 4. Test error handling (stop LMstudio first)
python -m src.dataset_agent.main "Test Dataset"

# 5. Test provider switching
export DATASET_AGENT_LLM_PROVIDER=ollama
python -m src.dataset_agent.main "Test Dataset"
export DATASET_AGENT_LLM_PROVIDER=lmstudio
python -m src.dataset_agent.main "Test Dataset"
```

## Benefits of This Implementation

1. ✅ **Zero Breaking Changes**: Existing Ollama and OpenRouter code untouched
2. ✅ **Reuses Existing Infrastructure**: Uses `langchain-openai` (already in deps)
3. ✅ **Follows Best Practices**: Based on official LMstudio documentation
4. ✅ **User-Friendly**: Clear error messages and health checks
5. ✅ **Well-Documented**: Comprehensive spec and implementation guide
6. ✅ **Easy to Test**: Clear testing strategy and checklist

## Common Pitfalls to Avoid

1. ❌ **Don't** modify existing Ollama or OpenRouter code
2. ❌ **Don't** assume LMstudio is always running (add health checks)
3. ❌ **Don't** use provider-specific parameters (top_k, top_p) - they're in LMstudio UI
4. ❌ **Don't** forget to update documentation
5. ❌ **Don't** skip testing error scenarios

## Success Criteria

Implementation is complete when:

- ✅ User can select `lmstudio` as provider
- ✅ Agent connects to LMstudio successfully
- ✅ All research features work (web search, URL validation, etc.)
- ✅ Error messages are clear and helpful
- ✅ Health check provides useful feedback
- ✅ Documentation is comprehensive
- ✅ Tests pass
- ✅ No breaking changes to existing providers

## Support Resources

| Resource | Location |
|----------|----------|
| **Full Specification** | [SPEC_LMSTUDIO_PROVIDER.md](./SPEC_LMSTUDIO_PROVIDER.md) |
| **Implementation Guide** | [IMPLEMENTATION_GUIDE_LMSTUDIO.md](./IMPLEMENTATION_GUIDE_LMSTUDIO.md) |
| **LMstudio Docs** | https://lmstudio.ai/docs/app/api/endpoints/openai |
| **OpenAI API Reference** | https://platform.openai.com/docs/api-reference |
| **LangChain Docs** | https://python.langchain.com/docs/integrations/chat/openai |

## Visual: Data Flow

```
┌─────────────┐
│    User     │
│   Request   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  config.py                                  │
│  • Loads DATASET_AGENT_LLM_PROVIDER         │
│  • Validates provider = "lmstudio"          │
│  • Gets LMSTUDIO_BASE_URL                   │
│  • Calls setup_dependencies()               │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  agent.py                                   │
│  • Creates LangChainAgent(provider="lmstudio") │
│  • Checks _check_lmstudio_health()          │
│  • Initializes ChatOpenAI with base_url     │
│  • Creates tool calling agent               │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  LMstudio (localhost:1234)                  │
│  • Receives POST /v1/chat/completions       │
│  • Processes with loaded model              │
│  • Returns OpenAI-compatible response       │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  usecases.py                                │
│  • Processes response                        │
│  • Calls tools (web_search, make_request)   │
│  • Generates dataset information            │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Result    │
│   (JSON)    │
└─────────────┘
```

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to LMstudio" | Start LMstudio server (Developer → Local Server) |
| "No models loaded" | Click ↔ icon in LMstudio to load a model |
| "Tool calling not working" | Try a different model that supports function calling |
| "Slow responses" | Use a smaller model or enable GPU acceleration |
| Port 1234 in use | Change port in LMstudio, update `LMSTUDIO_BASE_URL` |

---

## Next Steps

1. **Read the full spec**: [SPEC_LMSTUDIO_PROVIDER.md](./SPEC_LMSTUDIO_PROVIDER.md)
2. **Follow the implementation guide**: [IMPLEMENTATION_GUIDE_LMSTUDIO.md](./IMPLEMENTATION_GUIDE_LMSTUDIO.md)
3. **Set up LMstudio**: Download from https://lmstudio.ai
4. **Start coding**: Follow the step-by-step guide
5. **Test thoroughly**: Use the testing checklist
6. **Submit for review**: Create a pull request

---

**Estimated Total Effort**: 5-6 hours  
**Difficulty Level**: Medium  
**Risk Level**: Low (no breaking changes)

*This quick reference is part of the LMstudio provider integration project. For questions or issues, refer to the full specification.*

