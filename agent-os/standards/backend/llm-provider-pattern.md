## LLM Provider Pattern

### Provider Abstraction

- **Unified Interface**: All LLM providers (Ollama, OpenRouter, LMStudio) must implement a consistent interface through the agent adapter layer
- **Configuration-Based Selection**: Provider selection should be determined by configuration (environment variables or constructor parameters), not hardcoded logic
- **Health Check Pattern**: Each provider implementation should include health check methods to verify availability before use
- **Graceful Degradation**: Providers should fail gracefully with clear, actionable error messages when unavailable

### Implementation Pattern

```python
# ✅ Good: Provider abstraction with health checks
class LangChainAgent(AgentInterface):
    def __init__(self, provider: str = "ollama", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self._initialize_agent()
    
    def _initialize_agent(self):
        if self.provider == "ollama":
            if not self._check_ollama_health():
                logger.warning("Ollama not available")
            llm = ChatOllama(...)
        elif self.provider == "openrouter":
            llm = ChatOpenAI(api_key=self.api_key, base_url=self.base_url, ...)
        elif self.provider == "lmstudio":
            if not self._check_lmstudio_health():
                logger.warning("LMStudio not available")
            llm = ChatOpenAI(api_key=self.api_key or "lm-studio", base_url=self.base_url, ...)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
```

### Health Check Standards

- **Early Detection**: Check provider availability during initialization, not at first use
- **Informative Logging**: Log provider status with specific guidance for resolution
- **Non-Blocking**: Health checks should warn but not prevent initialization (allow first request to fail with clear error)
- **Timeout Handling**: Use reasonable timeouts (5 seconds) for health check requests
- **Model Verification**: For local providers (Ollama, LMStudio), verify that models are loaded and ready

### Provider-Specific Patterns

**Local Providers (Ollama, LMStudio)**:
- Check server availability via HTTP health endpoint
- Verify model is loaded and accessible
- Provide clear installation and setup guidance in error messages
- Include links to documentation for troubleshooting

**Cloud Providers (OpenRouter)**:
- Validate API key presence before initialization
- Handle authentication errors with clear messages
- Document rate limits and quota management

### Configuration Management

- **Environment Variables**: Use `DATASET_AGENT_LLM_PROVIDER` to specify provider selection
- **Provider-Specific Config**: Each provider should have its own configuration namespace (e.g., `LMSTUDIO_BASE_URL`, `OPENROUTER_API_KEY`)
- **Sensible Defaults**: Provide reasonable defaults for common configurations (port 1234 for LMStudio, port 11434 for Ollama)
- **Validation**: Validate provider configuration at initialization time, not at runtime

### Error Messages

- **Actionable Guidance**: Error messages should tell users exactly what to do (e.g., "Start LMStudio server at Developer → Local Server")
- **Context-Specific**: Tailor messages to the specific provider and failure mode
- **Documentation Links**: Include links to setup guides and troubleshooting documentation
- **Progressive Detail**: Log detailed errors for debugging while showing user-friendly messages to end users
