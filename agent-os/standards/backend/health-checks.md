## Health Check Standards

### Health Check Design Principles

- **Non-Blocking Initialization**: Health checks should warn but not prevent service initialization
- **Informative Logging**: Log health status with specific guidance for resolution
- **Timeout Handling**: Use reasonable timeouts (5 seconds) to avoid blocking startup
- **Progressive Verification**: Check availability first, then check models/resources loaded
- **Clear Error Messages**: Provide actionable error messages with specific troubleshooting steps

### LLM Provider Health Check Pattern

```python
# ✅ Good: Comprehensive health check implementation
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def _check_lmstudio_health(self) -> bool:
    """
    Check if LMStudio server is running and has models loaded.
    
    Returns:
        bool: True if LMStudio is healthy, False otherwise
    """
    try:
        base = self.base_url or "http://localhost:1234/v1"
        models_url = f"{base.rstrip('/')}/models"
        
        # Try to get list of loaded models with timeout
        response = requests.get(models_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            
            if models:
                model_names = [m.get("id", "unknown") for m in models]
                logger.info(
                    f"LMStudio is running with {len(models)} model(s) loaded: "
                    f"{', '.join(model_names)}"
                )
                return True
            else:
                logger.warning(
                    "LMStudio server is running but no models are loaded. "
                    "Please load a model in LMStudio before using the agent. "
                    "In LMStudio, click the ↔ icon to load a model into memory."
                )
                return False
        else:
            logger.warning(
                f"LMStudio server returned unexpected status: {response.status_code}"
            )
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Cannot connect to LMStudio at {base}. "
            "Ensure LMStudio is running and the server is enabled. "
            "In LMStudio, go to Developer → Local Server and click 'Start Server'."
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(
            f"LMStudio health check timed out after 5 seconds. "
            "The server may be overloaded or unresponsive."
        )
        return False
    except Exception as e:
        logger.error(f"LMStudio health check failed: {str(e)}")
        return False
```

### Health Check for Local Services

```python
# ✅ Good: Ollama health check pattern
def _check_ollama_health(self) -> bool:
    """Check if Ollama service is running and accessible."""
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Ollama is running with {len(models)} model(s) available")
            return True
        return False
        
    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Ensure Ollama is installed and running. "
            "Start Ollama with: ollama serve"
        )
        return False
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        return False
```

### API Health Endpoint Pattern

```python
# ✅ Good: FastAPI health endpoint with dependency checks
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint that verifies all system dependencies.
    
    Returns:
        Dict with status and details of each dependency
    """
    health_status = {
        "status": "healthy",
        "dependencies": {}
    }
    
    # Check database connectivity
    try:
        # Attempt simple database query
        db.execute("SELECT 1")
        health_status["dependencies"]["database"] = {
            "status": "healthy",
            "message": "Database is accessible"
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["database"] = {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}"
        }
    
    # Check LLM provider availability
    try:
        provider_healthy = check_llm_provider_health()
        health_status["dependencies"]["llm_provider"] = {
            "status": "healthy" if provider_healthy else "degraded",
            "message": "LLM provider is available" if provider_healthy else "LLM provider not responding"
        }
    except Exception as e:
        health_status["dependencies"]["llm_provider"] = {
            "status": "unhealthy",
            "message": f"LLM provider error: {str(e)}"
        }
    
    # Check file system
    try:
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        health_status["dependencies"]["filesystem"] = {
            "status": "healthy",
            "message": "Output directory is accessible"
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["filesystem"] = {
            "status": "unhealthy",
            "message": f"Filesystem error: {str(e)}"
        }
    
    # Return 503 if unhealthy
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status
```

### Troubleshooting Guidance in Error Messages

Error messages should include:

1. **What went wrong**: Clear description of the failure
2. **Where to fix it**: Specific location or setting to check
3. **How to fix it**: Step-by-step instructions
4. **Documentation link**: URL to relevant documentation (optional)

```python
# ✅ Good: Error message with complete troubleshooting guidance
logger.error(
    "LMStudio server is not responding at http://localhost:1234/v1. "
    "\n"
    "To fix this issue:\n"
    "1. Ensure LMStudio is installed (https://lmstudio.ai)\n"
    "2. Open LMStudio application\n"
    "3. Download and load a model (click the ↔ icon)\n"
    "4. Enable the local server:\n"
    "   - Go to Developer → Local Server\n"
    "   - Click 'Start Server'\n"
    "   - Verify port is set to 1234\n"
    "5. Test connection: curl http://localhost:1234/v1/models\n"
    "\n"
    "If using a custom port, set LMSTUDIO_BASE_URL environment variable."
)
```

### Health Check Timing

- **Startup**: Run health checks during initialization to detect issues early
- **Runtime**: Health endpoint should be callable at any time
- **Periodic**: Consider periodic background health checks for long-running services
- **Pre-Request**: For critical operations, optionally check health before execution

### Return Value Standards

- **Boolean Return**: Simple `True`/`False` for pass/fail checks
- **Status Object**: Return structured status for detailed health information
- **Logging Side Effects**: Always log health check results, don't rely only on return value
- **Non-Throwing**: Health checks should catch exceptions and return `False`, not raise
