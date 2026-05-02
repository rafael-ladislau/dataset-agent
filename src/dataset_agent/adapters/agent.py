"""
Agent implementation for LLM-based dataset research.
"""

import time
import logging
import os
from typing import List, Dict, Any, Optional

from ..domain.usecases import AgentInterface

# Set up logging
logger = logging.getLogger(__name__)


class LangChainAgent(AgentInterface):
    """
    Implementation of AgentInterface using LangChain with Ollama, OpenRouter, or LMStudio.
    
    Supports three LLM providers:
    - Ollama: Local inference with custom API
    - OpenRouter: Cloud-based inference with OpenAI-compatible API
    - LMStudio: Local inference with OpenAI-compatible API (https://lmstudio.ai)
    """
    
    def __init__(self, model_name: str = "gpt-oss:20b", temperature: float = 0.85, 
                 provider: str = "ollama", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, top_k: Optional[int] = None, top_p: Optional[float] = None):
        """
        Initialize the LangChain agent.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for the model
            provider: LLM provider (ollama, openrouter, or lmstudio)
            api_key: API key (for OpenRouter) or placeholder (for LMStudio)
            base_url: Base URL for API (OpenRouter or LMStudio)
            top_k: Top-K sampling (Ollama only, LMStudio configures via UI)
            top_p: Top-P sampling (Ollama only, LMStudio configures via UI)
        """
        # Model name precedence: explicit argument > env var > default
        self.model_name = model_name or os.environ.get("DATASET_AGENT_LLM_MODEL", "gpt-oss:20b")
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key
        # Base URL for OpenRouter (if used)
        self.base_url = base_url
        # Base URL for Ollama
        self.ollama_url = os.environ.get("DATASET_AGENT_URL", "http://localhost:11434")
        # Optional sampling controls for Ollama
        self.top_k = top_k
        self.top_p = top_p
        self.agent_executor = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent with tools and LLM."""
        try:
            # Import common LangChain components
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            from langchain_core.tools import Tool, tool
            from langgraph.prebuilt import create_react_agent
            
            # Initialize the LLM based on provider
            if self.provider == "ollama":
                from langchain_ollama import ChatOllama
                
                # Check if Ollama is available
                if not self._check_ollama_health():
                    logger.error("Ollama server is not running or not responding")
                    raise RuntimeError("Ollama server is not running or not responding")
                
                # Ensure correct model is loaded and others are unloaded
                try:
                    self._ensure_ollama_model_loaded()
                except Exception as e:
                    logger.warning(f"Could not ensure Ollama model state: {str(e)}")
                
                # Initialize Ollama LLM
                llm = ChatOllama(
                    base_url=self.ollama_url,
                    model=self.model_name,
                    temperature=self.temperature,
                    top_k=self.top_k if self.top_k is not None else 40,
                    top_p=self.top_p if self.top_p is not None else 0.95,
                    streaming=False
                )
                logger.info(f"Initialized Ollama LLM with model {self.model_name}")
                
            elif self.provider == "openrouter":
                from langchain_openai import ChatOpenAI
                
                # Verify API key is available
                if not self.api_key:
                    raise ValueError("API key is required for OpenRouter provider")
                
                # Initialize OpenRouter LLM
                llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key,
                    base_url=self.base_url or "https://openrouter.ai/api/v1",
                    streaming=False
                )
                logger.info(f"Initialized OpenRouter LLM with model {self.model_name}")
                
            elif self.provider == "lmstudio":
                from langchain_openai import ChatOpenAI
                
                # Set base URL with default
                lmstudio_base_url = self.base_url or "http://localhost:1234/v1"
                
                # Check if LMStudio is available (optional but recommended)
                if not self._check_lmstudio_health():
                    logger.warning(
                        f"LMStudio server is not responding at {lmstudio_base_url}. "
                        "Please ensure LMStudio is running with a loaded model. "
                        "Visit https://lmstudio.ai for installation instructions."
                    )
                    # Continue anyway - let the first request fail with clear error
                
                # Initialize LMStudio LLM using OpenAI-compatible API
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
                    # Note: Some parameters like top_k are model-specific in LMStudio
                )
                
                logger.info(
                    f"Initialized LMStudio LLM with model '{self.model_name}' "
                    f"at {lmstudio_base_url}"
                )
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Import tools
            from .tools import web_search, make_request
            
            # Create tools list
            tools = [web_search, make_request]
            
            # Create the react agent using langgraph
            try:
                # System prompt for the agent
                system_prompt = "You are a helpful assistant that can use tools to answer the user's question."
                
                # Create the agent using langgraph's create_react_agent
                self.agent_executor = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt=system_prompt
                )
                logger.info("Agent initialized successfully with create_react_agent")
                
            except Exception as e:
                logger.error(f"Error creating agent: {str(e)}")
                raise
                
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {str(e)}")
            raise ImportError(f"Failed to import required libraries: {str(e)}")
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama server is running and healthy."""
        import requests
        
        try:
            health_url = f"{self.ollama_url.rstrip('/')}/api/tags"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama server health check failed: {str(e)}")
            return False
    
    def _check_lmstudio_health(self) -> bool:
        """
        Check if LMStudio server is running and has models loaded.
        
        Returns:
            bool: True if LMStudio is healthy, False otherwise
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
                        f"LMStudio is running with {len(models)} model(s) loaded: "
                        f"{', '.join(model_names)}"
                    )
                    return True
                else:
                    logger.warning(
                        "LMStudio server is running but no models are loaded. "
                        "Please load a model in LMStudio before using the agent."
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
                "Ensure LMStudio is running and the server is enabled."
            )
            return False
        except Exception as e:
            logger.error(f"LMStudio health check failed: {str(e)}")
            return False

    def _ensure_ollama_model_loaded(self) -> None:
        """Ensure the desired model is available; if pull fails, delete all and retry, then warm-load."""
        import requests
        base = self.ollama_url.rstrip('/')

        def _pull_model() -> Optional[int]:
            try:
                resp = requests.post(f"{base}/api/pull", json={"model": self.model_name}, timeout=600)
                return resp.status_code
            except Exception as exc:
                logger.warning(f"Pull attempt failed for {self.model_name}: {str(exc)}")
                return None

        def _delete_all_models() -> None:
            try:
                tags = requests.get(f"{base}/api/tags", timeout=30)
                if tags.status_code == 200:
                    payload = tags.json() or {}
                    for m in payload.get("models", []):
                        name = m.get("name")
                        if not name:
                            continue
                        # Try DELETE first
                        try:
                            del_resp = requests.delete(f"{base}/api/delete", json={"name": name}, timeout=60)
                            if del_resp.status_code != 200:
                                # Fallback to POST if DELETE not supported
                                del_resp = requests.post(f"{base}/api/delete", json={"name": name}, timeout=60)
                            if del_resp.status_code == 200:
                                logger.info(f"Deleted model '{name}' from Ollama store")
                            else:
                                logger.warning(f"Failed to delete model '{name}': {del_resp.status_code} {del_resp.text}")
                        except Exception as de:
                            logger.warning(f"Exception deleting model '{name}': {str(de)}")
                else:
                    logger.warning(f"Failed to list models for deletion: {tags.status_code} {tags.text}")
            except Exception as e:
                logger.warning(f"Could not enumerate/delete models: {str(e)}")

        # First try pulling the requested model
        status = _pull_model()
        if status not in (200, 201):
            logger.warning(f"Initial pull of '{self.model_name}' failed (status={status}). Deleting all models and retrying pull.")
            _delete_all_models()
            status = _pull_model()
            if status not in (200, 201):
                logger.warning(f"Pull still failing for '{self.model_name}' (status={status}).")

        # Warm load desired model to memory
        try:
            warm = requests.post(
                f"{base}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "ping",
                    "stream": False,
                    "keep_alive": "30m"
                },
                timeout=180,
            )
            if warm.status_code not in (200, 201):
                logger.warning(f"Model warm-up returned status {warm.status_code}: {warm.text}")
        except Exception as e:
            logger.warning(f"Could not warm-load model {self.model_name}: {str(e)}")
    
    def get_information(self, prompt: str) -> str:
        """
        Get information from the agent based on the prompt.
        
        Args:
            prompt: The input prompt for the agent
            
        Returns:
            str: The response from the agent
        """
        if not self.agent_executor:
            self._initialize_agent()
            
        if not self.agent_executor:
            raise RuntimeError("Failed to initialize agent")
            
        logger.info(f"Agent request: {prompt[:100]}...")
        start_time = time.time()
        
        try:
            from langchain_core.messages import HumanMessage
            
            # Invoke the agent using langgraph format (messages-based)
            response = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 10}
            )
            execution_time = time.time() - start_time
            logger.info(f"Agent response received in {execution_time:.2f} seconds")
            
            # Extract the output from the last message
            messages = response.get("messages", [])
            if messages:
                # Get the last AI message content
                last_message = messages[-1]
                output = getattr(last_message, 'content', str(last_message))
            else:
                output = ""
                
            if not output:
                logger.warning("Empty response from agent, using fallback message")
                output = "No detailed information could be found for this request."
            
            return output
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return f"Error in agent execution: {str(e)}" 
