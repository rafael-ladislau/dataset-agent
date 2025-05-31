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
    Implementation of AgentInterface using LangChain with Ollama or OpenRouter.
    """
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.6, 
                 provider: str = "ollama", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        """
        Initialize the LangChain agent.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature parameter for the model
            provider: LLM provider (ollama or openrouter)
            api_key: API key for OpenRouter
            base_url: Base URL for OpenRouter API
        """
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.agent_executor = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent with tools and LLM."""
        try:
            # Import common LangChain components
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            from langchain_core.tools import Tool, tool
            from langchain.agents import create_tool_calling_agent, AgentExecutor
            from langchain_core.prompts import PromptTemplate
            
            # Initialize the LLM based on provider
            if self.provider == "ollama":
                from langchain_ollama import ChatOllama
                
                # Check if Ollama is available
                if not self._check_ollama_health():
                    logger.error("Ollama server is not running or not responding")
                    raise RuntimeError("Ollama server is not running or not responding")
                
                # Initialize Ollama LLM
                llm = ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature,
                    top_k=20,
                    top_p=0.95,
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
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Import tools
            from .tools import web_search, make_request
            
            # Create tools list
            tools = [web_search, make_request]
            
            # Import hub modules for prompt
            try:
                from langchain import hub
                from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                
                # Use the pre-built agent prompt from hub
                try:
                    prompt = hub.pull("hwchase17/openai-tools-agent")
                    logger.info("Using hub prompt for agent")
                except Exception as e:
                    logger.warning(f"Could not load hub prompt: {str(e)}")
                    
                    # Create a custom prompt if hub is not available
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant that can use tools to answer the user's question."),
                        ("user", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    logger.info("Using custom prompt for agent")
                
                # Create the agent
                agent = create_tool_calling_agent(llm, tools, prompt)
                
                # Create the agent executor
                self.agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5,
                    max_execution_time=120,
                    early_stopping_method="generate"
                )
                logger.info("Agent initialized successfully")
                
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
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama server health check failed: {str(e)}")
            return False
    
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
            # Invoke the agent
            response = self.agent_executor.invoke({"input": prompt})
            execution_time = time.time() - start_time
            logger.info(f"Agent response received in {execution_time:.2f} seconds")
            
            # Extract the output
            output = response.get("output", "")
            if not output:
                logger.warning("Empty response from agent, using fallback message")
                output = "No detailed information could be found for this request."
            
            return output
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return f"Error in agent execution: {str(e)}" 