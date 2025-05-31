"""
Dataset Research Agent

This script implements an LLM-powered agent specialized in researching datasets.
It uses the qwen3:30b model from Ollama and LangChain's bind_tools feature to
connect web search and request tools to the model.
"""

import json
import requests
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama  # Updated import for ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
import logging
import sys
import time

# Configure logging with separate handlers for file and console
# File handler with DEBUG level
file_handler = logging.FileHandler('dataset_research_agent.log')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Console handler with INFO level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# Set up the root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler], force=True)

# Get the module logger
logger = logging.getLogger(__name__)

# Filter out noisy logs from HTTP libraries
for module in ['httpcore', 'httpx', 'urllib3']:
    logging.getLogger(module).setLevel(logging.WARNING)

# Set logging levels for LangChain components
logging.getLogger('langchain').setLevel(logging.INFO)
logging.getLogger('langchain_community').setLevel(logging.INFO)
logging.getLogger('langchain_core').setLevel(logging.INFO)

logger.info("Logging configured: DEBUG to file, INFO to console")
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import ToolException
from langchain.callbacks.manager import CallbackManagerForToolRun

# Define tool schemas
class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""
    query: str = Field(
        ..., 
        description="The search query to find information. Be specific and include key terms for better results."
    )

class RequestsInput(BaseModel):
    """Input schema for the make_request tool."""
    url: str = Field(
        ..., 
        description="The complete URL to send a GET request to, including http:// or https:// prefix."
    )

# Define the web search tool
@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for the specified query.
    
    Args:
        query (str): The search query string
        
    Returns:
        str: Search results as a formatted string
    """
    logger.info(f"🔍 TOOL CALL: web_search with query: '{query}'")
    
    try:
        from duckduckgo_search import DDGS
        
        # Log the search start
        search_start = time.time()
        
        # Try different search backends with error handling
        results = []
        errors = []
        
        # First try the html backend
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=7))
            if results:
                logger.info(f"Successfully retrieved {len(results)} results from html backend")
        except Exception as e:
            error_msg = f"Error to search using html backend: {str(e)}"
            logger.info(error_msg)
            errors.append(error_msg)
            
            # If html backend fails, try lite backend
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=7, backend="lite"))
                if results:
                    logger.info(f"Successfully retrieved {len(results)} results from lite backend")
            except Exception as e:
                error_msg = f"Error to search using lite backend: {str(e)}"
                logger.info(error_msg)
                errors.append(error_msg)
        
        # Calculate and log search time
        search_time = time.time() - search_start
        logger.info(f"Web search completed in {search_time:.2f} seconds with {len(results)} results")
        
        # Format the results for the agent
        formatted_results = f"Search results for '{query}':\n\n"
        
        if not results:
            formatted_results += "No results found. "
            if errors:
                formatted_results += f"Errors encountered: {'; '.join(errors)}"
            return formatted_results
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            snippet = result.get('body', 'No snippet').replace('\n', ' ')
            
            formatted_results += f"[{i}] {title}\n"
            formatted_results += f"URL: {url}\n"
            formatted_results += f"Summary: {snippet}\n\n"
        
        # Log a sample of the results
        logger.info(f"Search results (sample): {formatted_results[:500]}...")
        
        return formatted_results
    
    except ImportError:
        error_msg = "Error: DuckDuckGo-Search package not installed. Install with 'pip install duckduckgo-search'"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error during web search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

# Define the custom requests tool with browser-like headers
@tool("make_request", args_schema=RequestsInput)
def make_request(url: str) -> str:
    """
    Make an HTTP request to the specified URL and return information about the response.
    
    Args:
        url (str): The URL to request
        
    Returns:
        str: Response information as a formatted string
    """
    logger.info(f"🌐 TOOL CALL: make_request to URL: '{url}'")
    
    try:
        import requests
        from requests.exceptions import RequestException
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
                logger.info(f"Added https:// prefix to URL: {url}")
            else:
                return "Error: Invalid URL format. URL must start with http:// or https://"
        
        # Make the request with a timeout
        logger.info(f"Making request to: {url}")
        response = requests.get(url, timeout=10)
        status_code = response.status_code
        
        # Prepare response information
        result = f"Status code: {status_code}\n"
        
        # Add details based on status code
        if status_code == 200:
            result += "Valid: True\n"
            
            # Try to determine content type
            content_type = response.headers.get('Content-Type', 'unknown')
            result += f"Content-Type: {content_type}\n"
            
            # Check if it's a data file or documentation
            if 'application/json' in content_type or 'text/csv' in content_type or 'application/xml' in content_type:
                result += "Content: DATA FILE\n"
                # Sample the content (first 500 chars)
                sample = response.text[:500] + ("..." if len(response.text) > 500 else "")
                result += f"Sample content: {sample}\n"
            elif 'text/html' in content_type:
                result += "Content: HTML PAGE\n"
                # Extract the title if possible
                import re
                title_match = re.search('<title>(.*?)</title>', response.text, re.IGNORECASE | re.DOTALL)
                if title_match:
                    result += f"Page title: {title_match.group(1).strip()}\n"
                # Check for dataset-related keywords
                keywords = ['dataset', 'data dictionary', 'documentation', 'schema', 'metadata', 'download']
                found_keywords = [kw for kw in keywords if kw.lower() in response.text.lower()]
                if found_keywords:
                    result += f"Dataset-related keywords found: {', '.join(found_keywords)}\n"
            elif 'application/pdf' in content_type:
                result += "Content: PDF DOCUMENT\n"
            else:
                result += f"Content appears to be: {content_type}\n"
                
        elif 300 <= status_code < 400:
            result += "Valid: False (Redirection)\n"
            if 'Location' in response.headers:
                result += f"Redirects to: {response.headers['Location']}\n"
        elif status_code == 404:
            result += "Valid: False (Not Found)\n"
            result += "This URL does not exist or has been removed.\n"
        elif 400 <= status_code < 500:
            result += "Valid: False (Client Error)\n"
            result += f"The server rejected this request with error {status_code}.\n"
        elif 500 <= status_code < 600:
            result += "Valid: False (Server Error)\n"
            result += f"The server encountered an error while processing this request: {status_code}.\n"
        
        # Log the result
        logger.info(f"Request result: {url} {status_code}")
        return result
    
    except RequestException as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        return f"Error: Failed to retrieve URL: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in make_request: {str(e)}", exc_info=True)
        return f"Error: An unexpected error occurred: {str(e)}"

def extract_list_from_text(text: str) -> List[str]:
    """
    Extract a Python list from a text response.
    
    Args:
        text (str): Text containing a Python list
        
    Returns:
        List[str]: Extracted list of strings, or empty list if extraction fails
    """
    import re
    
    # Log the text length for debugging
    logger.info(f"Extracting list from text of length {len(text)}")
    
    # Try to find a Python list pattern using regex
    # This will match lists like ["item1", "item2"] or ['item1', 'item2']
    list_pattern = r'\[(.*?)\]'
    match = re.search(list_pattern, text, re.DOTALL)
    
    if match:
        try:
            # Get the content inside brackets
            items_text = match.group(1)
            
            # Split by commas, handling both single and double quotes
            items = re.findall(r'(?:"([^"]*)")|(?:\'([^\']*)\')', items_text)
            
            # Extract the non-None group from each match tuple
            result = [item[0] if item[0] else item[1] for item in items]
            
            # If we found items, consider it a success
            if result:
                logger.info(f"Successfully extracted {len(result)} items using regex")
                logger.info(f"Extracted items: {result}")
                return result
    except Exception as e:
            logger.warning(f"Error parsing list with regex: {str(e)}")
    
    # Fallback methods if regex didn't work
    
    # Try using ast.literal_eval for more complex cases
    try:
        import ast
        # Find something that looks like a list
        list_matches = re.findall(r'\[.*?\]', text, re.DOTALL)
        for list_match in list_matches:
            try:
                result = ast.literal_eval(list_match)
                if isinstance(result, list) and all(isinstance(item, str) for item in result):
                    logger.info(f"Successfully extracted {len(result)} items using ast.literal_eval")
                    logger.info(f"Extracted items: {result}")
                    return result
            except:
                continue
    except Exception as e:
        logger.warning(f"Error using ast.literal_eval: {str(e)}")
    
    # Last resort: look for markdown list items or numbered lists
    try:
        # Try to find markdown list items like "- item1" or "* item1" or "1. item1"
        items = re.findall(r'(?:^|\n)\s*(?:-|\*|\d+\.)\s*(.*?)(?=\n|$)', text)
            if items:
            # Clean up items
            items = [item.strip() for item in items if item.strip()]
            if items:
                logger.info(f"Successfully extracted {len(items)} items from markdown list")
                logger.info(f"Extracted items: {items}")
                return items
    except Exception as e:
        logger.warning(f"Error extracting markdown list: {str(e)}")
    
    # If all else fails, return an empty list
    logger.warning("Failed to extract a list from the text")
        return []

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    logger.info(f"Extracting URLs from text of length {len(text)}")
    logger.debug(f"Text excerpt: {text[:100]}...")
    
    try:
        # More comprehensive URL pattern
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
        logger.debug(f"Using URL regex pattern: {url_pattern}")
        
        # Find all URLs
        urls = re.findall(url_pattern, text)
        
        # Log results
        if urls:
            logger.info(f"Found {len(urls)} URLs in text")
            for i, url in enumerate(urls[:5]):  # Log first 5 URLs
                logger.info(f"URL {i+1}: {url}")
            if len(urls) > 5:
                logger.info(f"... and {len(urls) - 5} more URLs")
        else:
            logger.warning("No URLs found in text")
            
        # Normalize URLs - ensure they all have proper http/https prefix
        normalized_urls = []
        for url in urls:
            if url.startswith('www.'):
                normalized_url = 'https://' + url
                logger.info(f"Normalized URL from {url} to {normalized_url}")
                normalized_urls.append(normalized_url)
            else:
                normalized_urls.append(url)
                
        return normalized_urls
    except Exception as e:
        logger.error(f"Error extracting URLs: {str(e)}", exc_info=True)
        logger.debug(f"Exception type: {type(e).__name__}")
        logger.debug(f"Exception args: {e.args}")
        return []

def create_dataset_research_agent(dataset_name: str, dataset_url: Optional[str] = None):
    """
    Create and run a dataset research agent.
    
    Args:
        dataset_name (str): The name of the dataset to research
        dataset_url (Optional[str]): An optional URL related to the dataset
        
    Returns:
        dict: JSON object with the dataset information
    """
    logger.info(f"Creating dataset research agent for '{dataset_name}'")
    logger.debug(f"Input dataset_name: {dataset_name}")
    logger.debug(f"Input dataset_url: {dataset_url}")
    
    # Track memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before agent creation: {memory_before:.2f} MB")
    except ImportError:
        logger.warning("psutil not installed, skipping memory usage tracking")
    except Exception as e:
        logger.warning(f"Unable to track memory usage: {str(e)}")
    
    # Log execution start for performance tracking
    start_time = time.time()
    step_start_time = start_time
    # Set up the system prompt
    system_prompt = """
    You are a dataset research expert agent specialized in extracting comprehensive information about datasets.
    You have access to web search and request tools to gather detailed information.
    
    Your goal is to research a dataset and collect the following information:
    1. Aliases: All forms of reference to the dataset (names, acronyms, DOIs, etc.)
    2. Flag terms: All organizations related to the dataset (with acronyms and full names)
    3. Description: Brief explanation of the dataset and its purpose
    4. Access Type: Whether it's "Open", "Restricted", or "Unknown"
    5. Data URL: Validated URL to download the dataset files
    6. Schema URL: Validated URL to the dataset schema/data dictionary
    7. Documentation URL: Validated URL to the dataset documentation
    
    Follow this workflow exactly, in order:
    1. Introduce yourself as a dataset research expert
    2. Research and extract a description of the dataset
    3. Research and identify all aliases of the dataset
    4. Research and identify all flag terms (organizations) related to the dataset
    5. Research and validate the data URL for downloading the dataset
    6. Research and validate the schema URL for the dataset
    7. Research and validate the documentation URL for the dataset
    8. Compile all findings into a structured JSON format
    
    IMPORTANT RULES:
    - ALWAYS use the web_search tool BEFORE suggesting any URLs. NEVER generate URLs from memory.
    - ONLY use URLs that were returned in web_search results.
    - For each URL type (data, schema, documentation), perform a specific web search with targeted keywords.
    - Use the requests tool to validate that each URL returns a 200 status code
    - Set the URL to null if it cannot be validated
    
    ALWAYS be thorough in your research. Use explicit search queries to find the information.
    NEVER skip any step in the workflow.
    
    ALWAYS use the provided tools instead of using your own knowledge.
    """
    
    # Initialize the model
    logger.info("Initializing ChatOllama model")
    try:
        logger.debug("Model parameters: model=qwen3:30b, temperature=0.6, streaming=True")
        llm = ChatOllama(
            model="qwen3:30b",  # Using a model better suited for function calling
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            streaming=True
        )
        logger.info("ChatOllama model initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize ChatOllama model: {str(e)}", exc_info=True)
        logger.debug(f"Error type: {type(e).__name__}")
        logger.debug(f"Error details: {e}")
        raise Exception(f"Failed to initialize model: {str(e)}")
    
    # Initialize conversation memory
    logger.info("Initializing conversation memory")
    try:
        memory = ConversationBufferMemory(return_messages=True)
        logger.debug("ConversationBufferMemory initialized with return_messages=True")
    except Exception as e:
        logger.error(f"Failed to initialize memory: {str(e)}", exc_info=True)
        raise Exception(f"Failed to initialize memory: {str(e)}")
    
    # Create tools list
    logger.info("Setting up tools")
    tools = [web_search, make_request]
    # Fix: Use tool.name instead of __name__ since StructuredTool objects don't have __name__
    logger.debug(f"Tools registered: {[tool.name for tool in tools]}")
    
    # Set up LangChain Agent with proper handling for Ollama
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # Create a proper prompt template for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    logger.info("Creating agent with tools")
    try:
        # Create agent with the model and tools
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            return_intermediate_steps=True
        )
        
        logger.info("Successfully created agent executor with tools")
    except Exception as e:
        logger.critical(f"Failed to create agent: {str(e)}", exc_info=True)
        logger.debug(f"Error type: {type(e).__name__}")
        logger.debug(f"Error details: {e}")
        raise Exception(f"Failed to create agent: {str(e)}")
    
    # Initialize result dictionary to store findings at each step
    result = {
        "aliases": [],
        "flag_terms": [],
        "description": "",
        "access_type": "Unknown",
        "data_url": None,
        "schema_url": None,
        "documentation_url": None
    }
    
    # Store chat history
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    chat_history = []
    
    # Initialize dictionary to track step times
    step_times = {}
    
    # Function to log the timing of each step
    def log_step_timing(step_name):
        nonlocal step_start_time
        step_time = time.time() - step_start_time
        logger.info(f"Step '{step_name}' completed in {step_time:.2f} seconds")
        step_start_time = time.time()
        
        # Log memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            memory_current = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage after step '{step_name}': {memory_current:.2f} MB")
        except:
            pass
        
        # Store step time in the dictionary
        step_times[step_name] = step_time
        
        return step_time
    
    # Step 1: Introduction
    logger.info("=" * 50)
    logger.info("Starting Step 1: Introduction")
    print("Step 1: Introduction")
    initial_input = f"I need to research information about the dataset named '{dataset_name}'"
    if dataset_url:
        initial_input += f" with URL {dataset_url}"
    
    # Send initial query
    logger.info(f"Sending to model: Initial query about '{dataset_name}'")
    messages = [HumanMessage(content=initial_input)]
    chat_history = messages.copy()
    logger.debug(f"Invoking model with initial input: {initial_input[:100]}...")
    try:
        response = agent_executor.invoke({"input": initial_input, "chat_history": []})
        logger.debug("Model response received successfully")
        logger.debug(f"Response type: {type(response)}")
        logger.info(f"Model response: {response['output'][:500]}" + ("..." if len(response['output']) > 500 else ""))
        # Store the response in chat history
        chat_history.append(AIMessage(content=response['output']))
    except Exception as e:
        logger.error(f"Failed to get model response for initial input: {str(e)}", exc_info=True)
        raise Exception(f"Failed to get model response: {str(e)}")
    
    # Log timing for Step 1
    log_step_timing("Introduction")
    
    # Step 2: Description Research
    logger.info("=" * 50)
    logger.info("Starting Step 2: Description Research")
    print("\nStep 2: Description Research")
    step_start_time = time.time()
    
    description_prompt = f"Please use the web_search tool to find a detailed description of the '{dataset_name}' dataset."
    logger.info(f"Sending to model: Request to search for description of '{dataset_name}'")
    response = agent_executor.invoke({"input": description_prompt, "chat_history": chat_history})
    logger.info(f"Model response: {response['output'][:500]}" + ("..." if len(response['output']) > 500 else ""))
    chat_history.append(HumanMessage(content=description_prompt))
    chat_history.append(AIMessage(content=response['output']))
    
    # Extract a description from the search results
    extract_prompt = "Based on your search, provide a concise description of the dataset in 150-200 words. Focus on what the dataset contains, who created it, its purpose, and key features. DO NOT include any markdown formatting or your thinking process."
    logger.info("Sending to model: Request to provide concise description")
    description_response = agent_executor.invoke({"input": extract_prompt, "chat_history": chat_history})
    logger.info(f"Model response (description): {description_response['output']}")
    chat_history.append(HumanMessage(content=extract_prompt))
    chat_history.append(AIMessage(content=description_response['output']))

    # Store description
    result["description"] = description_response['output']
    log_step_timing("Description Research")
    
    # Step 3: Aliases Research
    logger.info("=" * 50)
    logger.info("Starting Step 3: Aliases Research")
    print("\nStep 3: Aliases Research")
    step_start_time = time.time()
    
    aliases_prompt = f"Please use the web_search tool to find all names, acronyms, identifiers, and other aliases used to refer to the '{dataset_name}' dataset."
    logger.info(f"Sending to model: Request to search for aliases of '{dataset_name}'")
    response = agent_executor.invoke({"input": aliases_prompt, "chat_history": chat_history})
    logger.info(f"Model response: {response['output'][:500]}" + ("..." if len(response['output']) > 500 else ""))
    chat_history.append(HumanMessage(content=aliases_prompt))
    chat_history.append(AIMessage(content=response['output']))
    
    # Extract aliases from the response
    extract_aliases_prompt = "Based on your search, provide a list of all aliases for this dataset. Format as a Python list of strings."
    logger.info("Sending to model: Request to provide list of aliases")
    aliases_response = agent_executor.invoke({"input": extract_aliases_prompt, "chat_history": chat_history})
    logger.info(f"Model response (aliases): {aliases_response['output']}")
    chat_history.append(HumanMessage(content=extract_aliases_prompt))
    chat_history.append(AIMessage(content=aliases_response['output']))
    
    # Parse and save aliases
    aliases = extract_list_from_text(aliases_response['output'])
    if not aliases:
        # If extraction failed, ask the agent to fix the format
        fix_prompt = "Please format the aliases as a simple Python list of strings, e.g. ['alias1', 'alias2']"
        logger.info("Sending to model: Request to fix aliases format")
        fix_response = agent_executor.invoke({"input": fix_prompt, "chat_history": chat_history})
        logger.info(f"Model response (fixed aliases): {fix_response['output']}")
        chat_history.append(HumanMessage(content=fix_prompt))
        chat_history.append(AIMessage(content=fix_response['output']))
        
        # Try extraction again
        aliases = extract_list_from_text(fix_response['output'])
    
    # Store the aliases in the result dictionary
    logger.info(f"Extracted aliases: {aliases}")
    result["aliases"] = aliases
    
    # Log step timing
    step_time = log_step_timing("Aliases Research")
    
    # Step 4: Flag Terms Research
    logger.info("=" * 50)
    logger.info("Starting Step 4: Flag Terms Research")
    print("\nStep 4: Flag Terms Research")
    step_start_time = time.time()
    
    flag_terms_prompt = f"Please use the web_search tool to find all organizations related to the '{dataset_name}' dataset, including creators, funders, stewards, and hosts. Include both full names and acronyms."
    logger.info(f"Sending to model: Request to search for organizations related to '{dataset_name}'")
    response = agent_executor.invoke({"input": flag_terms_prompt, "chat_history": chat_history})
    logger.info(f"Model response: {response['output'][:500]}" + ("..." if len(response['output']) > 500 else ""))
    chat_history.append(HumanMessage(content=flag_terms_prompt))
    chat_history.append(AIMessage(content=response['output']))
    
    # Extract flag terms from the response
    extract_flag_terms_prompt = "Based on your search, provide a list of all organizations related to this dataset. Format as a Python list of strings."
    logger.info("Sending to model: Request to provide list of organizations")
    flag_terms_response = agent_executor.invoke({"input": extract_flag_terms_prompt, "chat_history": chat_history})
    logger.info(f"Model response (flag terms): {flag_terms_response['output']}")
    chat_history.append(HumanMessage(content=extract_flag_terms_prompt))
    chat_history.append(AIMessage(content=flag_terms_response['output']))
    
    # Parse and save flag terms
    flag_terms = extract_list_from_text(flag_terms_response['output'])
    if not flag_terms:
        # If extraction failed, ask the agent to fix the format
        fix_prompt = "Please format the organizations as a simple Python list of strings, e.g. ['org1', 'org2']"
        logger.info("Sending to model: Request to fix organizations format")
        fix_response = agent_executor.invoke({"input": fix_prompt, "chat_history": chat_history})
        logger.info(f"Model response (fixed organizations): {fix_response['output']}")
        chat_history.append(HumanMessage(content=fix_prompt))
        chat_history.append(AIMessage(content=fix_response['output']))
        
        # Try extraction again
        flag_terms = extract_list_from_text(fix_response['output'])
    
    # Store the flag terms in the result dictionary
    logger.info(f"Extracted flag terms: {flag_terms}")
    result["flag_terms"] = flag_terms
    
    # Determine access type
    access_type_prompt = "Based on your research, determine if this dataset has 'Open', 'Restricted', or 'Unknown' access. Just answer with one of these three words."
    logger.info("Sending to model: Request to determine access type")
    access_type_response = agent_executor.invoke({"input": access_type_prompt, "chat_history": chat_history})
    logger.info(f"Model response (access type): {access_type_response['output']}")
    chat_history.append(HumanMessage(content=access_type_prompt))
    chat_history.append(AIMessage(content=access_type_response['output']))
    
    # Store the access type in the result dictionary
    access_type = access_type_response['output'].strip().lower()
    # Normalize to one of the three allowed values
    if "open" in access_type:
        result["access_type"] = "Open"
    elif "restricted" in access_type:
        result["access_type"] = "Restricted"
    else:
        result["access_type"] = "Unknown"
    logger.info(f"Determined access type: {result['access_type']}")
    
    # Log step timing
    step_time = log_step_timing("Flag Terms Research")
    
    # Step 5: Data URL Research
    print("\nStep 5: Data URL Research")
    step_start_time = time.time()
    
    # Search for data URLs
    data_url_prompt = f"""Please use the web_search tool to find direct download links for the '{dataset_name}' dataset. 
You MUST:
1. Use the web_search tool first to find potential URLs
2. Only suggest URLs that appeared in the search results
3. Validate each URL with the make_request tool before confirming
4. Check both the status code AND content returned

Be specific and search for downloadable data files, not just informational pages."""
    logger.info(f"Sending to agent: Request to search for download links for '{dataset_name}'")
    data_url_response = agent_executor.invoke({"input": data_url_prompt, "chat_history": chat_history})
    logger.info(f"Agent response: {data_url_response['output'][:500]}" + ("..." if len(data_url_response['output']) > 500 else ""))
    
    # Add to chat history
    chat_history.append(HumanMessage(content=data_url_prompt))
    chat_history.append(AIMessage(content=data_url_response['output']))
    
    # Extract list of data URLs
    extract_data_urls_prompt = """Based on your search, provide a Python list of URLs where the dataset can be downloaded.
IMPORTANT:
1. ONLY include URLs that appeared in the search results
2. Each URL must be validated with the make_request tool
3. Format as a Python list of strings: ["url1", "url2", ...]
4. Do NOT suggest URLs that weren't in the search results"""
    logger.info("Sending to agent: Request to provide list of download URLs")
    data_urls_response = agent_executor.invoke({"input": extract_data_urls_prompt, "chat_history": chat_history})
    logger.info(f"Agent response (data URLs): {data_urls_response['output']}")
    
    # Add to chat history
    chat_history.append(HumanMessage(content=extract_data_urls_prompt))
    chat_history.append(AIMessage(content=data_urls_response['output']))
    
    # Parse and save data URLs
    potential_data_urls = extract_list_from_text(data_urls_response['output'])
    logger.info(f"Potential data URLs to validate: {potential_data_urls}")
    
    # Validate data URLs
    for url in potential_data_urls:
        validate_prompt = f"""Please use the make_request tool to validate this potential data URL: {url}
You MUST:
1. Call the make_request tool with this exact URL
2. Check the HTTP status code (200 is success)
3. Examine the content to confirm it's dataset related
4. Report whether the URL is valid and contains actual data

Report back with complete details."""
        logger.info(f"Sending to agent: Request to validate URL: {url}")
        validate_response = agent_executor.invoke({"input": validate_prompt, "chat_history": chat_history})
        logger.debug(f"Full validation response: {validate_response}")
        
        # Check if response contains success indicators
        is_valid = False
        if "200" in validate_response['output'] and ("valid" in validate_response['output'].lower() or "success" in validate_response['output'].lower()):
            is_valid = True
            logger.info(f"URL validation successful: {url}")
        else:
            logger.info(f"URL validation failed: {url}")
        
        # Add to chat history
        chat_history.append(HumanMessage(content=validate_prompt))
        chat_history.append(AIMessage(content=validate_response['output']))
        
        if is_valid:
            result["data_url"] = url
            logger.info(f"Valid data URL found: {url}")
            break
    
    log_step_timing("Data URL Research")
    
    # Step 6: Schema URL Research
    print("\nStep 6: Schema URL Research")
    step_start_time = time.time()
    
    # Search for schema URLs
    schema_url_prompt = f"""Please use the web_search tool to find schema or data dictionary URLs for the '{dataset_name}' dataset.
You MUST:
1. Use the web_search tool first to find potential schema URLs
2. Only suggest URLs that appeared in the search results
3. Look specifically for data dictionaries, field descriptions, or schema documentation
4. Validate each URL with the make_request tool before confirming

A schema URL typically describes the structure and fields of the dataset."""
    logger.info(f"Sending to agent: Request to search for schema URLs for '{dataset_name}'")
    schema_url_response = agent_executor.invoke({"input": schema_url_prompt, "chat_history": chat_history})
    logger.info(f"Agent response: {schema_url_response['output'][:500]}" + ("..." if len(schema_url_response['output']) > 500 else ""))
    
    # Add to chat history
    chat_history.append(HumanMessage(content=schema_url_prompt))
    chat_history.append(AIMessage(content=schema_url_response['output']))
    
    # Extract list of schema URLs
    extract_schema_urls_prompt = """Based on your search, provide a Python list of schema or data dictionary URLs for the dataset.
IMPORTANT:
1. ONLY include URLs that appeared in the search results
2. Each URL must be validated with the make_request tool
3. Format as a Python list of strings: ["url1", "url2", ...]
4. Do NOT suggest URLs that weren't in the search results"""
    logger.info("Sending to agent: Request to provide list of schema URLs")
    schema_urls_response = agent_executor.invoke({"input": extract_schema_urls_prompt, "chat_history": chat_history})
    logger.info(f"Agent response (schema URLs): {schema_urls_response['output']}")
    
    # Add to chat history
    chat_history.append(HumanMessage(content=extract_schema_urls_prompt))
    chat_history.append(AIMessage(content=schema_urls_response['output']))
    
    # Parse and save schema URLs
    potential_schema_urls = extract_list_from_text(schema_urls_response['output'])
    logger.info(f"Potential schema URLs to validate: {potential_schema_urls}")
    
    # Validate schema URLs
    for url in potential_schema_urls:
        validate_prompt = f"""Please use the make_request tool to validate this potential schema URL: {url}
You MUST:
1. Call the make_request tool with this exact URL
2. Check the HTTP status code (200 is success)
3. Examine the content to confirm it's schema/data dictionary related
4. Report whether it's a valid schema document

Report back with complete details."""
        logger.info(f"Sending to agent: Request to validate URL: {url}")
        validate_response = agent_executor.invoke({"input": validate_prompt, "chat_history": chat_history})
        logger.debug(f"Full validation response: {validate_response}")
        
        # Check if response contains success indicators
        is_valid = False
        if "200" in validate_response['output'] and ("valid" in validate_response['output'].lower() or "success" in validate_response['output'].lower()):
            is_valid = True
            logger.info(f"URL validation successful: {url}")
        else:
            logger.info(f"URL validation failed: {url}")
        
        # Add to chat history
        chat_history.append(HumanMessage(content=validate_prompt))
        chat_history.append(AIMessage(content=validate_response['output']))
        
        if is_valid:
            result["schema_url"] = url
            logger.info(f"Valid schema URL found: {url}")
            break
    
    log_step_timing("Schema URL Research")
    
    # Step 7: Documentation URL Research
    print("\nStep 7: Documentation URL Research")
    step_start_time = time.time()
    
    # Search for documentation URLs
    documentation_prompt = f"""Please use the web_search tool to find documentation URLs for the '{dataset_name}' dataset.
You MUST:
1. Use the web_search tool first to find potential documentation URLs
2. Only suggest URLs that appeared in the search results 
3. Look specifically for user guides, methodology documents, or technical pages
4. Validate each URL with the make_request tool before confirming

Good documentation explains how to use the dataset or details its methodology."""
    logger.info(f"Sending to agent: Request to search for documentation URLs for '{dataset_name}'")
    documentation_response = agent_executor.invoke({"input": documentation_prompt, "chat_history": chat_history})
    logger.info(f"Agent response: {documentation_response['output'][:500]}" + ("..." if len(documentation_response['output']) > 500 else ""))
    
    # Add to chat history
    chat_history.append(HumanMessage(content=documentation_prompt))
    chat_history.append(AIMessage(content=documentation_response['output']))
    
    # Extract list of documentation URLs
    extract_doc_urls_prompt = """Based on your search, provide a Python list of documentation URLs for the dataset.
IMPORTANT:
1. ONLY include URLs that appeared in the search results
2. Each URL must be validated with the make_request tool
3. Format as a Python list of strings: ["url1", "url2", ...]
4. Do NOT suggest URLs that weren't in the search results"""
    logger.info("Sending to agent: Request to provide list of documentation URLs")
    doc_urls_response = agent_executor.invoke({"input": extract_doc_urls_prompt, "chat_history": chat_history})
    logger.info(f"Agent response (documentation URLs): {doc_urls_response['output']}")
    
    # Add to chat history
    chat_history.append(HumanMessage(content=extract_doc_urls_prompt))
    chat_history.append(AIMessage(content=doc_urls_response['output']))
    
    # Parse and save documentation URLs
    potential_doc_urls = extract_list_from_text(doc_urls_response['output'])
    logger.info(f"Potential documentation URLs to validate: {potential_doc_urls}")
    
    # Validate documentation URLs
    for url in potential_doc_urls:
        validate_prompt = f"""Please use the make_request tool to validate this potential documentation URL: {url}
You MUST:
1. Call the make_request tool with this exact URL
2. Check the HTTP status code (200 is success)
3. Examine the content to confirm it's documentation related
4. Report whether it's a valid documentation resource

Report back with complete details."""
        logger.info(f"Sending to agent: Request to validate URL: {url}")
        validate_response = agent_executor.invoke({"input": validate_prompt, "chat_history": chat_history})
        logger.debug(f"Full validation response: {validate_response}")
        
        # Check if response contains success indicators
        is_valid = False
        if "200" in validate_response['output'] and ("valid" in validate_response['output'].lower() or "success" in validate_response['output'].lower()):
            is_valid = True
            logger.info(f"URL validation successful: {url}")
        else:
            logger.info(f"URL validation failed: {url}")
        
        # Add to chat history
        chat_history.append(HumanMessage(content=validate_prompt))
        chat_history.append(AIMessage(content=validate_response['output']))
        
        if is_valid:
            result["documentation_url"] = url
            logger.info(f"Valid documentation URL found: {url}")
            break
    
    log_step_timing("Documentation URL Research")
    
    # Final Result Compilation
    print("\nCompiling results...")
    compile_prompt = "Based on all the research we've done, please compile a final JSON summary of the dataset information."
    logger.info("Sending to agent: Request to compile final JSON summary")
    final_response = agent_executor.invoke({"input": compile_prompt, "chat_history": chat_history})
    logger.info(f"Agent response (final compilation): {final_response['output']}")

    # Add to chat history
    chat_history.append(HumanMessage(content=compile_prompt))
    chat_history.append(AIMessage(content=final_response['output']))
    print(final_response['output'])

    # Save the results to a file
    logger.info("Research completed in {:.2f} seconds".format(time.time() - start_time))
    logger.info("Research completed with results:")
    logger.info(f"Description length: {len(result.get('description', ''))} chars")
    logger.info(f"Found {len(result.get('aliases', []))} aliases: {result.get('aliases', [])}")
    logger.info(f"Found {len(result.get('flag_terms', []))} flag terms: {result.get('flag_terms', [])}")
    logger.info(f"Access type: {result.get('access_type', 'Unknown')}")
    logger.info(f"Data URL: {result.get('data_url', None)}")
    logger.info(f"Schema URL: {result.get('schema_url', None)}")
    logger.info(f"Documentation URL: {result.get('documentation_url', None)}")

    # Output the final result
    print("\nDataset Research Results:")
    output_file_name = f"{dataset_name.lower().replace(' ', '_')}_research.json"
    json_result = json.dumps(result, indent=2)
    print(json_result)
    logger.info(f"Saving results to file: {output_file_name}")
    with open(output_file_name, "w") as f:
        f.write(json_result)
    logger.info(f"Results successfully saved to {output_file_name}")

    print(f"\nResults saved to {output_file_name}")
    logger.info("Dataset research process completed successfully")
    
    # Return the completed result
    return result

def main():
    """Main function to run the dataset research agent."""
    import argparse
    import platform
    import sys
    import datetime
    
    # Log system information for debugging
    logger.info("=" * 80)
    logger.info(f"Dataset Research Agent starting at {datetime.datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Check key packages
    try:
        import langchain
        logger.info(f"LangChain version: {langchain.__version__}")
    except Exception as e:
        logger.warning(f"Failed to get LangChain version: {str(e)}")
        
    try:
        import requests
        logger.info(f"Requests version: {requests.__version__}")
    except Exception as e:
        logger.warning(f"Failed to get Requests version: {str(e)}")
        
    try:
        import duckduckgo_search
        logger.info(f"DuckDuckGo Search version: {duckduckgo_search.__version__}")
    except Exception as e:
        logger.warning(f"Failed to get DuckDuckGo Search version: {str(e)}")
        
    logger.info("=" * 80)
    
    # Set up argument parser
    logger.debug("Setting up argument parser")
    parser = argparse.ArgumentParser(description="Dataset Research Agent")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to research")
    parser.add_argument("--url", type=str, help="Optional URL related to the dataset", default=None)
    parser.add_argument("--output", type=str, help="Output JSON file path", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable additional debug output")
    
    # Parse arguments
    logger.debug("Parsing command line arguments")
    args = parser.parse_args()
    
    # If debug flag is set, increase logging verbosity even further
    if args.debug:
        logger.info("Debug mode enabled via command line")
        logging.getLogger().setLevel(logging.DEBUG)
        # Enable even more verbose logging for HTTP libraries in debug mode
        logging.getLogger('httpx').setLevel(logging.DEBUG)
        logging.getLogger('requests').setLevel(logging.DEBUG)
    
    try:
        # Log the input parameters
        logger.info(f"Researching dataset: {args.dataset_name}")
        if args.url:
            logger.info(f"Starting URL: {args.url}")
        else:
            logger.info("No starting URL provided")
            
        # Track execution time
        start_time = time.time()
        logger.info(f"Starting research at {time.strftime('%H:%M:%S')}")
        
        # Run the agent
        logger.info("Invoking dataset research agent")
        result = create_dataset_research_agent(args.dataset_name, args.url)
        
        # Log execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Research completed in {elapsed_time:.2f} seconds")
        
        # Log the results
        logger.info("Research completed with results:")
        logger.info(f"Description length: {len(result['description'])} chars")
        logger.info(f"Found {len(result['aliases'])} aliases: {result['aliases']}")
        logger.info(f"Found {len(result['flag_terms'])} flag terms: {result['flag_terms']}")
        logger.info(f"Access type: {result['access_type']}")
        logger.info(f"Data URL: {result['data_url']}")
        logger.info(f"Schema URL: {result['schema_url']}")
        logger.info(f"Documentation URL: {result['documentation_url']}")
        
        # Pretty print the results to console
        print("\nDataset Research Results:")
        formatted_json = json.dumps(result, indent=2)
        print(formatted_json)
        logger.debug(f"Formatted JSON output: {formatted_json}")
        
        # Save results to a JSON file
        output_filename = args.output or f"{args.dataset_name.replace(' ', '_').lower()}_research.json"
        logger.info(f"Saving results to file: {output_filename}")
        
        try:
            with open(output_filename, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results successfully saved to {output_filename}")
        except Exception as save_err:
            logger.error(f"Error saving results to file: {str(save_err)}", exc_info=True)
            print(f"Error saving results: {str(save_err)}")
        
        print(f"\nResults saved to {output_filename}")
        logger.info("Dataset research process completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (KeyboardInterrupt)")
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Critical error in main function: {str(e)}", exc_info=True)
        logger.debug(f"Exception type: {type(e).__name__}")
        logger.debug(f"Exception args: {e.args}")
        
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        
        logger.info("Execution terminated with error")
    finally:
        # Always log completion
        logger.info(f"Dataset Research Agent finished at {datetime.datetime.now().isoformat()}")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()