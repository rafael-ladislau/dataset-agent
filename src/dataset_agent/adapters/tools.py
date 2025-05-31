"""
Tool implementations for the dataset agent.

This module defines tools used by the agent for gathering information.
"""

import time
import logging
import requests
import os
from typing import Dict, Any, Literal

from langchain_core.tools import tool

# Set up logging
logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """
    Search the web for the specified query.
    
    Args:
        query (str): The search query string
        
    Returns:
        str: Search results as a formatted string
    """
    logger.info(f"Web search tool called with query: '{query}'")
    
    # Get the configured provider from environment
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "duckduckgo").lower()
    
    # Dispatch to the appropriate provider
    if provider == "tavily":
        return _tavily_search(query)
    else:
        return _duckduckgo_search(query)


def _tavily_search(query: str) -> str:
    """
    Search the web using Tavily for the specified query.
    
    Args:
        query (str): The search query string
        
    Returns:
        str: Search results as a formatted string
    """
    logger.info(f"Using Tavily search provider for query: '{query}'")
    tool_start_time = time.time()
    
    try:
        from tavily import TavilyClient
        
        # Get Tavily API key from environment
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.error("TAVILY_API_KEY not found in environment variables")
            return "Error: Tavily API key not found. Please set TAVILY_API_KEY environment variable."
        
        # Log the search start
        search_start = time.time()
        
        # Initialize Tavily client
        client = TavilyClient(api_key=tavily_api_key)
        
        # Perform search
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=7,
            include_answer=False,
            include_raw_content=False,
            include_images=False
        )
        
        results = response.get("results", [])
        
        # Calculate and log search time
        search_time = time.time() - search_start
        logger.info(f"Tavily search completed in {search_time:.2f} seconds with {len(results)} results")
        
        # Format results for the LLM
        if not results:
            return "No results found."
        
        formatted_results = "Search results for '" + query + "':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"[{i}] {result['title']}\n"
            formatted_results += f"URL: {result['url']}\n"
            formatted_results += f"Summary: {result['content']}\n\n"
            
            # Log a sample of results (first result only)
            if i == 1:
                logger.info(f"Search results (sample): {formatted_results[:500]}...")
        
        tool_execution_time = time.time() - tool_start_time
        logger.info(f"Tavily search tool completed in {tool_execution_time:.2f} seconds")
        return formatted_results
        
    except ImportError:
        logger.error("Tavily package not installed. Install with: pip install tavily-python")
        return "Error: Search capability is not available. Tavily package not installed."
    except Exception as e:
        logger.error(f"Tavily search failed with error: {str(e)}")
        return f"Error: Tavily search failed with: {str(e)}"


def _duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for the specified query.
    
    Args:
        query (str): The search query string
        
    Returns:
        str: Search results as a formatted string
    """
    logger.info(f"Using DuckDuckGo search provider for query: '{query}'")
    tool_start_time = time.time()
    
    try:
        from duckduckgo_search import DDGS
        
        # Log the search start
        search_start = time.time()
        
        # Try different search backends with error handling
        results = []
        
        # First try the html backend
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=7))
            if results:
                logger.info(f"Successfully retrieved {len(results)} results from html backend")
        except Exception as e1:
            logger.warning(f"HTML backend failed with error: {str(e1)}")
            
            # Try lite backend as fallback
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=7, backend="lite"))
                if results:
                    logger.info(f"Successfully retrieved {len(results)} results from lite backend")
            except Exception as e2:
                logger.error(f"Lite backend failed with error: {str(e2)}")
        
        # Calculate and log search time
        search_time = time.time() - search_start
        logger.info(f"Web search completed in {search_time:.2f} seconds with {len(results)} results")
        
        # Format results for the LLM
        if not results:
            return "No results found."
        
        formatted_results = "Search results for '" + query + "':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"[{i}] {result['title']}\n"
            formatted_results += f"URL: {result['href']}\n"
            formatted_results += f"Summary: {result['body']}\n\n"
            
            # Log a sample of results (first result only)
            if i == 1:
                logger.info(f"Search results (sample): {formatted_results[:500]}...")
        
        tool_execution_time = time.time() - tool_start_time
        logger.info(f"Web search tool completed in {tool_execution_time:.2f} seconds")
        return formatted_results
    except ImportError:
        logger.error("DuckDuckGo search package not installed. Install with: pip install duckduckgo-search")
        return "Error: Search capability is not available. DuckDuckGo search package not installed."
    except Exception as e:
        logger.error(f"Web search failed with error: {str(e)}")
        return f"Error: Web search failed with: {str(e)}"


@tool
def make_request(url: str) -> str:
    """
    Make a request to a URL and return the response.
    
    Args:
        url: URL to request
        
    Returns:
        str: Response 
    """
    logger.info(f"Request tool called with URL: {url}")
    start_time = time.time()
    
    try:
        # Add timeout to prevent hanging on slow responses
        response = requests.get(url, timeout=10)
        elapsed = time.time() - start_time
        
        # Format the response based on status code
        valid = response.status_code == 200
        logger.info(f"Request to {url} completed with status code {response.status_code} ({elapsed:.2f}s)")
        
        # Get the content type and length
        content_type = response.headers.get('Content-Type', 'unknown')
        content_length = len(response.text)
        
        # Prepare a preview of the content (first 500 chars)
        content_preview = response.text[:500] if valid else ""
        
        # Return formatted response
        return f"""URL: {url}
Status code: {response.status_code}
Valid: {valid}
Content type: {content_type}
Content length: {content_length} bytes
Request time: {elapsed:.2f} seconds
Content preview:
{content_preview}"""
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error making request to {url}: {str(e)} ({elapsed:.2f}s)")
        return f"""URL: {url}
Status code: Error
Valid: False
Error: {str(e)}
Request time: {elapsed:.2f} seconds""" 