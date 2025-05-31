"""
FastAPI application for dataset research agent API.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import APIKeyAuth
from .database import Database
from .main import run_research
from .domain.models import DatasetInfo

# Set up logger
logger = logging.getLogger(__name__)

# Models for request/response
class ResearchRequest(BaseModel):
    """Model for dataset research request."""
    dataset_name: str = Field(..., description="Name of the dataset to research")
    dataset_url: Optional[str] = Field(None, description="Optional URL for the dataset")

class ResearchTask(BaseModel):
    """Model for research task information."""
    id: str
    dataset_name: str
    dataset_url: Optional[str]
    status: str
    created_at: str
    updated_at: str

class ResearchResult(BaseModel):
    """Model for research result."""
    task_id: str
    dataset_name: str
    description: str
    aliases: List[str]
    organizations: List[str]
    access_type: str
    created_at: str

class TaskList(BaseModel):
    """Model for paginated task list."""
    tasks: List[ResearchTask]
    total: int
    limit: int
    offset: int

class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str

# Setup FastAPI app
app = FastAPI(
    title="Dataset Research API",
    description="API for dataset research agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dependencies
api_auth = APIKeyAuth()
db = Database(os.environ.get('SQLITE_DB_PATH', './data/research.db'))

def get_db():
    """Database dependency."""
    try:
        yield db
    finally:
        db.close_connection()

async def verify_api_key(x_api_key: str = Header(...)):
    """
    Dependency for API key verification.
    
    Args:
        x_api_key: API key from request header
        
    Returns:
        str: The API key if valid
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if not api_auth.is_valid_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

def process_research_task(task_id: str, dataset_name: str, dataset_url: Optional[str], client_id: str, db: Database):
    """
    Process a research task in the background.
    
    Args:
        task_id: Unique task identifier
        dataset_name: Name of the dataset to research
        dataset_url: Optional URL for the dataset
        client_id: Client identifier
        db: Database instance
    """
    logger.info(f"Starting background task {task_id} for client {client_id}")
    
    try:
        # Update task status to 'processing'
        db.update_task_status(task_id, 'processing')
        
        # Run the research process
        result = run_research(dataset_name, dataset_url)
        
        # Store the result
        db.store_result(task_id, result.to_dict())
        
        logger.info(f"Completed task {task_id} successfully")
    except Exception as e:
        # Update task status to 'failed'
        db.update_task_status(task_id, 'failed')
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)

@app.post(
    "/api/research", 
    response_model=ResearchTask, 
    status_code=202, 
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def create_research_task(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    db: Database = Depends(get_db)
):
    """
    Create a new dataset research task.
    
    Args:
        request: Research request details
        background_tasks: FastAPI background tasks
        api_key: Verified API key
        db: Database instance
        
    Returns:
        ResearchTask: Created task information
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    client_id = api_auth.get_client_id(api_key)
    
    try:
        # Create task record
        db.create_task(task_id, request.dataset_name, request.dataset_url, client_id)
        
        # Add task to background processing
        background_tasks.add_task(
            process_research_task,
            task_id,
            request.dataset_name,
            request.dataset_url,
            client_id,
            db
        )
        
        # Get the task data to return
        task = db.get_task(task_id)
        if task:
            return task
        
        raise HTTPException(
            status_code=500,
            detail="Failed to create research task"
        )
    except Exception as e:
        logger.error(f"Error creating research task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating research task: {str(e)}"
        )

@app.get(
    "/api/research/{task_id}", 
    response_model=ResearchTask,
    responses={404: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}
)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key),
    db: Database = Depends(get_db)
):
    """
    Get the status of a research task.
    
    Args:
        task_id: Task identifier
        api_key: Verified API key
        db: Database instance
        
    Returns:
        ResearchTask: Task information
    """
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return task

@app.get(
    "/api/research/{task_id}/result", 
    response_model=ResearchResult,
    responses={
        404: {"model": ErrorResponse}, 
        401: {"model": ErrorResponse},
        409: {"model": ErrorResponse}
    }
)
async def get_research_result(
    task_id: str,
    api_key: str = Depends(verify_api_key),
    db: Database = Depends(get_db)
):
    """
    Get research results for a task.
    
    Args:
        task_id: Task identifier
        api_key: Verified API key
        db: Database instance
        
    Returns:
        ResearchResult: Research results
    """
    # First check if the task exists
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    # Check if task is completed
    if task['status'] != 'completed':
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is not completed (current status: {task['status']})"
        )
    
    # Get the result
    result = db.get_result(task_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Result for task {task_id} not found"
        )
    
    return result

@app.get(
    "/api/research", 
    response_model=TaskList,
    responses={401: {"model": ErrorResponse}}
)
async def list_research_tasks(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key: str = Depends(verify_api_key),
    db: Database = Depends(get_db)
):
    """
    List research tasks with pagination.
    
    Args:
        limit: Maximum number of tasks to return
        offset: Pagination offset
        api_key: Verified API key
        db: Database instance
        
    Returns:
        TaskList: Paginated list of tasks
    """
    client_id = api_auth.get_client_id(api_key)
    tasks, total = db.list_tasks(client_id, limit, offset)
    
    return {
        "tasks": tasks,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()} 