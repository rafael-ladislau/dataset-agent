"""
Database module for SQLite persistence.
"""

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Thread-local storage for database connections
_local = threading.local()

class Database:
    """SQLite database implementation for the dataset research agent."""
    
    def __init__(self, db_path: str):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory_exists()
        self._init_db()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the directory for the database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created database directory: {db_dir}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a thread-local database connection.
        
        Returns:
            sqlite3.Connection: SQLite connection
        """
        if not hasattr(_local, 'connection'):
            _local.connection = sqlite3.connect(self.db_path)
            # Enable foreign keys
            _local.connection.execute("PRAGMA foreign_keys = ON")
            # Configure connection for better performance
            _local.connection.execute("PRAGMA journal_mode = WAL")
            _local.connection.execute("PRAGMA synchronous = NORMAL")
            # Return rows as dictionaries
            _local.connection.row_factory = sqlite3.Row
        
        return _local.connection
    
    def close_connection(self) -> None:
        """Close the thread-local database connection if it exists."""
        if hasattr(_local, 'connection'):
            _local.connection.close()
            del _local.connection
    
    def _init_db(self) -> None:
        """Initialize the database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create research_tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_tasks (
            id TEXT PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            dataset_url TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            client_id TEXT NOT NULL
        )
        ''')
        
        # Create research_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_results (
            task_id TEXT PRIMARY KEY,
            description TEXT,
            aliases TEXT,
            organizations TEXT,
            access_type TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (task_id) REFERENCES research_tasks(id) ON DELETE CASCADE
        )
        ''')
        
        # Create indexes for efficient lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON research_tasks(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_client_id ON research_tasks(client_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON research_tasks(created_at)')
        
        conn.commit()
        logger.info("Database initialized successfully")
    
    def create_task(self, task_id: str, dataset_name: str, dataset_url: Optional[str], client_id: str) -> None:
        """
        Create a new research task.
        
        Args:
            task_id: Unique identifier for the task
            dataset_name: Name of the dataset to research
            dataset_url: Optional URL for the dataset
            client_id: Identifier for the client making the request
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute(
            '''
            INSERT INTO research_tasks 
            (id, dataset_name, dataset_url, status, created_at, updated_at, client_id) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (task_id, dataset_name, dataset_url, 'pending', now, now, client_id)
        )
        
        conn.commit()
        logger.info(f"Created research task: {task_id} for dataset: {dataset_name}")
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """
        Update the status of a research task.
        
        Args:
            task_id: Task identifier
            status: New status value
            
        Returns:
            bool: True if the task was updated, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute(
            '''
            UPDATE research_tasks 
            SET status = ?, updated_at = ? 
            WHERE id = ?
            ''',
            (status, now, task_id)
        )
        
        conn.commit()
        return cursor.rowcount > 0
    
    def store_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Store the research results.
        
        Args:
            task_id: Task identifier
            result: Research results
            
        Returns:
            bool: True if the result was stored, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        try:
            # Serialize list fields to JSON strings
            aliases = ','.join(result.get('aliases', []))
            organizations = ','.join(result.get('organizations', []))
            
            cursor.execute(
                '''
                INSERT INTO research_results 
                (task_id, description, aliases, organizations, access_type, created_at) 
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    task_id, 
                    result.get('description', ''),
                    aliases,
                    organizations,
                    result.get('access_type', ''),
                    now
                )
            )
            
            # Update task status to completed
            self.update_task_status(task_id, 'completed')
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing research result for {task_id}: {str(e)}")
            conn.rollback()
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task information by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Optional[Dict[str, Any]]: Task data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM research_tasks WHERE id = ?',
            (task_id,)
        )
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get research result by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Optional[Dict[str, Any]]: Result data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            SELECT r.*, t.dataset_name, t.dataset_url, t.status, t.created_at as task_created_at 
            FROM research_results r
            JOIN research_tasks t ON r.task_id = t.id
            WHERE r.task_id = ?
            ''',
            (task_id,)
        )
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Convert comma-separated strings back to lists
            result['aliases'] = result['aliases'].split(',') if result['aliases'] else []
            result['organizations'] = result['organizations'].split(',') if result['organizations'] else []
            return result
        return None
    
    def list_tasks(self, client_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """
        List research tasks with pagination.
        
        Args:
            client_id: Optional client ID to filter by
            limit: Maximum number of tasks to return
            offset: Pagination offset
            
        Returns:
            Tuple[List[Dict[str, Any]], int]: List of tasks and total count
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM research_tasks'
        count_query = 'SELECT COUNT(*) FROM research_tasks'
        
        params = []
        if client_id:
            query += ' WHERE client_id = ?'
            count_query += ' WHERE client_id = ?'
            params.append(client_id)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        # Get total count
        cursor.execute(count_query, params[:-2] if params else [])
        total_count = cursor.fetchone()[0]
        
        # Get paginated tasks
        cursor.execute(query, params)
        tasks = [dict(row) for row in cursor.fetchall()]
        
        return tasks, total_count 