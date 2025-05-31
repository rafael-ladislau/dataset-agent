# Dataset Research API

A REST API for researching datasets using LLMs and web search, built with FastAPI and SQLite.

## Architecture

The API is organized with a clean architecture approach:

- **API Layer**: FastAPI endpoints for client interaction
- **Database Layer**: SQLite for persistent storage with thread-local connections
- **Authentication Layer**: API key-based authentication
- **Background Processing**: Asynchronous task execution

### File Structure

- `api.py`: FastAPI application and route definitions
- `server.py`: Server entry point and configuration
- `database.py`: SQLite database implementation
- `auth.py`: API key authentication
- `main.py`: Core dataset research functionality

## API Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/api/research` | POST | Submit research request | Required |
| `/api/research/{task_id}` | GET | Check task status | Required |
| `/api/research/{task_id}/result` | GET | Get research results | Required |
| `/api/research` | GET | List all research tasks | Required |
| `/api/health` | GET | Health check | Not required |

## Implementation Details

### Database

- SQLite database with thread-local connections for thread safety
- WAL journal mode for better concurrent performance
- Task and results tables with appropriate indexes
- Proper connection handling and resource cleanup

### Authentication

- API key validation from environment variables
- Support for multiple API keys
- Client identification based on API key

### Logging

- Rotating log files to prevent excessive disk usage
- Separate logs for general operations, errors, and API access
- Structured log format with timestamps and request context

### Background Processing

- Asynchronous task execution with FastAPI background tasks
- Task status tracking in the database
- Proper error handling and logging

## Setup and Usage

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r src/dataset_agent/requirements.txt
```

3. Create a `.env` file in the project root (copy from `.env.example`):

```
# Authentication
API_KEYS=key1,key2,key3

# Database
SQLITE_DB_PATH=./data/research.db

# Logging
LOG_PATH=./logs
LOG_LEVEL=INFO

# Other settings...
```

### Running the API Server

From the project root directory:

```bash
python -m src.dataset_agent.server
```

or from inside the src directory:

```bash
python -m dataset_agent.server
```

The API will be available at `http://localhost:8000/api`.

### API Documentation

Once the server is running, you can access the interactive API documentation:

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Testing the API

### Using curl

#### Submit a research request:

```bash
curl -X POST "http://localhost:8000/api/research" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"dataset_name": "NASS Census of Agriculture", "dataset_url": "https://www.nass.usda.gov/AgCensus/"}'
```

Response:
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "dataset_name": "NASS Census of Agriculture",
  "dataset_url": "https://www.nass.usda.gov/AgCensus/",
  "status": "pending",
  "created_at": "2023-07-21T15:22:47.123Z",
  "updated_at": "2023-07-21T15:22:47.123Z"
}
```

#### Check task status:

```bash
curl -X GET "http://localhost:8000/api/research/3fa85f64-5717-4562-b3fc-2c963f66afa6" \
  -H "X-API-Key: key1"
```

#### Get research results:

```bash
curl -X GET "http://localhost:8000/api/research/3fa85f64-5717-4562-b3fc-2c963f66afa6/result" \
  -H "X-API-Key: key1"
```

#### List all tasks:

```bash
curl -X GET "http://localhost:8000/api/research" \
  -H "X-API-Key: key1"
```

### Using Python Requests

Here's a simple Python script to interact with the API:

```python
import requests

API_URL = "http://localhost:8000/api"
API_KEY = "key1"
HEADERS = {"X-API-Key": API_KEY}

# Submit research request
response = requests.post(
    f"{API_URL}/research",
    headers=HEADERS,
    json={"dataset_name": "NASS Census of Agriculture"}
)
task = response.json()
task_id = task["id"]
print(f"Submitted task: {task_id}")

# Check status
status_response = requests.get(
    f"{API_URL}/research/{task_id}",
    headers=HEADERS
)
print(f"Status: {status_response.json()['status']}")

# Get results when completed
# Note: in real usage, you'd check the status first
result_response = requests.get(
    f"{API_URL}/research/{task_id}/result",
    headers=HEADERS
)
if result_response.status_code == 200:
    print(f"Result: {result_response.json()}")
else:
    print(f"Can't get results yet: {result_response.status_code}")
```

## Working with the SQLite Database

### Viewing the Database Structure

```bash
sqlite3 ./data/research.db .schema
```

### Querying Tasks

```bash
sqlite3 ./data/research.db "SELECT id, dataset_name, status FROM research_tasks"
```

### Viewing Research Results

```bash
sqlite3 ./data/research.db "SELECT r.task_id, t.dataset_name, r.description, r.access_type FROM research_results r JOIN research_tasks t ON r.task_id = t.id"
```

### Counting Tasks by Status

```bash
sqlite3 ./data/research.db "SELECT status, COUNT(*) FROM research_tasks GROUP BY status"
```

### Finding a Specific Task

```bash
sqlite3 ./data/research.db "SELECT * FROM research_tasks WHERE dataset_name LIKE '%NASS%'"
```

### Other Useful SQLite Commands

- List all tables:
  ```bash
  sqlite3 ./data/research.db .tables
  ```

- Export query results to CSV:
  ```bash
  sqlite3 -header -csv ./data/research.db "SELECT * FROM research_tasks" > tasks.csv
  ```

- Interactive SQLite session:
  ```bash
  sqlite3 ./data/research.db
  ```
  Then you can run commands like:
  ```
  .headers on
  .mode column
  SELECT * FROM research_tasks LIMIT 5;
  ```

## Troubleshooting

- **Database connection issues**: Ensure the directory for the database file exists
- **Authentication errors**: Check your API key in request headers and environment variables
- **Task stuck in 'processing'**: Check the logs for errors during task execution 