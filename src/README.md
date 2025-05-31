# Dataset Research API

A REST API for researching datasets using LLMs and web search, built with FastAPI and SQLite.

## Quick Start

1. **Setup Environment**:
   - Create a `.env` file in the `src` directory using the provided template
   - Set your API keys in the file

```
# Copy and paste this to create your .env file
API_KEYS=your_key1,your_key2,your_key3
SQLITE_DB_PATH=./data/research.db
LOG_PATH=./logs
LOG_LEVEL=INFO
OLLAMA_HOST=http://localhost:11434
```

2. **Install Dependencies**:
```bash
pip install -r src/dataset_agent/requirements.txt
```

3. **Run the Server**:
```bash
# From project root
python -m src.dataset_agent.server

# OR from src directory
cd src
python -m dataset_agent.server
```

4. **Test the API**:
```bash
# Submit a research task
curl -X POST "http://localhost:8000/api/research" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key1" \
  -d '{"dataset_name": "NASS Census of Agriculture"}'

# Replace TASK_ID with the ID from the response
curl -X GET "http://localhost:8000/api/research/TASK_ID" \
  -H "X-API-Key: your_key1"

curl -X GET "http://localhost:8000/api/research/TASK_ID/result" \
  -H "X-API-Key: your_key1"
```

5. **Access Documentation**:
   - Swagger UI: `http://localhost:8000/api/docs`
   - ReDoc: `http://localhost:8000/api/redoc`

## Database Commands

Explore the SQLite database:

```bash
# Show database schema
sqlite3 ./data/research.db .schema

# List tasks
sqlite3 ./data/research.db "SELECT id, dataset_name, status FROM research_tasks"

# View results
sqlite3 ./data/research.db "SELECT task_id, description FROM research_results"

# Interactive session
sqlite3 ./data/research.db
```

Interactive SQLite commands:
```
.headers on
.mode column
SELECT * FROM research_tasks LIMIT 5;
.tables
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research` | POST | Submit research request |
| `/api/research/{task_id}` | GET | Check task status |
| `/api/research/{task_id}/result` | GET | Get research results |
| `/api/research` | GET | List all research tasks |
| `/api/health` | GET | Health check |

For more detailed information, see the [API Documentation](/src/dataset_agent/README.md). 