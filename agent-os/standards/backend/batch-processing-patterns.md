## Batch Processing Patterns

### CSV Processing Standards

- **CSV Parsing**: Use Python's built-in `csv` module for parsing CSV files to handle quoted fields and commas correctly
- **Header Handling**: Skip header rows explicitly and process data rows with clear indexing
- **Field Extraction**: Extract fields by column index or name, with validation for expected structure
- **Error Isolation**: Process each row independently so that one failure doesn't stop the entire batch

### Bash Script Pattern for Batch Processing

```bash
#!/bin/bash

# ✅ Good: Comprehensive batch processing script
set -e  # Exit on error (but handle errors explicitly where needed)

# Color output for readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CSV_FILE="/path/to/datasets.csv"
VENV_PATH="./venv"
OUTPUT_DIR="./output"
LOG_FILE="batch_processing.log"

# Counters for summary
TOTAL_DATASETS=0
SUCCESSFUL=0
FAILED=0

# Pre-flight checks
echo "Checking prerequisites..."

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}Error: CSV file not found at $CSV_FILE${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

# Check if LMStudio is running (if using LMStudio)
if ! curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: LMStudio is not running on port 1234${NC}"
    echo "Please start LMStudio before continuing."
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Process CSV file
echo "Processing datasets from $CSV_FILE..."
echo "Results will be saved to $OUTPUT_DIR/"
echo ""

# Read CSV and process each line
tail -n +2 "$CSV_FILE" | while IFS=, read -r name description url; do
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    
    # Remove quotes from fields if present
    name=$(echo "$name" | sed 's/^"//;s/"$//')
    description=$(echo "$description" | sed 's/^"//;s/"$//')
    url=$(echo "$url" | sed 's/^"//;s/"$//')
    
    echo -e "${YELLOW}Processing dataset $TOTAL_DATASETS: $name${NC}"
    
    # Run the agent with error handling
    if python dataset_research.py "$name" \
        --url "$url" \
        --llm-provider lmstudio \
        --llm-model "gpt-oss-120b" \
        >> "$LOG_FILE" 2>&1; then
        
        echo -e "${GREEN}✓ Successfully processed: $name${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo -e "${RED}✗ Failed to process: $name${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Print summary
echo "========================================="
echo "Batch Processing Summary"
echo "========================================="
echo "Total datasets: $TOTAL_DATASETS"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "Log file: $LOG_FILE"
echo "========================================="
```

### Python Batch Processing Pattern

```python
# ✅ Good: Python-based batch processing with error handling
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class BatchResult:
    """Result of batch processing operation."""
    total: int
    successful: int
    failed: int
    errors: List[Dict[str, str]]

def process_csv_batch(
    csv_path: Path,
    processor_func: callable,
    skip_header: bool = True
) -> BatchResult:
    """
    Process datasets from CSV file in batch.
    
    Args:
        csv_path: Path to CSV file
        processor_func: Function to process each row
        skip_header: Whether to skip first row as header
    
    Returns:
        BatchResult with processing statistics
    """
    results = BatchResult(total=0, successful=0, failed=0, errors=[])
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f) if skip_header else csv.reader(f)
        
        for idx, row in enumerate(reader, start=1):
            results.total += 1
            
            try:
                # Process each row independently
                dataset_name = row.get('name') or row[0]
                dataset_url = row.get('url') or row[2]
                
                logging.info(f"Processing dataset {idx}: {dataset_name}")
                
                # Call processor function
                processor_func(dataset_name, dataset_url)
                
                results.successful += 1
                logging.info(f"✓ Successfully processed: {dataset_name}")
                
            except Exception as e:
                results.failed += 1
                error_info = {
                    'row': idx,
                    'dataset': dataset_name,
                    'error': str(e)
                }
                results.errors.append(error_info)
                logging.error(f"✗ Failed to process {dataset_name}: {str(e)}")
                
                # Continue processing remaining rows
                continue
    
    return results
```

### Logging Standards for Batch Operations

- **Structured Logging**: Use consistent log format with timestamps, severity, and context
- **Progress Indicators**: Log progress at regular intervals (e.g., every N items)
- **Error Details**: Log full error details including stack traces to log file
- **Summary Statistics**: Provide summary at the end with counts of successful/failed operations
- **Colored Console Output**: Use ANSI color codes for terminal output (green=success, red=error, yellow=warning)

### Error Handling in Batch Processing

- **Isolation**: Wrap each item's processing in try-except to prevent cascade failures
- **Error Collection**: Collect errors in a list for reporting at the end
- **Partial Success**: Allow batch to complete even if some items fail
- **Retry Logic**: Consider implementing retry with exponential backoff for transient failures
- **Validation Before Processing**: Validate all inputs before starting batch to fail fast on configuration errors

### Pre-flight Checks

Always verify prerequisites before starting batch processing:

1. **Input File Exists**: Check that CSV file exists and is readable
2. **Output Directory**: Ensure output directory exists or can be created
3. **Service Availability**: Check that required services (LMStudio, databases) are running
4. **Configuration**: Validate all required configuration is present
5. **Disk Space**: Verify sufficient disk space for output files
6. **Dependencies**: Ensure all required Python packages are installed

### Performance Considerations

- **Sequential vs Parallel**: For I/O-bound operations with rate limits, sequential processing is often sufficient
- **Memory Management**: Process large CSV files line-by-line, not loading entire file into memory
- **Resource Cleanup**: Ensure resources (file handles, connections) are properly closed after each item
- **Rate Limiting**: Implement delays between requests if calling external APIs
- **Checkpointing**: For very large batches, consider saving progress and supporting resume
