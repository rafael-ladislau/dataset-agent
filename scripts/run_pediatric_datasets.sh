#!/bin/bash

################################################################################
# Script: run_pediatric_datasets.sh
# Purpose: Run dataset research agent for pediatric datasets from CSV file
# Usage: ./run_pediatric_datasets.sh
################################################################################

# Configuration
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/dataset_research_batch.log"
OUTPUT_DIR="$SCRIPT_DIR/../results"

# CSV file path
CSV_FILE="/Volumes/LadislauHD/Developer/digitalScience/discovered_dataset_pediatric_with_references.csv"

# LMStudio Configuration
LLM_PROVIDER="lmstudio"
LLM_MODEL="mlx-community/gpt-oss-120b"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

run_research() {
    local dataset_name="$1"
    local dataset_description="$2"
    local dataset_num="$3"
    
    log "=========================================="
    log "Processing Dataset #$dataset_num: $dataset_name"
    log "Description: ${dataset_description:0:100}..."
    log "=========================================="
    
    # Create a temporary prompt that includes the description as context
    # We'll pass this to the agent which will use it during research
    local full_context="$dataset_name

Context from publication: $dataset_description"
    
    # Run the research with the dataset name
    # The description will be discovered by the agent through web search
    $VENV_PYTHON -m src.dataset_agent.main "$dataset_name" \
        --llm-provider "$LLM_PROVIDER" \
        --llm-model "$LLM_MODEL" \
        --output-dir "$OUTPUT_DIR"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        success "Dataset #$dataset_num completed successfully"
        echo ""
        return 0
    else
        error "Dataset #$dataset_num failed with exit code $exit_code"
        echo ""
        return 1
    fi
}

parse_csv_line() {
    local line="$1"
    local IFS=','
    local fields=()
    local field=""
    local in_quotes=false
    local i=0
    
    # Parse CSV line handling quoted fields with commas
    while IFS= read -r -n1 char; do
        if [ "$char" = '"' ]; then
            if [ "$in_quotes" = true ]; then
                in_quotes=false
            else
                in_quotes=true
            fi
        elif [ "$char" = ',' ] && [ "$in_quotes" = false ]; then
            fields+=("$field")
            field=""
        else
            field="${field}${char}"
        fi
    done <<< "$line"
    
    # Add last field
    fields+=("$field")
    
    # Return fields as space-separated values
    # Fields: name, num_references, score, description, domain
    echo "${fields[@]}"
}

################################################################################
# Main Script
################################################################################

# Change to script directory
cd "$SCRIPT_DIR/.." || exit 1

# Initialize log file
echo "Dataset Research Batch Process - $(date)" > "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    error "CSV file not found at $CSV_FILE"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    error "Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

# Check if LMStudio is running
log "Checking LMStudio connection..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:1234/v1/models | grep -q "200"; then
    success "LMStudio is running and accessible"
else
    warning "LMStudio may not be running. Please ensure it's started on port 1234"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

log "Starting batch processing of datasets from CSV..."
log "CSV file: $CSV_FILE"
echo ""

# Track statistics
TOTAL_DATASETS=0
SUCCESSFUL=0
FAILED=0

# Read CSV file and process each dataset
# Skip header (line 1) and process lines 2-14
line_num=0
while IFS= read -r line; do
    ((line_num++))
    
    # Skip header
    if [ $line_num -eq 1 ]; then
        continue
    fi
    
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Stop after line 14 (13 datasets after header)
    if [ $line_num -gt 14 ]; then
        break
    fi
    
    # Parse CSV line using Python for proper CSV handling
    dataset_info=$(python3 -c "
import csv
import sys
line = '''$line'''
reader = csv.reader([line])
for row in reader:
    if len(row) >= 4:
        name = row[0].strip()
        description = row[3].strip()
        print(f'{name}|||{description}')
    else:
        print('|||')
")
    
    # Extract name and description
    dataset_name=$(echo "$dataset_info" | cut -d'|' -f1)
    dataset_description=$(echo "$dataset_info" | cut -d'|' -f4)
    
    # Skip if name is empty
    if [ -z "$dataset_name" ]; then
        continue
    fi
    
    ((TOTAL_DATASETS++))
    
    # Run research for this dataset
    if run_research "$dataset_name" "$dataset_description" "$TOTAL_DATASETS"; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
    fi
    
done < "$CSV_FILE"

################################################################################
# Summary
################################################################################

log "=========================================="
log "BATCH PROCESSING COMPLETE"
log "=========================================="
log "Total datasets processed: $TOTAL_DATASETS"
success "Successful: $SUCCESSFUL"
if [ $FAILED -gt 0 ]; then
    error "Failed: $FAILED"
else
    log "Failed: 0"
fi
log "=========================================="
log "Results saved to: $OUTPUT_DIR"
log "Full log saved to: $LOG_FILE"
log "=========================================="

# Exit with appropriate code
if [ $FAILED -eq 0 ]; then
    success "All datasets processed successfully!"
    exit 0
else
    warning "Some datasets failed. Check the log for details."
    exit 1
fi


