#!/bin/bash

################################################################################
# Script: run_asthma_datasets.sh
# Purpose: Run dataset research agent for multiple asthma-related datasets
# Usage: ./run_asthma_datasets.sh
################################################################################

# Configuration
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/dataset_research_batch.log"
OUTPUT_DIR="$SCRIPT_DIR/../results"

# LMStudio Configuration (from .env)
LLM_PROVIDER="lmstudio"
LLM_MODEL="openai/gpt-oss-120b"

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
    local dataset_url="$2"
    local dataset_num="$3"
    
    log "=========================================="
    log "Processing Dataset #$dataset_num: $dataset_name"
    log "URL: $dataset_url"
    log "=========================================="
    
    # Run the research
    if [ -z "$dataset_url" ]; then
        # No URL provided, run without URL
        $VENV_PYTHON -m src.dataset_agent.main "$dataset_name" \
            --llm-provider "$LLM_PROVIDER" \
            --llm-model "$LLM_MODEL" \
            --output-dir "$OUTPUT_DIR"
    else
        # URL provided
        $VENV_PYTHON -m src.dataset_agent.main "$dataset_name" \
            --url "$dataset_url" \
            --llm-provider "$LLM_PROVIDER" \
            --llm-model "$LLM_MODEL" \
            --output-dir "$OUTPUT_DIR"
    fi
    
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

################################################################################
# Main Script
################################################################################

# Change to script directory
cd "$SCRIPT_DIR/.." || exit 1

# Initialize log file
echo "Dataset Research Batch Process - $(date)" > "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    error "Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

# Check if LMStudio is running (optional check)
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

log "Starting batch processing of datasets..."
echo ""

# Track statistics
TOTAL_DATASETS=8
SUCCESSFUL=0
FAILED=0

################################################################################
# Dataset 1: BRFSS Asthma Call-back Survey
################################################################################
if run_research \
    "BRFSS Asthma Call-back Survey" \
    "https://www.cdc.gov/brfss/acbs/index.htm" \
    "1"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 2: National Health Interview Survey (NHIS)
################################################################################
if run_research \
    "National Health Interview Survey (NHIS)" \
    "https://www.cdc.gov/asthma/nhis/2021/data.htm" \
    "2"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 3: National Survey of Children's Health (NSCH)
################################################################################
if run_research \
    "National Survey of Children's Health (NSCH)" \
    "https://www.childhealthdata.org/" \
    "3"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 4: Asthma Emergency Department Visits and Hospitalizations
################################################################################
if run_research \
    "Asthma Emergency Department Visits and Hospitalizations Among Children and Adults" \
    "https://www.cdc.gov/asthma/healthcare-use/2020/table_a.html" \
    "4"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 5: Hospital Inpatient Stays with Asthma
################################################################################
if run_research \
    "Number and rate of hospital inpatient stays with asthma as the first-listed diagnosis per 10,000 population, by selected patient characteristics: United States, 2020" \
    "https://www.cdc.gov/asthma/healthcare-use/2020/table_b.html" \
    "5"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 6: NHANES
################################################################################
if run_research \
    "National Health and Nutrition Examination Survey (NHANES)" \
    "https://wwwn.cdc.gov/nchs/nhanes/default.aspx" \
    "6"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 7: HCUP NEDS
################################################################################
if run_research \
    "HCUP National (Nationwide) Emergency Department Sample (NEDS)" \
    "" \
    "7"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 8: HCUP NIS
################################################################################
if run_research \
    "HCUP National (Nationwide) Inpatient Sample (NIS)" \
    "" \
    "8"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

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

