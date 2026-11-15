#!/bin/bash

################################################################################
# Script: run_employment_datasets.sh
# Purpose: Run dataset research agent for employment and economic datasets
# Usage: 
#   ./run_employment_datasets.sh                    # Run all datasets
#   START_FROM=7 ./run_employment_datasets.sh       # Start from dataset 7
################################################################################

# Configuration
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/dataset_research_employment_batch.log"
OUTPUT_DIR="$SCRIPT_DIR/../results"

# Start from specific dataset (default: 1)
# Set via environment variable: START_FROM=7 ./run_employment_datasets.sh
START_FROM=${START_FROM:-1}

# LMStudio Configuration (from .env)
LLM_PROVIDER="lmstudio"
LLM_MODEL="openai/gpt-oss-120b"
# LLM_MODEL="glm-4.5-air-mlx"
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
    
    # Skip if dataset number is less than START_FROM
    if [ $dataset_num -lt $START_FROM ]; then
        warning "Skipping Dataset #$dataset_num: $dataset_name (START_FROM=$START_FROM)"
        return 2  # Return special code for skipped
    fi
    
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
            --output-dir "$OUTPUT_DIR" --log-file "$OUTPUT_DIR/${dataset_name}.log"
    else
        # URL provided
        $VENV_PYTHON -m src.dataset_agent.main "$dataset_name" \
            --url "$dataset_url" \
            --llm-provider "$LLM_PROVIDER" \
            --llm-model "$LLM_MODEL" \
            --output-dir "$OUTPUT_DIR" --log-file "$OUTPUT_DIR/${dataset_name}.log"
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
echo "Employment Dataset Research Batch Process - $(date)" > "$LOG_FILE"
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

log "Starting batch processing of employment datasets..."
if [ $START_FROM -gt 1 ]; then
    log "Starting from dataset #$START_FROM (skipping datasets 1-$((START_FROM-1)))"
fi
echo ""

# Track statistics
TOTAL_DATASETS=15
SUCCESSFUL=0
FAILED=0
SKIPPED=0

################################################################################
# Dataset 1: LEHD Origin-Destination Employment Statistics (LODES)
################################################################################
run_research \
    "LEHD Origin-Destination Employment Statistics (LODES)" \
    "https://lehd.ces.census.gov/data/" \
    "1"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 2: National Longitudinal Survey of Youth (NLSY)
################################################################################
run_research \
    "National Longitudinal Survey of Youth (NLSY)" \
    "https://www.bls.gov/nls/" \
    "2"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 3: U.S. Decennial Census
################################################################################
run_research \
    "U.S. Decennial Census" \
    "https://www.census.gov/programs-surveys/decennial-census.html" \
    "3"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 4: Current Population Survey (CPS)
################################################################################
run_research \
    "Current Population Survey (CPS)" \
    "https://www.census.gov/programs-surveys/cps.html" \
    "4"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 5: O*NET Online Database
################################################################################
run_research \
    "O*NET Online Database" \
    "https://www.onetcenter.org/database.html" \
    "5"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 6: RAIS - Brazilian Employer-Employee Dataset
################################################################################
run_research \
    "RAIS - Relacao Anual de Informacoes Sociais (Brazilian Employer-Employee Dataset)" \
    "https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/rais" \
    "6"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 7: American Time Use Survey (ATUS)
################################################################################
run_research \
    "American Time Use Survey (ATUS)" \
    "https://www.bls.gov/tus/" \
    "7"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 8: Longitudinal Business Database (LBD)
################################################################################
run_research \
    "Longitudinal Business Database (LBD)" \
    "https://www.census.gov/programs-surveys/ces/data/restricted-use-data/longitudinal-business-database.html" \
    "8"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 9: Health and Retirement Study (HRS)
################################################################################
run_research \
    "Health and Retirement Study (HRS)" \
    "https://hrs.isr.umich.edu/" \
    "9"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 10: Swedish Register Data
################################################################################
run_research \
    "Swedish Register Data (Swedish Administrative Registers)" \
    "https://www.scb.se/en/services/guidance-for-researchers-and-universities/" \
    "10"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 11: Dingel and Neiman Remote Work Classification
################################################################################
run_research \
    "Dingel and Neiman Remote Work Feasibility Classification" \
    "https://github.com/jdingel/DingelNeiman-workathome" \
    "11"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 12: Panel Study of Income Dynamics (PSID)
################################################################################
run_research \
    "Panel Study of Income Dynamics (PSID)" \
    "https://psidonline.isr.umich.edu/" \
    "12"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 13: USPTO Patent Data
################################################################################
run_research \
    "USPTO Patent Data" \
    "https://www.uspto.gov/ip-policy/economic-research/research-datasets" \
    "13"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 14: Revelio Labs Education File
################################################################################
run_research \
    "Revelio Labs Workforce Data - Education File" \
    "https://www.reveliolabs.com/" \
    "14"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi

################################################################################
# Dataset 15: FRBNY Consumer Credit Panel / Equifax
################################################################################
run_research \
    "FRBNY Consumer Credit Panel Equifax (CCP)" \
    "https://www.newyorkfed.org/microeconomics/hhdc" \
    "15"
exit_code=$?
if [ $exit_code -eq 0 ]; then
    ((SUCCESSFUL++))
elif [ $exit_code -eq 2 ]; then
    ((SKIPPED++))
else
    ((FAILED++))
fi


################################################################################
# Summary
################################################################################

log "=========================================="
log "BATCH PROCESSING COMPLETE"
log "=========================================="
log "Total datasets: $TOTAL_DATASETS"
if [ $SKIPPED -gt 0 ]; then
    warning "Skipped: $SKIPPED"
fi
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
