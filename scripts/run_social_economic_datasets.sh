#!/bin/bash

################################################################################
# Script: run_social_economic_datasets.sh
# Purpose: Run dataset research agent for social, economic, and health datasets
# Usage: 
#   ./run_social_economic_datasets.sh                    # Run all datasets
#   START_FROM=7 ./run_social_economic_datasets.sh       # Start from dataset 7
################################################################################

# Configuration
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/dataset_research_batch.log"
OUTPUT_DIR="$SCRIPT_DIR/../results"

# Start from specific dataset (default: 1)
# Set via environment variable: START_FROM=7 ./run_social_economic_datasets.sh
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
if [ $START_FROM -gt 1 ]; then
    log "Starting from dataset #$START_FROM (skipping datasets 1-$((START_FROM-1)))"
fi
echo ""

# Track statistics
TOTAL_DATASETS=13
SUCCESSFUL=0
FAILED=0
SKIPPED=0

################################################################################
# Dataset 1: Pregnancy Risk Assessment Monitoring System (PRAMS)
################################################################################
run_research \
    "Pregnancy Risk Assessment Monitoring System (PRAMS)" \
    "https://www.cdc.gov/prams/index.html" \
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
# Dataset 2: Nutrition Data Systems for Research (NDSR)
################################################################################
run_research \
    "Nutrition Data Systems for Research (NDSR)" \
    "https://www.ncc.umn.edu/" \
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
# Dataset 3: Social Vulnerability Index (SVI)
################################################################################
run_research \
    "Social Vulnerability Index (SVI)" \
    "https://www.atsdr.cdc.gov/place-health/php/svi/index.html" \
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
# Dataset 4: Integrated Public Use Microdata Series Current Population Survey (IPUMS CPS)
################################################################################
run_research \
    "Integrated Public Use Microdata Series Current Population Survey (IPUMS CPS)" \
    "https://cps.ipums.org/cps/" \
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
# Dataset 5: University of Michigan Health and Retirement Study (HRS)
################################################################################
run_research \
    "University of Michigan Health and Retirement Study (HRS)" \
    "https://hrs.isr.umich.edu/about" \
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
# Dataset 6: Canadian Community Health Survey (CCHS)
################################################################################
run_research \
    "Canadian Community Health Survey (CCHS)" \
    "https://www.statcan.gc.ca/en/survey/household/3226" \
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
# Dataset 7: Occupational Employment and Wage Statistics (OEWS)
################################################################################
run_research \
    "Occupational Employment and Wage Statistics (OEWS)" \
    "https://www.bls.gov/oes/" \
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
# Dataset 8: National Longitudinal Surveys (NLS)
################################################################################
run_research \
    "National Longitudinal Surveys (NLS)" \
    "https://www.bls.gov/nls/" \
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
# Dataset 9: Current Population Survey (CPS)
################################################################################
run_research \
    "Current Population Survey (CPS)" \
    "https://www.census.gov/programs-surveys/cps.html" \
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
# Dataset 10: American Community Survey (ACS)
################################################################################
run_research \
    "American Community Survey (ACS)" \
    "https://www.census.gov/programs-surveys/acs.html" \
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
# Dataset 11: Food and Nutrient Database for Dietary Studies (FNDDS)
################################################################################
run_research \
    "Food and Nutrient Database for Dietary Studies (FNDDS)" \
    "https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds/" \
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
# Dataset 12: Food Patterns Equivalent Database (FPED)
################################################################################
run_research \
    "Food Patterns Equivalent Database (FPED)" \
    "https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fped-overview/" \
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
# Dataset 13: Panel Study of Income Dynamics (PSID)
################################################################################
run_research \
    "Panel Study of Income Dynamics (PSID)" \
    "https://psidonline.isr.umich.edu/" \
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
