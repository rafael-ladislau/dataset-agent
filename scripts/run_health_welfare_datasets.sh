#!/bin/bash

################################################################################
# Script: run_health_welfare_datasets.sh
# Purpose: Run dataset research agent for health, welfare, and nutrition datasets
# Usage: 
#   ./run_health_welfare_datasets.sh                    # Run all datasets
#   START_FROM=7 ./run_health_welfare_datasets.sh       # Start from dataset 7
################################################################################

# Configuration
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/health_welfare_batch.log"
OUTPUT_DIR="$SCRIPT_DIR/../results"

# Start from specific dataset (default: 1)
# Set via environment variable: START_FROM=7 ./run_health_welfare_datasets.sh
START_FROM=${START_FROM:-1}

# LMStudio Configuration (from .env)
LLM_PROVIDER="lmstudio"
# LLM_MODEL="openai/gpt-oss-120b"
# LLM_MODEL="glm-4.5-air-mlx"
LLM_MODEL="gpt-oss-120b"
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
echo "Health & Welfare Dataset Research Batch Process - $(date)" > "$LOG_FILE"
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
TOTAL_DATASETS=12
SUCCESSFUL=0
FAILED=0
SKIPPED=0

################################################################################
# Dataset 1: National Health Interview Survey (NHIS)
# Domain: Public health / epidemiology
################################################################################
run_research \
    "National Health Interview Survey (NHIS)" \
    "https://www.cdc.gov/nchs/nhis/index.html" \
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
# Dataset 2: American Time Use Survey (ATUS)
# Domain: Time-use research
################################################################################
run_research \
    "American Time Use Survey (ATUS)" \
    "https://www.bls.gov/tus/" \
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
# Dataset 3: Behavioral Risk Factor Surveillance Survey Fruit and Vegetable Module (BRFSS FV module)
# Domain: Nutrition / Public Health
################################################################################
run_research \
    "Behavioral Risk Factor Surveillance Survey Fruit and Vegetable Module (BRFSS FV module)" \
    "https://www.cdc.gov/nutrition/data-statistics/data-users-guide.html" \
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
# Dataset 4: Food Environment Atlas (USDA ERS)
# Domain: Nutrition / Food Access
################################################################################
run_research \
    "Food Environment Atlas (USDA ERS)" \
    "https://www.ers.usda.gov/data-products/food-environment-atlas" \
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
# Dataset 5: Fragile Families and Child Wellbeing Study
# Domain: Social Science / Family Health
################################################################################
run_research \
    "Fragile Families and Child Wellbeing Study" \
    "https://www.icpsr.umich.edu/web/DSDR/studies/31622" \
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
# Dataset 6: Household Pulse Survey (Phase 1)
# Domain: Public health, economics, sociology
################################################################################
run_research \
    "Household Pulse Survey (Phase 1)" \
    "https://www.census.gov/programs-surveys/household-pulse-survey/data.html" \
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
# Dataset 7: USDA 18-item Household Food Security Survey Module (HFSSM)
# Domain: Nutrition / Food Security
################################################################################
run_research \
    "USDA 18-item Household Food Security Survey Module (HFSSM)" \
    "https://www.ers.usda.gov/topics/food-nutrition-assistance/food-security-in-the-us/survey-tools" \
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
# Dataset 8: RAND HRS Health Care and Nutrition Study (HCNS)
# Domain: Nutrition / Health Care
################################################################################
run_research \
    "RAND HRS Health Care and Nutrition Study (HCNS)" \
    "https://hrsdata.isr.umich.edu/data-products/2013-health-care-and-nutrition-study-hcns" \
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
# Dataset 9: University of Kentucky Center for Poverty Research National Welfare Data (UKCPR)
# Domain: Public policy / Welfare economics
################################################################################
run_research \
    "University of Kentucky Center for Poverty Research National Welfare Data (UKCPR)" \
    "https://ukcpr.uky.edu/resources/national-welfare-data" \
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
# Dataset 10: American College Health Association - National College Health Assessment III (ACHA-NCHA III)
# Domain: Higher Education Health / Public Health
################################################################################
run_research \
    "American College Health Association - National College Health Assessment III (ACHA-NCHA III)" \
    "https://www.acha.org/ncha/" \
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
# Dataset 11: Feeding America Map the Meal Gap (MMG)
# Domain: Nutrition / Food security
################################################################################
run_research \
    "Feeding America Map the Meal Gap (MMG)" \
    "https://map.feedingamerica.org/" \
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
# Dataset 12: RAND HRS Longitudinal Dataset
# Domain: Aging, Social Sciences
################################################################################
run_research \
    "RAND HRS Longitudinal Dataset" \
    "https://www.rand.org/well-being/social-and-behavioral-policy/portfolios/aging-longevity/dataprod.html" \
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
