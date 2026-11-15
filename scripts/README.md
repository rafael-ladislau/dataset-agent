# Scripts

Utility scripts for running the dataset research agent in batch mode and processing its output.

> **Note:** All scripts must be run from the **project root** directory (they handle this automatically via `$SCRIPT_DIR/..`). Results are saved to `results/` and logs to `logs/`.

---

## Batch Research Scripts

These scripts run the agent against a predefined list of datasets in a specific domain. Each calls `python -m src.dataset_agent.main` for every dataset and saves the JSON output to `results/`.

### Prerequisites
1. A virtual environment at `venv/` with dependencies installed (`pip install -r requirements.txt`)
2. An LLM provider running — by default, **LMStudio** on `localhost:1234` (see [docs/lmstudio-integration-guide.md](../docs/lmstudio-integration-guide.md))
3. Configure provider/model in the script header (`LLM_PROVIDER`, `LLM_MODEL`) or via `.env`

### `run_asthma_datasets.sh`
Runs research for **8 asthma-related datasets** (BRFSS, NHIS, hospitalizations data, etc.).

```bash
./scripts/run_asthma_datasets.sh
```

### `run_employment_datasets.sh`
Runs research for **employment and economic datasets** (~13 datasets: CPS, PSID, ACS, etc.).  
Supports resuming from a specific dataset number:

```bash
./scripts/run_employment_datasets.sh              # run all
START_FROM=7 ./scripts/run_employment_datasets.sh # resume from dataset 7
```

### `run_health_welfare_datasets.sh`
Runs research for **health, welfare, and nutrition datasets** (~13 datasets: PRAMS, NHANES, SVI, etc.).  
Supports `START_FROM`:

```bash
./scripts/run_health_welfare_datasets.sh
START_FROM=4 ./scripts/run_health_welfare_datasets.sh
```

### `run_pediatric_datasets.sh`
Runs research for **pediatric health datasets** (~8 datasets: NSCH, YRBS, Project Viva, etc.).

```bash
./scripts/run_pediatric_datasets.sh
```

### `run_social_economic_datasets.sh`
Runs research for **social and economic datasets** (~15 datasets: CPS, IPUMS, SVI, PSID, etc.).  
Supports `START_FROM`:

```bash
./scripts/run_social_economic_datasets.sh
START_FROM=10 ./scripts/run_social_economic_datasets.sh
```

---

## Output Processing Scripts

### `dataset_metadata_spreadsheet.py`
Converts a folder of research result JSON files into a multi-sheet Excel spreadsheet.

**Sheets produced:**
- `Main Data` — dataset name, URLs, description, access type
- `Aliases` — all known aliases per dataset
- `Organizations` — all organizations associated with each dataset

**Usage:**
```bash
python scripts/dataset_metadata_spreadsheet.py <input_folder> <output_file>

# Example: export all results to Excel
python scripts/dataset_metadata_spreadsheet.py results/ data/datasets_metadata.xlsx
```

**Requirements:** `pandas`, `openpyxl` (included in `requirements.txt`)

---

## Log Files
Batch run logs are written to `logs/` (gitignored). Check them after a run to review per-dataset success/failure details.
