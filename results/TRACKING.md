# Results Tracking

This file documents each batch run of the dataset research agent, including the date, model used, datasets processed, and output location. Use it to track how the agent's output quality evolves over time.

> Results are stored as JSON files. Each file follows the naming convention `{dataset_slug}_research.json`.

---

## Batch History

### Batch 1 — 2025-08-09
- **Folder:** `results/`
- **Script:** Direct CLI invocations (pre-batch-script era)
- **LLM Provider:** Ollama (pre-LMStudio integration)
- **Model:** Unknown (early agent version)
- **Datasets (33):**
  - Academic Research and Development Survey
  - Agricultural Resource Management Survey (ARMS)
  - Burning Glass Technologies Data
  - Business R&D and Innovation Survey
  - Census of Agriculture
  - Current Population Survey Food Security Supplement
  - Early Career Doctorates Survey
  - Farm to School Census
  - FFRDC Research and Development Survey
  - Federal Science and Engineering Support to Universities Survey
  - Food Access Research Atlas
  - Food Acquisition and Purchase Survey
  - Higher Education R&D Survey
  - Household Food Security Survey Module
  - Information Resources Inc (IRI) InfoScan
  - Local Food Marketing Practices Survey
  - National Science Foundation Annual Business Survey
  - National Science Foundation Survey of Earned Doctorates
  - National Survey of Recent College Graduates
  - Nonprofit Research Activities Survey
  - NSF Science Engineering Indicators
  - Quarterly Food-at-Home Price Database
  - RUCC
  - Science Engineering Indicators
  - Scientists and Engineers Statistical Data System (SESTAT)
  - Survey of Doctorate Recipients
  - Survey of Federal Funds for Research and Development
  - Survey of Graduate Students and Postdoctorates in Science and Engineering
  - Survey of Science and Engineering Research Facilities
  - Survey of State Government Research and Development
  - Tenure Ownership and Transition of Agricultural Land
  - Transition of Agricultural Land Survey
  - Women, Minorities, and Persons with Disabilities

---

### Batch 2 — 2025-08-09 (same day, separate run)
- **Folder:** `workforce/`
- **Script:** Direct CLI invocations
- **LLM Provider:** Ollama (pre-LMStudio integration)
- **Model:** Unknown (early agent version)
- **Note:** Workforce/labor market datasets batch — agent did **not** yet have `official_name` / `relationship_type` fields
- **Datasets (11):**
  - Burning Glass Technologies Data
  - Lightcast Data
  - Longitudinal Employer-Household Dynamics Data (LEHD)
  - Multipurpose Occupational Systems Analysis Inventory Data (O\*NET predecessor)
  - National Labor Exchange Data (NLx)
  - Occupational Information Network Data (O\*NET)
  - Occupational Requirements Survey (ORS)
  - Program for the International Assessment of Adult Competencies (PIAAC)
  - Revelio Labs Data
  - State Longitudinal Data Systems Data (SLDS)
  - Unemployment Insurance Wage Record Data

---

### Batch 3 — 2025-08-21
- **Folder:** `results/`
- **Script:** Manual re-run / correction
- **LLM Provider:** Ollama
- **Model:** Unknown
- **Datasets (1):**
  - NASS Census of Agriculture *(re-run / correction)*

---

### Batch 4 — 2025-10-11 (Asthma batch)
- **Folder:** `results/`
- **Script:** `scripts/run_asthma_datasets.sh`
- **LLM Provider:** LMStudio
- **Model:** `openai/gpt-oss-120b`
- **Note:** First batch using LMStudio; includes `official_name` and `relationship_type` fields
- **Datasets (9):**
  - Asthma Emergency Department Visits and Hospitalizations Among Children and Adults
  - BRFSS Asthma Call-back Survey
  - Environmental Quality Index (EQI)
  - HCUP National/Nationwide Emergency Department Sample (NEDS)
  - HCUP National/Nationwide Inpatient Sample (NIS)
  - National Health and Nutrition Examination Survey (NHANES)
  - National Health Interview Survey (NHIS)
  - National Survey of Children's Health (NSCH)
  - Number and Rate of Hospital Inpatient Stays (Asthma, 2020)

---

### Batch 5 — 2025-11-14 (Pediatric / Clinical Research batch)
- **Folder:** `results/`
- **Script:** `scripts/run_pediatric_datasets.sh`
- **LLM Provider:** LMStudio
- **Model:** `openai/gpt-oss-120b`
- **Datasets (13):**
  - 1000 Genomes Project Phase 1 EUR
  - Community Wellbeing Index (CWB)
  - GALA II (Genes, Environments, Admixture in Latino Americans)
  - GTEx (Genotype-Tissue Expression Database)
  - Medical Expenditure Panel Survey (MEPS 1996)
  - NHANES III
  - Norwegian Mother and Child Cohort Study (MoBa)
  - Norwegian Prescription Database
  - Nurses' Health Study II
  - Project Viva Cohort
  - SAGE II (Study of African Americans, Asthma, Genes & Environments)
  - UK CPRD Gold (Clinical Practice Research Datalink)
  - World Health Organization Mortality Database

---

### Batch 6 — 2025-11-20
- **Folder:** `results/`
- **Script:** Manual / partial run
- **LLM Provider:** LMStudio
- **Model:** `openai/gpt-oss-120b`
- **Datasets (2):**
  - Nutrition Data Systems for Research (NDSR)
  - Pregnancy Risk Assessment Monitoring System (PRAMS)

---

### Batch 7 — 2025-11-21 (Social Economic / Health Welfare batch)
- **Folder:** `results/`
- **Script:** `scripts/run_social_economic_datasets.sh` / `scripts/run_health_welfare_datasets.sh`
- **LLM Provider:** LMStudio
- **Model:** `openai/gpt-oss-120b`
- **Datasets (11):**
  - American Community Survey (ACS)
  - Canadian Community Health Survey (CCHS)
  - Current Population Survey (CPS)
  - Food and Nutrient Database for Dietary Studies (FNDDS)
  - Food Patterns Equivalent Database (FPED)
  - Integrated Public Use Microdata Series CPS (IPUMS CPS)
  - National Longitudinal Surveys (NLS)
  - Occupational Employment and Wage Statistics (OEWS)
  - Panel Study of Income Dynamics (PSID)
  - Social Vulnerability Index (SVI)
  - University of Michigan Health and Retirement Study (HRS)

---

### Batch 8 — 2025-12-23 (Employment batch)
- **Folder:** `output/`
- **Script:** `scripts/run_employment_datasets.sh`
- **LLM Provider:** LMStudio
- **Model:** `openai/gpt-oss-120b`
- **Datasets (12):**
  - American Time Use Survey (ATUS)
  - Current Population Survey (CPS)
  - Dingel and Neiman Remote Work Feasibility Classification
  - FRBNY Consumer Credit Panel / Equifax CCP
  - Health and Retirement Study (HRS)
  - Longitudinal Business Database (LBD)
  - National Longitudinal Survey of Youth (NLSY)
  - Panel Study of Income Dynamics (PSID)
  - RAIS — Relação Anual de Informações Sociais (Brazilian employer-employee dataset)
  - Swedish Register Data / Swedish Administrative Registers
  - US Decennial Census
  - USPTO Patent Data

---

### Batch 9 — 2026-01-26
- **Folder:** `results/`
- **Script:** Manual / single dataset run
- **LLM Provider:** LMStudio
- **Model:** Unknown
- **Datasets (1):**
  - MIDRC Data Commons *(re-run / update)*

---

## Summary Table

| Batch | Date | Folder | # Datasets | LLM Provider | Model |
|-------|------|--------|-----------|--------------|-------|
| 1 | 2025-08-09 | results/ | 33 | Ollama | Unknown |
| 2 | 2025-08-09 | workforce/ | 11 | Ollama | Unknown |
| 3 | 2025-08-21 | results/ | 1 | Ollama | Unknown |
| 4 | 2025-10-11 | results/ | 9 | LMStudio | gpt-oss-120b |
| 5 | 2025-11-14 | results/ | 13 | LMStudio | gpt-oss-120b |
| 6 | 2025-11-20 | results/ | 2 | LMStudio | gpt-oss-120b |
| 7 | 2025-11-21 | results/ | 11 | LMStudio | gpt-oss-120b |
| 8 | 2025-12-23 | output/ | 12 | LMStudio | gpt-oss-120b |
| 9 | 2026-01-26 | results/ | 1 | LMStudio | Unknown |
| **Total** | | | **93** | | |

---

## Notes on Agent Evolution

- **Batches 1–3** (pre-Oct 2025): Agent lacked `official_name`, `relationship_type`, and `official_name_reasoning` fields. Results in `workforce/` reflect this older schema.
- **Batch 4+** (Oct 2025+): LMStudio integration enabled local inference. New schema fields added for official name detection.
- **Future runs:** Update this file after each batch, recording model, provider, date, and dataset count.
