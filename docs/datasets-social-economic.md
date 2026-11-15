# Social, Economic, and Health Datasets Information

This document contains information about the 15 datasets configured in `run_social_economic_datasets.sh`.

## Dataset Details

### 1. Pregnancy Risk Assessment Monitoring System (PRAMS)
- **Alias**: PRAMS
- **URL**: https://www.cdc.gov/prams/index.html
- **Organization**: CDC, Division of Reproductive Health
- **Description**: A joint surveillance project collecting state-specific, population-based data on maternal behaviors before, during, and after pregnancy to reduce infant morbidity and mortality.

### 2. National Health and Nutrition Examination Survey (NHANES)
- **Alias**: NHANES
- **URL**: https://www.cdc.gov/nchs/nhanes/index.html
- **Organization**: CDC, National Center for Health Statistics (NCHS)
- **Description**: A survey designed to assess the health and nutritional status of adults and children in the United States, combining interviews and physical examinations.

### 3. Social Vulnerability Index (SVI)
- **Alias**: SVI, CDC/ATSDR SVI
- **URL**: https://www.atsdr.cdc.gov/place-health/php/svi/index.html
- **Organization**: CDC/ATSDR
- **Description**: A place-based index, database, and mapping application designed to identify and quantify communities experiencing social vulnerability using U.S. Census data.

### 4. Integrated Public Use Microdata Series Current Population Survey (IPUMS CPS)
- **Alias**: IPUMS CPS, IPUMS-CPS
- **URL**: https://cps.ipums.org/cps/
- **Organization**: University of Minnesota
- **Description**: Harmonized microdata from the monthly U.S. labor force survey (CPS), covering 1962 to present, including demographic information and rich employment data.

### 5. University of Michigan Health and Retirement Study (HRS)
- **Alias**: HRS
- **URL**: https://hrs.isr.umich.edu/about
- **Organization**: University of Michigan, NIA
- **Description**: A longitudinal panel study surveying a representative sample of more than 20,000 Americans over age 50 every two years to study aging, retirement, and health.

### 6. Canadian Community Health Survey (CCHS)
- **Alias**: CCHS
- **URL**: https://www.statcan.gc.ca/en/survey/household/3226
- **Organization**: Statistics Canada
- **Description**: Collects health-related data at the community level in Canada to support local health units by providing information for program evaluation and design.

### 7. Occupational Employment and Wage Statistics (OEWS)
- **Alias**: OEWS, OES
- **URL**: https://www.bls.gov/oes/
- **Organization**: U.S. Bureau of Labor Statistics (BLS)
- **Description**: A semiannual survey producing employment and wage estimates for approximately 830 occupations based on surveys of business establishments.

### 8. National Longitudinal Surveys (NLS)
- **Alias**: NLS
- **URL**: https://www.bls.gov/nls/
- **Organization**: U.S. Bureau of Labor Statistics (BLS)
- **Description**: A set of surveys gathering information at multiple points in time on labor market activities and significant life events of several groups of men and women.

### 9. Current Population Survey (CPS)
- **Alias**: CPS
- **URL**: https://www.census.gov/programs-surveys/cps.html
- **Organization**: U.S. Census Bureau, U.S. Bureau of Labor Statistics
- **Description**: A monthly survey of about 60,000 U.S. households, serving as the primary source of labor force statistics for the U.S. population.

### 10. American Community Survey (ACS)
- **Alias**: ACS
- **URL**: https://www.census.gov/programs-surveys/acs.html
- **Organization**: U.S. Census Bureau
- **Description**: The premier source of detailed information about the nation's people and housing, collecting detailed social, economic, housing, and demographic information from a sample of households annually.

### 11. Food and Nutrient Database for Dietary Studies (FNDDS)
- **Alias**: FNDDS
- **URL**: https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds/
- **Organization**: USDA, Agricultural Research Service
- **Description**: An application database created for analyzing dietary intakes from What We Eat In America (WWEIA), NHANES, converting foods and beverages into gram amounts and nutrient values.

### 12. Food Patterns Equivalent Database (FPED)
- **Alias**: FPED
- **URL**: https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fped-overview/
- **Organization**: USDA, Agricultural Research Service
- **Description**: Converts foods and beverages in FNDDS to 37 USDA Food Patterns components, serving as a research tool to evaluate food and beverage intakes with respect to Dietary Guidelines recommendations.

### 13. Occupational Information Network (O*NET)
- **Alias**: O*NET, ONET
- **URL**: https://www.onetcenter.org/overview.html
- **Organization**: U.S. Department of Labor
- **Description**: The nation's primary source of occupational information, containing hundreds of job definitions with detailed descriptions of the world-of-work for use by job seekers, workforce development professionals, and researchers.

### 14. Panel Study of Income Dynamics (PSID)
- **Alias**: PSID
- **URL**: https://psidonline.isr.umich.edu/
- **Organization**: University of Michigan, Survey Research Center
- **Description**: A longitudinal panel survey of American families measuring economic, social, and health factors over the life course of families across multiple generations.

### 15. Nutrition Data Systems for Research (NDSR)
- **Alias**: NDSR
- **URL**: https://www.ncc.umn.edu/
- **Organization**: University of Minnesota, Nutrition Coordinating Center
- **Description**: A Windows-based dietary analysis software widely used for collection and coding of 24-hour dietary recalls and analysis of food records, menus, and recipes, with a comprehensive research-quality food and nutrient database.

## Usage

To run the research agent for all datasets:

```bash
./run_social_economic_datasets.sh
```

Make sure:
1. The virtual environment is set up at `./venv/`
2. LMStudio is running on port 1234 with a model loaded
3. All dependencies are installed

## Output

Results will be saved to the `output/` directory as JSON files, with one file per dataset containing:
- Dataset name and home URL
- Comprehensive description
- Aliases
- Organizations
- Access type
- Data, schema, and documentation URLs
- Processing metadata
