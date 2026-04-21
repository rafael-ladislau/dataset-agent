"""User prompt templates (LOGICA_DO_PROJETO.md)."""

from __future__ import annotations


def description_and_home_url_prompt(dataset_name: str, dataset_url: str | None) -> str:
    """Combined prompt: discover description AND official home URL in one call."""
    url = dataset_url if dataset_url else "None"
    return f"""Research the dataset '{dataset_name}' and provide TWO outputs.

Reference URL (if provided): {url}

Use web_search to find the official website or repository for this dataset.

===EXAMPLE OUTPUT===
For "Current Population Survey":

===DESCRIPTION===
The Current Population Survey (CPS) is a monthly survey of approximately 60,000 households conducted by the U.S. Census Bureau for the Bureau of Labor Statistics. It serves as the primary source of labor force statistics for the United States population, providing data on employment, unemployment, earnings, and hours of work. The survey covers the civilian noninstitutional population aged 16 and older. Key metrics derived from CPS include the national unemployment rate and labor force participation rate. Data collection occurs during the calendar week containing the 12th of each month. The survey has been conducted since 1940 and includes supplemental questions on topics such as income, health insurance, and school enrollment.
===HOME_URL===
https://www.census.gov/programs-surveys/cps.html

===YOUR TASK===
Now do the same for '{dataset_name}':

===DESCRIPTION===
<150-200 words: what it contains, who maintains it, purpose, typical uses. Factual style, no first person, start directly with content.>
===HOME_URL===
<Official URL or "None">"""


def organizations_prompt(dataset_name: str, description: str, home_url: str | None) -> str:
    """Find organizations related to the dataset."""
    url_hint = f"Official site: {home_url}" if home_url else "No official URL known."
    return f"""Find organizations related to the dataset '{dataset_name}'.

Description: {description}
{url_hint}

===EXAMPLE===
For "Current Population Survey", the correct output is:
["Census Bureau", "U.S. Census Bureau", "Bureau of Labor Statistics", "BLS", "Department of Commerce", "Department of Labor"]

For "NASS Census of Agriculture":
["USDA", "United States Department of Agriculture", "NASS", "National Agricultural Statistics Service"]

===RULES===
- Return ONLY organization names, no descriptions
- Include both full names AND acronyms as separate entries
- Split "Name (ACRONYM)" into two entries: "Name" and "ACRONYM"

Use web_search: "{dataset_name} dataset creator publisher organization"

===YOUR TASK===
Return a Python list for '{dataset_name}':
["Organization Name", "ACRONYM", "Another Organization"]"""


def aliases_prompt(
    dataset_name: str,
    description: str,
    home_url: str | None,
    organizations: list[str] | None = None,
) -> str:
    """Find aliases/citation forms for the dataset."""
    url_hint = f"Official site: {home_url}" if home_url else ""
    orgs_hint = f"Known organizations: {', '.join(organizations[:5])}" if organizations else ""
    context = "\n".join(filter(None, [url_hint, orgs_hint]))

    return f"""Find alternative names and acronyms for the dataset '{dataset_name}'.

Description: {description}
{context}

===EXAMPLE===
For "Occupational Requirements Survey", the correct output is:
["Occupational Requirements Survey", "ORS", "BLS Occupational Requirements Survey", "BLS ORS"]

For "Current Population Survey":
["Current Population Survey", "CPS", "CPS Basic Monthly", "CPS-ASEC", "Annual Social and Economic Supplement"]

For "American Community Survey":
["American Community Survey", "ACS", "ACS 1-Year", "ACS 5-Year", "Census ACS"]

===RULES===
- Return ONLY simple names/acronyms, one per entry
- Include the original name "{dataset_name}"
- Include common abbreviations and variations
- Include DOI URL if one exists (just the URL)
- DO NOT include citation formats (no "APA:", "MLA:", "Retrieved from")
- DO NOT include explanatory notes or commentary

Use web_search: "{dataset_name} abbreviation acronym alternative name"

===YOUR TASK===
Return a Python list for '{dataset_name}':
["Full Name", "ACRONYM", "Alternative Name"]"""


def refine_dataset_aliases_prompt(
    main_dataset_name: str,
    description: str,
    dataset_names: list[str],
    flag_terms: list[str],
) -> str:
    """Ask LLM to return dataset_names for literature search without embedding org/sponsor tokens."""
    names_lines = "\n".join(f"  - {n!r}" for n in dataset_names[:20])
    flags_lines = "\n".join(f"  - {f!r}" for f in flag_terms[:20])
    if not names_lines:
        names_lines = "  (none)"
    if not flags_lines:
        flags_lines = "  (none)"

    return f"""You are cleaning **dataset name aliases** for academic literature string search.

Primary dataset title: {main_dataset_name!r}

=== DESCRIPTION (context) ===
{description[:2000]}

=== CURRENT dataset_names (may wrongly mix sponsors with product names) ===
{names_lines}

=== flag_terms (organizations — already used separately in search; do NOT repeat here) ===
{flags_lines}

=== YOUR TASK ===
Return **only** a JSON object with one key:
{{"dataset_names": [<strings>]}}

Rules:
- Each string must name the **dataset/product** (official title, acronym, or spelling variant), suitable to appear in paper titles/abstracts.
- **Remove** sponsor/agency prefixes/suffixes from aliases when the canonical product name does not include them (e.g. "Bls O*net" → prefer "O*NET" and add "ONET" if useful; do not keep "Bls" in dataset_names because BLS is already in flag_terms).
- **Do not** include any phrase that duplicates an entry in flag_terms (case-insensitive).
- Include useful **spelling variants** when justified (e.g. O*NET vs ONET).
- Omit junk; max ~15 items; empty list only if nothing remains after cleaning.

Respond with ONLY valid JSON, no markdown fences."""


def urls_and_access_prompt(
    dataset_name: str,
    description: str,
    home_url: str | None,
) -> str:
    """Combined prompt: find data/schema/documentation URLs AND determine access type."""
    url_hint = f"Official site (start here): {home_url}" if home_url else "No official URL known yet."
    return f"""For the dataset '{dataset_name}', find specific URLs and determine access type.

Description: {description}
{url_hint}

===EXAMPLE OUTPUT===
For "Current Population Survey" with home_url "https://www.census.gov/programs-surveys/cps.html":

===DATA_URL===
https://www.census.gov/programs-surveys/cps/data.html
===SCHEMA_URL===
https://www.census.gov/programs-surveys/cps/technical-documentation/codebooks.html
===DOCUMENTATION_URL===
https://www.census.gov/programs-surveys/cps/technical-documentation.html
===ACCESS_TYPE===
Open

===TASKS===
1. Find URLs (validate with make_request):
   - data_url: where to download actual data files
   - schema_url: data dictionary, field definitions, codebook
   - documentation_url: user guides, methodology, technical docs

2. Determine access type:
   - Open: freely downloadable without login
   - Restricted: requires registration, approval, or payment
   - Unknown: cannot determine

===YOUR TASK===
Return for '{dataset_name}':

===DATA_URL===
<URL or "None">
===SCHEMA_URL===
<URL or "None">
===DOCUMENTATION_URL===
<URL or "None">
===ACCESS_TYPE===
<Open or Restricted or Unknown>"""


# Legacy prompts kept for backward compatibility
def access_type_prompt(dataset_name: str, description: str, dataset_url: str | None) -> str:
    """Standalone access type prompt (legacy, prefer urls_and_access_prompt)."""
    url = dataset_url if dataset_url else "None"
    return f"""Determine access type for '{dataset_name}': Open, Restricted, or Unknown.

Description: {description}
Reference URL: {url}

Examples:
- "Census Bureau surveys" -> Open (free public data)
- "NCHS Restricted Data" -> Restricted (requires application)
- "Proprietary commercial data" -> Restricted (requires payment)

Use web_search: "{dataset_name} dataset access download"

Return ONLY one word: Open, Restricted, or Unknown."""


URL_SPECS: dict[str, tuple[str, str]] = {
    "data": ("download data access", "for downloading the data"),
    "schema": ("schema data dictionary fields", "for the data dictionary or schema"),
    "documentation": ("documentation guide manual", "for documentation or guides"),
}


def typed_url_prompt(
    url_type: str,
    dataset_name: str,
    description: str,
    dataset_url: str | None,
) -> str:
    """Single URL prompt (legacy, prefer urls_and_access_prompt)."""
    suffix, type_description = URL_SPECS[url_type]
    url = dataset_url if dataset_url else "None"
    return f"""Find a URL {type_description} for '{dataset_name}'.

Reference URL: {url}

Search: "{dataset_name} {suffix}"
Validate with make_request (must return 200).

Return ONLY the URL, no explanation. If none found, return "None"."""
