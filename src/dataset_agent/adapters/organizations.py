"""Organization / flag_terms post-processing (US-centric acronym map from LOGICA)."""

from __future__ import annotations

import re

from dataset_agent.adapters.llm_output_cleanup import is_llm_list_entry_junk

ACRONYM_MAP: dict[str, list[str]] = {
    "USDA": ["United States Department of Agriculture", "U.S. Department of Agriculture"],
    "NASS": ["National Agricultural Statistics Service"],
    "NIH": ["National Institutes of Health"],
    "CDC": ["Centers for Disease Control and Prevention"],
    "NSF": ["National Science Foundation"],
    "NASA": ["National Aeronautics and Space Administration"],
    "NOAA": ["National Oceanic and Atmospheric Administration"],
    "NBER": ["National Bureau of Economic Research"],
    "EPA": ["Environmental Protection Agency"],
    "FDA": ["Food and Drug Administration"],
    "BLS": ["Bureau of Labor Statistics", "U.S. Bureau of Labor Statistics"],
    "SSA": ["Social Security Administration"],
    "DOL": ["Department of Labor", "U.S. Department of Labor"],
    "OSHA": ["Occupational Safety and Health Administration"],
    "NCHS": ["National Center for Health Statistics"],
    "CENSUS": ["U.S. Census Bureau", "Census Bureau"],
}


def _extract_clean_names(text: str) -> list[str]:
    """Extract clean organization names, handling numbered lists, quotes, and 'and' patterns."""
    t = text.strip()
    # Remove leading numbers like "1. " or "2) "
    t = re.sub(r'^\d+[\.\)]\s*', '', t)
    
    results: list[str] = []
    
    # Handle "Name and Acronym" patterns -> split into multiple entries
    # e.g., '"Bureau of Labor Statistics" and "BLS"' -> ['Bureau of Labor Statistics', 'BLS']
    if ' and "' in t.lower() or " and '" in t.lower():
        parts = re.split(r'\s+and\s+', t, flags=re.IGNORECASE)
        for p in parts:
            p = re.sub(r'^["\']|["\']$', '', p.strip())
            if p and len(p) > 1:
                results.append(p)
        return results
    
    # Remove surrounding quotes for single entry
    t = re.sub(r'^["\']|["\']$', '', t)
    if t and len(t) > 1:
        results.append(t)
    
    return results


def process_organizations(organizations: list[str]) -> list[str]:
    """Process organization list: clean, dedupe, expand known acronyms."""
    if not organizations:
        return []
    
    processed: set[str] = set()
    found_acronyms: set[str] = set()
    
    for org in organizations:
        # Extract clean names (may return multiple from "Name and Acronym" patterns)
        clean_names = _extract_clean_names(org)
        
        for clean_org in clean_names:
            if not clean_org:
                continue
            if is_llm_list_entry_junk(clean_org):
                continue
            
            # Skip very short entries that aren't known acronyms
            if len(clean_org) <= 2 and clean_org.upper() not in ACRONYM_MAP:
                continue
            
            processed.add(clean_org)
            
            # Handle "Name (ACRONYM)" pattern
            m = re.search(r"(.*?)\s*\(([A-Z]{2,})\)", clean_org)
            if m:
                full_name, acronym = m.group(1).strip(), m.group(2).strip()
                if full_name and len(full_name) > 2:
                    processed.add(full_name)
                processed.add(acronym)
                found_acronyms.add(acronym)
            
            # Check if this entry IS a known acronym
            upper = clean_org.upper()
            if upper in ACRONYM_MAP:
                found_acronyms.add(upper)
                processed.add(upper)
            
            # Check if entry contains a known acronym as a word
            words = set(w.upper() for w in clean_org.split())
            for acronym in ACRONYM_MAP:
                if acronym in words:
                    found_acronyms.add(acronym)
    
    # Only expand acronyms that were actually found in the input
    for acronym in found_acronyms:
        if acronym in ACRONYM_MAP:
            processed.update(ACRONYM_MAP[acronym])
            processed.add(acronym)
    
    # Final cleanup: remove any remaining junk
    result = [
        org for org in processed
        if org and len(org) > 1 and not is_llm_list_entry_junk(org)
    ]
    
    return sorted(set(result))
