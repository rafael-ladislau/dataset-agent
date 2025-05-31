"""
Use cases for the dataset agent.

This module defines the interfaces and abstract base classes for the core business logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .models import DatasetInfo


class AgentInterface(ABC):
    """Base interface for agent implementations."""
    
    @abstractmethod
    def get_information(self, prompt: str) -> str:
        """
        Get information from the agent.
        
        Args:
            prompt: The input prompt for the agent
            
        Returns:
            str: The response from the agent
        """
        pass


class TextExtractor(ABC):
    """Interface for text extraction operations."""
    
    @abstractmethod
    def extract_list(self, text: str) -> List[str]:
        """
        Extract a list from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List[str]: Extracted list
        """
        pass
    
    @abstractmethod
    def extract_url(self, text: str) -> Optional[str]:
        """
        Extract a URL from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Optional[str]: Extracted URL or None
        """
        pass


class DatasetInfoRepository(ABC):
    """Interface for storing and retrieving dataset information."""
    
    @abstractmethod
    def save(self, dataset_info: DatasetInfo) -> str:
        """
        Save dataset information.
        
        Args:
            dataset_info: The dataset information to save
            
        Returns:
            str: The path or identifier of the saved information
        """
        pass
    
    @abstractmethod
    def load(self, identifier: str) -> DatasetInfo:
        """
        Load dataset information.
        
        Args:
            identifier: The identifier of the dataset information
            
        Returns:
            DatasetInfo: The loaded dataset information
        """
        pass


class DatasetResearchUseCase:
    """
    Use case for researching a dataset.
    
    This class orchestrates the process of gathering information about a dataset.
    """
    
    def __init__(
        self, 
        agent: AgentInterface,
        extractor: TextExtractor,
        repository: DatasetInfoRepository
    ):
        self.agent = agent
        self.extractor = extractor
        self.repository = repository
    
    def execute(self, dataset_name: str, dataset_url: Optional[str] = None) -> DatasetInfo:
        """
        Research a dataset and return information about it.
        
        Args:
            dataset_name: Name of the dataset
            dataset_url: Optional URL for the dataset
            
        Returns:
            DatasetInfo: Information about the dataset
        """
        # Create a new dataset info object
        dataset_info = DatasetInfo(name=dataset_name, home_url=dataset_url)
        
        # Track timing information
        timings = {}
        
        # Get description
        description = self._get_description(dataset_name, dataset_url)
        dataset_info.description = description
        
        # Get organizations
        organizations = self._get_organizations(dataset_name, description, dataset_url)
        dataset_info.organizations = organizations
        
        # Get aliases
        aliases = self._get_aliases(dataset_name, description, dataset_url, organizations)
        dataset_info.aliases = aliases
        
        # Get access type
        access_type = self._get_access_type(dataset_name, description, dataset_url)
        dataset_info.access_type = access_type
        
        # Get data URL
        data_url = self._get_url("data", dataset_name, description, dataset_url)
        dataset_info.data_url = data_url
        
        # Get schema URL
        schema_url = self._get_url("schema", dataset_name, description, dataset_url)
        dataset_info.schema_url = schema_url
        
        # Get documentation URL
        documentation_url = self._get_url("documentation", dataset_name, description, dataset_url)
        dataset_info.documentation_url = documentation_url
        
        # Save to repository
        self.repository.save(dataset_info)
        
        return dataset_info
    
    def _get_description(self, dataset_name: str, dataset_url: Optional[str]) -> str:
        """Get a description of the dataset."""
        prompt = f"""Research the dataset named '{dataset_name}'. 
        
        If a URL was provided for reference, it is: {dataset_url if dataset_url else 'None'}
        
        I need a concise description (150-200 words) that includes:
        - What the dataset contains
        - Who created it
        - Its purpose and use cases
        - Key features or unique aspects
        
        Use the web_search tool to find information about this dataset.
        """
        
        response = self.agent.get_information(prompt)
        return self._clean_description(response)
    
    def _get_organizations(self, dataset_name: str, description: str, dataset_url: Optional[str]) -> List[str]:
        """Get organizations related to the dataset."""
        prompt = f"""Find all organizations related to the dataset '{dataset_name}'.

Dataset description: {description}
If a URL was provided for reference, it is: {dataset_url if dataset_url else 'None'}

Find all organizations associated with this dataset, including:
- Dataset creators
- Publishers
- Funders
- Hosting institutions
- Research collaborators

For each organization:
- Include both full names and acronyms (e.g., "United States Department of Agriculture" and "USDA")
- If you see an acronym, search for its full name and include both
- If you see a combined name with acronym like "United States Department of Agriculture (USDA)", separate them into distinct entries

Use the web_search tool to search for "{dataset_name} dataset organization creator publisher funder".

Return your findings as a Python list of strings like this: ["Organization1", "Organization2", "USDA", "United States Department of Agriculture"]
"""
        
        response = self.agent.get_information(prompt)
        organizations = self.extractor.extract_list(response)
        return self._process_organizations(organizations)
    
    def _get_aliases(
        self, dataset_name: str, description: str, dataset_url: Optional[str], organizations: List[str]
    ) -> List[str]:
        """Get aliases for the dataset."""
        prompt = f"""Find all aliases, names, acronyms, and identifiers for the dataset '{dataset_name}'.

Dataset description: {description}
If a URL was provided for reference, it is: {dataset_url if dataset_url else 'None'}

For the purpose of this task, aliases refer to how publications' authors cite, acknowledge, and credit the dataset in their publications. Search the web for instructions on how to cite, acknowledge, and credit this dataset to help find the aliases information.

Examples of aliases include:
- Alternative names that appear in academic papers or documentation
- How researchers formally cite the dataset in publications
- Shortened versions or commonly used abbreviations
- DOIs, accession numbers, URLs, or other formal identifiers

Specifically consider these common patterns:
- Adjective form of the main noun (e.g., "National" instead of "Nation")
- Organization name + key terms 
- Abbreviations of key terms
- Re-ordering of terms
- With and without "of" or other connecting words
- Common shorthand variations used by researchers

IMPORTANT: 
1. Use the web_search tool to find information about how this dataset is cited and referenced.
2. Include the original dataset name as one of the aliases.
3. Be comprehensive and thorough; find ALL possible variations.

Return your answer as a Python list of strings. For example: ["Name 1", "Name 2", "Acronym", "http://example.com/identifier"]
"""

        response = self.agent.get_information(prompt)
        aliases = self.extractor.extract_list(response)
        
        # Add the original dataset name if not already present
        if dataset_name not in aliases and dataset_name.lower() not in [a.lower() for a in aliases]:
            aliases.append(dataset_name)
            
        return self._filter_aliases(aliases, organizations)
    
    def _get_access_type(self, dataset_name: str, description: str, dataset_url: Optional[str]) -> str:
        """Determine the access type for the dataset."""
        prompt = f"""Determine if the dataset '{dataset_name}' is freely accessible (Open), requires registration or payment (Restricted), or if this information is unclear (Unknown).

Dataset description: {description}
If a URL was provided for reference, it is: {dataset_url if dataset_url else 'None'}

Research the dataset's accessibility and licensing. Consider:
- Can anyone download the data without login or payment?
- Is registration, approval, or payment required?
- Are there usage restrictions like non-commercial only?

Search for policies, access information, download pages, or API documentation.

After research, simply respond with one of these three words:
Open - If the dataset is freely accessible without any login or payment
Restricted - If the dataset requires registration, approval, or payment
Unknown - If you can't determine the access type

Use the web_search tool to search for "{dataset_name} dataset access download availability"
"""

        response = self.agent.get_information(prompt)
        response = self._clean_description(response)
        
        if "open" in response.lower():
            return "Open"
        elif "restricted" in response.lower():
            return "Restricted"
        else:
            return "Unknown"
    
    def _get_url(self, url_type: str, dataset_name: str, description: str, dataset_url: Optional[str]) -> Optional[str]:
        """Get a URL of a specified type for the dataset."""
        # Determine search terms and description based on type
        if url_type == "data":
            search_suffix = "dataset download link data access"
            type_description = "for downloading the dataset's data"
        elif url_type == "schema":
            search_suffix = "dataset schema data dictionary field definitions metadata"
            type_description = "for the dataset's data dictionary, schema, or field definitions"
        elif url_type == "documentation":
            search_suffix = "dataset documentation user guide technical manual help"
            type_description = "for documentation, user guides, or technical manuals"
        else:
            return None
        
        prompt = f"""Find a valid URL {type_description} for the dataset '{dataset_name}'.

Dataset description: {description}
If a URL was provided for reference, it is: {dataset_url if dataset_url else 'None'}

First, use the web_search tool to search for "{dataset_name} {search_suffix}".
Then, validate any URL you find using the make_request tool to ensure it returns a 200 status.

If you find multiple URLs, choose the most official and comprehensive one.
If no perfect URL is found, return the closest valid URL that comes from an official source.
DO NOT return "Not found" or an empty response. Return the best URL you can find, even if it's just a landing page.

Return ONLY the URL with no additional text or explanation.
"""

        response = self.agent.get_information(prompt)
        url = self.extractor.extract_url(response)
        
        # Fallback to dataset_url if no URL found
        if not url and dataset_url:
            return dataset_url
            
        return url
    
    def _clean_description(self, description: str) -> str:
        """Clean the description by removing thinking sections and other artifacts."""
        from ..utils.text_processing import clean_description
        
        # Use the improved clean_description function from text_processing.py
        return clean_description(description)
    
    def _process_organizations(self, organizations: List[str]) -> List[str]:
        """Process organization names to expand acronyms and split combined names."""
        import re
        
        if not organizations:
            return []
        
        # Common organization acronyms and their expansions
        acronym_map = {
            "USDA": ["United States Department of Agriculture", "U.S. Department of Agriculture"],
            "DOA": ["Department of Agriculture"],
            "NASS": ["National Agricultural Statistics Service"],
            "ERS": ["Economic Research Service"],
            "NAL": ["National Agricultural Library"],
            "ARS": ["Agricultural Research Service"],
            "FS": ["Forest Service"],
            "APHIS": ["Animal and Plant Health Inspection Service"],
            "NRCS": ["Natural Resources Conservation Service"],
            "FSA": ["Farm Service Agency"],
            "FAS": ["Foreign Agricultural Service"],
            "RMA": ["Risk Management Agency"],
            "FEMA": ["Federal Emergency Management Agency"],
            "EPA": ["Environmental Protection Agency"],
            "FDA": ["Food and Drug Administration"],
            "NIH": ["National Institutes of Health"],
            "CDC": ["Centers for Disease Control and Prevention"],
            "NSF": ["National Science Foundation"],
            "NOAA": ["National Oceanic and Atmospheric Administration"],
            "NASA": ["National Aeronautics and Space Administration"],
            "DOE": ["Department of Energy"],
            "DOI": ["Department of the Interior"],
            "DOC": ["Department of Commerce"],
            "DOD": ["Department of Defense"],
            "DOL": ["Department of Labor"],
            "DOJ": ["Department of Justice"],
            "DOS": ["Department of State"],
            "DOT": ["Department of Transportation"],
            "VA": ["Department of Veterans Affairs", "Veterans Affairs"],
            "HHS": ["Department of Health and Human Services", "Health and Human Services"],
            "ED": ["Department of Education", "Education Department"],
            "HUD": ["Department of Housing and Urban Development", "Housing and Urban Development"],
        }
        
        processed_orgs = set()
        
        # First pass: Extract acronyms from parentheses and split combined names
        for org in organizations:
            # Clean the organization name
            clean_org = org.strip()
            
            # Skip empty strings
            if not clean_org:
                continue
                
            # Add the original organization name
            processed_orgs.add(clean_org)
            
            # Extract acronym from parentheses - pattern: Name (ACRONYM)
            acronym_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', clean_org)
            if acronym_match:
                full_name = acronym_match.group(1).strip()
                acronym = acronym_match.group(2).strip()
                
                # Add both the full name and acronym separately
                processed_orgs.add(full_name)
                processed_orgs.add(acronym)
                
                # Also check if we know expansions for this acronym
                if acronym in acronym_map:
                    processed_orgs.update(acronym_map[acronym])
            
            # Check for known acronyms in the text
            for acronym, expansions in acronym_map.items():
                if acronym in clean_org.split() or acronym == clean_org:
                    # Add the acronym's expansions
                    processed_orgs.update(expansions)
        
        # Remove duplicates and sort
        result = sorted(list(processed_orgs))
        return result
    
    def _filter_aliases(self, aliases: List[str], organizations: List[str] = None) -> List[str]:
        """Filter aliases to remove redundancy."""
        import re
        from ..utils.text_processing import filter_aliases_by_substrings
        
        if not aliases:
            return []
        
        # Initialize result
        if organizations is None:
            organizations = []
        
        # Process each alias
        processed_aliases = []
        
        # First pass: clean up aliases
        for alias in aliases:
            # Skip empty aliases
            if not alias or not alias.strip():
                continue
                
            # Skip if it's just a single letter or number
            if len(alias.strip()) <= 1:
                continue
                
            # Check if alias is a URL or identifier (DOI, etc.)
            is_url_or_id = (
                alias.startswith(('http://', 'https://')) or
                'doi.org' in alias.lower() or
                re.search(r'issn:\s*[\d-]+', alias.lower()) or
                re.search(r'doi:\s*[\d./]+', alias.lower())
            )
            
            # If it's a URL or identifier, add it directly
            if is_url_or_id:
                processed_aliases.append(alias)
                continue
                
            # Otherwise, process it
            processed = alias.strip()
            
            # Remove years and version numbers (like "2022 Census of Agriculture")
            processed = re.sub(r'\b(19|20)\d{2}\b', '', processed)
            
            # Remove extra whitespace
            processed = re.sub(r'\s+', ' ', processed).strip()
            
            # Capitalize properly
            processed = ' '.join(w.capitalize() if w.lower() not in ('of', 'and', 'the', 'in', 'for') else w 
                                for w in processed.split())
            
            # Only add if not empty after processing
            if processed and len(processed) >= 2:
                processed_aliases.append(processed)
        
        # Apply the substring filtering logic using the new function
        result = filter_aliases_by_substrings(processed_aliases)
        
        return result 