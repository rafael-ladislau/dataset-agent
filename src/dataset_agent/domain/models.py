"""
Domain models for the dataset agent.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DatasetInfo:
    """
    Data class representing information about a dataset.
    """
    name: str
    home_url: Optional[str] = None
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    access_type: str = "Unknown"
    data_url: Optional[str] = None
    schema_url: Optional[str] = None
    documentation_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DatasetInfo to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of DatasetInfo
        """
        return {
            "dataset_name": self.name,
            "home_url": self.home_url,
            "description": self.description,
            "aliases": self.aliases,
            "organizations": self.organizations,
            "access_type": self.access_type,
            "data_url": self.data_url,
            "schema_url": self.schema_url,
            "documentation_url": self.documentation_url,
            "_metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """
        Create a DatasetInfo instance from a dictionary.
        
        Args:
            data: Dictionary containing dataset information
            
        Returns:
            DatasetInfo: New DatasetInfo instance
        """
        return cls(
            name=data.get("dataset_name", ""),
            home_url=data.get("home_url"),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            organizations=data.get("organizations", []),
            access_type=data.get("access_type", "Unknown"),
            data_url=data.get("data_url"),
            schema_url=data.get("schema_url"),
            documentation_url=data.get("documentation_url"),
            metadata=data.get("_metadata", {})
        ) 