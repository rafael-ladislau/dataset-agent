"""
Storage implementations for saving and loading dataset information.
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional

from ..domain.usecases import DatasetInfoRepository
from ..domain.models import DatasetInfo

# Set up logging
logger = logging.getLogger(__name__)


class JSONFileRepository(DatasetInfoRepository):
    """
    Implementation of DatasetInfoRepository that stores data in JSON files.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the JSON file repository.
        
        Args:
            output_dir: Directory to store JSON files (default: current directory)
        """
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_filename(self, dataset_name: str) -> str:
        """
        Create a filename based on the dataset name.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            str: Sanitized filename
        """
        # Sanitize the dataset name for use as a filename
        filename = re.sub(r'[^\w\s-]', '', dataset_name.lower())
        filename = re.sub(r'[-\s]+', '_', filename)
        return f"{filename}_research.json"
    
    def save(self, dataset_info: DatasetInfo) -> str:
        """
        Save dataset information to a JSON file.
        
        Args:
            dataset_info: DatasetInfo object to save
            
        Returns:
            str: Path to the saved file
        """
        # Get filename
        filename = self._get_filename(dataset_info.name)
        file_path = os.path.join(self.output_dir, filename)
        
        # Convert to dictionary
        data = dataset_info.to_dict()
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved dataset information to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving dataset information: {str(e)}")
            raise
    
    def load(self, identifier: str) -> DatasetInfo:
        """
        Load dataset information from a JSON file.
        
        Args:
            identifier: Path to the JSON file or dataset name
            
        Returns:
            DatasetInfo: Loaded dataset information
        """
        file_path = identifier
        
        # If the identifier is not a file path, assume it's a dataset name
        if not os.path.isfile(identifier):
            file_path = os.path.join(self.output_dir, self._get_filename(identifier))
        
        # Load from file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded dataset information from {file_path}")
            return DatasetInfo.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading dataset information: {str(e)}")
            raise 