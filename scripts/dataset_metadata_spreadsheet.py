import os
import json
import pandas as pd
import argparse
from pathlib import Path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process dataset metadata JSON files and create an Excel spreadsheet')
    parser.add_argument('input_folder', type=str, help='Path to folder containing JSON metadata files')
    parser.add_argument('output_file', type=str, help='Path to output Excel file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Define the results directory and output path
    results_dir = Path(args.input_folder)
    output_file = Path(args.output_file)
    
    # Validate that input folder exists
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: Input folder '{results_dir}' does not exist or is not a directory")
        return 1
    
    # Lists to store data for each sheet
    main_data = []
    aliases_data = []
    organizations_data = []
    
    # Read all JSON files in the results directory
    for json_file in results_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                
                # Extract main metadata
                metadata = {
                    'dataset_name': data.get('dataset_name', ''),
                    'home_url': data.get('home_url', ''),
                    'description': data.get('description', ''),
                    'access_type': data.get('access_type', ''),
                    'data_url': data.get('data_url', ''),
                    'schema_url': data.get('schema_url', ''),
                    'documentation_url': data.get('documentation_url', '')
                }
                main_data.append(metadata)
                
                # Extract aliases
                dataset_name = data.get('dataset_name', '')
                for alias in data.get('aliases', []):
                    aliases_data.append({
                        'dataset_name': dataset_name,
                        'alias': alias
                    })
                
                # Extract organizations
                for org in data.get('organizations', []):
                    organizations_data.append({
                        'dataset_name': dataset_name,
                        'organization': org
                    })
                    
            except json.JSONDecodeError:
                print(f"Error parsing JSON file: {json_file}")
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
    
    # Create DataFrames
    main_df = pd.DataFrame(main_data)
    aliases_df = pd.DataFrame(aliases_data)
    organizations_df = pd.DataFrame(organizations_data)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write each DataFrame to a different worksheet
        main_df.to_excel(writer, sheet_name='Main Data', index=False)
        aliases_df.to_excel(writer, sheet_name='Aliases', index=False)
        organizations_df.to_excel(writer, sheet_name='Organizations', index=False)
    
    print(f"Spreadsheet created successfully: {output_file}")
    return 0

if __name__ == "__main__":
    exit(main()) 