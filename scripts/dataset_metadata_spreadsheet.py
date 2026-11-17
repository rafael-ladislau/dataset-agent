"""Export dataset metadata JSON files to a multi-sheet Excel workbook.

Reads every ``*.json`` file in an input folder and writes three sheets:
``Main Data``, ``Aliases``, and ``Organizations``.

Supports both the legacy schema (``dataset_name`` / ``aliases`` /
``organizations``) and the current ``DatasetRecord`` schema
(``main_dataset_name`` / ``dataset_names`` / ``flag_terms``).

Usage:
    python scripts/dataset_metadata_spreadsheet.py results/ datasets_metadata.xlsx

Requires:
    pip install -e ".[spreadsheet]"   # pandas + openpyxl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _name(data: dict[str, Any]) -> str:
    return data.get("dataset_name") or data.get("main_dataset_name") or ""


def _aliases(data: dict[str, Any]) -> list[str]:
    return data.get("aliases") or data.get("dataset_names") or []


def _organizations(data: dict[str, Any]) -> list[str]:
    return data.get("organizations") or data.get("flag_terms") or []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process dataset metadata JSON files into an Excel spreadsheet"
    )
    parser.add_argument("input_folder", type=str, help="Folder containing JSON metadata files")
    parser.add_argument("output_file", type=str, help="Path to the output Excel file")
    args = parser.parse_args()

    results_dir = Path(args.input_folder)
    output_file = Path(args.output_file)

    if not results_dir.is_dir():
        print(f"Error: Input folder '{results_dir}' does not exist or is not a directory")
        return 1

    main_data: list[dict[str, Any]] = []
    aliases_data: list[dict[str, str]] = []
    organizations_data: list[dict[str, str]] = []

    for json_file in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Error parsing JSON file: {json_file}")
            continue
        except OSError as exc:
            print(f"Error reading file {json_file}: {exc}")
            continue

        if not isinstance(data, dict):
            continue

        name = _name(data)
        main_data.append(
            {
                "dataset_name": name,
                "home_url": data.get("home_url", ""),
                "description": data.get("description", ""),
                "access_type": data.get("access_type", ""),
                "data_url": data.get("data_url", ""),
                "schema_url": data.get("schema_url", ""),
                "documentation_url": data.get("documentation_url", ""),
            }
        )
        for alias in _aliases(data):
            aliases_data.append({"dataset_name": name, "alias": alias})
        for org in _organizations(data):
            organizations_data.append({"dataset_name": name, "organization": org})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        pd.DataFrame(main_data).to_excel(writer, sheet_name="Main Data", index=False)
        pd.DataFrame(aliases_data).to_excel(writer, sheet_name="Aliases", index=False)
        pd.DataFrame(organizations_data).to_excel(
            writer, sheet_name="Organizations", index=False
        )

    print(f"Spreadsheet created successfully: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
