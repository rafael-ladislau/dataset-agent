"""JSON file persistence."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from dataset_agent.domain.models import DatasetRecord
from dataset_agent.domain.ports import DatasetRepositoryPort

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name.lower())
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return (s[:120] or "dataset") + "_research.json"


class JsonDatasetRepository(DatasetRepositoryPort):
    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, main_dataset_name: str) -> Path:
        return self._output_dir / _safe_filename(main_dataset_name)

    def load(self, main_dataset_name: str) -> DatasetRecord | None:
        """The stored research record of *main_dataset_name*; None when there is none or it
        cannot be read (logged)."""
        path = self.path_for(main_dataset_name)
        if not path.is_file():
            return None
        try:
            return DatasetRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("JSON load: cannot read %s: %s", path, exc)
            return None

    def save(self, record: DatasetRecord) -> Path:
        path = self.path_for(record.main_dataset_name)
        logger.info(
            "JSON persist: main_dataset_name=%r -> %s",
            record.main_dataset_name,
            path,
        )
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path
