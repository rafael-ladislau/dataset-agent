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

    def save(self, record: DatasetRecord) -> Path:
        path = self._output_dir / _safe_filename(record.main_dataset_name)
        logger.info(
            "JSON persist: main_dataset_name=%r -> %s",
            record.main_dataset_name,
            path,
        )
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path
