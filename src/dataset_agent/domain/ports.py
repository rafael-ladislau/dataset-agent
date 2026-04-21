"""Abstract ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataset_agent.domain.models import DatasetRecord


class AgentPort(ABC):
    @abstractmethod
    def get_information(self, prompt: str) -> str:
        pass


class TextExtractorPort(ABC):
    @abstractmethod
    def extract_list(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def extract_url(self, text: str) -> str | None:
        pass

    @abstractmethod
    def extract_section(self, text: str, section_name: str) -> str:
        """Extract content from a ===SECTION=== block."""
        pass

    @abstractmethod
    def extract_sections(self, text: str) -> dict[str, str]:
        """Extract all ===SECTION=== blocks into a dict."""
        pass


class DatasetRepositoryPort(ABC):
    @abstractmethod
    def save(self, record: DatasetRecord) -> Path:
        pass


class LiteratureGateResult:
    __slots__ = ("passed", "indicator", "detail")

    def __init__(self, passed: bool, indicator: float = 0.0, detail: dict | None = None):
        self.passed = passed
        self.indicator = indicator
        self.detail = detail or {}


class LiteratureGatePort(ABC):
    """Post-save screening (Dimensions + lexical + optional LLM sample)."""

    @abstractmethod
    def assess(
        self,
        record: DatasetRecord,
        sample_size: int = 10,
        llm_batch_size: int = 25,
    ) -> LiteratureGateResult:
        pass


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRepositoryPort(ABC):
    @abstractmethod
    def create_task(self, payload: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        result_path: str | None = None,
        error: str | None = None,
    ) -> None:
        pass
