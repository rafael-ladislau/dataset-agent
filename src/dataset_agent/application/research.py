"""Dataset research use case."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from dataset_agent.application import prompts
from dataset_agent.adapters.llm_output_cleanup import is_llm_list_entry_junk
from dataset_agent.adapters.organizations import process_organizations
from dataset_agent.adapters.text_processing import (
    clean_description,
    filter_aliases_by_substrings,
    normalize_access_label,
)
from dataset_agent.adapters.literature import evaluate_terms_with_llm
from dataset_agent.domain.models import (
    DatasetRecord,
    LiteratureValidation,
    ResearchRequest,
    TermsEvaluation,
    build_record_from_pipeline,
)
from dataset_agent.domain.ports import (
    AgentPort,
    DatasetRepositoryPort,
    LiteratureGatePort,
    TextExtractorPort,
)

if TYPE_CHECKING:
    from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)


class LiteratureGateFailed(RuntimeError):
    """Raised when literature screening fails after all attempts."""


def _filter_alias_entries(aliases: list[str], organizations: list[str]) -> list[str]:
    _ = organizations  # reserved for future rules
    processed: list[str] = []
    for alias in aliases:
        if not alias or not alias.strip():
            continue
        a = alias.strip()
        if is_llm_list_entry_junk(a):
            continue
        if len(a) <= 1:
            continue
        is_url_or_id = (
            a.startswith(("http://", "https://"))
            or "doi.org" in a.lower()
            or re.search(r"doi:\s*[\d./]+", a.lower())
        )
        if is_url_or_id:
            processed.append(a)
            continue
        s = re.sub(r"\b(19|20)\d{2}\b", "", a)
        s = re.sub(r"\s+", " ", s).strip()
        s = " ".join(
            w.capitalize() if w.lower() not in ("of", "and", "the", "in", "for") else w
            for w in s.split()
        )
        if len(s) >= 2:
            processed.append(s)
    return filter_aliases_by_substrings(processed)


class DatasetResearchUseCase:
    def __init__(
        self,
        agent: AgentPort,
        extractor: TextExtractorPort,
        repository: DatasetRepositoryPort,
        settings: Settings,
        literature_gate: LiteratureGatePort | None = None,
    ):
        self._agent = agent
        self._extractor = extractor
        self._repository = repository
        self._settings = settings
        self._literature_gate = literature_gate

    def execute(self, request: ResearchRequest) -> tuple[DatasetRecord, Path]:
        max_attempts = max(1, self._settings.literature_max_attempts)
        logger.info(
            "Research started: dataset_name=%r literature_max_attempts=%s gate=%s",
            request.dataset_name,
            max_attempts,
            type(self._literature_gate).__name__ if self._literature_gate else "none",
        )

        record: DatasetRecord | None = None
        path: Path | None = None
        last_result = None
        last_attempt = 0

        for attempt in range(1, max_attempts + 1):
            last_attempt = attempt
            logger.info("Research cycle %s/%s", attempt, max_attempts)
            record = self._run_pipeline_once(request)
            logger.info("Saving record JSON (dataset=%r)...", request.dataset_name)
            path = self._repository.save(record)
            logger.info("File saved: %s", path)

            if self._literature_gate is None:
                logger.info("No literature gate configured; research complete.")
                return record, path

            logger.info(
                "Running literature gate: %s (sample_size=%s llm_batch_size=%s)",
                type(self._literature_gate).__name__,
                request.sample_size,
                request.llm_batch_size,
            )
            result = self._literature_gate.assess(
                record,
                sample_size=request.sample_size,
                llm_batch_size=request.llm_batch_size,
            )
            last_result = result
            logger.info(
                "Literature gate returned: passed=%s indicator=%s detail=%s",
                result.passed,
                result.indicator,
                result.detail,
            )

            if result.passed:
                logger.info(
                    "Literature gate OK (indicator=%s detail=%s)",
                    result.indicator,
                    result.detail,
                )
                # Save validation result in record
                record.literature_validation = LiteratureValidation(
                    passed=True,
                    indicator=result.indicator,
                    attempts=attempt,
                    detail=result.detail,
                )
                # Extract publications_total to record level
                record.publications_total = result.detail.get("publications_total")
                
                # Always evaluate terms effectiveness
                validation_details = result.detail.get("validation_details", [])
                logger.info("Running terms evaluation...")
                eval_result = evaluate_terms_with_llm(
                    dataset_name=record.main_dataset_name,
                    description=record.description,
                    dataset_names=record.dataset_names,
                    flag_terms=record.flag_terms,
                    validation_details=validation_details,
                    agent=self._agent,
                )
                record.terms_evaluation = TermsEvaluation(
                    is_effective=eval_result["is_effective"],
                    dataset_names_score=eval_result["dataset_names_score"],
                    flag_terms_score=eval_result["flag_terms_score"],
                    issues=eval_result["issues"],
                    suggested_dataset_names=eval_result["suggested_dataset_names"],
                    suggested_flag_terms=eval_result["suggested_flag_terms"],
                    reasoning=eval_result["reasoning"],
                )
                
                path = self._repository.save(record)
                return record, path

            logger.warning("Literature gate failed (attempt %s): %s", attempt, result.detail)

        # All attempts exhausted - save with validation info and return (no exception)
        if record and last_result:
            logger.warning(
                "Literature gate did not pass after %s attempts; completing with validation info",
                last_attempt,
            )
            record.literature_validation = LiteratureValidation(
                passed=False,
                indicator=last_result.indicator,
                attempts=last_attempt,
                detail=last_result.detail,
            )
            # Extract publications_total to record level
            record.publications_total = last_result.detail.get("publications_total")
            
            # Always evaluate terms effectiveness
            validation_details = last_result.detail.get("validation_details", [])
            logger.info("Running terms evaluation...")
            eval_result = evaluate_terms_with_llm(
                dataset_name=record.main_dataset_name,
                description=record.description,
                dataset_names=record.dataset_names,
                flag_terms=record.flag_terms,
                validation_details=validation_details,
                agent=self._agent,
            )
            record.terms_evaluation = TermsEvaluation(
                is_effective=eval_result["is_effective"],
                dataset_names_score=eval_result["dataset_names_score"],
                flag_terms_score=eval_result["flag_terms_score"],
                issues=eval_result["issues"],
                suggested_dataset_names=eval_result["suggested_dataset_names"],
                suggested_flag_terms=eval_result["suggested_flag_terms"],
                reasoning=eval_result["reasoning"],
            )
            
            path = self._repository.save(record)
            logger.info("Final record saved with literature_validation and terms_evaluation")

        return record, path  # type: ignore[return-value]

    def _run_pipeline_once(self, request: ResearchRequest) -> DatasetRecord:
        name = request.dataset_name

        # Step 1: Description + Home URL (combined, 1 LLM call)
        logger.info("Step 1/4: description + home_url (LLM + web search)")
        desc_home_raw = self._agent.get_information(
            prompts.description_and_home_url_prompt(name, None)
        )
        sections = self._extractor.extract_sections(desc_home_raw)
        description = clean_description(sections.get("DESCRIPTION", ""))
        home_url = self._extractor.extract_url(sections.get("HOME_URL", "")) or None
        if not description:
            description = clean_description(desc_home_raw)
        logger.info(
            "Step 1 done: description=%s chars, home_url=%r",
            len(description),
            home_url,
        )

        # Step 2: All URLs + Access Type (combined, 1 LLM call)
        logger.info("Step 2/4: URLs (data/schema/doc) + access_type (LLM + validation)")
        urls_raw = self._agent.get_information(
            prompts.urls_and_access_prompt(name, description, home_url)
        )
        url_sections = self._extractor.extract_sections(urls_raw)
        data_url = self._extractor.extract_url(url_sections.get("DATA_URL", ""))
        schema_url = self._extractor.extract_url(url_sections.get("SCHEMA_URL", ""))
        doc_url = self._extractor.extract_url(url_sections.get("DOCUMENTATION_URL", ""))
        access_type = normalize_access_label(url_sections.get("ACCESS_TYPE", "Unknown"))
        # Fallback: use home_url if specific URLs not found
        if not data_url and home_url:
            data_url = home_url
        if not doc_url and home_url:
            doc_url = home_url
        logger.info(
            "Step 2 done: data_url=%r, schema_url=%r, doc_url=%r, access=%r",
            data_url,
            schema_url,
            doc_url,
            access_type,
        )

        # Step 3: Organizations (1 LLM call, uses home_url context)
        logger.info("Step 3/4: organizations (LLM)")
        org_raw = self._agent.get_information(
            prompts.organizations_prompt(name, description, home_url)
        )
        logger.debug("Organizations raw output: %r", org_raw[:500] if org_raw else "")
        org_list = self._extractor.extract_list(org_raw)
        logger.debug("Organizations extracted list: %r", org_list)
        flag_terms = process_organizations(org_list)
        logger.info("Step 3 done: %s context terms", len(flag_terms))

        # Step 4: Aliases (1 LLM call, uses orgs + home_url context)
        logger.info("Step 4/4: aliases (LLM)")
        alias_raw = self._agent.get_information(
            prompts.aliases_prompt(name, description, home_url, flag_terms)
        )
        logger.debug("Aliases raw output: %r", alias_raw[:500] if alias_raw else "")
        aliases = self._extractor.extract_list(alias_raw)
        logger.debug("Aliases extracted list: %r", aliases)
        if name not in aliases and name.lower() not in {a.lower() for a in aliases}:
            aliases.append(name)
        dataset_names = _filter_alias_entries(aliases, flag_terms)
        logger.info("Step 4 done: %s names after filters", len(dataset_names))

        logger.info("Building DatasetRecord with config defaults")
        return build_record_from_pipeline(
            request,
            description=description,
            dataset_names=dataset_names,
            flag_terms=flag_terms,
            access_type=access_type,
            data_url=data_url,
            schema_url=schema_url,
            documentation_url=doc_url,
            home_url=home_url,
            defaults=self._settings,
        )
