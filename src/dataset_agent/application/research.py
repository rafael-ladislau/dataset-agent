"""Dataset research use case."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from dataset_agent.application import prompts
from dataset_agent.adapters.llm_output_cleanup import is_llm_list_entry_junk
from dataset_agent.adapters.tools import TOOL_DEFINITIONS, make_request, web_search
from dataset_agent.adapters.organizations import process_organizations
from dataset_agent.adapters.text_processing import (
    clean_description,
    dedupe_strings_ci_preserve_order,
    filter_aliases_by_substrings,
    hostname_from_http_url,
    is_alias_likely_sentence,
    is_alias_version_token_only,
    is_whole_alias_generic_token,
    normalize_access_label,
    remove_compound_aliases,
)
from dataset_agent.adapters.dataset_aliases import (
    detect_subdataset_aliases,
    refine_dataset_names_with_llm,
    refine_flag_terms_with_llm,
    validate_no_flag_alias_overlap,
)
from dataset_agent.adapters.literature import (
    evaluate_terms_with_llm,
    fetch_dimensions_publication_metrics,
    promote_dimensions_metrics_from_detail,
)
from dataset_agent.adapters.fp_analysis import web_search_alias_meanings
from dataset_agent.adapters.query_heuristics import generate_multilingual_aliases, check_suffix_necessity
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
    DimensionsDslPort,
    LiteratureGatePort,
    TextExtractorPort,
)

if TYPE_CHECKING:
    from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)


def _attach_dimensions_preview_if_needed(
    record: DatasetRecord,
    settings: Settings,
    sample_size: int,
) -> None:
    """When literature gate is noop, still fetch Dimensions count + DSL if API key is set."""
    if settings.literature_gate == "dimensions":
        return
    if not (settings.dimensions_api_key or "").strip():
        return
    try:
        q, total = fetch_dimensions_publication_metrics(
            record, sample_size=sample_size, settings=settings
        )
        record.dsl_query = q
        record.publications_total = total
        logger.info(
            "Dimensions preview: publications_total=%s dsl_query=%s",
            total,
            (q[:120] + "…") if len(q) > 120 else q,
        )
    except Exception as exc:
        logger.warning("Dimensions preview failed: %s", exc)


class LiteratureGateFailed(RuntimeError):
    """Raised when literature screening fails after all attempts."""


_MAX_ALIAS_LEN = 80


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
        if not is_url_or_id:
            if len(a) > _MAX_ALIAS_LEN:
                continue
            if is_alias_version_token_only(a):
                continue
            if is_whole_alias_generic_token(a):
                continue
            if is_alias_likely_sentence(a):
                continue
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
    processed = dedupe_strings_ci_preserve_order(processed)
    processed = remove_compound_aliases(processed)
    return filter_aliases_by_substrings(processed)


class DatasetResearchUseCase:
    def __init__(
        self,
        agent: AgentPort,
        extractor: TextExtractorPort,
        repository: DatasetRepositoryPort,
        settings: Settings,
        literature_gate: LiteratureGatePort | None = None,
        dsl_port: DimensionsDslPort | None = None,
    ):
        self._agent = agent
        self._extractor = extractor
        self._repository = repository
        self._settings = settings
        self._literature_gate = literature_gate
        self._dsl_port = dsl_port

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
                _attach_dimensions_preview_if_needed(
                    record, self._settings, request.sample_size
                )
                path = self._repository.save(record)
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
                promote_dimensions_metrics_from_detail(record, result.detail)
                _attach_dimensions_preview_if_needed(
                    record, self._settings, request.sample_size
                )

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
                    suggested_exclude_terms=list(
                        eval_result.get("suggested_exclude_terms") or []
                    ),
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
            promote_dimensions_metrics_from_detail(record, last_result.detail)
            _attach_dimensions_preview_if_needed(
                record, self._settings, request.sample_size
            )

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
                suggested_exclude_terms=list(
                    eval_result.get("suggested_exclude_terms") or []
                ),
                reasoning=eval_result["reasoning"],
            )
            
            path = self._repository.save(record)
            logger.info("Final record saved with literature_validation and terms_evaluation")

        return record, path  # type: ignore[return-value]

    def _run_pipeline_once(self, request: ResearchRequest) -> DatasetRecord:
        name = request.dataset_name

        # Step 1: Description + Home URL (combined, 1 LLM call)
        logger.info("Step 1/6: description + home_url (deterministic web search + fetch)")
        search_results = web_search(name)
        fetched_page = ""
        if request.dataset_url:
            fetched_page = make_request(request.dataset_url)
        desc_home_raw = self._agent.get_information(
            prompts.description_and_home_url_prompt(
                name, request.dataset_url, search_results, fetched_page
            ),
            tools=TOOL_DEFINITIONS,
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
        logger.info("Step 2/6: URLs (data/schema/doc) + access_type (LLM + validation)")
        urls_raw = self._agent.get_information(
            prompts.urls_and_access_prompt(name, description, home_url),
            tools=TOOL_DEFINITIONS,
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
        logger.info("Step 3/6: organizations (LLM)")
        org_raw = self._agent.get_information(
            prompts.organizations_prompt(name, description, home_url),
            tools=TOOL_DEFINITIONS,
        )
        logger.debug("Organizations raw output: %r", org_raw[:500] if org_raw else "")
        org_list = self._extractor.extract_list(org_raw)
        logger.debug("Organizations extracted list: %r", org_list)
        flag_terms = process_organizations(org_list)
        logger.info("Step 3 done: %s context terms (before refine)", len(flag_terms))

        flags_before_refine = list(flag_terms)
        logger.info("Step 4/6: refine flag_terms (LLM)")
        flag_terms = refine_flag_terms_with_llm(
            self._agent,
            main_dataset_name=name,
            description=description,
            flag_terms=flag_terms,
        )
        if not flag_terms:
            logger.warning(
                "refine_flag_terms returned empty; keeping %s pre-refine flag terms",
                len(flags_before_refine),
            )
            flag_terms = flags_before_refine
        logger.info("Step 4 done: %s flag terms for literature", len(flag_terms))

        # Step 5: Aliases (1 LLM call, uses orgs + home_url context)
        logger.info("Step 5/6: aliases (LLM)")
        alias_raw = self._agent.get_information(
            prompts.aliases_prompt(
                name,
                description,
                home_url,
                flag_terms,
                dataset_url=request.dataset_url,
            ),
            tools=TOOL_DEFINITIONS,
        )
        logger.debug("Aliases raw output: %r", alias_raw[:500] if alias_raw else "")
        aliases = self._extractor.extract_list(alias_raw)
        logger.debug("Aliases extracted list: %r", aliases)
        ref_for_host = (request.dataset_url or "").strip() or (home_url or "")
        host_alias = hostname_from_http_url(ref_for_host)
        if host_alias and host_alias.lower() not in {str(a).lower() for a in aliases if a}:
            aliases.append(host_alias)
            logger.info("Added hostname as alias candidate: %s", host_alias)
        if name not in aliases and name.lower() not in {a.lower() for a in aliases}:
            aliases.append(name)
        dataset_names = _filter_alias_entries(aliases, flag_terms)
        logger.info("Step 5 done: %s names after filters", len(dataset_names))

        if self._settings.research_multilingual_aliases:
            ml_aliases = generate_multilingual_aliases(name, self._agent)
            if ml_aliases:
                before_ml = len(dataset_names)
                ml_lower = {n.lower() for n in dataset_names}
                dataset_names.extend(a for a in ml_aliases if a.lower() not in ml_lower)
                dataset_names = dedupe_strings_ci_preserve_order(dataset_names)
                logger.info(
                    "Multilingual alias expansion: %s -> %s names",
                    before_ml,
                    len(dataset_names),
                )

        names_before_refine = list(dataset_names)
        logger.info("Step 6/6: refine dataset_names vs flag_terms (LLM)")
        dataset_names = refine_dataset_names_with_llm(
            self._agent,
            main_dataset_name=name,
            description=description,
            dataset_names=dataset_names,
            flag_terms=flag_terms,
        )
        if not dataset_names:
            logger.warning(
                "refine_dataset_names returned empty; keeping %s pre-refine names",
                len(names_before_refine),
            )
            dataset_names = names_before_refine
        dataset_names, flag_terms, overlap_removed = validate_no_flag_alias_overlap(
            dataset_names,
            flag_terms,
        )
        if overlap_removed:
            logger.info(
                "Removed %s aliases overlapping flag_terms (sample=%r)",
                len(overlap_removed),
                overlap_removed[:5],
            )
        dataset_names, sub_removed = detect_subdataset_aliases(
            self._agent,
            name,
            dataset_names,
        )
        if sub_removed:
            logger.info(
                "Subdataset alias filter removed %s entries (sample=%r)",
                len(sub_removed),
                sub_removed[:8],
            )
        logger.info("Step 6 done: %s dataset name aliases for literature", len(dataset_names))

        # Step 7: Homonym / collision web check
        collision_domains: dict[str, list[str]] = {}
        collision_max = self._settings.research_collision_max_aliases
        if collision_max > 0 and dataset_names:
            logger.info("Step 7: collision check on top %s aliases", collision_max)
            for alias in dataset_names[:collision_max]:
                try:
                    domains = web_search_alias_meanings(self._agent, alias)
                    for domain, terms in domains.items():
                        collision_domains.setdefault(domain, []).extend(terms)
                except Exception as exc:
                    logger.warning("Collision check failed for %r: %s", alias, exc)
            if collision_domains:
                logger.info(
                    "Collision check: %s domains, %s total candidate exclude terms",
                    len(collision_domains),
                    sum(len(v) for v in collision_domains.values()),
                )

        # Step 8: Suffix necessity check
        suffix_forms: dict[str, str] = {}
        suffix_max = self._settings.research_suffix_check_max_aliases
        if suffix_max > 0 and dataset_names:
            logger.info("Step 8: suffix necessity check on top %s aliases", suffix_max)
            for alias in dataset_names[:suffix_max]:
                result = None
                # Try DSL-based suffix check first
                if self._dsl_port is not None:
                    try:
                        loop = asyncio.new_event_loop()
                        result = loop.run_until_complete(
                            check_suffix_necessity(self._dsl_port, alias)
                        )
                        loop.close()
                        if result:
                            suffix_forms[alias] = result
                            logger.info("Suffix promoted: %r -> %r", alias, result)
                            continue
                    except RuntimeError as exc:
                        if "already running" in str(exc):
                            logger.debug("Cannot run nested event loop; will use LLM fallback for suffix check")
                        else:
                            logger.warning("Suffix check failed for %r: %s", alias, exc)
                    except Exception as exc:
                        logger.warning("Suffix check failed for %r: %s", alias, exc)
                # LLM fallback (also used when nested event loop fails)
                if result is None and alias in suffix_forms:
                    continue
                try:
                    prompt = (
                        f"Dataset: {name!r}. Alias: {alias!r}.\n\n"
                        "Would this alias benefit from adding a suffix like 'dataset', "
                        "'survey', 'data', or 'study' for a literature search? "
                        "If yes, return the suggested suffixed form. If no, return '' (empty string)."
                    )
                    raw = self._agent.get_structured(prompt, {"type": "string"})
                    if isinstance(raw, str) and raw.strip():
                        suffix_forms[alias] = raw.strip()
                        logger.info("Suffix promoted (LLM): %r -> %r", alias, raw.strip())
                except Exception as exc:
                    logger.warning("LLM suffix fallback failed for %r: %s", alias, exc)
        logger.info("Building DatasetRecord with config defaults")
        record = build_record_from_pipeline(
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
        record.alias_collisions = collision_domains
        if suffix_forms:
            # Store suffix forms in a way that's accessible; for now attach to record
            # We don't have a dedicated field, so use a private detail or extend aliases
            # Let's add suffix-promoted forms to dataset_names if not already present
            existing_lower = {n.lower() for n in record.dataset_names}
            for original, suffixed in suffix_forms.items():
                if suffixed.lower() not in existing_lower:
                    record.dataset_names.append(suffixed)
                    existing_lower.add(suffixed.lower())
            record.dataset_names = dedupe_strings_ci_preserve_order(record.dataset_names)
        return record
