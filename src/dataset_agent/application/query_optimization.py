"""Four-phase Dimensions query optimization orchestration (spec §7)."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Literal

from dataset_agent.adapters.ai_model_research import (
    MODEL_TERMS_TOOL,
    HubEvidence,
    assemble_candidates,
    classify_model_name,
    contains_word,
    fetch_hub_evidence,
    mine_context_terms,
    model_terms_prompt,
)
from dataset_agent.adapters.dimensions_dsl import (
    AsyncThrottledDimensionsDsl,
    DimensionsDslError,
    build_for_clause_count_dsl,
    build_search_publications_paged_dsl,
    fetch_top_n,
    inner_quoted_term,
    is_dsl_syntax_error,
    parse_total_count,
    publication_title,
    publications_from_result,
    run_alias_count,
    short_dsl_error,
)
from dataset_agent.adapters.fp_analysis import (
    classify_alias_mention,
    classify_title,
    derive_exclude_terms_from_fps,
    identify_fp_domains,
    run_scope_comparison,
    scan_abstracts_for_aliases,
    score_title_signal3,
    web_search_alias_meanings,
)
from dataset_agent.adapters.extraction_client import ExtractionClient, ExtractionError
from dataset_agent.adapters.fulltext_evidence import Evidence, build_evidence
from dataset_agent.adapters.fulltext_recall import (
    RecallCandidate,
    evidence_quotes,
    judge_complement,
    recall_candidates,
)
from dataset_agent.adapters.fulltext_refiner import refine_candidate
from dataset_agent.adapters.literature import evaluate_for_clause_sync
from dataset_agent.adapters.name_grounding import (
    GROUNDING_TOOL,
    content_tokens,
    defined_names,
    grounding_prompt,
    is_organization_alias,
    needs_grounding,
    screen_candidates,
)
from dataset_agent.adapters.name_history import HISTORY_TOOL, history_phrases, history_prompt
from dataset_agent.adapters.tools import web_search

#: Funding agencies too broad to gate a query when the dataset's own organizations exist.
BROAD_FUNDER_FLAGS = frozenset(
    x.casefold()
    for x in (
        "NSF", "National Science Foundation", "U.S. National Science Foundation",
        "NIH", "National Institutes of Health", "U.S. National Institutes of Health",
    )
)
from dataset_agent.adapters.seed_mapping import (
    QueryCandidate,
    ai_model_seed_clause,
    candidate_to_ai_model_seed,
    candidate_clause,
    candidate_to_seed,
    complement_candidate,
    complement_clause,
)
from dataset_agent.adapters.query_builder import (
    _or_group,
    build_for_clause,
    default_variant_build_order,
    run_variant,
)
from dataset_agent.adapters.query_heuristics import (
    _SUFFIX_CANDIDATES,
    apply_short_acronym_heuristic,
    check_suffix_necessity,
    detect_rhetorical_name,
    is_acronym_shaped_name,
    is_acronym_suffix_form,
    is_short_acronym,
    platform_exclude_suggestions,
    rhetorical_exclusions,
)
from dataset_agent.application.proposal_service import build_proposal, merge_diffs
from dataset_agent.adapters.term_sanitizer import sanitize_terms
from dataset_agent.adapters.text_processing import dedupe_strings_ci_preserve_order
from dataset_agent.domain.models import (
    AliasCountEntry,
    ConfidenceLevel,
    DatasetRecord,
    IterationMetrics,
    OptimizeRequest,
    QueryOptimizationAliases,
    QueryOptimizationRecord,
    ResearchRequest,
    TermDecision,
    VariantTestedEntry,
)
from dataset_agent.domain.proposals import TermDiff
from dataset_agent.domain.ports import AgentPort, DimensionsDslPort

if TYPE_CHECKING:
    from dataset_agent.application.research import DatasetResearchUseCase
    from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)

_VARIANT_ORDER = {"V4": 0, "V3": 1, "V2": 2, "V1": 3, "V5": 4}


def _alias_risk_label(c_alias: int, c_full: int, has_full_name: bool) -> Literal["safe", "risky"]:
    """Same rules as ``classify_alias_risk`` without extra Dimensions round-trips."""
    if not has_full_name:
        return "risky" if c_alias > 50_000 else "safe"
    if c_full <= 0:
        return "risky" if c_alias > 0 else "safe"
    if c_alias > 10.0 * float(c_full):
        return "risky"
    return "safe"


def _pick_start_variant(tested: list[VariantTestedEntry]) -> VariantTestedEntry:
    """Best Phase-2 variant: non-empty results first, then lowest FP, then variant preference."""
    return min(
        tested,
        key=lambda v: (
            0 if (v.expected_count or 0) > 0 else 1,
            float(v.fp_rate_pct or 0.0),
            _VARIANT_ORDER.get(v.label, 99),
        ),
    )


def _fp_keywords_active(fp_keywords: dict[str, list[str]]) -> bool:
    return any(isinstance(v, list) and v for v in fp_keywords.values())


def _fp_rate_and_noise_terms(
    titles: list[str],
    fp_keywords: dict[str, list[str]],
) -> tuple[float, list[str]]:
    """Proxy FP rate from title keyword hits across all FP domains + noise for NOT refinement."""
    hits = 0
    noise_terms: list[str] = []
    n = 0
    for title in titles:
        if not (title or "").strip():
            continue
        n += 1
        m = classify_title(title, fp_keywords)
        matched: list[str] = []
        for kws in m.values():
            if isinstance(kws, list):
                matched.extend(str(x) for x in kws if isinstance(x, str) and x.strip())
        if matched:
            hits += 1
            noise_terms.extend(matched)
    pct = (100.0 * hits / n) if n else 0.0
    return pct, dedupe_strings_ci_preserve_order(noise_terms)


@dataclasses.dataclass
class _ModelRun:
    """State of one AI-model run shared between term measurement and query selection."""

    name: str
    family: str
    hub: HubEvidence
    description: str = ""
    developer: str = ""
    #: Every counted name of the model, longest first (so a qualified name is matched
    #: before the bare word it contains).
    names: list[str] = dataclasses.field(default_factory=list)
    #: Names that may be flag-gated: the model's own name(s) and caller-given aliases.
    own_keys: set[str] = dataclasses.field(default_factory=set)
    #: OR group of paper titles and safe names: papers that certainly refer to the model.
    reference_clause: str = ""
    reference_count: int = 0
    decisions: list[TermDecision] = dataclasses.field(default_factory=list)
    #: for clause -> (safe, risky, flags, excludes) it was built from.
    states: dict[str, tuple[list[str], list[str], list[str], list[str]]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def context(self) -> str:
        dev = f" Developer: {self.developer}." if self.developer else ""
        return f"{(self.description or '').strip()[:400]}{dev}".strip()

    def decide(self, term: str, kind: str, decision: str, reason: str,
               count: int | None = None, **stats) -> None:
        self.decisions.append(TermDecision(
            term=term, kind=kind, decision=decision, count=count, reason=reason, stats=stats,
        ))

    def remember(self, clause: str, safe, risky, flags, excludes) -> None:
        self.states[clause] = (list(safe), list(risky), list(flags), list(excludes))


class QueryOptimizationUseCase:
    """Runs Phase 1–4: alias counts + risk → variant probes → light FP proxy → selection."""

    def __init__(
        self,
        *,
        dsl_port: DimensionsDslPort,
        research: DatasetResearchUseCase,
        agent: AgentPort,
        settings: Settings,
        extraction_client: ExtractionClient | None = None,
        proposal_repo=None,
    ) -> None:
        self._proposal_repo = proposal_repo
        self._dsl = dsl_port
        self._research = research
        self._agent = agent
        self._settings = settings
        self._extraction = extraction_client
        self._dims_calls = 0
        self._cap_extra = 0

    def _call_cap(self) -> int:
        """Dimensions call cap of the current run (0 = unlimited)."""
        cap = int(self._settings.optimize_max_dimensions_calls)
        return cap + self._cap_extra if cap > 0 else cap

    def _fulltext_enabled(self) -> bool:
        return bool(self._settings.fulltext_refinement_enabled)

    def _get_extraction(self) -> ExtractionClient:
        if self._extraction is None:
            self._extraction = ExtractionClient.for_target(
                self._settings, self._settings.probe_target
            )
        return self._extraction

    async def _probe_candidate(
        self, candidate: QueryCandidate, request: OptimizeRequest, *, target: int | None = None
    ) -> Evidence:
        """Run one extraction probe for *candidate*; the probe task is cleaned up by the client."""
        s = self._settings
        seed = candidate_to_seed(
            candidate,
            group_name=s.default_group_name,
            years=(s.default_years_start, s.default_years_end),
            publication_types=list(s.default_publication_types),
            filter_us_affiliation=s.default_filter_us_affiliation,
            extra={"home_url": request.dataset_url},
        )
        client = self._get_extraction()
        out = await asyncio.to_thread(
            functools.partial(
                client.run_probe_topup,
                seed,
                target=target if target is not None else s.fulltext_target_delivered,
                batch_size=s.fulltext_batch_size,
                max_batches=s.fulltext_max_batches,
                pull_cap=s.probe_pull_cap,
            )
        )
        ev = build_evidence(out.results)
        ev.target = out.target
        ev.batches = out.batches
        ev.stop_reason = out.stop_reason
        ev.budget_exhausted = out.budget_exhausted
        return ev

    def _budget_left(self) -> bool:
        cap = self._call_cap()
        if cap <= 0:
            return True
        return self._dims_calls < cap

    def _scope_budget_left(self) -> bool:
        cap = self._call_cap()
        if cap <= 0:
            return True
        return self._dims_calls + 2 <= cap

    def _phase1_budget_left(self) -> bool:
        """Room for one more phase-1 check while keeping ``refine_impact_budget_reserve``
        calls (plus the next gate's two) for the full-text refiner."""
        cap = int(self._settings.optimize_max_dimensions_calls)
        if cap <= 0:
            return True
        reserve = int(self._settings.refine_impact_budget_reserve)
        return self._dims_calls + reserve + 2 < cap

    async def _counted_execute(self, dsl: str):
        """Throttled Dimensions call that counts against the call budget."""
        if not self._budget_left():
            raise DimensionsDslError("Dimensions call budget exhausted")
        self._dims_calls += 1
        return await self._dsl.execute_dsl(dsl)

    def _make_eval_runner(self, loop: asyncio.AbstractEventLoop):
        """Sync callable for worker threads: routes the evaluator's DSL calls through ``self._dsl``."""

        def _run(dsl: str):
            return asyncio.run_coroutine_threadsafe(self._counted_execute(dsl), loop).result()

        return _run

    async def _exclude_impact(self, for_clause: str, term: str) -> int:
        """How many results of *for_clause* also contain *term* (what ``NOT term`` removes).

        Raises once the call budget is down to ``refine_impact_budget_reserve``, which is
        kept for the next gate evaluation."""
        cap = self._call_cap()
        reserve = int(self._settings.refine_impact_budget_reserve)
        if cap > 0 and self._dims_calls >= cap - reserve:
            raise DimensionsDslError("Dimensions budget reserve reached")
        self._dims_calls += 1
        clause = f'({for_clause}) AND \\"{inner_quoted_term(term)}\\"'
        result = await self._dsl.execute_dsl(build_for_clause_count_dsl(clause))
        return parse_total_count(result)

    async def _count(self, alias: str, *, search_in: str = "full_data") -> int:
        if not self._budget_left():
            return 0
        self._dims_calls += 1
        return await run_alias_count(self._dsl, alias, search_in=search_in)

    async def _count_alias(self, alias: str, notes_parts: list[str]) -> int | None:
        """Count one alias; ``None`` (with a note) when Dimensions cannot parse it.

        Only lexer/syntax errors drop the alias; transient and other errors propagate."""
        try:
            return await self._count(alias)
        except DimensionsDslError as exc:
            if not is_dsl_syntax_error(exc):
                raise
            logger.warning("Dropping alias %r: Dimensions could not parse it", alias)
            notes_parts.append(f"alias '{alias}' dropped: Dimensions could not parse it")
            return None

    async def _screen_research_flags(self, flags: list[str], notes_parts: list[str]) -> list[str]:
        """Drop research flags that admit almost any paper, while other flags remain: a
        bare short acronym ("GAGE" is also a gene family and a spelling of gauge) or a
        term with more than ``optimize_flag_max_hits`` Dimensions hits."""
        s = self._settings
        kept = list(flags)
        dropped: list[str] = []
        if s.optimize_drop_short_acronym_flags:
            short = [f for f in kept if is_short_acronym(f)]
            if short and len(short) < len(kept):
                kept = [f for f in kept if not is_short_acronym(f)]
                dropped += [f"{f} (bare short acronym)" for f in short]
        if s.optimize_flag_max_hits:
            broad: list[tuple[str, int]] = []
            for f in kept:
                if not self._phase1_budget_left():
                    notes_parts.append("research flag counts stopped: Dimensions budget kept for refinement")
                    break
                try:
                    c = await self._count(f)
                except DimensionsDslError as exc:
                    logger.warning("optimize: flag %r not counted: %s", f, exc)
                    continue
                if c > s.optimize_flag_max_hits:
                    broad.append((f, c))
            if broad and len(broad) < len(kept):
                keys = {f.casefold() for f, _c in broad}
                kept = [f for f in kept if f.casefold() not in keys]
                dropped += [f"{f} ({c} Dimensions hits)" for f, c in broad]
        if dropped:
            notes_parts.append("dropped research flag(s): " + "; ".join(dropped))
            logger.info("optimize: dropped research flag(s) %s", dropped)
        return kept

    async def _top_fields(self, name: str) -> list[str]:
        """Top three fields of research of the papers matching *name* (diagnostic)."""
        if not self._phase1_budget_left():
            return []
        dsl = (
            f'search publications in full_data for "\\"{inner_quoted_term(name)}\\"" '
            "return category_for limit 3"
        )
        try:
            result = await self._counted_execute(dsl)
        except DimensionsDslError as exc:
            logger.warning("optimize: fields of %r not read: %s", name, exc)
            return []
        data = getattr(result, "json", result)
        rows = data.get("category_for") if isinstance(data, dict) else None
        return [
            f"{x.get('name')} ({x.get('count')})"
            for x in (rows if isinstance(rows, list) else [])
            if isinstance(x, dict)
        ]

    async def _added_name_risk(
        self,
        name: str,
        count: int,
        c_full: int,
        has_fn: bool,
        flag_terms: list[str],
        notes_parts: list[str],
    ) -> Literal["safe", "risky"]:
        """Risk of a grounded or history name, from what its own papers are about.

        An acronym-shaped token ("BINEX", "COCONet") is flag-gated. Otherwise the share
        of its hits that also contain a flag term decides: at least
        ``optimize_grounded_safe_min_flag_share`` is searched alone, less is flag-gated.
        The share never rejects a name: a real collection name shared with unrelated
        networks ("Network of the Americas", 7%) is still worth searching with flags."""
        s = self._settings
        risk = _alias_risk_label(count, c_full, has_fn)
        if s.optimize_name_field_diagnostic:
            fields = await self._top_fields(name)
            if fields:
                notes_parts.append(f"fields of {name!r}: {', '.join(fields)}")
                logger.info("optimize: fields of %r: %s", name, fields)
        if not flag_terms:
            return risk
        if is_acronym_shaped_name(name):
            notes_parts.append(f"{name!r} flag-gated: acronym-shaped name")
            return "risky"
        if count <= 0:
            return risk
        if not self._phase1_budget_left():
            notes_parts.append(f"{name!r}: flag share not measured (Dimensions budget kept for refinement)")
            return risk
        # A flag sharing a word with the name ("UNAVCO" in "UNAVCO GAGE", "EarthScope
        # Consortium" next to "EarthScope GAGE") co-occurs with it by construction and is
        # no evidence of what the name's papers are about.
        name_tokens = content_tokens(name)
        independent = [f for f in flag_terms if not (content_tokens(f) & name_tokens)]
        if not independent:
            notes_parts.append(f"{name!r}: every flag shares a word with it; flag share not measured")
            return risk
        group = " OR ".join(f'\\"{inner_quoted_term(f)}\\"' for f in independent)
        clause = f'\\"{inner_quoted_term(name)}\\" AND ({group})'
        try:
            with_flags = parse_total_count(
                await self._counted_execute(build_for_clause_count_dsl(clause))
            )
        except DimensionsDslError as exc:
            logger.warning("optimize: flag share of %r unknown: %s", name, exc)
            return risk
        share = with_flags / count
        label: Literal["safe", "risky"] = (
            "safe" if share >= s.optimize_grounded_safe_min_flag_share else "risky"
        )
        notes_parts.append(
            f"{name!r}: {with_flags}/{count} hits ({share:.0%}) contain a flag term; {label}"
        )
        logger.info("optimize: flag share of %r = %d/%d -> %s", name, with_flags, count, label)
        return label

    async def _ground_names(
        self,
        request: OptimizeRequest,
        rec: DatasetRecord,
        alias_entries: list[AliasCountEntry],
        flag_terms: list[str],
        notes_parts: list[str],
    ) -> list[tuple[str, int]]:
        """Proper names for a descriptive title, each with its Dimensions count.

        Candidates come from the terms evaluation's suggestions and one structured LLM
        call over the description and a web search; see ``adapters.name_grounding``."""
        s = self._settings
        name = request.dataset_name
        suggested = list(rec.terms_evaluation.suggested_dataset_names) if rec.terms_evaluation else []
        try:
            web = await asyncio.to_thread(web_search, f"{name} data network archive name")
        except Exception as exc:  # web context is best effort
            logger.warning("grounding: web search failed: %s", exc)
            web = ""
        prompt = grounding_prompt(
            name, rec.description or "", [(e.alias, e.count) for e in alias_entries], flag_terms, web
        )
        try:
            raw = await asyncio.to_thread(self._agent.get_structured, prompt, GROUNDING_TOOL)
        except Exception as exc:
            logger.warning("grounding: LLM call failed: %s", exc)
            raw = {}
        llm_names = raw.get("names") if isinstance(raw, dict) else None
        llm_names = [n for n in llm_names if isinstance(n, str)] if isinstance(llm_names, list) else []
        # Names defined with an acronym in the web results or description need more hits:
        # no model vouched for them.
        # A spelled-out acronym of the title's own words ("Global Navigation Satellite
        # System" for GNSS) is a technology, not the collection's name.
        title_words = {w.casefold() for w in re.split(r"[\W_]+", name) if w}
        defined: list[str] = []
        spelled: list[str] = []
        for d_name, acronym in defined_names(f"{rec.description or ''}\n{web}"):
            (spelled if acronym.casefold() in title_words else defined).append(d_name)
        vouched = {n.casefold() for n in suggested + llm_names}
        unvouched = {n.casefold() for n in defined} - vouched
        candidates = suggested + llm_names + defined
        kept, rejected = screen_candidates(
            candidates,
            dataset_name=name,
            known_aliases=[e.alias for e in alias_entries],
            flag_terms=flag_terms,
            limit=s.optimize_grounding_max_candidates,
        )
        found: list[tuple[str, int]] = []
        for cand in kept:
            if not self._budget_left():
                rejected.append(f"{cand} (Dimensions call budget)")
                continue
            n = await self._count_alias(cand, notes_parts)
            if n is None:
                continue
            floor = (
                s.optimize_grounding_min_hits
                if cand.casefold() in unvouched
                else s.optimize_grounding_min_alias_hits
            )
            if n < floor:
                rejected.append(f"{cand} ({n} Dimensions hits)")
                continue
            ceiling = s.optimize_grounding_unvouched_max_hits
            if cand.casefold() in unvouched and ceiling and n > ceiling:
                rejected.append(f"{cand} ({n} Dimensions hits: too common for an unvouched name)")
                continue
            found.append((cand, n))
        added = ", ".join(f"{a!r} ({n} hits)" for a, n in found) or "none"
        notes_parts.append(f"grounding: title has no Dimensions hits; added {added}")
        rejected += [f"{n} (spells out a title acronym)" for n in spelled]
        if rejected:
            notes_parts.append("grounding: rejected " + "; ".join(rejected))
        logger.info("grounding: added=%s rejected=%s", found, rejected)
        return found

    async def _name_history(
        self,
        request: OptimizeRequest,
        rec: DatasetRecord,
        alias_entries: list[AliasCountEntry],
        flag_terms: list[str],
        notes_parts: list[str],
    ) -> tuple[list[tuple[str, int]], list[str]]:
        """Former and successor names of the collection (each with its Dimensions count)
        and of its organizations (new flag terms); see ``adapters.name_history``."""
        s = self._settings
        name = request.dataset_name
        # The collection's own names lead; an organization-only alias would search the
        # organization's history instead.
        leading = sorted(
            [
                (e.alias, e.count)
                for e in alias_entries
                if e.count > 0 and not is_organization_alias(e.alias, flag_terms, name)
            ],
            key=lambda x: -x[1],
        )[:3] or [(name, 0)]
        # The plain search finds the history; with an organization added it stays on this
        # collection when the name is shared ("Network of the Americas" is also the name of
        # unrelated networks). Junk from the plain search is filtered by the anchors below.
        queries = [f'"{lead}" formerly OR renamed OR merged OR successor' for lead, _n in leading[:2]]
        if flag_terms:
            queries.append(f'"{leading[0][0]}" "{flag_terms[0]}" history formerly merged')
        web_parts: list[str] = []
        for query in queries:
            try:
                web_parts.append(await asyncio.to_thread(web_search, query))
            except Exception as exc:  # web context is best effort
                logger.warning("name history: web search failed: %s", exc)
        web = "\n".join(web_parts)
        yr = rec.years_range
        years = f"{yr.start_year}-{yr.end_year}" if yr else "recent years"
        try:
            raw = await asyncio.to_thread(
                self._agent.get_structured,
                history_prompt(name, leading, flag_terms, years, web),
                HISTORY_TOOL,
            )
        except Exception as exc:
            logger.warning("name history: LLM call failed: %s", exc)
            raw = {}
        raw = raw if isinstance(raw, dict) else {}

        def _items(key: str) -> list[dict]:
            v = raw.get(key)
            return [x for x in v if isinstance(x, dict) and isinstance(x.get("name"), str)] if isinstance(v, list) else []

        relation = {x["name"].strip().casefold(): x for x in _items("dataset_names")}
        llm_names = [x["name"].strip() for x in _items("dataset_names")]
        # Web phrases count only in results that mention a leading name or an organization.
        phrased = history_phrases(web, [a for a, _n in leading] + list(flag_terms))
        unvouched = {n.casefold() for n in phrased} - set(relation)
        kept, rejected = screen_candidates(
            llm_names + phrased,
            dataset_name=name,
            known_aliases=[e.alias for e in alias_entries],
            flag_terms=flag_terms,
            limit=s.optimize_name_history_max_candidates,
        )
        found: list[tuple[str, int]] = []
        for cand in kept:
            if not self._budget_left():
                rejected.append(f"{cand} (Dimensions call budget)")
                continue
            n = await self._count_alias(cand, notes_parts)
            if n is None:
                continue
            floor = (
                s.optimize_grounding_min_hits
                if cand.casefold() in unvouched
                else s.optimize_grounding_min_alias_hits
            )
            ceiling = s.optimize_grounding_unvouched_max_hits
            if n < floor:
                rejected.append(f"{cand} ({n} Dimensions hits)")
            elif cand.casefold() in unvouched and ceiling and n > ceiling:
                rejected.append(f"{cand} ({n} Dimensions hits: too common for an unvouched name)")
            else:
                found.append((cand, n))

        # An organization's other name is a context term only when it renames a listed one
        # or is what a listed one merged into. A partner absorbed by the same merger
        # ("merged_from": IRIS next to UNAVCO in EarthScope) may work in another field.
        listed = {f.casefold() for f in flag_terms}
        new_flags: list[str] = []
        for x in _items("organizations"):
            org, of = x["name"].strip(), str(x.get("of") or "").strip()
            if (
                x.get("relation") in ("former", "successor", "merged_into")
                and of.casefold() in listed
                and org.casefold() not in listed
                and org.casefold() not in BROAD_FUNDER_FLAGS
                and any(ch.isupper() for ch in org)
                and not is_short_acronym(org)
                and org.casefold() not in {f.casefold() for f in new_flags}
            ):
                new_flags.append(org)

        def _why(cand: str) -> str:
            x = relation.get(cand.casefold())
            if not x:
                return "named as a former/successor in web results"
            year = f", {x['year']}" if isinstance(x.get("year"), int) else ""
            return f"{x.get('relation', 'related')}{year}"

        if found:
            notes_parts.append(
                "name history: added "
                + ", ".join(f"{a!r} ({_why(a)}; {n} hits)" for a, n in found)
            )
        if new_flags:
            notes_parts.append(f"name history: added organization name(s) {new_flags}")
        if rejected:
            notes_parts.append("name history: rejected " + "; ".join(rejected))
        logger.info("name history: added=%s flags=%s rejected=%s", found, new_flags, rejected)
        return found, new_flags

    async def _refine_count(self, term: str) -> int:
        """Refiner-only count: like ``_exclude_impact`` it raises (instead of returning 0)
        once the budget is down to ``refine_impact_budget_reserve``."""
        cap = self._call_cap()
        reserve = int(self._settings.refine_impact_budget_reserve)
        if cap > 0 and self._dims_calls >= cap - reserve:
            raise DimensionsDslError("Dimensions budget reserve reached")
        self._dims_calls += 1
        return await run_alias_count(self._dsl, term, search_in="full_data")

    async def _try_suffix_promote(
        self,
        alias: str,
        bare_count: int,
        *,
        notes_parts: list[str],
        suffix_checked: int,
        c_full: int | None = None,
        has_fn: bool = True,
    ) -> tuple[str | None, int]:
        """Return (suffixed safe form, updated suffix_checked) or (None, suffix_checked)."""
        if not self._settings.optimize_suffix_check_enabled:
            return None, suffix_checked
        if suffix_checked >= self._settings.optimize_suffix_check_max_aliases:
            return None, suffix_checked
        if bare_count <= 0 or not self._budget_left():
            return None, suffix_checked
        ratio = (
            float(self._settings.optimize_suffix_ratio_short_acronym)
            if is_short_acronym(alias)
            else float(self._settings.optimize_suffix_ratio_threshold)
        )
        counts: dict[str, int] = {}
        suffixed = await check_suffix_necessity(
            self._dsl,
            alias,
            bare_count=bare_count,
            ratio_threshold=ratio,
            counts_out=counts,
        )
        self._dims_calls += len(_SUFFIX_CANDIDATES)
        suffix_checked += 1
        if suffixed and c_full is not None and suffixed in counts:
            # A promoted form keeps the risk rules: it must pass them on its own count.
            if _alias_risk_label(counts[suffixed], c_full, has_fn) == "risky":
                notes_parts.append(
                    f"suffix_rejected: {suffixed!r} still risky "
                    f"({counts[suffixed]} hits); {alias!r} stays risky"
                )
                return None, suffix_checked
        if suffixed:
            notes_parts.append(f"suffix_promoted: {alias!r} -> {suffixed!r}")
        return suffixed, suffix_checked

    async def _recall_check(
        self,
        request: OptimizeRequest,
        current: QueryCandidate,
        *,
        main_names: list[str],
        known_aliases: list[str],
        flag_terms: list[str],
        evidence_against: set[str],
    ) -> tuple[QueryCandidate, list[TermDiff], list[str], float | None]:
        """Propose missing canonical names as safe aliases; returns
        ``(candidate, add diffs, notes, min complement precision of promoted terms)``.

        Every note starts with ``recall check:``. A candidate is promoted only when a
        complement probe (papers matching it but not the current query) is precise enough."""
        s = self._settings
        notes: list[str] = []
        diffs: list[TermDiff] = []
        precisions: list[float] = []
        eligible, rejected = recall_candidates(
            current, main_names=main_names, known_aliases=known_aliases,
            flag_terms=flag_terms, evidence_against=evidence_against,
        )
        if rejected:
            more = f"; and {len(rejected) - 8} more" if len(rejected) > 8 else ""
            notes.append("recall check: not eligible: " + "; ".join(rejected[:8]) + more)
        if not eligible:
            return current, diffs, notes, None

        # Count stage: the Dimensions hits each candidate would add (tier order, bounded).
        counted: list[tuple[RecallCandidate, int]] = []
        count_cap = 3 * s.fulltext_recall_max_candidates
        for cand in eligible[count_cap:]:
            notes.append(f"recall check: '{cand.term}' not evaluated: over the counting cap")
        for cand in eligible[:count_cap]:
            if not self._budget_left():
                notes.append("recall check stopped: Dimensions call budget exhausted")
                break
            try:
                k = parse_total_count(
                    await self._counted_execute(
                        build_for_clause_count_dsl(complement_clause(cand.term, current))
                    )
                )
            except DimensionsDslError as exc:
                why = ("Dimensions rejected the complement query" if is_dsl_syntax_error(exc)
                       else f"Dimensions count failed: {short_dsl_error(str(exc))}")
                notes.append(f"recall check: '{cand.term}' rejected: {why}")
                continue
            if k < s.fulltext_recall_min_evidence:
                notes.append(f"recall check: '{cand.term}' rejected: adds only {k} Dimensions hits")
            elif k > s.refine_flag_max_hits:
                notes.append(f"recall check: '{cand.term}' rejected: {k} hits, too broad")
            else:
                counted.append((cand, k))
        counted.sort(key=lambda ck: (ck[0].tier, -ck[1]))
        chosen = counted[: s.fulltext_recall_max_candidates]
        for cand, k in counted[s.fulltext_recall_max_candidates :]:
            notes.append(f"recall check: '{cand.term}' not evaluated: over the candidate cap")

        target = max(s.fulltext_recall_target_delivered, s.fulltext_recall_min_evidence)
        for cand, k in chosen:
            term = cand.term
            if diffs:
                # Measure beyond the aliases promoted so far, so the gains stay additive.
                if not self._budget_left():
                    notes.append("recall check stopped: Dimensions call budget exhausted")
                    break
                try:
                    k = parse_total_count(
                        await self._counted_execute(
                            build_for_clause_count_dsl(complement_clause(term, current))
                        )
                    )
                except DimensionsDslError as exc:
                    notes.append(
                        f"recall check: '{term}' not evaluated: Dimensions count failed: "
                        f"{short_dsl_error(str(exc))}"
                    )
                    continue
                if k < s.fulltext_recall_min_evidence:
                    notes.append(
                        f"recall check: '{term}' rejected: adds only {k} hits beyond "
                        f"{', '.join(repr(d.term) for d in diffs)}"
                    )
                    continue
            try:
                evidence = await self._probe_candidate(
                    complement_candidate(term, current), request, target=target
                )
            except ExtractionError as exc:
                notes.append(f"recall check: '{term}' not evaluated: {exc}; remaining candidates skipped")
                break
            except ValueError as exc:
                notes.append(f"recall check: '{term}' not evaluated: {exc}")
                continue
            ok, reason, stats = judge_complement(term, evidence, k, s)
            stats = {**stats, "source": cand.source,
                     "complement_clause": complement_clause(term, current)}
            if not ok:
                notes.append(f"recall check: '{term}' rejected: {reason}")
                continue
            was_gated = any(r.lower() == term.lower() for r in current.risky)
            if was_gated:
                stats["source"] = "alias"
                reason += " (was flag-gated)"
            current = dataclasses.replace(
                current,
                safe=current.safe + [term],
                risky=[r for r in current.risky if r.lower() != term.lower()],
            )
            precisions.append(float(stats["complement_precision"]))
            diffs.append(TermDiff(
                kind="alias", term=term, action="add", status="verified",
                reason=reason, stats=stats, evidence=evidence_quotes(term, evidence),
            ))
            notes.append(f"recall check: promoted '{term}': {reason}")
        return current, diffs, notes, (min(precisions) if precisions else None)

    # ------------------------------------------------------------------ AI models

    async def _clause_count(self, clause: str) -> int | None:
        """Hits of a boolean clause; ``None`` when the call budget is spent or
        Dimensions cannot parse the clause. Other errors propagate."""
        if not clause or not self._budget_left():
            return None
        try:
            return parse_total_count(
                await self._counted_execute(build_for_clause_count_dsl(clause))
            )
        except DimensionsDslError as exc:
            if not is_dsl_syntax_error(exc):
                raise
            return None

    async def _model_exclude_check(self, model: _ModelRun, term: str) -> tuple[bool, str, dict]:
        """An exclude is kept only when it removes (almost) none of the papers that
        certainly refer to the model."""
        if not model.reference_clause or model.reference_count <= 0:
            return False, "no precise name or paper title to validate it against", {}
        hit = await self._clause_count(f"{model.reference_clause} AND {_or_group([term])}")
        if hit is None:
            return False, "not measured (Dimensions call budget or unparseable term)", {}
        loss = hit / model.reference_count
        stats = {"precise_matches_removed": hit, "precise_matches": model.reference_count}
        if loss > self._settings.ai_model_exclude_max_anchor_loss:
            return (
                False,
                f"would remove {hit} of {model.reference_count} precise matches ({loss:.0%})",
                stats,
            )
        return True, f"removes {hit} of {model.reference_count} precise matches ({loss:.1%})", stats

    async def _mine_model_context(
        self, model: _ModelRun, model_names: list[str]
    ) -> list[tuple[str, int]]:
        """Context-term candidates from the abstracts of the precise matches (one call)."""
        s = self._settings
        n = int(s.ai_model_context_mining_sample)
        if not model.reference_clause or n <= 0 or s.ai_model_max_mined_context_terms <= 0:
            return []
        if not self._budget_left():
            return []
        try:
            result = await self._counted_execute(
                build_search_publications_paged_dsl(
                    model.reference_clause, limit=n, fields="title+abstract"
                )
            )
        except DimensionsDslError as exc:
            logger.warning("model research: context mining failed: %s", exc)
            return []
        texts = [
            f"{publication_title(p)} {p.get('abstract') or ''}"
            for p in publications_from_result(result)
        ]
        return mine_context_terms(
            texts,
            model_names=model_names,
            min_docs=max(3, len(texts) // 50),
            limit=s.ai_model_max_mined_context_terms,
        )

    async def _refine_model_terms(
        self,
        model: _ModelRun,
        ev: dict,
        safe: list[str],
        risky: list[str],
        flags: list[str],
        excludes: list[str],
        notes_parts: list[str],
        iteration: int,
        for_clause: str = "",
    ) -> tuple[list[str], list[str], list[str]] | None:
        """One refinement step from the abstract check's false positives.

        A safe name with two or more false positives, and no more genuine mentions than
        false ones, stops being safe: it becomes
        flag-gated when it is the model's own name and flags exist, and is dropped
        otherwise. An exclude phrase the LLM derives from the false positives is kept only
        if it occurs in one of their snippets, passes ``_model_exclude_check`` and removes
        a small part of the current result (a phrase that removes much is describing the
        topic, not the false positive). Returns ``None`` when nothing changes."""
        fp_hits = [h for h in (ev.get("fp_hits") or []) if isinstance(h, dict)]
        if not fp_hits:
            return None
        per_alias: dict[str, int] = {}
        for h in fp_hits:
            key = str(h.get("matched_alias") or "").casefold()
            per_alias[key] = per_alias.get(key, 0) + 1
        genuine = {
            str(k).casefold(): int(v)
            for k, v in (ev.get("genuine_by_alias") or {}).items()
            if isinstance(v, int)
        }
        new_safe: list[str] = []
        new_risky = list(risky)
        changed = False
        for alias in safe:
            n_fp = per_alias.get(alias.casefold(), 0)
            # At least two false positives and at least as many as genuine mentions.
            if n_fp < 2 or n_fp < genuine.get(alias.casefold(), 0):
                new_safe.append(alias)
                continue
            changed = True
            if flags and alias.casefold() in model.own_keys:
                new_risky.append(alias)
                notes_parts.append(
                    f"gate_iter{iteration}: {alias!r} flag-gated ({n_fp} false positives in abstracts)"
                )
            else:
                notes_parts.append(
                    f"gate_iter{iteration}: {alias!r} dropped ({n_fp} false positives in abstracts)"
                )
        try:
            suggested = await asyncio.to_thread(
                derive_exclude_terms_from_fps,
                self._agent,
                dataset_name=model.name,
                fp_hits=fp_hits,
            )
        except Exception as exc:  # the LLM step is best effort
            logger.warning("model refine: exclude derivation failed: %s", exc)
            suggested = []
        new_excl = list(excludes)
        taken = {t.casefold() for t in [*excludes, *flags, *model.names]}
        tried = {d.term.casefold() for d in model.decisions if d.kind == "exclude"}
        snippets = " ".join(str(h.get("snippet") or "") for h in fp_hits).casefold()
        total = int(ev.get("publications_total") or 0)
        max_removed = max(1, int(total * self._settings.ai_model_refine_exclude_max_share))
        added = 0
        for term in suggested[: self._settings.ai_model_max_exclude_terms]:
            key = term.casefold()
            if key in taken or key in tried or is_short_acronym(term):
                continue
            tried.add(key)
            if added >= self._settings.ai_model_refine_max_excludes:
                break
            if key not in snippets:
                model.decide(term, "exclude", "rejected",
                             "from abstract false positives; not a phrase of any of them")
                continue
            ok, reason, stats = await self._model_exclude_check(model, term)
            if ok and for_clause:
                removed = await self._clause_count(f"({for_clause}) AND {_or_group([term])}")
                stats["removed"] = removed
                if removed is None or removed > max_removed:
                    ok = False
                    reason = (
                        f"would remove {removed} of {total} results: too broad for a phrase "
                        "taken from a few false positives"
                        if removed is not None else "effect not measured"
                    )
                elif removed == 0:
                    ok, reason = False, "removes nothing from the query"
            added += 1 if ok else 0
            model.decide(
                term, "exclude", "exclude" if ok else "rejected",
                f"from abstract false positives; {reason}", **stats,
            )
            if ok:
                new_excl.append(term)
                changed = True
        if not changed:
            return None
        return new_safe, dedupe_strings_ci_preserve_order(new_risky), new_excl

    async def _execute_ai_model(
        self, request: OptimizeRequest, notes_parts: list[str], run_id: str, t0: float
    ) -> QueryOptimizationRecord:
        """Query optimization for an AI model.

        Candidate terms come from ``adapters.ai_model_research``; every one is measured in
        Dimensions before it is used:

        * a paper title is an anchor only when the model card cites it and it names the
          model or mostly co-occurs with the model name;
        * a name is safe, flag-gated or rejected by its shape and by the share of its hits
          that quote an anchor title (``classify_model_name``);
        * an exclude must leave the precise matches (titles and safe names) intact;
        * a context term becomes a flag only when most papers it admits for the gated
          name are precise matches already.

        Phases 2-4 are shared with datasets (``_select_and_finish``)."""
        s = self._settings
        name = request.dataset_name
        family = request.model_family or name
        self._dims_calls = 0
        self._cap_extra = int(s.ai_model_extra_dimensions_calls)
        phase1 = "phase1_alias_counting"

        hub = await asyncio.to_thread(fetch_hub_evidence, request.hugging_face_id)
        model = _ModelRun(name=name, family=family, hub=hub)
        for err in hub.errors:
            notes_parts.append(f"hub: {err}")
        if hub.found:
            notes_parts.append(
                f"hub: {hub.hugging_face_id} owner={hub.developer or hub.author!r} "
                f"papers={hub.paper_titles or 'none linked'}"
            )
        try:
            web = await asyncio.to_thread(
                web_search, f'"{name}" {hub.developer or hub.author} AI model paper'.replace("  ", " ")
            )
        except Exception as exc:  # web context is best effort
            logger.warning("model research: web search failed: %s", exc)
            web = ""
        try:
            raw = await asyncio.to_thread(
                self._agent.get_structured,
                model_terms_prompt(name, family, hub, list(request.dataset_names), web),
                MODEL_TERMS_TOOL,
            )
        except Exception as exc:
            logger.warning("model research: LLM call failed: %s", exc)
            notes_parts.append(f"model research: LLM call failed ({short_dsl_error(str(exc))})")
            raw = {}
        llm = dict(raw) if isinstance(raw, dict) else {}
        if not llm:
            notes_parts.append("model research: no LLM terms; using Hub and name heuristics only")

        def _strs(v: object) -> list[str]:
            return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []

        # Caller-given flag and exclude terms are candidates too: measured like the rest.
        llm["context_terms"] = list(request.flag_terms) + _strs(llm.get("context_terms"))
        llm["other_meanings"] = list(request.exclude_terms) + _strs(llm.get("other_meanings"))
        cands = assemble_candidates(
            model_name=name,
            family=family,
            given_aliases=list(request.dataset_names),
            hub=hub,
            llm=llm,
            max_names=s.ai_model_max_names,
            max_context=s.ai_model_max_context_terms,
            max_excludes=s.ai_model_max_exclude_terms,
        )
        model.description = cands.description
        model.developer = cands.developer
        for kind, term, reason in cands.rejected:
            model.decide(term, kind, "rejected", reason)

        def q(*terms: str) -> str:
            return _or_group(list(terms))

        def fail(error: str) -> QueryOptimizationRecord:
            return QueryOptimizationRecord(
                success=False,
                failed_phase=phase1,
                errors=[error],
                entity_type="ai_model",
                term_decisions=model.decisions,
                notes=" ".join(notes_parts),
            )

        base_keys = {name.casefold(), family.casefold()}
        given_keys = {a.casefold() for a in request.dataset_names}
        try:
            # 1. Paper titles: anchors for everything measured below.
            titles: list[str] = []
            for title in cands.paper_titles:
                c = await self._clause_count(q(title))
                if not c:
                    model.decide(title, "paper_title", "rejected",
                                 "0 Dimensions hits" if c == 0 else "not measured", c)
                    continue
                from_hub = title.casefold() in cands.hub_title_keys
                names_model = contains_word(title, name) or contains_word(title, family)
                if not from_hub:
                    # A title the LLM recalls proves nothing: "Aurora kinase inhibitors
                    # in cancer therapy" also names "Aurora". Only titles cited on the
                    # model card (or linked from it) anchor the measurements.
                    model.decide(
                        title, "paper_title", "rejected",
                        "not cited on the model card: an unverified title cannot be an anchor",
                        c,
                    )
                    continue
                if names_model:
                    reason = "cited on the model card; the title names the model"
                else:
                    both = await self._clause_count(f"{q(title)} AND {q(name, family)}")
                    if both is None:
                        model.decide(title, "paper_title", "rejected",
                                     "not measured (Dimensions call budget)", c)
                        continue
                    share = both / c
                    if c > s.ai_model_title_max_hits:
                        model.decide(title, "paper_title", "rejected",
                                     f"cited on the model card, but quoted by {c} papers: too "
                                     "widely cited to single out this model", c)
                        continue
                    if share < s.ai_model_title_min_name_share:
                        model.decide(
                            title, "paper_title", "rejected",
                            f"cited on the model card, but only {share:.0%} of the {c} papers "
                            f"quoting it name {name!r}", c, name_share=round(share, 3),
                        )
                        continue
                    reason = (
                        f"cited on the model card; {share:.0%} of the papers quoting it "
                        f"name {name!r}"
                    )
                titles.append(title)
                model.decide(title, "paper_title", "anchor", reason, c)
            title_group = q(*titles) if titles else ""

            # 2. Names: count, then classify by shape and by title confirmation.
            safe: list[str] = []
            gate: list[tuple[str, int, str]] = []
            counts: dict[str, int] = {}
            for nm, _kind in cands.names:
                c = await self._clause_count(q(nm))
                if c is None:
                    model.decide(nm, "alias", "rejected", "not measured (budget or unparseable)")
                    continue
                if c == 0:
                    model.decide(nm, "alias", "rejected", "0 Dimensions hits", 0)
                    continue
                counts[nm] = c
                confirmation: float | None = None
                if title_group:
                    both = await self._clause_count(f"{q(nm)} AND {title_group}")
                    confirmation = (both / c) if both is not None else None
                verdict, reason = classify_model_name(
                    nm,
                    count=c,
                    confirmation=confirmation,
                    has_titles=bool(title_group),
                    developer=model.developer,
                    safe_share=s.ai_model_safe_min_context_share,
                    context_share=s.ai_model_context_min_anchor_precision,
                    plain_word_max_hits=s.ai_model_plain_word_max_safe_hits,
                )
                stats = {} if confirmation is None else {"title_confirmation": round(confirmation, 3)}
                own = nm.casefold() in base_keys or nm.casefold() in given_keys
                if own:
                    model.own_keys.add(nm.casefold())
                if verdict == "safe":
                    safe.append(nm)
                    model.decide(nm, "alias", "safe", reason, c, **stats)
                    continue
                if own:
                    gate.append((nm, c, reason))
                    continue
                cover = next((g for g, _c, _r in gate if contains_word(nm, g)), None)
                if cover:
                    model.decide(nm, "alias", "rejected",
                                 f"{reason}; already covered by flag-gated {cover!r}", c, **stats)
                else:
                    model.decide(nm, "alias", "rejected",
                                 f"{reason}; not confirmed as a name of this model", c, **stats)
            if s.ai_model_paper_titles_as_aliases:
                safe.extend(titles)

            # 3. Reference set: papers that certainly refer to the model.
            reference = dedupe_strings_ci_preserve_order(titles + safe)
            model.reference_clause = q(*reference) if reference else ""
            model.reference_count = (
                await self._clause_count(model.reference_clause) or 0 if reference else 0
            )

            # 4. Excludes (other meanings of the name).
            excludes: list[str] = []
            for term in cands.other_meanings:
                ok, reason, stats = await self._model_exclude_check(model, term)
                model.decide(term, "exclude", "exclude" if ok else "rejected", reason, **stats)
                if ok:
                    excludes.append(term)
            not_x = f" NOT {q(*excludes)}" if excludes else ""

            # 5. Context flags, only when a name needs gating.
            flags: list[str] = []
            mined: dict[str, int] = {}
            gated = [g for g, _c, _r in gate]
            if not gated:
                for term in cands.context_terms:
                    model.decide(term, "flag", "rejected", "not needed: no name requires gating")
            else:
                g_group = q(*gated)
                context_terms = list(cands.context_terms)
                taken = {t.casefold() for t in context_terms} | {
                    d.term.casefold() for d in model.decisions if d.kind == "flag"
                }
                for term, n_docs in await self._mine_model_context(model, gated + safe):
                    if term.casefold() not in taken:
                        taken.add(term.casefold())
                        context_terms.append(term)
                        mined[term.casefold()] = n_docs
                for term in context_terms:
                    if not model.reference_clause or model.reference_count <= 0:
                        # Nothing certain to measure against: an unverified flag would
                        # let the ambiguous name through on trust.
                        model.decide(term, "flag", "rejected",
                                     "cannot be verified: no precise name or cited paper title")
                        continue
                    admitted = await self._clause_count(f"({g_group} AND {q(term)}){not_x}")
                    if not admitted:
                        model.decide(
                            term, "flag", "rejected",
                            f"never occurs with {gated}" if admitted == 0
                            else "not measured (Dimensions call budget or unparseable term)",
                            admitted,
                        )
                        continue
                    confirmed = await self._clause_count(
                        f"({model.reference_clause} AND {g_group} AND {q(term)}){not_x}"
                    )
                    if confirmed is None:
                        model.decide(term, "flag", "rejected",
                                     "precision not measured (Dimensions call budget)", admitted)
                        continue
                    precision = confirmed / admitted
                    stats = {"admitted": admitted, "confirmed": confirmed,
                             "anchor_precision": round(precision, 3)}
                    detail = (
                        f"admits {admitted} papers with {gated}; {confirmed} "
                        f"({precision:.0%}) are precise matches"
                    )
                    if term.casefold() in mined:
                        detail = (
                            f"found in {mined[term.casefold()]} abstracts of precise matches; "
                            + detail
                        )
                    passes = (
                        precision >= s.ai_model_flag_min_anchor_precision
                        and confirmed >= s.ai_model_flag_min_anchor_hits
                    )
                    own_hits = await self._clause_count(q(term)) if passes else None
                    if passes and (own_hits is None or own_hits > s.ai_model_flag_max_hits):
                        # Shape cannot tell "AI" or "Weather-Forecasting" from a coined
                        # name; its own count can.
                        model.decide(
                            term, "flag", "context",
                            f"{detail}, but "
                            + (
                                f"it has {own_hits} hits on its own: too common to gate an "
                                "ambiguous name"
                                if own_hits is not None
                                else "its own hit count was not measured, so it gates nothing"
                            ),
                            admitted, **stats, own_hits=own_hits,
                        )
                    elif passes:
                        stats["own_hits"] = own_hits
                        flags.append(term)
                        model.decide(term, "flag", "flag", detail, admitted, **stats)
                    elif precision >= s.ai_model_context_min_anchor_precision:
                        model.decide(term, "flag", "context",
                                     f"{detail}: too loose to gate an ambiguous name",
                                     admitted, **stats)
                    else:
                        model.decide(term, "flag", "rejected",
                                     f"{detail}: mostly other meanings", admitted, **stats)

            # 6. Keep only excludes that remove something from the query as built.
            positive = build_for_clause(
                safe=safe, variant="V3", risky=gated if flags else [], flag_terms=flags
            )
            kept_excludes: list[str] = []
            for term in excludes:
                removed = await self._clause_count(f"({positive}) AND {q(term)}") if positive else 0
                decision = next(
                    d for d in model.decisions if d.kind == "exclude" and d.term == term
                )
                decision.count = removed
                if removed:
                    kept_excludes.append(term)
                    decision.reason = f"removes {removed} papers from the query; {decision.reason}"
                else:
                    decision.decision = "rejected"
                    decision.reason = (
                        "removes nothing from the query as built"
                        if removed == 0 else "effect not measured"
                    ) + f"; {decision.reason}"
            excludes = kept_excludes

            risky: list[str] = []
            for g, c, reason in gate:
                if flags:
                    risky.append(g)
                    model.decide(g, "alias", "gated",
                                 f"{reason}; searched only together with a context flag", c)
                else:
                    model.decide(g, "alias", "rejected",
                                 f"{reason}; no context flag qualified, so it is not searched", c)
        except DimensionsDslError as e:
            return fail(str(e))

        safe = dedupe_strings_ci_preserve_order(safe)
        if not safe and not risky:
            return fail(
                f"No name of {name!r} can be searched precisely: every candidate is ambiguous "
                "or has no Dimensions hits. Add a qualified alias (dataset_names) or a "
                "Hugging Face id with a linked paper."
            )
        model.names = sorted(
            dedupe_strings_ci_preserve_order(safe + risky), key=lambda n: (-len(n), n)
        )
        alias_entries = [
            AliasCountEntry(alias=a, count=counts.get(a, 0), risk="safe") for a in safe
            if a in counts
        ] + [AliasCountEntry(alias=a, count=counts.get(a, 0), risk="risky") for a in risky]
        logger.info(
            "optimize_phase run_id=%s phase=phase1_ai_model dims=%s safe=%s risky=%s flags=%s "
            "excludes=%s",
            run_id, self._dims_calls, safe, risky, flags, excludes,
        )

        out = await self._select_and_finish(
            request, rec=None, safe=safe, risky=risky, flag_terms=flags,
            exclude_terms=excludes, alias_entries=alias_entries, notes_parts=notes_parts,
            c_full=counts.get(name, 0), desc_for_signal3=model.description[:2000],
            run_id=run_id, t0=t0, model=model,
        )
        out.entity_type = "ai_model"
        out.term_decisions = model.decisions
        if not out.success or not out.for_clause:
            return out
        return await self._finish_ai_model(request, model, out)

    async def _finish_ai_model(
        self, request: OptimizeRequest, model: _ModelRun, out: QueryOptimizationRecord
    ) -> QueryOptimizationRecord:
        """Align the record with the selected clause and build the extraction seed."""
        s = self._settings
        state = model.states.get(out.for_clause or "")
        if state is None:  # every evaluated clause is registered; keep the record's terms
            state = (list(out.aliases.safe), list(out.aliases.risky),
                     list(out.flag_terms), list(out.exclusion_terms))
        safe, risky, flags, excludes = state
        out.aliases = QueryOptimizationAliases(safe=safe, risky=risky)
        out.flag_terms, out.exclusion_terms = list(flags), list(excludes)
        risk = {a.casefold(): "safe" for a in safe} | {a.casefold(): "risky" for a in risky}
        out.alias_counts = [
            e.model_copy(update={"risk": risk[e.alias.casefold()]})
            for e in out.alias_counts if e.alias.casefold() in risk
        ]

        # Decisions describe the selected query, not a later refinement that lost.
        safe_k = {a.casefold() for a in safe}
        risky_k = {a.casefold() for a in risky}
        excl_k = {a.casefold() for a in excludes}
        for d in model.decisions:
            key = d.term.casefold()
            if d.kind == "alias" and d.decision in ("safe", "gated"):
                if key in safe_k:
                    d.decision = "safe"
                elif key in risky_k:
                    if d.decision == "safe":
                        d.reason += "; flag-gated after false positives in abstracts"
                    d.decision = "gated"
                else:
                    d.decision = "rejected"
                    d.reason += "; not in the selected query"
            elif d.kind == "exclude" and d.decision == "exclude" and key not in excl_k:
                d.decision = "rejected"
                d.reason += "; not in the selected query"

        years_end = request.years_end or max(s.default_years_end, time.gmtime().tm_year)
        known = [y for y in [model.hub.created_year, *model.hub.paper_years] if y]
        years_start = request.years_start or (min(known) if known else s.default_years_start)
        years_start = min(years_start, years_end)
        out.dsl_query = (
            f'search publications in full_data for "{out.for_clause}" '
            f"where year in [{years_start}:{years_end}] "
            "return publications[id+title+doi+year+times_cited] sort by times_cited"
        )

        notes = [out.notes] if out.notes else []
        threshold = float(s.optimize_fp_threshold_pct)
        evaluated = "~gate" in (out.selected_variant or "")
        if not evaluated or float(out.fp_rate_pct or 0.0) > threshold:
            # Never hand the pipeline a query the abstract check did not clear.
            why = (
                f"the abstract check measured {float(out.fp_rate_pct or 0.0):.0f}% false "
                f"positives (limit {threshold:.0f}%)"
                if evaluated else "the abstract check did not run on the selected query"
            )
            out.confidence = ConfidenceLevel.LOW
            notes.append(f"seed withheld: {why}; review term_decisions before using this query")
            out.notes = " ".join(notes)
            return out
        candidate = QueryCandidate(
            main_dataset_name=model.name, safe=safe, risky=risky, flag_terms=flags,
            exclude_terms=excludes, description=model.description,
        )
        if risky and flags and safe:
            # The extraction pipeline applies flags to every name. If that loses more than
            # the gated name adds, send the safe names only.
            gated_count = await self._clause_count(
                ai_model_seed_clause(safe + risky, flags, excludes)
            )
            safe_count = await self._clause_count(ai_model_seed_clause(safe, [], excludes))
            if gated_count is not None and safe_count is not None and safe_count > gated_count:
                candidate = dataclasses.replace(candidate, risky=[], flag_terms=[])
                notes.append(
                    f"seed: safe names alone find {safe_count} papers, more than the "
                    f"flag-gated seed ({gated_count}); the seed uses the safe names only"
                )
        try:
            seed = candidate_to_ai_model_seed(
                candidate, model_name=model.name, model_family=model.family,
                hugging_face_id=request.hugging_face_id, years=(years_start, years_end),
            )
        except ValueError as exc:
            notes.append(f"seed not built: {exc}")
            out.notes = " ".join(notes)
            return out
        seed_names = [seed["model_name"], *seed["model_aliases"]]
        out.ai_model_seed = seed
        out.ai_model_seed_clause = ai_model_seed_clause(
            seed_names, seed["flag_terms"], seed["exclude_terms"]
        )
        out.ai_model_seed_count = await self._clause_count(out.ai_model_seed_clause)
        if seed["model_name"].casefold() != model.name.casefold():
            notes.append(
                f"seed: model_name is {seed['model_name']!r} because {model.name!r} is not "
                "searched on its own"
            )
        if seed["flag_terms"] and safe:
            notes.append(
                "seed: flag terms apply to every name in the extraction pipeline; "
                f"it finds {out.ai_model_seed_count} papers (all years)"
            )
        out.notes = " ".join(notes)
        return out

    async def _signal3_fp_rate_pct(
        self,
        request: OptimizeRequest,
        dataset_description: str,
        titles: list[str],
    ) -> float | None:
        """LLM Signal 3: share of titles scored as likely off-topic (bounded sample)."""
        cap = int(self._settings.optimize_signal3_max_titles)
        if cap <= 0:
            return None
        min_sum = int(self._settings.optimize_signal3_min_relevance_sum)
        sample = [t for t in titles if (t or "").strip()][:cap]
        if not sample:
            return None
        fp_hits = 0

        def _score_one(title: str) -> tuple[int, int]:
            raw = score_title_signal3(
                self._agent,
                dataset_name=request.dataset_name,
                dataset_description=dataset_description,
                title=title,
            )
            ms = int(raw.get("mention_score") or 0)
            cs = int(raw.get("context_score") or 0)
            return ms, cs

        for title in sample:
            ms, cs = await asyncio.to_thread(_score_one, title)
            if ms + cs < min_sum:
                fp_hits += 1
        return 100.0 * fp_hits / float(len(sample))

    def _build_where_dsl(self, for_clause: str) -> str:
        ys = self._settings.default_years_start
        ye = self._settings.default_years_end
        return (
            f'search publications in full_data for "{for_clause}" '
            f"where year in [{ys}:{ye}] and type=\"article\" "
            "return publications[id+title+doi+year+times_cited] sort by times_cited"
        )

    async def _run_research(self, request: OptimizeRequest) -> DatasetRecord:
        req = ResearchRequest(
            dataset_name=request.dataset_name,
            dataset_url=request.dataset_url,
            sample_size=min(50, self._settings.fp_sample_size),
        )
        record, _path = await asyncio.to_thread(self._research.execute, req)
        return record

    async def execute(self, request: OptimizeRequest) -> QueryOptimizationRecord:
        t0 = time.perf_counter()
        notes_parts: list[str] = []

        key = (self._settings.dimensions_api_key or "").strip()
        if not key:
            return QueryOptimizationRecord(
                success=False,
                failed_phase="precheck",
                errors=["Dimensions API key not configured (DATASET_AGENT_DIMENSIONS_API_KEY)"],
                notes="Cannot run query optimization without Dimensions credentials.",
            )

        if not isinstance(self._dsl, AsyncThrottledDimensionsDsl):
            notes_parts.append(f"DSL client type: {type(self._dsl).__name__}")

        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "optimize_start run_id=%s dataset_name=%r",
            run_id,
            request.dataset_name,
        )

        self._cap_extra = 0
        if request.entity_type == "ai_model":
            return await self._execute_ai_model(request, notes_parts, run_id, t0)

        desc_for_signal3 = ""
        rec: DatasetRecord | None = None
        if request.dataset_names:
            aliases = dedupe_strings_ci_preserve_order(list(request.dataset_names))
            flag_terms = dedupe_strings_ci_preserve_order(list(request.flag_terms))
            exclude_terms = dedupe_strings_ci_preserve_order(list(request.exclude_terms))
        else:
            try:
                rec = await self._run_research(request)
            except Exception as e:
                logger.exception("optimize: research failed")
                return QueryOptimizationRecord(
                    success=False,
                    failed_phase="research",
                    errors=[f"Research pipeline failed: {e}"],
                    notes="Provide dataset_names + flag_terms to skip research.",
                )
            desc_for_signal3 = (rec.description or "")[:2000]
            aliases = dedupe_strings_ci_preserve_order(list(rec.dataset_names))
            flag_terms = dedupe_strings_ci_preserve_order(list(rec.flag_terms))
            exclude_terms = dedupe_strings_ci_preserve_order(
                list(rec.exclude_terms) + list(request.exclude_terms)
            )

        aliases = sanitize_terms(aliases, "alias")
        # A lone generic word ("dataset", "data") matches every paper; it is never a name.
        generic = [a for a in aliases if not content_tokens(a)]
        if generic:
            aliases = [a for a in aliases if content_tokens(a)]
            notes_parts.append(f"dropped generic alias(es) {generic}")
            logger.warning("optimize: dropped generic alias(es) %s", generic)
        flag_terms = sanitize_terms(flag_terms, "flag term")
        if rec is not None and self._settings.optimize_drop_broad_funder_flags:
            # A national funding agency is thanked in a large share of all papers; next to
            # the dataset's own organizations it only lets funding acknowledgments through.
            broad = [f for f in flag_terms if f.casefold() in BROAD_FUNDER_FLAGS]
            if broad and len(broad) < len(flag_terms):
                flag_terms = [f for f in flag_terms if f.casefold() not in BROAD_FUNDER_FLAGS]
                notes_parts.append(f"dropped broad funder flag(s) {broad}")
                logger.info("optimize: dropped broad funder flag(s) %s", broad)
        exclude_terms = sanitize_terms(exclude_terms, "exclude term")

        if not aliases:
            return QueryOptimizationRecord(
                success=False,
                failed_phase="phase1_alias_counting",
                errors=["No dataset name aliases available after research/inputs."],
                notes="",
            )

        self._dims_calls = 0
        phase1 = "phase1_alias_counting"
        t_phase1 = time.perf_counter()
        try:
            if rec is not None and len(flag_terms) > 1:
                flag_terms = await self._screen_research_flags(flag_terms, notes_parts)
            st0, rk0 = apply_short_acronym_heuristic(aliases, [])
            c_full = await self._count(request.dataset_name)
            alias_entries: list[AliasCountEntry] = []
            risky: list[str] = []
            safe: list[str] = []
            has_fn = bool((request.dataset_name or "").strip())
            suffix_checked = 0
            dropped_aliases = 0

            for a in rk0:
                if not self._budget_left():
                    notes_parts.append("Stopped short-acronym counting early (Dimensions call budget).")
                    break
                c = await self._count_alias(a, notes_parts)
                if c is None:
                    dropped_aliases += 1
                    continue
                alias_entries.append(AliasCountEntry(alias=a, count=c, risk="risky"))
                if c <= 0:
                    risky.append(a)
                    continue
                suffixed, suffix_checked = await self._try_suffix_promote(
                    a, c, notes_parts=notes_parts, suffix_checked=suffix_checked,
                    c_full=c_full, has_fn=has_fn,
                )
                if suffixed:
                    safe.append(suffixed)
                else:
                    risky.append(a)

            for a in st0:
                if not self._budget_left():
                    notes_parts.append("Stopped alias counting early (Dimensions call budget).")
                    break
                c = await self._count_alias(a, notes_parts)
                if c is None:
                    dropped_aliases += 1
                    continue
                risk = _alias_risk_label(c, c_full, has_fn)
                alias_entries.append(AliasCountEntry(alias=a, count=c, risk=risk))
                if c <= 0:
                    continue
                if risk == "risky":
                    suffixed, suffix_checked = await self._try_suffix_promote(
                        a, c, notes_parts=notes_parts, suffix_checked=suffix_checked,
                        c_full=c_full, has_fn=has_fn,
                    )
                    if suffixed:
                        safe.append(suffixed)
                    else:
                        risky.append(a)
                else:
                    safe.append(a)

            # Aliases that only name a listed organization neither show the dataset is cited
            # nor keep grounding from looking for its real names.
            org_aliases = (
                [e.alias for e in alias_entries if is_organization_alias(e.alias, flag_terms, request.dataset_name)]
                if rec is not None
                else []
            )
            org_keys = {a.casefold() for a in org_aliases}
            added_names: list[str] = []
            if (
                rec is not None
                and self._settings.optimize_grounding_enabled
                and needs_grounding(
                    c_full,
                    [e.count for e in alias_entries if e.alias.casefold() not in org_keys],
                    self._settings.optimize_grounding_min_hits,
                )
            ):
                for g_alias, g_count in await self._ground_names(
                    request, rec, alias_entries, flag_terms, notes_parts
                ):
                    g_risk = await self._added_name_risk(
                        g_alias, g_count, c_full, has_fn, flag_terms, notes_parts
                    )
                    alias_entries.append(AliasCountEntry(alias=g_alias, count=g_count, risk=g_risk))
                    (risky if g_risk == "risky" else safe).append(g_alias)
                    added_names.append(g_alias)

            if rec is not None and self._settings.optimize_name_history_enabled:
                h_names, h_flags = await self._name_history(
                    request, rec, alias_entries, flag_terms, notes_parts
                )
                flag_terms.extend(h_flags)
                for h_alias, h_count in h_names:
                    h_risk = await self._added_name_risk(
                        h_alias, h_count, c_full, has_fn, flag_terms, notes_parts
                    )
                    alias_entries.append(AliasCountEntry(alias=h_alias, count=h_count, risk=h_risk))
                    (risky if h_risk == "risky" else safe).append(h_alias)
                    added_names.append(h_alias)

            # Once real names were found, an organization-only alias just admits the
            # organization's other papers.
            if org_aliases and added_names:
                safe = [a for a in safe if a.casefold() not in org_keys]
                risky = [a for a in risky if a.casefold() not in org_keys]
                alias_entries = [e for e in alias_entries if e.alias.casefold() not in org_keys]
                notes_parts.append(
                    f"dropped organization alias(es) {org_aliases} in favor of {added_names}"
                )

            # "WR dataset" still matches other meanings of "WR", whether research listed it
            # or the suffix check promoted it: flag-gated. Without flag terms a risky alias
            # would never be searched, so it stays safe then.
            if flag_terms and not self._settings.optimize_suffix_short_acronym_safe:
                for s_alias in [x for x in safe if is_acronym_suffix_form(x)]:
                    risky.append(s_alias)
                    for entry in alias_entries:
                        if entry.alias == s_alias:
                            entry.risk = "risky"
                    note = f"suffix_risky: {s_alias!r} kept risky (short acronym)"
                    if note not in notes_parts:
                        notes_parts.append(note)
            risky = dedupe_strings_ci_preserve_order(risky)
            r_low = {r.lower() for r in risky}
            safe = dedupe_strings_ci_preserve_order([s for s in safe if s.lower() not in r_low])
            if dropped_aliases and dropped_aliases >= len(rk0) + len(st0):
                return QueryOptimizationRecord(
                    success=False,
                    failed_phase=phase1,
                    errors=["Dimensions could not parse any alias."],
                    notes=" ".join(notes_parts),
                )
            if not safe and not risky:
                return QueryOptimizationRecord(
                    success=False,
                    failed_phase=phase1,
                    errors=["All aliases returned zero Dimensions hits."],
                    alias_counts=alias_entries,
                    notes=" ".join(notes_parts),
                )
            logger.info(
                "optimize_phase run_id=%s phase=phase1_alias dims=%s elapsed_s=%.2f safe=%s risky=%s",
                run_id,
                self._dims_calls,
                time.perf_counter() - t_phase1,
                len(safe),
                len(risky),
            )
        except DimensionsDslError as e:
            return QueryOptimizationRecord(
                success=False,
                failed_phase=phase1,
                errors=[str(e)],
                notes=" ".join(notes_parts),
            )

        return await self._select_and_finish(
            request, rec=rec, safe=safe, risky=risky, flag_terms=flag_terms,
            exclude_terms=exclude_terms, alias_entries=alias_entries, notes_parts=notes_parts,
            c_full=c_full, desc_for_signal3=desc_for_signal3, run_id=run_id, t0=t0,
        )

    async def _select_and_finish(
        self,
        request: OptimizeRequest,
        *,
        rec: DatasetRecord | None,
        safe: list[str],
        risky: list[str],
        flag_terms: list[str],
        exclude_terms: list[str],
        alias_entries: list[AliasCountEntry],
        notes_parts: list[str],
        c_full: int,
        desc_for_signal3: str,
        run_id: str,
        t0: float,
        model: _ModelRun | None = None,
    ) -> QueryOptimizationRecord:
        """Phases 2-4: probe variants, evaluate the gate, refine, select.

        With *model* (AI-model runs) the terms were measured already: no heuristic
        excludes are added, full-text probes are skipped, and refinement keeps to
        the model rules (see ``_refine_model_terms``)."""
        ex_extra: list[str] = []
        if model is None and detect_rhetorical_name(request.dataset_name):
            ex_extra.extend(rhetorical_exclusions(request.dataset_name))
            notes_parts.append("Applied rhetorical NOT heuristics.")
        if model is None:
            ex_extra.extend(platform_exclude_suggestions(request.dataset_url))
        exclusion_terms = dedupe_strings_ci_preserve_order(exclude_terms + ex_extra)

        phase2 = "phase2_variant_building"
        seq = default_variant_build_order(
            safe,
            risky,
            flag_terms,
            exclusion_terms,
            include_v2_diagnostic=False,
        )
        if not seq:
            return QueryOptimizationRecord(
                success=False,
                failed_phase=phase2,
                errors=["No query variants could be built (missing flag terms for risky aliases?)."],
                aliases=QueryOptimizationAliases(safe=safe, risky=risky),
                alias_counts=alias_entries,
                flag_terms=flag_terms,
                exclusion_terms=exclusion_terms,
                notes=" ".join(notes_parts),
            )

        tested: list[VariantTestedEntry] = []
        errors: list[str] = []
        t_phase2 = time.perf_counter()

        for label, for_clause in seq:
            if not for_clause.strip():
                continue
            if not self._budget_left():
                notes_parts.append("Skipped remaining variants (Dimensions call budget).")
                break
            try:
                vr = await run_variant(self._dsl, label, for_clause, limit=20)
                self._dims_calls += 1
            except DimensionsDslError as e:
                errors.append(f"{label}: {e}")
                continue
            tested.append(
                VariantTestedEntry(
                    label=label,
                    for_clause=for_clause,
                    expected_count=vr.get("expected_count"),
                    fp_rate_pct=0.0,
                    top_titles=list(vr.get("top_titles") or [])[:10],
                )
            )

        if not tested:
            return QueryOptimizationRecord(
                success=False,
                failed_phase=phase2,
                errors=errors or ["All variant probes failed."],
                aliases=QueryOptimizationAliases(safe=safe, risky=risky),
                alias_counts=alias_entries,
                flag_terms=flag_terms,
                exclusion_terms=exclusion_terms,
                notes=" ".join(notes_parts),
            )

        if all((v.expected_count or 0) == 0 for v in tested):
            return QueryOptimizationRecord(
                success=False,
                failed_phase=phase2,
                errors=["All candidate variants returned zero publications in Dimensions."],
                aliases=QueryOptimizationAliases(safe=safe, risky=risky),
                alias_counts=alias_entries,
                flag_terms=flag_terms,
                exclusion_terms=exclusion_terms,
                all_variants_tested=tested,
                notes=" ".join(notes_parts),
            )

        logger.info(
            "optimize_phase run_id=%s phase=phase2_variants dims=%s elapsed_s=%.2f n_variants=%s",
            run_id,
            self._dims_calls,
            time.perf_counter() - t_phase2,
            len(tested),
        )

        # Phase 3: iterative gate evaluation loop (replaces proxy-FP phase)
        phase3 = "phase3_gate_evaluation"
        t_phase3 = time.perf_counter()

        # Collect all aliases for the evaluator
        alias_pool = [request.dataset_name]
        if request.dataset_names:
            alias_pool.extend(request.dataset_names)
        if rec is not None:
            alias_pool.extend(rec.dataset_names)
        if model is not None:
            # Longest first: a hit on "Microsoft Aurora" is not attributed to "Aurora".
            alias_pool = list(model.names)
        aliases = dedupe_strings_ci_preserve_order([a for a in alias_pool if a])

        exc_work = dedupe_strings_ci_preserve_order(list(exclusion_terms))
        current_safe = list(safe)
        current_risky = list(risky)
        current_flags = list(flag_terms)

        # Start with the best Phase-2 variant: lowest FP, then non-empty result,
        # then the variant preference order (V4 > V3 > V2 > V1 > V5).
        best_variant = _pick_start_variant(tested)
        current_for_clause = best_variant.for_clause
        current_variant_label = best_variant.label.split("~", 1)[0]

        best_ev: dict | None = None
        phase3_error: str | None = None
        ft_probe_error: str | None = None
        ft_eval_error: tuple[int, str] | None = None
        best_variant_entry: VariantTestedEntry | None = None

        max_iters = max(1, self._settings.optimize_max_iterations)
        threshold = float(self._settings.optimize_fp_threshold_pct)

        ft_enabled = self._fulltext_enabled() and model is None
        if model is not None:
            if self._fulltext_enabled():
                notes_parts.append(
                    "full-text refinement skipped: probes are not supported for AI-model seeds"
                )
            # One clause per term state, so the selected clause maps back to its terms.
            for v_label, v_clause in seq:
                model.remember(
                    v_clause,
                    safe,
                    risky if v_label != "V1" else [],
                    flag_terms if v_label != "V1" else [],
                    exclusion_terms if v_label == "V4" else [],
                )
            current_for_clause = build_for_clause(
                safe=safe, variant="V4", risky=risky, flag_terms=flag_terms,
                exclusion_terms=exc_work,
            )
            current_variant_label = "V4"
            model.remember(current_for_clause, safe, risky, flag_terms, exc_work)
        ft_iterations: list[IterationMetrics] = []
        ft_candidate = QueryCandidate(
            main_dataset_name=request.dataset_name,
            safe=list(current_safe),
            risky=list(current_risky),
            flag_terms=list(current_flags),
            exclude_terms=list(exc_work),
            description=desc_for_signal3,
        )
        ft_entries: dict[int, VariantTestedEntry] = {}
        ft_candidates: dict[int, QueryCandidate] = {}
        ft_diffs: dict[int, list[TermDiff]] = {}
        if ft_enabled:
            current_for_clause = candidate_clause(ft_candidate)
            current_variant_label = "V4"

        for iteration in range(max_iters):
            logger.info(
                "optimize_phase run_id=%s gate_iter=%s for_clause=%s",
                run_id,
                iteration,
                (current_for_clause[:80] + "...") if len(current_for_clause) > 80 else current_for_clause,
            )

            ev = await asyncio.to_thread(
                evaluate_for_clause_sync,
                for_clause=current_for_clause,
                aliases=aliases,
                dataset_name=request.dataset_name,
                settings=self._settings,
                agent=self._agent,
                dsl_runner=self._make_eval_runner(asyncio.get_running_loop()),
                **({"entity_context": model.context} if model is not None else {}),
            )

            if ev.get("error"):
                # A failed evaluation says nothing about precision; never read it as 0% FP.
                phase3_error = str(ev["error"])
                reason = str(ev.get("error_reason") or short_dsl_error(phase3_error))
                if phase3_error.startswith("dimensions_query_failed"):
                    notes_parts.append(
                        f"gate_iter{iteration}: Dimensions query failed ({reason}); "
                        "iteration not evaluated; evaluation failed, result is partial "
                        "and confidence is capped at LOW."
                    )
                    if ft_enabled:
                        ft_eval_error = (iteration, reason)
                else:
                    notes_parts.append(
                        f"gate_iter{iteration}: evaluation failed ({phase3_error}); "
                        "result is partial and confidence is capped at LOW."
                    )
                if ft_enabled:
                    ft_iterations.append(
                        IterationMetrics(
                            iteration=iteration,
                            for_clause=current_for_clause,
                            error=reason,
                        )
                    )
                break

            fp_rate = float(ev.get("fp_rate_pct") or 0.0)
            expected_count = int(ev.get("publications_total") or 0)
            coverage = float(ev.get("coverage_pct") or 0.0)

            label = f"{current_variant_label}~gate{iteration}"
            entry = VariantTestedEntry(
                label=label,
                for_clause=current_for_clause,
                expected_count=expected_count,
                fp_rate_pct=fp_rate,
                top_titles=[],
            )
            tested.append(entry)
            notes_parts.append(
                f"gate_iter{iteration} fp_rate={fp_rate:.1f}% coverage={coverage:.1f}% total={expected_count}"
            )

            if best_ev is None or fp_rate < (best_ev.get("fp_rate_pct") or 100.0):
                best_ev = ev
                best_variant_entry = entry

            if ft_enabled:
                ft_entries[iteration] = entry
                ft_candidates[iteration] = ft_candidate
                try:
                    evidence = await self._probe_candidate(ft_candidate, request)
                except (ExtractionError, ValueError) as exc:
                    logger.warning("full-text probe failed: %s", exc)
                    ft_probe_error = str(exc)
                    ft_iterations.append(
                        IterationMetrics(
                            iteration=iteration,
                            for_clause=current_for_clause,
                            abstract_fp_rate_pct=fp_rate,
                            error=str(exc),
                        )
                    )
                    notes_parts.append(f"gate_iter{iteration}: full-text probe failed: {exc}")
                    break
                metrics = evidence.metrics(iteration, current_for_clause, fp_rate)
                ft_iterations.append(metrics)
                shortfall = evidence.shortfall_note
                if shortfall:
                    notes_parts.append(f"gate_iter{iteration}: full-text shortfall: {shortfall}")
                prec = metrics.fulltext_precision
                min_ev = self._settings.fulltext_min_evidence
                if prec is not None:
                    notes_parts.append(
                        f"gate_iter{iteration}: full-text precision={prec:.2f} "
                        f"(n_evaluated={metrics.n_evaluated})"
                    )
                if (
                    prec is not None
                    and metrics.n_evaluated >= min_ev
                    and prec >= 1.0 - threshold / 100.0
                ):
                    notes_parts.append(f"gate_iter{iteration}: full-text precision within threshold.")
                    break
                if iteration == max_iters - 1:
                    break
                # Term changes are driven by full-text evidence only; the abstract FP
                # signal can at most mark an alias "suspect".
                loop = asyncio.get_running_loop()
                result = await asyncio.to_thread(
                    refine_candidate,
                    ft_candidate,
                    evidence,
                    settings=self._settings,
                    agent=self._agent,
                    dataset_name=request.dataset_name,
                    count_fn=lambda t: asyncio.run_coroutine_threadsafe(
                        self._refine_count(t), loop
                    ).result(),
                    reference_count=c_full,
                    abstract_fp_aliases={
                        str(h.get("matched_alias") or "") for h in (ev.get("fp_hits") or [])
                    },
                    impact_fn=lambda t, fc=current_for_clause: asyncio.run_coroutine_threadsafe(
                        self._exclude_impact(fc, t), loop
                    ).result(),
                )
                ft_diffs[iteration] = result.diffs
                notes_parts.extend(f"gate_iter{iteration}: {n}" for n in result.notes)
                if not result.changed:
                    notes_parts.append(f"gate_iter{iteration}: refiner found nothing to change.")
                    break
                ft_candidate = result.candidate
                new_fc = candidate_clause(ft_candidate)
                if not new_fc.strip() or new_fc == current_for_clause:
                    break
                current_for_clause = new_fc
                continue

            if fp_rate <= threshold:
                notes_parts.append(f"gate_iter{iteration}: FP rate within threshold.")
                break

            if model is not None:
                refined = await self._refine_model_terms(
                    model, ev, current_safe, current_risky, current_flags, exc_work,
                    notes_parts, iteration, current_for_clause,
                )
                if refined is None:
                    notes_parts.append(f"gate_iter{iteration}: nothing to refine; stopping.")
                    break
                current_safe, current_risky, exc_work = refined
                new_fc = build_for_clause(
                    safe=current_safe, variant="V4", risky=current_risky,
                    flag_terms=current_flags, exclusion_terms=exc_work,
                )
                if not new_fc.strip() or new_fc == current_for_clause:
                    notes_parts.append(f"gate_iter{iteration}: for_clause unchanged; stopping.")
                    break
                current_for_clause = new_fc
                model.remember(new_fc, current_safe, current_risky, current_flags, exc_work)
                continue

            # Refine: derive excludes from FP hits
            fp_hits = ev.get("fp_hits", [])
            if fp_hits and self._agent:
                def _derive() -> list[str]:
                    return derive_exclude_terms_from_fps(
                        self._agent,
                        dataset_name=request.dataset_name,
                        fp_hits=fp_hits,
                    )

                suggested = await asyncio.to_thread(_derive)
                if suggested:
                    exc_work = dedupe_strings_ci_preserve_order(exc_work + suggested[:25])
                    notes_parts.append(
                        f"gate_iter{iteration}: added {len(suggested)} excludes from FP hits"
                    )

            # Rebuild V4 with updated excludes
            if not (current_risky and current_flags):
                notes_parts.append(f"gate_iter{iteration}: no risky aliases+flags to refine.")
                break

            new_fc = build_for_clause(
                safe=current_safe,
                variant="V4",
                risky=current_risky,
                flag_terms=current_flags,
                exclusion_terms=exc_work,
            )
            if not new_fc.strip() or new_fc == current_for_clause:
                notes_parts.append(f"gate_iter{iteration}: for_clause unchanged; stopping.")
                break
            current_for_clause = new_fc
            current_variant_label = "V4"

        logger.info(
            "optimize_phase run_id=%s phase=phase3_gate dims=%s elapsed_s=%.2f",
            run_id,
            self._dims_calls,
            time.perf_counter() - t_phase3,
        )

        best_ft_iter: int | None = None
        if ft_enabled:
            scored = [
                (m.fulltext_precision, -m.iteration)
                for m in ft_iterations
                if m.fulltext_precision is not None
                and m.n_evaluated >= self._settings.fulltext_min_evidence
                and m.iteration in ft_entries
            ]
            if scored:
                best_ft_iter = -max(scored)[1]
                best_variant_entry = ft_entries[best_ft_iter]
                bc = ft_candidates[best_ft_iter]
                safe, risky = list(bc.safe), list(bc.risky)
                flag_terms, exc_work = list(bc.flag_terms), list(bc.exclude_terms)
                notes_parts.append(
                    f"Selected iteration {best_ft_iter} by full-text precision "
                    f"({max(scored)[0]:.2f})."
                )

        recall_diffs: list[TermDiff] = []
        recall_min_precision: float | None = None
        recall_final: QueryCandidate | None = None
        s_cfg = self._settings
        if ft_enabled and s_cfg.fulltext_recall_enabled and s_cfg.fulltext_recall_max_candidates > 0:
            if best_ft_iter is None:
                notes_parts.append(
                    "recall check skipped: no full-text evidence for the selected query"
                )
            elif ft_probe_error is not None:
                notes_parts.append("recall check skipped: a full-text probe failed earlier")
            elif ft_candidates[best_ft_iter].raw_clause:
                notes_parts.append("recall check skipped: the selected query is a raw clause")
            else:
                sel_c = ft_candidates[best_ft_iter]
                names = dedupe_strings_ci_preserve_order(
                    [request.dataset_name]
                    + ([rec.main_dataset_name, rec.official_name] if rec is not None else [])
                )
                names = [n for n in names if n]
                known = dedupe_strings_ci_preserve_order(
                    list(sel_c.risky)
                    + [e.alias for e in alias_entries if e.count > 0]
                )
                flags = dedupe_strings_ci_preserve_order(
                    list(sel_c.flag_terms) + list(ft_candidates[0].flag_terms)
                )
                against = {
                    d.term
                    for ds in ft_diffs.values()
                    for d in ds
                    if d.kind == "alias" and d.action in ("drop", "demote")
                }
                final_c, recall_diffs, rnotes, recall_min_precision = await self._recall_check(
                    request, sel_c, main_names=names, known_aliases=known,
                    flag_terms=flags, evidence_against=against,
                )
                notes_parts.extend(rnotes)
                if recall_diffs:
                    recall_final = final_c
                    safe, risky = list(final_c.safe), list(final_c.risky)
                    new_clause = candidate_clause(final_c)
                    count: int | None = None
                    if self._budget_left():
                        try:
                            count = parse_total_count(
                                await self._counted_execute(build_for_clause_count_dsl(new_clause))
                            )
                        except DimensionsDslError:
                            count = None
                    prev = best_variant_entry
                    best_variant_entry = VariantTestedEntry(
                        label=f"{prev.label}+recall" if prev is not None else "V4+recall",
                        for_clause=new_clause,
                        expected_count=count,
                        fp_rate_pct=prev.fp_rate_pct if prev is not None else 0.0,
                    )
                    tested.append(best_variant_entry)
                    notes_parts.append(
                        f"recall check: added {len(recall_diffs)} alias(es); "
                        "abstract FP rate not re-evaluated"
                    )

        # Prefer best gate-evaluated variant; fall back to lowest-FP among all tested
        if best_variant_entry is not None:
            best = best_variant_entry
            notes_parts.append(
                f"Selected gate-evaluated variant {best.label} (fp_rate={best.fp_rate_pct:.1f}%)."
            )
        else:
            candidates: list[tuple[float, int, int, VariantTestedEntry]] = []
            for ent in tested:
                cnt = ent.expected_count or 0
                fp = float(ent.fp_rate_pct or 0.0)
                if fp > 5.0:
                    continue
                score = float(cnt) * (1.0 - fp / 100.0)
                pref = _VARIANT_ORDER.get(ent.label, 99)
                candidates.append((score, -pref, cnt, ent))
            if not candidates:
                best = tested[0]
                notes_parts.append("No variant under 5% proxy FP; picked first probe result.")
            else:
                best = max(candidates, key=lambda x: (x[0], x[1], x[2]))[3]

        fp_sel = float(best.fp_rate_pct or 0.0)
        if fp_sel < 1.0:
            confidence = ConfidenceLevel.HIGH
        elif fp_sel < 3.0:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        if phase3_error is not None:
            confidence = ConfidenceLevel.LOW
        if ft_enabled:
            # Full-text precision is the primary metric: the reported confidence
            # never exceeds what the selected iteration's full-text evidence supports.
            ft_conf = ConfidenceLevel.LOW
            if best_ft_iter is not None:
                sel = next(m for m in ft_iterations if m.iteration == best_ft_iter)
                prec = float(sel.fulltext_precision or 0.0)
                if recall_min_precision is not None:
                    # A promoted complement may dominate the result set and was accepted
                    # at its own (possibly lower) bar; never report more than it supports.
                    prec = min(prec, recall_min_precision)
                if prec >= 1.0 - threshold / 100.0:
                    ft_conf = ConfidenceLevel.HIGH
                elif prec >= 0.5:
                    ft_conf = ConfidenceLevel.MEDIUM
            rank = {ConfidenceLevel.LOW: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.HIGH: 2}
            if rank[ft_conf] < rank[confidence]:
                confidence = ft_conf

        proposal = None
        if ft_enabled and ft_candidates:
            use_it = best_ft_iter if best_ft_iter is not None else 0
            order = sorted(i for i in ft_diffs if i < use_it)
            try:
                proposal = build_proposal(
                    dataset_name=request.dataset_name,
                    initial=ft_candidates[0],
                    proposed=recall_final if recall_final is not None else ft_candidates[use_it],
                    diffs=merge_diffs([ft_diffs[i] for i in order] + ([recall_diffs] if recall_diffs else [])),
                    iterations=ft_iterations,
                    settings=self._settings,
                    notes=[
                        n
                        for n in notes_parts
                        if "full-text" in n
                        or "refiner" in n
                        or "Dimensions query failed" in n
                        or n.startswith("recall check")
                    ],
                    seed_kw={"extra": {"home_url": request.dataset_url}},
                )
            except ValueError as exc:
                logger.warning("proposal_unbuildable run_id=%s error=%s", run_id, exc)
                notes_parts.append(f"full-text proposal skipped: {exc}")
                proposal = None
            if proposal is not None and best_ft_iter is None:
                most = max((m.n_evaluated for m in ft_iterations), default=0)
                if most > 0:
                    proposal.notes.append(
                        f"Full-text evidence stayed below the minimum sample (at most {most} "
                        f"evaluated paper(s), fulltext_min_evidence="
                        f"{self._settings.fulltext_min_evidence}); the proposal equals the "
                        "initial query. Confidence is LOW."
                    )
                else:
                    proposal.notes.append(
                        "No full-text evidence was obtained; the proposal equals the initial "
                        "query. Confidence is LOW."
                    )
            if proposal is not None and ft_probe_error is not None:
                proposal.notes.append(
                    f"A full-text probe failed ({ft_probe_error}); later iterations were not probed."
                )
            if proposal is not None and ft_eval_error is not None:
                proposal.notes.append(
                    f"Gate iteration {ft_eval_error[0]} could not be evaluated "
                    f"(Dimensions query failed: {ft_eval_error[1]}); the proposal is based "
                    "on earlier iterations."
                )
            if proposal is not None and self._proposal_repo is not None:
                self._proposal_repo.save(proposal)

        dsl_query = self._build_where_dsl(best.for_clause)
        elapsed = round(time.perf_counter() - t0, 2)
        notes_parts.append(f"dimensions_api_calls≈{self._dims_calls} elapsed_s={elapsed}")
        logger.info(
            "optimize_done run_id=%s success=true selected=%s dims=%s elapsed_s=%s",
            run_id,
            best.label,
            self._dims_calls,
            elapsed,
        )

        return QueryOptimizationRecord(
            success=True,
            selected_variant=best.label,
            for_clause=best.for_clause,
            dsl_query=dsl_query,
            expected_count=best.expected_count,
            fp_rate_pct=best.fp_rate_pct,
            confidence=confidence,
            aliases=QueryOptimizationAliases(safe=safe, risky=risky),
            alias_counts=alias_entries,
            flag_terms=flag_terms,
            exclusion_terms=exc_work,
            all_variants_tested=tested,
            fulltext_iterations=ft_iterations,
            proposal=proposal,
            notes=" ".join(notes_parts),
        )
