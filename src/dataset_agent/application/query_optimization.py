"""Four-phase Dimensions query optimization orchestration (spec §7)."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import math
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
    is_distinctive,
    is_plain_word,
    mine_context_terms,
    mine_joined_versions,
    model_terms_prompt,
    version_sibling,
)
from dataset_agent.adapters.dimensions_dsl import (
    AsyncThrottledDimensionsDsl,
    DimensionsDslError,
    build_for_clause_count_dsl,
    build_search_publications_paged_dsl,
    inner_quoted_term,
    is_dsl_syntax_error,
    parse_total_count,
    publication_title,
    publications_from_result,
    raise_for_dsl_errors,
    run_alias_count,
    short_dsl_error,
)
from dataset_agent.adapters.dimensions_phrase import (
    ampersand_additions,
    phrase_contains,
    same_search,
    search_key,
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
from dataset_agent.adapters.extraction_client import (
    STOP_POOL_EXHAUSTED,
    ExtractionClient,
    ExtractionError,
)
from dataset_agent.adapters.fulltext_evidence import Evidence, _phrase_in, _tokens, build_evidence
from dataset_agent.adapters.fulltext_recall import (
    MAX_ADDED_HITS,
    RecallCandidate,
    evidence_quotes,
    judge_complement,
    recall_candidates,
)
from dataset_agent.adapters.fulltext_refiner import refine_candidate
from dataset_agent.adapters.literature import _stem_text, evaluate_for_clause_sync
from dataset_agent.adapters.name_grounding import (
    GROUNDING_TOOL,
    content_tokens,
    defined_names,
    grounding_prompt,
    is_organization_alias,
    needs_grounding,
    screen_candidates,
)
from dataset_agent.adapters.name_history import HISTORY_TOOL, history_prompt
from dataset_agent.adapters.tools import web_search

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
    apply_short_acronym_heuristic,
    detect_rhetorical_name,
    is_acronym_shaped_name,
    is_acronym_suffix_form,
    is_short_acronym,
)
from dataset_agent.adapters.referent_check import check_referents, model_profile
from dataset_agent.adapters.resource_map import program_datasets, program_datasets_summary
from dataset_agent.adapters.scope_check import (
    acronym_form,
    check_scope,
    generated_data_phrases,
    phrase_program,
)
from dataset_agent.application.proposal_service import build_proposal, merge_diffs
from dataset_agent.adapters.term_sanitizer import sanitize_term, sanitize_terms
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
    ResourceMapDecision,
    TermDecision,
    VariantTestedEntry,
)
from dataset_agent.domain.proposals import TermDiff
from dataset_agent.domain.ports import AgentPort, DimensionsDslPort
from dataset_agent.settings import MEASURED_SCOPE_VOTES

if TYPE_CHECKING:
    from dataset_agent.application.research import DatasetResearchUseCase
    from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)

_VARIANT_ORDER = {"V4": 0, "V3": 1, "V2": 2, "V1": 3, "V5": 4}


def covered_safe_names(safe: list[str]) -> dict[str, str]:
    """``{name: covering name}`` for each safe name whose every Dimensions match is a
    match of another, broader safe name (compared the way Dimensions searches: hyphens
    fold to spaces, a dot stays inside its token, so "GPT-4.5" is not covered by "GPT-4")."""
    out: dict[str, str] = {}
    for a in safe:
        cover = next(
            (b for b in safe if not same_search(a, b) and phrase_contains(a, b)), None
        )
        if cover:
            out[a] = cover
    return out


def _pct_text(value: float | None) -> str:
    """A percentage for notes; ``unknown`` when the abstract check classified nothing."""
    return "unknown" if value is None else f"{value:.1f}%"


def _alias_risk_label(c_alias: int, c_full: int, has_full_name: bool) -> Literal["safe", "risky"]:
    """Risky when count > 10x the title count, or any hits when the title has none; 50,000 when there is no title."""
    if not has_full_name:
        return "risky" if c_alias > 50_000 else "safe"
    if c_full <= 0:
        return "risky" if c_alias > 0 else "safe"
    if c_alias > 10.0 * float(c_full):
        return "risky"
    return "safe"


def _big_addition_notes(recall_diffs: list[TermDiff], final_count: int | None) -> list[str]:
    """One note per recall promotion that is more than half of the final count: its share of
    the final count and its usage rate in the recall probe. The final query is not
    re-measured after a promotion, so this term's papers dominate an unmeasured result."""
    if not final_count:
        return []
    out = []
    for d in recall_diffs:
        st = d.stats or {}
        added = int(st.get("added_hits") or 0)
        if added * 2 <= final_count:
            continue
        n = int(st.get("n_evaluated") or 0)
        used = int(st.get("used") or 0)
        usage = f"{used} of {n} recall-probe papers use the data ({used / n:.2f})" if n else "usage unknown"
        out.append(
            f"recall check: big addition '{d.term}' is {100.0 * added / final_count:.0f}% of the "
            f"final count (+{added} of {final_count} hits); {usage}"
        )
    return out


def _recall_target_max() -> int:
    """Upper bound of ``fulltext_recall_target_delivered``; a re-probe's doubled target is
    clamped to it."""
    from dataset_agent.settings import Settings as _Settings

    field = _Settings.model_fields["fulltext_recall_target_delivered"]
    return int(next(m.le for m in field.metadata if getattr(m, "le", None) is not None))


def _fp_better(new: float | None, best: float | None) -> bool:
    """True when abstract FP rate *new* beats *best*. Unknown (None) is the worst rate and
    never beats anything; a measured 0.0 is the best rate, not a missing value."""
    if new is None:
        return False
    return best is None or float(new) < float(best)


def _pick_start_variant(tested: list[VariantTestedEntry]) -> VariantTestedEntry:
    """Best Phase-2 variant: non-empty results first, then variant preference."""
    return min(
        tested,
        key=lambda v: (
            0 if (v.expected_count or 0) > 0 else 1,
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of *k*/*n*; (0, 1) when *n* is 0."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def ai_model_confidence(k: int, n: int, *, released: bool) -> ConfidenceLevel:
    """AI models: HIGH when the Wilson upper bound of the abstract false-positive rate is
    at most 10%, MEDIUM for any other released seed, LOW only when withheld."""
    if not released:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.HIGH if wilson_interval(k, n)[1] <= 0.10 else ConfidenceLevel.MEDIUM


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
    #: The safe names in the reference set (the titles are ``anchors``).
    reference_names: list[str] = dataclasses.field(default_factory=list)
    #: Hits of reference clauses already counted (each is counted once).
    clause_counts: dict[str, int | None] = dataclasses.field(default_factory=dict)
    decisions: list[TermDecision] = dataclasses.field(default_factory=list)
    #: Anchor paper titles, and the first publication year of those already known
    #: (from the top-cited pull), keyed by the punctuation-folded title.
    anchors: list[str] = dataclasses.field(default_factory=list)
    anchor_years: dict[str, int] = dataclasses.field(default_factory=dict)
    #: The model's own base names (request name and family), case-folded.
    base_keys: set[str] = dataclasses.field(default_factory=set)
    #: Own coined base names that are safe by shape or by a partial title share only.
    weak_bases: list[str] = dataclasses.field(default_factory=list)
    #: Measured flags held out of the query; they gate a weak base name that fails the
    #: abstract check.
    reserve_flags: list[str] = dataclasses.field(default_factory=list)
    #: Reserve flags were measured (up front or at re-gate time); never twice.
    reserve_measured: bool = False
    #: Titles and abstracts already pulled (top-cited hits), mined for joined versions.
    pulled_texts: list[str] = dataclasses.field(default_factory=list)
    #: Year window of the shipped query (start, end), known once anchors are decided.
    years: tuple[int, int] | None = None
    #: Anchor titles that may flag a gated own name, with why: ``(title, reason)``.
    title_flags: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    #: Excludes accepted from abstract false positives in this run (set-wide loss bound).
    refine_excludes: list[str] = dataclasses.field(default_factory=list)
    #: Context-flag candidates of the run, kept for a measurement at re-gate time.
    context_terms: list[str] = dataclasses.field(default_factory=list)
    #: Why the seed must be withheld when the abstract check fails (names kept unsafe).
    withhold_reasons: list[str] = dataclasses.field(default_factory=list)
    #: for clause -> (fp k, judged n, counted without judgement, coverage %) of its
    #: abstract check.
    abstract_stats: dict[str, tuple[int, int, int, float]] = dataclasses.field(
        default_factory=dict
    )
    #: for clause -> (safe, risky, flags, excludes) it was built from.
    states: dict[str, tuple[list[str], list[str], list[str], list[str]]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def where(self) -> str:
        return f"year in [{self.years[0]}:{self.years[1]}]" if self.years else ""

    @property
    def context(self) -> str:
        """The model profile: description (and developer), plus the first anchor title
        naming the model (else the first anchor) and its first year. The same string is
        given to the abstract judge, the referent question and the seed description."""
        dev = f" Developer: {self.developer}." if self.developer else ""
        base = f"{(self.description or '').strip()[:400]}{dev}".strip()
        if not self.anchors:
            return base
        anchor = next(
            (t for t in self.anchors
             if contains_word(t, self.name) or contains_word(t, self.family)),
            self.anchors[0],
        )
        year = self.anchor_years.get(_stem_text(anchor)) or (self.years[0] if self.years else None)
        return model_profile(base, [(anchor, year)])

    def decide(self, term: str, kind: str, decision: str, reason: str,
               count: int | None = None, **stats) -> None:
        self.decisions.append(TermDecision(
            term=term, kind=kind, decision=decision, count=count, reason=reason, stats=stats,
        ))

    def remember(self, clause: str, safe, risky, flags, excludes) -> None:
        self.states[clause] = (list(safe), list(risky), list(flags), list(excludes))


def resource_map_description(description: str, rmap) -> str:
    """*description* plus the program and data resource names of a resource map."""
    if rmap is None:
        return description
    return (f"{description} Program names: {', '.join(rmap.program_names)}. "
            f"Data resource names: {', '.join(rmap.data_resource_names)}.").strip()


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
        referent_agent: AgentPort | None = None,
        main_overridden: bool = False,
    ) -> None:
        self._proposal_repo = proposal_repo
        #: The request overrode the main agent's model or sampling; the referent judge
        #: and the scope check still run on the configured model (``referent_agent``).
        self._main_overridden = main_overridden
        self._dsl = dsl_port
        self._research = research
        self._agent = agent
        self._settings = settings
        self._extraction = extraction_client
        #: Agent for the full-text referent question (None = the main agent).
        self._referent_agent = referent_agent
        self._dims_calls = 0
        self._cap_extra = 0
        self._release_reserve = 0
        self._resource_map = None
        self._resource_map_error: str | None = None
        self._map_decision = None
        self._dataset_description = ""
        self._scope_calls = 0

    def _call_cap(self) -> int:
        """Dimensions call cap of the current run (0 = unlimited)."""
        cap = int(self._settings.optimize_max_dimensions_calls)
        return cap + self._cap_extra if cap > 0 else cap

    def _room(self, keep: int = 0) -> bool:
        """Room for one more Dimensions call while keeping *keep* calls free (cap 0 = unlimited),
        plus the calls reserved for the release sample (``_release_reserve``)."""
        cap = self._call_cap()
        return cap <= 0 or self._dims_calls + 1 + keep + self._release_reserve <= cap

    def _reserve_release_calls(self) -> None:
        """Hold back the Dimensions call of the release sample (its year facet) when a
        random sample will be drawn (it decides, or the shadow records it), so the recall
        check and the gates cannot spend it. Released by ``_year_counts``."""
        s = self._settings
        will_draw = self._fulltext_enabled() and (
            self._random_release() or bool(s.fulltext_release_shadow))
        self._release_reserve = 1 if will_draw else 0

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
            years=self._dataset_years(),
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
        await self._apply_scope_check(ev, request)
        return ev

    def _judge_config(self) -> dict:
        """What the referent judge runs with (``Settings.referent_judge_config``); without
        a referent agent the question runs on the main agent."""
        cfg = self._settings.referent_judge_config()
        if self._referent_agent is None and cfg["agent"] == "referent":
            cfg.update(agent="main", thinking_budget=0, temperature=None, top_p=None,
                       top_k=None, measured_configuration=False)
        if self._main_overridden and self._referent_agent is not None:
            # The request's overrides apply to the main agent only: these questions ran
            # on a separate agent with the configured model, not on that 'main' agent.
            if cfg["agent"] == "main":
                cfg["agent"] = "configured"
            cfg["model"] = self._settings.llm_endpoint().model
        return cfg

    def _scope_check_config(self) -> dict:
        """The scope check's configuration: the referent judge plus its asks per paper;
        measured when the judge is the measured one and it asks once."""
        cfg = self._judge_config()
        votes = int(self._settings.scope_votes)
        cfg["votes"] = votes
        cfg["measured_configuration"] = bool(cfg["measured_configuration"]
                                             and votes == MEASURED_SCOPE_VOTES)
        return cfg

    @staticmethod
    def _judge_config_note(label: str, cfg: dict, asks: str, noun: str = "configuration") -> str:
        what = (f"the measured {noun}" if cfg.get("measured_configuration")
                else f"not the measured {noun}")
        fb = ("; fallback: the API style is not anthropic, thinking and sampling dropped"
              if cfg.get("fallback") else "")
        return (f"{label}: {cfg['agent']} agent, thinking budget {cfg['thinking_budget']}, "
                f"temperature {cfg['temperature']}, top_p {cfg['top_p']}, top_k {cfg['top_k']}, "
                f"{cfg['votes']} {asks} ({what}{fb})")

    async def _apply_scope_check(self, ev: Evidence, request: OptimizeRequest) -> None:
        """With a resource map, papers that only discuss the program stop counting as
        genuine (``adapters.scope_check``); the refiner and recall check then judge on
        corrected verdicts. Without a map nothing runs."""
        rmap = self._resource_map
        if rmap is None or ev.entity != "dataset":
            return
        programs = list(rmap.program_names)
        res = await asyncio.to_thread(
            check_scope, ev, agent=self._referent_agent or self._agent,
            dataset_name=request.dataset_name, description=self._dataset_description,
            program_names=programs, data_names=list(rmap.data_resource_names),
            data_terms=generated_data_phrases(programs), votes=self._settings.scope_votes,
        )
        self._scope_calls += res.calls
        logger.info("scope check: %s", {k: v for k, v in vars(res).items() if k != "papers"})

    def _budget_left(self) -> bool:
        return self._room()

    def _phase1_budget_left(self) -> bool:
        """Room for one more phase-1 check while keeping ``refine_impact_budget_reserve``
        calls (plus the next gate's two) for the full-text refiner."""
        return self._room(keep=int(self._settings.refine_impact_budget_reserve) + 2)

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
        if not self._room(keep=int(self._settings.refine_impact_budget_reserve)):
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
        term with more than ``optimize_flag_max_hits`` Dimensions hits. A flag that is a
        whole-word phrase inside another kept flag ("EarthScope" in "EarthScope
        Consortium") is dropped too: the refiner refuses that pair for its own proposals."""
        s = self._settings
        kept: list[str] = []
        dropped: list[str] = []
        # Flags with the same words ("O*NET" / "O-NET", "UNAVCO Inc." / "UNAVCO, Inc"):
        # keep the first one only, so neither is read as the other's broader form.
        for f in flags:
            same = next((k for k in kept if _tokens(k) == _tokens(f)), None)
            if same is None:
                kept.append(f)
            else:
                dropped.append(f"{f} (same words as {same})")
        broader: list[tuple[str, str]] = []
        for f in kept:
            g = next(
                (
                    g
                    for g in kept
                    if _tokens(g) != _tokens(f) and _phrase_in(f, g, plural_ok=False)
                ),
                None,
            )
            if g is not None:
                broader.append((f, g))
        if broader and len(broader) < len(kept):
            drop = {f for f, _g in broader}
            kept = [f for f in kept if f not in drop]
            dropped += [f"{f} (broader form of {g})" for f, g in broader]
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

    async def _count_candidates(
        self,
        kept: list[str],
        unvouched: set[str],
        rejected: list[str],
        notes_parts: list[str],
    ) -> list[tuple[str, int]]:
        """Count each kept candidate in Dimensions; apply the hit floor and the unvouched ceiling."""
        s = self._settings
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
        return found

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
        found = await self._count_candidates(kept, unvouched, rejected, notes_parts)
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
        org_keys: set[str],
        notes_parts: list[str],
    ) -> list[tuple[str, int]]:
        """Former and successor names of the collection, each with its Dimensions count;
        see ``adapters.name_history``."""
        s = self._settings
        name = request.dataset_name
        # The collection's own names lead; an organization-only alias would search the
        # organization's history instead.
        leading = sorted(
            [
                (e.alias, e.count)
                for e in alias_entries
                if e.count > 0 and e.alias.casefold() not in org_keys
            ],
            key=lambda x: -x[1],
        )[:3] or [(name, 0)]
        # The plain search finds the history; with an organization added it stays on this
        # collection when the name is shared ("Network of the Americas" is also the name of
        # unrelated networks).
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
        kept, rejected = screen_candidates(
            llm_names,
            dataset_name=name,
            known_aliases=[e.alias for e in alias_entries],
            flag_terms=flag_terms,
            limit=s.optimize_name_history_max_candidates,
        )
        found = await self._count_candidates(kept, set(), rejected, notes_parts)

        def _why(cand: str) -> str:
            key = cand.casefold()
            x = relation.get(key) or next(
                (v for k, v in relation.items() if k.startswith(key + " (")), {}
            )
            year = f", {x['year']}" if isinstance(x.get("year"), int) else ""
            return f"{x.get('relation', 'related')}{year}"

        if found:
            notes_parts.append(
                "name history: added "
                + ", ".join(f"{a!r} ({_why(a)}; {n} hits)" for a, n in found)
            )
        if rejected:
            notes_parts.append("name history: rejected " + "; ".join(rejected))
        logger.info("name history: added=%s rejected=%s", found, rejected)
        return found

    async def _refine_count(self, term: str) -> int:
        """Refiner-only count: like ``_exclude_impact`` it raises (instead of returning 0)
        once the budget is down to ``refine_impact_budget_reserve``."""
        if not self._room(keep=int(self._settings.refine_impact_budget_reserve)):
            raise DimensionsDslError("Dimensions budget reserve reached")
        self._dims_calls += 1
        return await run_alias_count(self._dsl, term, search_in="full_data")

    async def _recall_check(
        self,
        request: OptimizeRequest,
        current: QueryCandidate,
        *,
        main_names: list[str],
        known_aliases: list[str],
        flag_terms: list[str],
        evidence_against: set[str],
        base_count: int | None = None,
    ) -> tuple[QueryCandidate, list[TermDiff], list[str], float | None, list[str]]:
        """Propose missing canonical names as safe aliases; returns
        ``(candidate, add diffs, notes, min complement precision of promoted terms,
        terms rejected only on the Wilson lower bound)``.

        Every note starts with ``recall check:``. A candidate is promoted only when a
        complement probe (papers matching it but not the current query) is precise enough."""
        s = self._settings
        notes: list[str] = []
        diffs: list[TermDiff] = []
        precisions: list[float] = []
        lower_bound_only: list[str] = []
        eligible, rejected = recall_candidates(
            current, main_names=main_names, known_aliases=known_aliases,
            flag_terms=flag_terms, evidence_against=evidence_against,
            data_phrases=(self._standalone_data_phrases(self._resource_map)
                          if self._resource_map is not None else None),
        )
        if rejected:
            more = f"; and {len(rejected) - 8} more" if len(rejected) > 8 else ""
            notes.append("recall check: not eligible: " + "; ".join(rejected[:8]) + more)
        # A candidate whose complement cannot be expressed is dropped before any count call.
        expressible: list[RecallCandidate] = []
        for cand in eligible:
            try:
                complement_candidate(cand.term, current)
            except ValueError as exc:
                notes.append(f"recall check: '{cand.term}' not evaluated: {exc}")
                continue
            expressible.append(cand)
        eligible = expressible
        if not eligible:
            return current, diffs, notes, None, lower_bound_only

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
            elif k > MAX_ADDED_HITS:
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
            ok, reason, stats = judge_complement(term, evidence, k, s, base_count=base_count)
            first_lb = bool(stats.get("rejected_on_lower_bound"))
            if not ok and first_lb and self._budget_left():
                # Only the interval rejected it: probe the same complement once more on a
                # larger sample and judge on that probe alone (it re-reads the same
                # top-cited head, so the two probes are never added together).
                try:
                    evidence2 = await self._probe_candidate(
                        complement_candidate(term, current), request,
                        target=min(2 * target, _recall_target_max()),
                    )
                except (ExtractionError, ValueError) as exc:
                    notes.append(f"recall check: '{term}' not re-probed: {exc}")
                else:
                    notes.append(
                        f"recall check: '{term}' re-probed on {evidence2.n_evaluated} papers "
                        "after a lower-bound rejection"
                    )
                    evidence = evidence2
                    ok, reason, stats = judge_complement(term, evidence, k, s, base_count=base_count)
            stats = {**stats, "source": cand.source,
                     "complement_clause": complement_clause(term, current)}
            if not ok:
                notes.append(f"recall check: '{term}' rejected: {reason}")
                # A re-probe that rejects on the point or on too few papers does not erase
                # the first probe's lower-bound rejection (the confidence cap needs it).
                if first_lb or stats.get("rejected_on_lower_bound"):
                    lower_bound_only.append(term)
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
        return current, diffs, notes, (min(precisions) if precisions else None), lower_bound_only

    # ------------------------------------------------------------------ AI models

    async def _clause_count(self, clause: str, where: str = "") -> int | None:
        """Hits of a boolean clause (filtered by *where*, e.g. ``year in [a:b]``);
        ``None`` when the call budget is spent or Dimensions cannot parse the clause.
        Other errors propagate."""
        if not clause or not self._budget_left():
            return None
        try:
            return parse_total_count(
                await self._counted_execute(build_for_clause_count_dsl(clause, where=where))
            )
        except DimensionsDslError as exc:
            if not is_dsl_syntax_error(exc):
                raise
            return None

    async def _model_exclude_check(
        self, model: _ModelRun, term: str, reference: tuple[str, int] | None = None,
        also: list[str] | None = None,
    ) -> tuple[bool, str, dict]:
        """An exclude is kept only when it removes (almost) none of the papers that
        certainly refer to the model; *reference* ``(clause, count)`` replaces the
        model's reference set. With *also* (excludes already accepted) the bound covers
        the set: *term* OR any of them."""
        ref_clause, ref_count = reference or (model.reference_clause, model.reference_count)
        if not ref_clause or ref_count <= 0:
            return False, "no precise name or paper title to validate it against", {}
        group = dedupe_strings_ci_preserve_order([term, *(also or [])])
        hit = await self._clause_count(f"{ref_clause} AND {_or_group(group)}")
        if hit is None:
            return False, "not measured (Dimensions call budget or unparseable term)", {}
        loss = hit / ref_count
        stats = {"precise_matches_removed": hit, "precise_matches": ref_count}
        if loss > self._settings.ai_model_exclude_max_anchor_loss:
            return (
                False,
                f"would remove {hit} of {ref_count} precise matches ({loss:.0%})",
                stats,
            )
        return True, f"removes {hit} of {ref_count} precise matches ({loss:.1%})", stats

    async def _reference_without(self, model: _ModelRun, names: set[str]) -> tuple[str, int, str]:
        """``(clause, count, label)`` of the papers that certainly refer to the model
        without depending on *names* (casefolded): the paper titles plus the safe names
        that contain none of them. Each clause is counted once."""
        terms = dedupe_strings_ci_preserve_order(model.anchors + [
            n for n in model.reference_names if not any(contains_word(n, a) for a in names)
        ])
        clause = _or_group(terms) if terms else ""
        if clause and clause not in model.clause_counts:
            model.clause_counts[clause] = await self._clause_count(clause)
        count = model.clause_counts.get(clause) or 0 if clause else 0
        return clause, count, f"precise matches without {sorted(names)}"

    async def _model_title_check(
        self, model: _ModelRun, title: str, bases: list[str], snippet: str = ""
    ) -> tuple[bool, str, int | None]:
        """A title not cited on the model card anchors only when it names a coined base
        name of the model, has Dimensions hits, and the model-aware mention check (the
        title, or *snippet* when given, with the model description as context) calls it
        this model."""
        unverified = "not cited on the model card: an unverified title cannot be an anchor"
        base = next((b for b in bases if contains_word(title, b)), None)
        if base is None:
            return False, unverified, None
        c = await self._clause_count(_or_group([title]))
        if not c:
            return False, "0 Dimensions hits" if c == 0 else "not measured", c
        if not model.context:
            return False, f"{unverified} (no model description to check it against)", c
        genuine = await asyncio.to_thread(
            classify_alias_mention, self._agent, dataset_name=model.name, alias=base,
            snippet=snippet or title, entity_context=model.context,
        )
        if genuine is None:
            return False, f"{unverified} (the model-aware check failed)", c
        if not genuine:
            return False, (
                f"names {base!r}, but the model-aware check reads it as another meaning"
            ), c
        return True, f"names the coined name {base!r}; the model-aware check confirms it", c

    async def _top_cited_titles(
        self, model: _ModelRun, bases: list[str], decided: set[str]
    ) -> list[tuple[str, str, int | None]]:
        """Name-bearing titles among the top-cited hits of each safe coined base name,
        passed through ``_model_title_check``: ``(title, reason, count)`` per anchor."""
        s = self._settings
        cap, sample = int(s.ai_model_top_cited_titles), int(s.ai_model_top_cited_sample)
        out: list[tuple[str, str, int | None]] = []
        for base in bases:
            if cap <= 0 or sample <= 0 or len(out) >= cap or not self._budget_left():
                break
            try:
                result = await self._counted_execute(build_search_publications_paged_dsl(
                    _or_group([base]), limit=sample, fields="title+year+abstract",
                ))
                total = parse_total_count(result)
            except DimensionsDslError as exc:
                if not is_dsl_syntax_error(exc):
                    raise
                continue
            verdict, _r = classify_model_name(
                base, count=total, confirmation=None, developer=model.developer,
                safe_share=s.ai_model_safe_min_context_share,
                plain_word_max_hits=s.ai_model_plain_word_max_safe_hits,
                coined_min_share=s.ai_model_coined_min_title_share,
            )
            if verdict != "safe":
                continue
            judged = 0
            pubs = publications_from_result(result)
            model.pulled_texts.extend(
                f"{publication_title(p) or ''} {p.get('abstract') or ''}" for p in pubs
            )
            years: dict[str, int] = {}
            for pub in pubs:
                # Dimensions folds punctuation: "Arrival Time" and "arrival-time" are one key.
                t, y = _stem_text(publication_title(pub) or ""), pub.get("year")
                if t and isinstance(y, int):
                    years[t] = min(y, years.get(t, y))
            for pub in pubs:
                title = sanitize_term(publication_title(pub))
                key = _stem_text(title or "")
                if (
                    not title or key in decided or len(title.split()) < 4
                    or not contains_word(title, base)
                ):
                    continue
                decided.add(key)
                abstract = str(pub.get("abstract") or "").strip()
                ok, reason, c = await self._model_title_check(
                    model, title, [base],
                    snippet=f"{title}. {abstract[:600]}" if abstract else "",
                )
                judged += 1
                reason = f"top-cited hit of {base!r}; {reason}"
                if ok:
                    out.append((title, reason, c))
                    if key in years:
                        model.anchor_years[key] = years[key]
                    model.decide(title, "paper_title", "anchor", reason, c)
                else:
                    model.decide(title, "paper_title", "rejected", reason, c)
                if len(out) >= cap or judged >= 2 * cap:
                    break
        return out

    async def _model_years(self, request: OptimizeRequest, model: _ModelRun) -> tuple[int, int]:
        """Year window of the shipped query: from the caller, else the earliest of the
        card's years and the first name-bearing anchor's year, else the default start."""
        s = self._settings
        this_year = time.gmtime().tm_year
        years_end = request.years_end or max(
            s.default_years_end, this_year + (1 if s.ai_model_window_next_year else 0)
        )
        known = [y for y in [model.hub.created_year, *model.hub.paper_years] if y]
        if not request.years_start:
            # A versioned or re-uploaded card must not cut the model's own first papers.
            try:
                first = await self._first_anchor_year(model)
            except DimensionsDslError as exc:
                logger.warning("model research: anchor year lookup failed: %s", exc)
                first = None
            if first:
                known.append(first)
        years_start = request.years_start or (min(known) if known else s.default_years_start)
        return min(years_start, years_end), years_end

    async def _first_anchor_year(self, model: _ModelRun) -> int | None:
        """Earliest publication year of the anchors whose title names the model (the
        top-cited pull's year, else one title_only lookup per title). Anchors that do
        not name the model never move the window."""
        years: list[int] = []
        for title in model.anchors:
            if not (contains_word(title, model.name) or contains_word(title, model.family)):
                continue
            known = model.anchor_years.get(_stem_text(title))
            if known is None and self._budget_left():
                try:
                    result = await self._counted_execute(build_search_publications_paged_dsl(
                        _or_group([title]), search_in="title_only", limit=20,
                        fields="title+year",
                    ))
                except DimensionsDslError as exc:
                    if not is_dsl_syntax_error(exc):
                        raise
                    continue
                found = [
                    p["year"] for p in publications_from_result(result)
                    if isinstance(p.get("year"), int)
                ]
                known = min(found) if found else None
            if known is not None:
                years.append(known)
        return min(years) if years else None

    async def _mine_model_context(
        self, model: _ModelRun, model_names: list[str], clause: str | None = None
    ) -> list[tuple[str, int]]:
        """Context-term candidates from the abstracts of the precise matches (one call);
        *clause* replaces the model's reference clause."""
        s = self._settings
        n = int(s.ai_model_context_mining_sample)
        clause = model.reference_clause if clause is None else clause
        if not clause or n <= 0 or s.ai_model_max_mined_context_terms <= 0:
            return []
        if not self._budget_left():
            return []
        try:
            result = await self._counted_execute(
                build_search_publications_paged_dsl(
                    clause, limit=n, fields="title+abstract"
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

    async def _measure_flags(
        self,
        model: _ModelRun,
        group: list[str],
        candidates: list[str],
        safe: list[str],
        *,
        ref_clause: str,
        ref_ok: bool,
        unverified: str,
        reserve: bool,
        not_x: str,
    ) -> list[str]:
        """Measure context-flag *candidates* (plus terms mined from the precise matches)
        for the names in *group* against *ref_clause*. Returns the flags that pass; in
        *reserve* mode passing flags go to ``model.reserve_flags`` instead."""
        s = self._settings
        flags: list[str] = []
        mined: dict[str, int] = {}

        g_group = _or_group(group)
        context_terms = list(candidates)
        taken = {t.casefold() for t in context_terms} | {
            d.term.casefold() for d in model.decisions if d.kind == "flag"
        }
        if ref_ok:
            for term, n_docs in await self._mine_model_context(
                model, group + safe, ref_clause
            ):
                if term.casefold() not in taken:
                    taken.add(term.casefold())
                    context_terms.append(term)
                    mined[term.casefold()] = n_docs
        n_measured = n_passed = 0
        for term in context_terms:
            if not ref_ok:
                # Nothing certain to measure against: an unverified flag would
                # let the ambiguous name through on trust.
                model.decide(term, "flag", "rejected", unverified)
                continue
            if reserve and (
                n_measured >= s.ai_model_reserve_flag_max_candidates
                or n_passed >= s.ai_model_reserve_flag_max_passing
            ):
                model.decide(term, "flag", "rejected",
                             "not measured: the reserve-flag limit was reached")
                continue
            n_measured += 1
            admitted = await self._clause_count(f"({g_group} AND {_or_group([term])}){not_x}")
            if not admitted:
                model.decide(
                    term, "flag", "rejected",
                    f"never occurs with {group}" if admitted == 0
                    else "not measured (Dimensions call budget or unparseable term)",
                    admitted,
                )
                continue
            confirmed = await self._clause_count(
                f"({ref_clause} AND {g_group} AND {_or_group([term])}){not_x}"
            )
            if confirmed is None:
                model.decide(term, "flag", "rejected",
                             "precision not measured (Dimensions call budget)", admitted)
                continue
            precision = confirmed / admitted
            stats = {"admitted": admitted, "confirmed": confirmed,
                     "anchor_precision": round(precision, 3)}
            detail = (
                f"admits {admitted} papers with {group}; {confirmed} "
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
            own_hits = await self._clause_count(_or_group([term])) if passes else None
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
            elif passes and reserve:
                stats["own_hits"] = own_hits
                n_passed += 1
                model.reserve_flags.append(term)
                model.decide(
                    term, "flag", "context",
                    f"reserve flag for {group}: {detail}; kept out of the query "
                    "unless the name fails the abstract check",
                    admitted, **stats,
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

        return flags

    async def _measure_reserve_flags(
        self, model: _ModelRun, base: str, safe: list[str], excludes: list[str]
    ) -> None:
        """Measure flags for *base* at re-gate time, once per run, against the papers
        that do not depend on it (``_reference_without``); passing flags become
        ``model.reserve_flags``. Earlier "not needed" flag rows are replaced."""
        model.reserve_measured = True
        clause, count, _label = await self._reference_without(model, {base.casefold()})
        model.decisions = [
            d for d in model.decisions
            if not (d.kind == "flag" and d.reason.startswith("not needed"))
        ]
        await self._measure_flags(
            model, [base], model.context_terms, safe,
            ref_clause=clause, ref_ok=bool(clause) and count > 0,
            unverified=f"cannot be verified: no cited paper title or precise name without {[base]}",
            reserve=True,
            not_x=f" NOT {_or_group(excludes)}" if excludes else "",
        )

    async def _add_title_flags(
        self, model: _ModelRun, group: list[str], flags: list[str], not_x: str
    ) -> list[str]:
        """Join the model's paper titles (``model.title_flags``) to *flags*, the flags of
        the gated own names in *group*. A title the name never occurs with gates
        nothing. Returns the flags with the titles that joined."""
        if not self._settings.ai_model_title_flag_enabled or not model.title_flags:
            return list(flags)
        out = list(flags)
        taken = {f.casefold() for f in flags}
        decided = {
            d.term.casefold() for d in model.decisions
            if d.kind == "flag" and d.reason.startswith("model paper title")
        }
        for title, why in model.title_flags:
            key = title.casefold()
            if key in taken:
                continue
            admitted = await self._clause_count(
                f"({_or_group(group)} AND {_or_group([title])}){not_x}"
            )
            if not admitted:
                if key not in decided:
                    decided.add(key)
                    model.decide(
                        title, "flag", "rejected",
                        f"model paper title ({why}): "
                        + (f"never occurs with {group}" if admitted == 0
                           else "not measured (Dimensions call budget)"),
                        admitted,
                    )
                continue
            taken.add(key)
            out.append(title)
            if key not in decided:
                decided.add(key)
                model.decide(
                    title, "flag", "flag",
                    f"model paper title ({why}); admits {admitted} papers with {group}",
                    admitted, admitted=admitted,
                )
        return out

    async def _title_only_note(
        self, model: _ModelRun, alias: str, titles: list[str], excludes: list[str], ev: dict
    ) -> str:
        """Note for a name gated by the paper title alone: what it keeps of the name's
        papers and the genuine papers lost, estimated from the abstract check."""
        not_x = f" NOT {_or_group(excludes)}" if excludes else ""
        n = await self._clause_count(f"{_or_group([alias])}{not_x}")
        kept = await self._clause_count(f"({_or_group([alias])} AND {_or_group(titles)}){not_x}")
        note = f"{alias!r} gated by the model paper title only"
        if not n or kept is None:
            return note + "; the recall loss was not measured"
        note += f": keeps {kept} of {n} papers with the name"
        genuine = int(ev.get("genuine_count") or 0)
        judged = genuine + int(ev.get("fp_count") or 0)
        if genuine and judged:
            est = n * genuine / judged
            loss = max(0.0, 1.0 - kept / est)
            note += (
                f"; the abstract check puts genuine use at about {round(est)} papers, so "
                f"about {loss:.0%} of genuine papers are lost (estimate from {judged} abstracts)"
            )
        return note

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
    ) -> tuple[list[str], list[str], list[str], list[str]] | None:
        """One refinement step from the abstract check's false positives.

        Excludes come first. An exclude phrase the LLM derives from the false positives
        is kept only if it occurs in one of their snippets and passes
        ``_model_exclude_check``. When a name whose false positives contain the phrase
        is itself in the reference set, the check uses the papers that do not depend
        on that name (``_reference_without``): with at least
        ``ai_model_refine_reference_min_papers`` of them, the exclude is accepted when
        it removes at most ``ai_model_exclude_max_anchor_loss`` of them and something
        from the query. Otherwise the phrase must also remove only a small part of the
        current result (a phrase that removes much describes the topic, not the false
        positive).

        Then names. A safe name with two or more false positives, and no more genuine
        mentions than false ones, stops being safe: it becomes flag-gated when it is the
        model's own name and flags (or measured reserve flags) exist. An own base name
        is never dropped: with no flag to gate it, it is kept and the seed is withheld.
        Other names are dropped. "X model" forms follow their base name. When an exclude
        was accepted in this step, own names are not re-gated in it: the next abstract
        check judges the excluded query first.

        Returns ``(safe, risky, flags, excludes)``, or ``None`` when nothing changes."""
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
        changed = False

        # 1. Excludes.
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
        s = self._settings
        new_excl = list(excludes)
        taken = {t.casefold() for t in [*excludes, *flags, *model.names]}
        tried = {d.term.casefold() for d in model.decisions if d.kind == "exclude"}
        snippets = " ".join(str(h.get("snippet") or "") for h in fp_hits).casefold()
        in_reference = {n.casefold() for n in model.reference_names}
        total = int(ev.get("publications_total") or 0)
        max_removed = max(1, int(total * s.ai_model_refine_exclude_max_share))
        added = 0
        accepted_now: list[str] = []
        for term in suggested[: s.ai_model_max_exclude_terms]:
            key = term.casefold()
            if key in taken or key in tried or is_short_acronym(term):
                continue
            if added >= s.ai_model_refine_max_excludes:
                break
            tried.add(key)
            holder = next((e for e in new_excl if phrase_contains(term, e)), None)
            if holder is not None:
                # Every paper it matches is already excluded: no call, no slot.
                model.decide(term, "exclude", "rejected",
                             f"from abstract false positives; contains the accepted exclude {holder!r}")
                continue
            if key not in snippets:
                model.decide(term, "exclude", "rejected",
                             "from abstract false positives; not a phrase of any of them")
                continue
            # Names whose false positives hold the phrase and that vouch for the
            # reference set themselves: measuring against them would be circular.
            refined = {
                str(h.get("matched_alias") or "").casefold() for h in fp_hits
                if key in str(h.get("snippet") or "").casefold()
            } & in_reference
            reference: tuple[str, int] | None = None
            if refined:
                clause, count, _label = await self._reference_without(model, refined)
                if count >= s.ai_model_refine_reference_min_papers:
                    reference = (clause, count)
            if reference is not None:
                # The loss bound covers every exclude accepted this way in the run.
                ok, reason, stats = await self._model_exclude_check(
                    model, term, reference, also=model.refine_excludes
                )
                stats["reference_without"] = sorted(refined)
                if model.refine_excludes:
                    stats["loss_set"] = [*model.refine_excludes, term]
            else:
                ok, reason, stats = await self._model_exclude_check(model, term, reference)
            if ok and for_clause:
                beyond = f" NOT {_or_group(accepted_now)}" if accepted_now else ""
                removed = await self._clause_count(
                    f"(({for_clause}){beyond}) AND {_or_group([term])}", where=model.where
                )
                stats["removed"] = removed
                if removed is None:
                    ok, reason = False, "effect not measured"
                elif removed == 0:
                    ok, reason = False, (
                        "removes nothing beyond the excludes already accepted"
                        if accepted_now else "removes nothing from the query"
                    )
                elif reference is None and removed > max_removed:
                    ok = False
                    reason = (
                        f"would remove {removed} of {total} results: too broad for a phrase "
                        "taken from a few false positives"
                    )
                elif reference is not None:
                    reason += f" without {sorted(refined)}; removes {removed} of {total} results"
            added += 1 if ok else 0
            model.decide(
                term, "exclude", "exclude" if ok else "rejected",
                f"from abstract false positives; {reason}", **stats,
            )
            if ok:
                new_excl.append(term)
                accepted_now.append(term)
                if reference is not None:
                    model.refine_excludes.append(term)
                changed = True
        exclude_accepted = added > 0

        # 2. Names.
        new_safe: list[str] = []
        new_risky = list(risky)
        new_flags = list(flags)
        regated: set[str] = set()

        def _base_of(alias: str) -> str | None:
            key = alias.casefold()
            return next((b for b in model.base_keys
                         if key in (f"{b} model", f"{b} foundation model")), None)

        followers = [a for a in safe if _base_of(a)]
        for alias in [a for a in safe if not _base_of(a)]:
            key = alias.casefold()
            n_fp = per_alias.get(key, 0)
            # At least two false positives; a name that is not the model's own also needs
            # at least as many false positives as genuine mentions. An own name is
            # re-gated, never dropped, so the release threshold already bounds it.
            if n_fp < 2 or (key not in model.own_keys and n_fp < genuine.get(key, 0)):
                new_safe.append(alias)
                continue
            if (
                key in model.own_keys and not exclude_accepted and not new_flags
                and not model.reserve_flags and not model.reserve_measured
            ):
                # Flags were never measured for this name: measure them now, against
                # the papers that do not depend on it.
                await self._measure_reserve_flags(model, alias, safe, excludes)
            gate_flags: list[str] = []
            if key in model.own_keys and not exclude_accepted:
                gate_flags = await self._add_title_flags(
                    model, [alias], new_flags or list(model.reserve_flags),
                    f" NOT {_or_group(excludes)}" if excludes else "",
                )
            if key in model.own_keys and exclude_accepted:
                new_safe.append(alias)
                notes_parts.append(
                    f"gate_iter{iteration}: {alias!r} not re-gated: an exclude was accepted "
                    "first, the next abstract check judges it"
                )
            elif key in model.own_keys and gate_flags:
                changed = True
                regated.add(key)
                new_risky.append(alias)
                measured = new_flags or list(model.reserve_flags)
                titles = [f for f in gate_flags if f not in measured]
                if new_flags and not titles:
                    notes_parts.append(
                        f"gate_iter{iteration}: {alias!r} flag-gated ({n_fp} false positives in abstracts)"
                    )
                elif measured:
                    notes_parts.append(
                        f"gate_iter{iteration}: {alias!r} flag-gated with "
                        + ("flags" if new_flags else "reserve flags") + f" {measured}"
                        + (f" and the model paper title {titles}" if titles else "")
                        + f" ({n_fp} false positives in abstracts)"
                    )
                else:
                    notes_parts.append(
                        f"gate_iter{iteration}: "
                        + await self._title_only_note(model, alias, titles, excludes, ev)
                        + f" ({n_fp} false positives in abstracts)"
                    )
                new_flags = gate_flags
            elif key in model.base_keys:
                # Dropping the model's own name would ship only its rare variants.
                new_safe.append(alias)
                measured = [
                    f"{d.term} {d.stats['anchor_precision']:.0%}" for d in model.decisions
                    if d.kind == "flag" and "anchor_precision" in d.stats
                ]
                why = (
                    f"{alias!r} kept: no measured flag can gate it "
                    f"({', '.join(measured) or 'no flag candidate was measured'})"
                )
                if why not in model.withhold_reasons:
                    model.withhold_reasons.append(why)
                notes_parts.append(f"gate_iter{iteration}: {why}; {n_fp} false positives in abstracts")
            else:
                changed = True
                notes_parts.append(
                    f"gate_iter{iteration}: {alias!r} dropped ({n_fp} false positives in abstracts)"
                )
        for alias in followers:
            if _base_of(alias) in regated:
                changed = True
                new_risky.append(alias)
            else:
                new_safe.append(alias)
        if not changed:
            return None
        return new_safe, dedupe_strings_ci_preserve_order(new_risky), new_flags, new_excl

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
            version_siblings=s.ai_model_version_siblings,
            max_titles=s.ai_model_max_paper_titles,
        )
        model.description = cands.description
        model.developer = cands.developer
        # The caller's excludes are their own list: decided once the gate is known.
        caller_excl = {
            (sanitize_term(t) or t).casefold() for t in request.exclude_terms if isinstance(t, str)
        }
        caller_screened: list[tuple[str, str]] = []
        for kind, term, reason in cands.rejected:
            if kind == "exclude" and term.casefold() in caller_excl:
                caller_screened.append((term, reason))
                continue
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
        model.base_keys = set(base_keys)
        given_keys = {a.casefold() for a in request.dataset_names}
        try:
            # 1. Paper titles: anchors for everything measured below.
            titles: list[str] = []
            # Coined base names: only a title naming one of them can anchor off the card.
            coined_bases = [
                b for b in dedupe_strings_ci_preserve_order([name, family])
                if is_distinctive(b) and not is_plain_word(b) and not is_short_acronym(b)
            ]
            for title in cands.paper_titles:
                from_hub = title.casefold() in cands.hub_title_keys
                if not from_hub:
                    # A title the LLM recalls proves little: "Aurora kinase inhibitors
                    # in cancer therapy" also names "Aurora". Off the model card, only a
                    # title naming a coined name that the model-aware check confirms
                    # anchors the measurements.
                    ok, reason, c = await self._model_title_check(model, title, coined_bases)
                    if ok:
                        titles.append(title)
                        model.decide(title, "paper_title", "anchor",
                                     f"not cited on the model card; {reason}", c)
                        if contains_word(title, name) or contains_word(title, family):
                            model.title_flags.append(
                                (title, "names the model; confirmed by the model-aware check")
                            )
                    else:
                        model.decide(title, "paper_title", "rejected", reason, c)
                    continue
                c = await self._clause_count(q(title))
                if not c:
                    model.decide(title, "paper_title", "rejected",
                                 "0 Dimensions hits" if c == 0 else "not measured", c)
                    continue
                names_model = contains_word(title, name) or contains_word(title, family)
                if names_model:
                    reason = "cited on the model card; the title names the model"
                    flag_why = "cited on the model card and names the model"
                else:
                    if c > s.ai_model_title_max_hits:
                        model.decide(title, "paper_title", "rejected",
                                     f"cited on the model card, but quoted by {c} papers: too "
                                     "widely cited to single out this model", c)
                        continue
                    both = await self._clause_count(f"{q(title)} AND {q(name, family)}")
                    if both is None:
                        model.decide(title, "paper_title", "rejected",
                                     "not measured (Dimensions call budget)", c)
                        continue
                    share = both / c
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
                    flag_why = reason
                model.title_flags.append((title, flag_why))
                titles.append(title)
                model.decide(title, "paper_title", "anchor", reason, c)
            decided = {_stem_text(t) for t in cands.paper_titles} | {
                _stem_text(d.term) for d in model.decisions if d.kind == "paper_title"
            }
            for title, _reason, _c in await self._top_cited_titles(model, coined_bases, decided):
                titles.append(title)
                model.title_flags.append(
                    (title, "a top-cited title naming the model; confirmed by the model-aware check")
                )
            # Joined version spellings ("FourCastNetv2") seen in the texts pulled above:
            # candidates like any name, within the name limit.
            searched = {search_key(n) for n, _k in cands.names} | {
                search_key(t) for k, t, _r in cands.rejected if k == "alias"
            }
            mined_names: list[str] = []
            mined = mine_joined_versions(model.pulled_texts, coined_bases)
            if s.ai_model_version_siblings:
                mined = [t for tok in mined
                         for t in (tok, version_sibling(tok, coined_bases)) if t]
            for tok in mined:
                if search_key(tok) in searched:
                    continue
                searched.add(search_key(tok))
                if len(cands.names) >= s.ai_model_max_names:
                    model.decide(tok, "alias", "rejected",
                                 "joined version found in pulled abstracts; over the candidate limit")
                    continue
                cands.names.append((tok, "version"))
                mined_names.append(tok)
            if mined_names:
                notes_parts.append(f"model research: joined versions found in abstracts {mined_names}")
            model.anchors = list(titles)
            title_group = q(*titles) if titles else ""
            # The window that ships: the abstract check and the recorded counts use it.
            model.years = await self._model_years(request, model)

            # 2. Names: count, then classify by shape and by title confirmation.
            safe: list[str] = []
            gate: list[tuple[str, int, str]] = []
            counts: dict[str, int] = {}
            for nm, _kind in cands.names:
                follows = next((b for b in base_keys
                                if nm.casefold() in (f"{b} model", f"{b} foundation model")), None)
                if follows and follows not in {g.casefold() for g, _c, _r in gate}:
                    # "X model" adds papers only when the bare name is not searched alone.
                    continue
                both: int | None = None
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
                    plain_word_max_hits=s.ai_model_plain_word_max_safe_hits,
                    coined_min_share=s.ai_model_coined_min_title_share,
                )
                stats = {} if confirmation is None else {"title_confirmation": round(confirmation, 3)}
                if nm.casefold() in base_keys and title_group and both is not None:
                    # Recorded only: hits of the base name that also quote an anchor.
                    stats["with_anchor_titles"] = both
                if (
                    verdict == "safe" and nm.casefold() in base_keys and is_distinctive(nm)
                    and (confirmation is None
                         or confirmation < s.ai_model_safe_min_context_share / 2)
                ):
                    model.weak_bases.append(nm)
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
                cover = next((g for g, _c, _r in gate if phrase_contains(nm, g)), None)
                if cover:
                    model.decide(nm, "alias", "rejected",
                                 f"{reason}; already covered by flag-gated {cover!r}", c, **stats)
                else:
                    model.decide(nm, "alias", "rejected",
                                 f"{reason}; not confirmed as a name of this model", c, **stats)

            # 3. Reference set: papers that certainly refer to the model.
            reference = dedupe_strings_ci_preserve_order(titles + safe)
            model.reference_names = list(safe)
            model.reference_clause = q(*reference) if reference else ""
            model.reference_count = (
                await self._clause_count(model.reference_clause) or 0 if reference else 0
            )
            # A safe name containing another safe name adds no papers to the query.
            covered = covered_safe_names(safe)
            for a, cover in covered.items():
                d = next(d for d in model.decisions if d.kind == "alias" and d.term == a)
                d.decision = "rejected"
                d.reason += (f"; covered by {cover!r}: kept in the reference set, left out "
                             "of the query")
            safe = [a for a in safe if a not in covered]

            # 4. Excludes (other meanings of the name).
            excludes: list[str] = []
            if not gate:
                # Nothing is gated: a caller-given exclude has nothing to act on, and a
                # check against the query's own papers would be circular.
                for term in [t for t, _r in caller_screened] + [
                    t for t in cands.other_meanings if t.casefold() in caller_excl
                ]:
                    model.decide(term, "exclude", "rejected",
                                 "caller-given; not applied: no name is gated")
            else:
                for term, reason in caller_screened:
                    model.decide(term, "exclude", "rejected", reason)
            for term in cands.other_meanings:
                if not gate and term.casefold() in caller_excl:
                    continue
                if not gate:
                    # Every name is safe: no measured other meaning exists to exclude,
                    # and a check against the query's own papers would be circular.
                    model.decide(term, "exclude", "rejected",
                                 "not needed: no name has another measured meaning")
                    continue
                ok, reason, stats = await self._model_exclude_check(model, term)
                model.decide(term, "exclude", "exclude" if ok else "rejected", reason, **stats)
                if ok:
                    excludes.append(term)
            not_x = f" NOT {q(*excludes)}" if excludes else ""

            # 5. Context flags, only when a name needs gating. When nothing is gated but
            # an own coined base name is safe on weak evidence (shape, or a partial title
            # share), flags are still measured, against a reference that does not
            # contain that name, and held in reserve for the abstract check.
            flags: list[str] = []
            model.context_terms = list(cands.context_terms)
            gated = [g for g, _c, _r in gate]
            weak = [] if gated else [b for b in model.weak_bases if b in safe]
            reserve = bool(weak)
            group = gated or weak
            if not group:
                for term in cands.context_terms:
                    model.decide(term, "flag", "rejected", "not needed: no name requires gating")
            else:
                if reserve:
                    ref_terms = dedupe_strings_ci_preserve_order(titles + [
                        n for n in safe if not any(contains_word(n, b) for b in weak)
                    ])
                    ref_clause = q(*ref_terms) if ref_terms else ""
                    ref_ok = bool(ref_terms)
                    unverified = (
                        f"cannot be verified: no cited paper title or precise name without {weak}"
                    )
                else:
                    ref_clause = model.reference_clause
                    ref_ok = bool(model.reference_clause) and model.reference_count > 0
                    unverified = "cannot be verified: no precise name or cited paper title"
                flags = await self._measure_flags(
                    model, group, list(cands.context_terms), safe,
                    ref_clause=ref_clause, ref_ok=ref_ok, unverified=unverified,
                    reserve=reserve, not_x=not_x,
                )
                model.reserve_measured = reserve
                if gated:
                    flags = await self._add_title_flags(model, gated, flags, not_x)

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
        out.ai_model_judge_context = model.context or None
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
        flag_k = {a.casefold() for a in flags}
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
            elif d.kind == "flag" and d.decision == "context" and key in flag_k:
                d.decision = "flag"
                d.reason += "; flag-gates the name after false positives in abstracts"
            elif d.kind == "exclude" and d.decision == "exclude" and key not in excl_k:
                d.decision = "rejected"
                d.reason += "; not in the selected query"

        if model.years is None:
            model.years = await self._model_years(request, model)
        years_start, years_end = model.years
        out.dsl_query = (
            f'search publications in full_data for "{out.for_clause}" '
            f"where year in [{years_start}:{years_end}] "
            "return publications[id+title+doi+year+times_cited] sort by times_cited"
        )

        notes = [out.notes] if out.notes else []
        stats = model.abstract_stats.get(out.for_clause or "")
        if stats is not None:
            fp_k, judged, unjudged, coverage = stats
            lo, hi = wilson_interval(fp_k, judged)
            out.abstract_fp_count, out.abstract_n_judged = fp_k, judged
            out.abstract_n_counted_without_judgement = unjudged
            out.abstract_fp_wilson_95 = [round(lo, 4), round(hi, 4)]
            out.abstract_coverage_pct = round(coverage, 1)
            notes.append(
                f"abstract check: {fp_k} of {judged} false positives (Wilson 95% "
                f"{lo:.1%}-{hi:.1%}), coverage {coverage:.1f}%"
            )
        threshold = float(s.optimize_fp_threshold_pct)
        evaluated = "~gate" in (out.selected_variant or "")
        ft = bool(s.ai_model_fulltext_enabled)
        abstract_why = ""
        if not evaluated or out.fp_rate_pct is None or float(out.fp_rate_pct) > threshold:
            # Never hand the pipeline a query the abstract check did not clear (unless
            # full text decides, below).
            if not evaluated:
                why = "the abstract check did not run on the selected query"
            elif out.fp_rate_pct is None:
                why = "the abstract check classified none of the papers, so no false-positive rate exists"
            else:
                why = (
                    f"the abstract check measured {float(out.fp_rate_pct):.0f}% false "
                    f"positives (limit {threshold:.0f}%)"
                )
            out.confidence = ConfidenceLevel.LOW
            if model.withhold_reasons:
                why += "; " + "; ".join(model.withhold_reasons)
            if not ft:
                notes.append(f"seed withheld: {why}; review term_decisions before using this query")
                out.notes = " ".join(notes)
                return out
            abstract_why = why
        else:
            # Released: confidence from the Wilson upper bound of the abstract check.
            out.confidence = ai_model_confidence(*(stats or (0, 0, 0, 0.0))[:2], released=True)
        if not await self._attach_model_seed(request, model, out, notes):
            out.notes = " ".join(notes)
            return out
        if ft:
            await self._fulltext_release(request, model, out, notes, abstract_why)
        out.notes = " ".join(notes)
        return out

    async def _attach_model_seed(
        self, request: OptimizeRequest, model: _ModelRun, out: QueryOptimizationRecord,
        notes: list[str],
    ) -> bool:
        """Build the extraction seed for the record's terms and count it next to the
        selected query. False (with a note) when no seed can be built."""
        safe, risky = list(out.aliases.safe), list(out.aliases.risky)
        flags, excludes = list(out.flag_terms), list(out.exclusion_terms)
        years_start, years_end = model.years
        candidate = QueryCandidate(
            main_dataset_name=model.name, safe=safe, risky=risky, flag_terms=flags,
            exclude_terms=excludes, description=model.context,
        )
        try:
            seed = candidate_to_ai_model_seed(
                candidate, model_name=model.name, model_family=model.family,
                hugging_face_id=request.hugging_face_id, years=(years_start, years_end),
            )
        except ValueError as exc:
            notes.append(f"seed not built: {exc}")
            return False
        seed_names = [seed["model_name"], *seed["model_aliases"]]
        out.ai_model_seed = seed
        out.ai_model_seed_clause = ai_model_seed_clause(
            seed_names, seed["flag_terms"], seed["exclude_terms"]
        )
        out.ai_model_seed_count = await self._clause_count(
            out.ai_model_seed_clause, where=model.where
        )
        out.ai_model_query_count = await self._clause_count(out.for_clause, where=model.where)
        out.ai_model_query_count_all_years = await self._clause_count(out.for_clause)
        if seed["model_name"].casefold() != model.name.casefold():
            notes.append(
                f"seed: model_name is {seed['model_name']!r} because {model.name!r} is not "
                "searched on its own"
            )
        if seed["flag_terms"] and safe:
            notes.append(
                "seed: flag terms apply to every name in the extraction pipeline, so the "
                "safe names are sent as flag terms too; "
                f"it finds {out.ai_model_seed_count} papers ({years_start}-{years_end})"
            )
        if out.ai_model_seed_count != out.ai_model_query_count:
            notes.append(
                f"seed: the seed finds {out.ai_model_seed_count} papers and the selected "
                f"query {out.ai_model_query_count} ({years_start}-{years_end}); the counts differ"
            )
        return True

    async def _probe_model_seed(
        self, model: _ModelRun, seed: dict, *, max_batches: int | None = None
    ) -> Evidence | None:
        """Probe *seed* exactly on extraction full text (cleaned up by the client), then
        run the referent check on the genuine papers. None when the probe failed."""
        s = self._settings
        try:
            # Inside the try: a missing or unsupported extraction target is a
            # note ("full text not evaluated"), never a failed run.
            client = self._get_extraction()
            top = await asyncio.to_thread(functools.partial(
                client.run_probe_topup, seed, target=s.fulltext_target_delivered,
                batch_size=s.fulltext_batch_size,
                max_batches=max_batches or s.fulltext_max_batches,
                pull_cap=s.probe_pull_cap, entity="ai_model",
            ))
        except (ExtractionError, ValueError) as exc:
            logger.warning("model full-text probe failed: %s", exc)
            self._model_probe_error = str(exc)
            return None
        ev = build_evidence(top.results, entity="ai_model")
        ev.target, ev.batches = top.target, top.batches
        ev.stop_reason, ev.budget_exhausted = top.stop_reason, top.budget_exhausted
        await asyncio.to_thread(functools.partial(
            check_referents, ev, agent=self._referent_agent or self._agent,
            model_name=model.name, profile=model.context, anchor_titles=list(model.anchors),
            votes=s.referent_votes,
        ))
        return ev

    @staticmethod
    def _ft_summary(ev: Evidence) -> str:
        n, k = ev.n_evaluated, len(ev.genuine)
        p = k / n if n else 0.0
        lo, hi = wilson_interval(k, n)
        return (
            f"{k} of {n} genuine ({p:.2f}, Wilson 95% {lo:.2f}-{hi:.2f}; judge "
            f"{(ev.raw_precision or 0):.2f} before {ev.n_referent_rejected} referent rejections)"
        )

    async def _measure_seed(
        self, model: _ModelRun, out: QueryOptimizationRecord, notes: list[str], iteration: int
    ) -> Evidence | None:
        """Probe the record's seed, top it up once when borderline, and record it."""
        s = self._settings
        bar = s.ai_model_fulltext_min_precision
        ev = await self._probe_model_seed(model, out.ai_model_seed)
        if ev is not None and ev.n_evaluated:
            p = ev.corrected_precision or 0.0
            if (
                abs(p - bar) <= 0.07 and ev.n_evaluated < s.fulltext_target_delivered
                and ev.stop_reason != STOP_POOL_EXHAUSTED
            ):
                notes.append(f"full text borderline at {p:.2f} on {ev.n_evaluated}: topped up once")
                ev = await self._probe_model_seed(
                    model, out.ai_model_seed, max_batches=2 * s.fulltext_max_batches
                ) or ev
        if ev is None:
            why = getattr(self, "_model_probe_error", None)
            notes.append(f"full text probe {iteration + 1} failed"
                         + (f" ({why})" if why else "") + "; nothing measured")
            return None
        out.fulltext_iterations.append(
            ev.metrics(iteration, out.for_clause or "", out.fp_rate_pct)
        )
        return ev

    def _withhold(self, out: QueryOptimizationRecord, notes: list[str], why: str) -> None:
        out.ai_model_seed = None
        out.confidence = ConfidenceLevel.LOW
        notes.append(f"seed withheld: {why}; review term_decisions before using this query")

    def _ft_confidence(self, ev: Evidence) -> ConfidenceLevel:
        """Full text decided: HIGH at >= 0.90 on a sample that reached its target,
        otherwise MEDIUM."""
        p = ev.corrected_precision or 0.0
        full = ev.target is None or ev.n_delivered >= ev.target
        return ConfidenceLevel.HIGH if p >= 0.90 and full else ConfidenceLevel.MEDIUM

    async def _fulltext_release(
        self, request: OptimizeRequest, model: _ModelRun, out: QueryOptimizationRecord,
        notes: list[str], abstract_why: str,
    ) -> None:
        """S4: full text decides release both ways when enough papers were judged.

        n >= ``ai_model_fulltext_min_papers`` and corrected precision >= the bar: release.
        Below the bar: re-gate the own names once (measured, reserve or title flags),
        probe again, release at the bar, else withhold. Fewer papers: the abstract rule
        stands and full text is reported only."""
        s = self._settings
        need, bar = s.ai_model_fulltext_min_papers, s.ai_model_fulltext_min_precision
        # The judge that decides release: recorded, and marked when it is not the measured
        # configuration (e.g. the openai-style fallback drops thinking and sampling).
        judge = {**self._judge_config(), "votes": int(s.referent_votes)}
        out.referent_judge_config = judge
        notes.append(self._judge_config_note(
            "full-text judge configuration", judge, "vote(s) per paper", noun="judge configuration"))
        ev = await self._measure_seed(model, out, notes, 0)
        if ev is None or ev.n_evaluated < need:
            n = 0 if ev is None else ev.n_evaluated
            notes.append(
                f"full text not decisive: {n} papers judged (fewer than {need})"
                + (f"; {self._ft_summary(ev)}" if ev is not None and n else "")
            )
            if abstract_why:
                self._withhold(out, notes, abstract_why)
            return
        if (ev.corrected_precision or 0.0) >= bar:
            notes.append(f"full text released the seed: {self._ft_summary(ev)}")
            out.confidence = self._ft_confidence(ev)
            return
        first = self._ft_summary(ev)
        if not await self._regate_after_fulltext(request, model, out, notes):
            self._withhold(out, notes, f"full text measured {first}; no flag can gate the names")
            return
        notes.append(f"re-gated after full text measured {first}")
        ev2 = await self._measure_seed(model, out, notes, 1)
        if ev2 is not None and ev2.n_evaluated >= need and (ev2.corrected_precision or 0.0) >= bar:
            notes.append(f"full text released the re-gated seed: {self._ft_summary(ev2)}")
            out.confidence = self._ft_confidence(ev2)
            return
        self._withhold(
            out, notes,
            "full text measured "
            + (self._ft_summary(ev2) if ev2 is not None and ev2.n_evaluated else first)
            + " after one re-gate",
        )

    async def _regate_after_fulltext(
        self, request: OptimizeRequest, model: _ModelRun, out: QueryOptimizationRecord,
        notes: list[str],
    ) -> bool:
        """Gate the record's safe own names with the existing flags (else reserve flags,
        measured once) plus the model's paper titles; rebuild the query and the seed."""
        safe, risky = list(out.aliases.safe), list(out.aliases.risky)
        flags, excludes = list(out.flag_terms), list(out.exclusion_terms)
        own = [a for a in safe if a.casefold() in model.own_keys]
        if not own:
            return False
        if not flags and not model.reserve_flags and not model.reserve_measured:
            await self._measure_reserve_flags(model, own[0], safe, excludes)
        not_x = f" NOT {_or_group(excludes)}" if excludes else ""
        gate = await self._add_title_flags(model, own, flags or list(model.reserve_flags), not_x)
        if not gate:
            return False
        safe = [a for a in safe if a not in own]
        risky = dedupe_strings_ci_preserve_order(risky + own)
        clause = build_for_clause(safe=safe, variant="V4", risky=risky, flag_terms=gate,
                                  exclusion_terms=excludes)
        model.remember(clause, safe, risky, gate, excludes)
        out.for_clause, out.selected_variant = clause, f"{out.selected_variant}~ft"
        out.aliases = QueryOptimizationAliases(safe=safe, risky=risky)
        out.flag_terms = list(gate)
        lo, hi = model.years
        out.dsl_query = (
            f'search publications in full_data for "{clause}" where year in [{lo}:{hi}] '
            "return publications[id+title+doi+year+times_cited] sort by times_cited"
        )
        return await self._attach_model_seed(request, model, out, notes)

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

    def _split_release(self) -> bool:
        """A dataset the resource map splits (the broad program query): its release bar is
        the unrelated share, not precision."""
        rmap = self._resource_map
        return rmap is not None and bool(rmap.program_names)

    def _release_bar_text(self) -> str:
        s = self._settings
        if self._split_release():
            return (f"split dataset: unrelated share <= {s.fulltext_split_unrelated_max:.2f} "
                    f"and confirmed data share >= {s.fulltext_split_data_min:.2f}")
        return f"precision >= {s.fulltext_recall_min_precision:.2f}"

    def _need(self, need: int | None) -> int:
        """The minimum sample: *need* when given (the release decision passes
        ``_release_min_sample``), else ``fulltext_recall_min_evidence`` (the loop)."""
        return int(self._settings.fulltext_recall_min_evidence if need is None else need)

    def _release_min_sample(self) -> int:
        """Min evaluated papers for the release bar: ``fulltext_split_min_sample`` for a
        split dataset, ``fulltext_recall_min_evidence`` for every other dataset."""
        s = self._settings
        return int(s.fulltext_split_min_sample if self._split_release()
                   else s.fulltext_recall_min_evidence)

    def _random_release(self) -> bool:
        """Whether the random release sample decides (``fulltext_release_sampling``)."""
        mode = self._settings.fulltext_release_sampling
        return mode == "random_all" or (mode == "random_split" and self._split_release())

    def _meets_unrelated_bar(self, m: IterationMetrics, need: int | None = None) -> bool:
        """Split dataset: unrelated share <= ``fulltext_split_unrelated_max`` on at least
        *need* (default ``fulltext_recall_min_evidence``) evaluated papers (data,
        program-only and references-only papers all count toward it)."""
        s = self._settings
        return (m.n_evaluated >= self._need(need)
                and m.unrelated_share is not None
                and m.unrelated_share <= s.fulltext_split_unrelated_max)

    def _meets_data_floor(self, m: IterationMetrics, need: int | None = None) -> bool:
        """Split dataset: confirmed data share >= ``fulltext_split_data_min`` on at least
        *need* evaluated papers (the unrelated bar's sample)."""
        s = self._settings
        return (m.n_evaluated >= self._need(need)
                and self._confirmed_data_share(m) >= s.fulltext_split_data_min)

    def _meets_split_target(self, m: IterationMetrics, need: int | None = None) -> bool:
        """Split dataset: the unrelated bar and the data floor; the full-text loop refines
        until this holds or ``optimize_max_iterations`` rounds have run."""
        return self._meets_unrelated_bar(m, need) and self._meets_data_floor(m, need)

    def _meets_release_bar(self, m: IterationMetrics, precision: float | None = None,
                           need: int | None = None) -> bool:
        """The release bar on at least ``fulltext_recall_min_evidence`` evaluated papers.
        A split dataset: the unrelated bar and the data floor (``fulltext_split_data_min``
        on the confirmed data share); any other dataset: precision (*precision*, else the
        iteration's) >= ``fulltext_recall_min_precision``. ``release_bar_met``."""
        s = self._settings
        if m.n_evaluated < self._need(need):
            return False
        if self._split_release():
            return self._meets_split_target(m, need)
        p = m.fulltext_precision if precision is None else precision
        return p is not None and p >= s.fulltext_recall_min_precision

    def _released(self, m: IterationMetrics, precision: float | None = None,
                  need: int | None = None) -> bool:
        """Whether the selected iteration is released (MEDIUM confidence or above). Any
        dataset: the release bar. A split dataset that meets the unrelated bar but misses
        the data floor after the round limit is released with a warning unless
        ``fulltext_split_floor_required`` (default false: the owner's decision); it is
        never HIGH and ``release_bar_met`` stays false."""
        if self._meets_release_bar(m, precision, need):
            return True
        return (self._split_release() and self._meets_unrelated_bar(m, need)
                and not self._settings.fulltext_split_floor_required)

    def _selection_key(self, m: IterationMetrics) -> tuple[bool, bool, float, float, int]:
        """Rank full-text iterations. Any other dataset: the release bar, then precision.
        A split dataset: the unrelated limit, then the data floor, then precision, then
        the confirmed data share; so an iteration that reached the floor wins, and
        precision-then-share decides only among iterations on the same side of the floor
        (the owner's rule for the release below the floor). Then the earlier iteration
        (always the last element)."""
        if self._split_release():
            met = self._meets_unrelated_bar(m)
            floor = met and self._meets_data_floor(m)
            data = self._confirmed_data_share(m) if met else 0.0
            return (met, floor, float(m.fulltext_precision or 0.0), data, -m.iteration)
        return (self._meets_release_bar(m), False, float(m.fulltext_precision or 0.0), 0.0,
                -m.iteration)

    @staticmethod
    def _confirmed_data_count(m: IterationMetrics) -> int:
        return sum(1 for p in (m.paper_details or [])
                   if p.get("verdict") in ("used", "mentioned") and p.get("scope") == "data")

    def _data_floor(self, m: IterationMetrics, rounds: int, need: int | None = None) -> dict:
        """The data floor for *m*: min, share, wilson95, met, required, rounds."""
        s = self._settings
        k, n = self._confirmed_data_count(m), m.n_evaluated
        lo, hi = wilson_interval(k, n) if n else (0.0, 0.0)
        return {"min": s.fulltext_split_data_min, "share": round(k / n, 4) if n else 0.0,
                "wilson95": [round(lo, 4), round(hi, 4)], "met": self._meets_data_floor(m, need),
                "required": s.fulltext_split_floor_required, "rounds": rounds}

    @staticmethod
    def _confirmed_data_share(m: IterationMetrics) -> float:
        """Genuine papers the scope check labelled ``data`` / papers evaluated. A paper
        whose scope call was ``unclear`` (a failed call) is not counted as data."""
        if not m.paper_details or not m.n_evaluated:
            return 0.0
        return QueryOptimizationUseCase._confirmed_data_count(m) / m.n_evaluated

    def _release_note(self, bar: str, met: bool, m: IterationMetrics | None,
                      gate_error: bool, floor: dict | None = None,
                      need: int | None = None) -> str:
        """The bar applied, whether it was met and why, and for a split dataset the
        four-way split with Wilson intervals and the data-only precision."""
        need = self._need(need)
        if m is None:
            why = "not met: no full-text sample reached the minimum sample"
        elif m.n_evaluated < need:
            why = f"not met: {m.n_evaluated} of {need} papers, below the minimum sample"
        elif gate_error:
            why = "not met: a gate evaluation failed"
        else:
            why = "met" if met else "not met"
            if floor is not None and not floor["met"] and self._meets_unrelated_bar(m, need):
                rounds = floor["rounds"]
                lo, hi = floor["wilson95"]
                missed = (f"data floor {floor['min']:.2f} not reached after {rounds} "
                          f"round{'s' if rounds != 1 else ''}")
                why = (f"not met: {missed}" if floor["required"] else
                       f"not met, released with a warning: {missed}; released at data share "
                       f"{floor['share']:.2f} (Wilson 95% {lo:.2f}-{hi:.2f})")
        note = f"full-text release bar: {bar} ({why})"
        if floor is not None:
            lo, hi = floor["wilson95"]
            note += (f"; confirmed data share {floor['share']:.2f} (Wilson 95% "
                     f"{lo:.2f}-{hi:.2f}) vs floor {floor['min']:.2f}")
        if m is None:
            return note
        if m.four_way:
            parts = [
                f"{k} {v['count']} ({v['share']:.2f}, Wilson 95% "
                f"{v['wilson95'][0]:.2f}-{v['wilson95'][1]:.2f})"
                for k, v in m.four_way.items()
            ]
            note += f": {', '.join(parts)}; data-only precision {(m.data_share or 0):.2f}"
            if not m.four_way.get("data", {}).get("count"):
                note += " (warning: no sampled paper refers to the data resource)"
            note += " next to"
        else:
            note += ":"
        return note + f" precision {(m.fulltext_precision or 0):.2f} on {m.n_evaluated} papers"

    # -- the release sample (adapters.release_sample) ------------------------------------

    async def _year_counts(self, for_clause: str) -> dict[int, int]:
        """Per-year Dimensions counts of *for_clause* in the dataset window (one facet call),
        over the population the probe seed asks for: its publication types and, when set,
        the US-affiliation filter (``research_org_countries = "US"``, as extraction applies)."""
        import json

        from dataset_agent.adapters.release_sample import year_counts_from_result

        s = self._settings
        ys, ye = self._dataset_years()
        types = list(s.default_publication_types) or ["article"]
        where = f"year in [{ys}:{ye}] and type in {json.dumps(types)}"
        if s.default_filter_us_affiliation:
            where += ' and research_org_countries = "US"'
        dsl = (f'search publications in full_data for "{for_clause}" '
               f'where {where} return year limit 50')
        self._release_reserve = 0  # this is the call that was held back
        result = await self._counted_execute(dsl)
        raise_for_dsl_errors(result)
        return year_counts_from_result(result)

    def _sample_summary(self, m: IterationMetrics) -> dict:
        """Point estimates with Wilson 95% intervals for a release sample's metrics."""
        n = m.n_evaluated
        k = sum(1 for _pid, v in m.papers if v in ("used", "mentioned"))

        def est(count: int) -> dict:
            lo, hi = wilson_interval(count, n) if n else (0.0, 0.0)
            return {"count": count, "value": round(count / n, 4) if n else None,
                    "wilson95": [round(lo, 4), round(hi, 4)]}

        out: dict = {"n_sent": m.n_publications, "n_delivered": m.n_fulltext_delivered,
                     "n_evaluated": n, "precision": est(k)}
        if self._split_release():
            unrel = (m.four_way or {}).get("unrelated", {}).get("count")
            out["unrelated_share"] = est(int(unrel or 0)) if m.four_way else None
            out["data_share"] = est(self._confirmed_data_count(m))
            out["four_way"] = m.four_way
        return out

    async def _random_release_sample(self, candidate: QueryCandidate, request: OptimizeRequest,
                                     for_clause: str, iteration: int, *, decides: bool
                                     ) -> tuple[IterationMetrics | None, dict]:
        """Draw the random, year-stratified release sample of *candidate* and measure it
        (scope check included). Returns (metrics or None on failure, record)."""
        from dataset_agent.adapters.release_sample import allocate, new_seed

        s = self._settings
        need = self._release_min_sample()
        target = max(need, int(s.fulltext_target_delivered))
        rng_seed = int(s.fulltext_release_seed or new_seed())
        info: dict = {"sampling": "random_year_stratified", "decides": decides,
                      "seed": rng_seed, "target": target, "min_sample": need,
                      "year_counts": {}, "allocation": {}, "years": {}, "error": None}
        try:
            counts = await self._year_counts(for_clause)
            alloc = allocate(counts, target)
            info.update(year_counts=counts, allocation=alloc)
            if not alloc:
                raise ValueError("the Dimensions year facet returned no papers")
            seed = candidate_to_seed(
                candidate, group_name=s.default_group_name, years=self._dataset_years(),
                publication_types=list(s.default_publication_types),
                filter_us_affiliation=s.default_filter_us_affiliation,
                extra={"home_url": request.dataset_url})
            client = self._get_extraction()
            out = await asyncio.to_thread(functools.partial(
                client.run_year_sample, seed, allocation=alloc, year_counts=counts,
                rng_seed=rng_seed, batch_size=s.fulltext_batch_size,
                max_batches_per_year=s.fulltext_release_max_batches_per_year))
        except (ExtractionError, ValueError, DimensionsDslError) as exc:
            logger.warning("release sample failed: %s", exc)
            info["error"] = str(exc)
            return None, info
        ev = build_evidence(out.results)
        years = out.years
        ev.target = target
        ev.batches = sum(int(y.get("batches") or 0) for y in years.values())
        short = [y for y in years.values() if y.get("stop_reason") not in (None, "target_met")]
        ev.stop_reason = short[0]["stop_reason"] if short else "target_met"
        await self._apply_scope_check(ev, request)
        m = ev.metrics(iteration, for_clause, None)
        year_of = {str(p.get("publication_id")): p.get("sample_year")
                   for p in (out.results.get("publications") or []) if isinstance(p, dict)}
        info.update(years=years, **self._sample_summary(m))
        info["pool_capped_years"] = [y for y, v in sorted(years.items())
                                     if not v.get("pool_covers_year", True)]
        info["papers"] = [{"id": v.publication_id, "year": year_of.get(v.publication_id),
                           "verdict": v.verdict, "four_way": v.four_way if ev.scope_checked
                           else None} for v in ev.verdicts]
        return m, info

    def _top_cited_sample_info(self, m: IterationMetrics, *, decides: bool = True) -> dict:
        return {"sampling": "top_cited", "decides": decides, "iteration": m.iteration,
                "target": m.fulltext_target, "min_sample": self._release_min_sample(),
                **self._sample_summary(m)}

    @staticmethod
    def _release_sample_note(info: dict) -> str:
        def est(name: str, e: dict | None) -> str:
            if not e or e.get("value") is None:
                return ""
            lo, hi = e["wilson95"]
            return f"; {name} {e['value']:.2f} (Wilson 95% {lo:.2f}-{hi:.2f})"

        role = "decides release" if info.get("decides") else "recorded only, does not decide"
        if info["sampling"] == "top_cited":
            head = (f"full-text release sample: top-cited papers of iteration "
                    f"{info.get('iteration')} ({role}; {info.get('n_evaluated', 0)} evaluated, "
                    f"minimum {info['min_sample']})")
        elif info.get("error"):
            return (f"full-text release sample: random year-stratified, seed {info['seed']} "
                    f"({role}): release sample failed ({info['error']}); nothing measured")
        else:
            capped = info.get("pool_capped_years") or []
            head = (f"full-text release sample: random year-stratified, seed {info['seed']} "
                    f"({role}; allocation {info['allocation']} of year counts "
                    f"{info['year_counts']}; {info.get('n_evaluated', 0)} evaluated of "
                    f"{info['target']} aimed, minimum {info['min_sample']}"
                    + (f"; years drawn from their 1,000 most-cited papers {capped}"
                       if capped else "") + ")")
        return (head + est("unrelated share", info.get("unrelated_share"))
                + est("confirmed data share", info.get("data_share"))
                + est("precision", info.get("precision")))

    @staticmethod
    def _own_program_names(rmap) -> list[str]:
        """The program's own names: program names minus former or parent names (those
        go through the count and risk path, see ``_screen_former_program_names``)."""
        def key(name: str) -> str:  # stripped, case-folded, whitespace collapsed
            return " ".join(name.split()).casefold()

        former = {key(n) for n in (rmap.former_or_parent_names or [])}
        return [p for p in rmap.program_names if key(p) not in former]

    @classmethod
    def _standalone_data_phrases(cls, rmap) -> list[str]:
        """Generated data phrases that may stand alone: built from one of the program's own
        names that is not a short acronym (acronym forms keep the acronym guards; phrases
        of a former or parent name keep their phase-1 label)."""
        programs = cls._own_program_names(rmap)
        return [p for p in generated_data_phrases(programs)
                if not acronym_form(phrase_program(p, programs) or "")]

    async def _screen_former_program_names(
        self, rmap, safe: list[str], risky: list[str], alias_entries: list[AliasCountEntry],
        *, c_full: int, has_fn: bool, notes_parts: list[str],
    ) -> tuple[list[str], list[str]]:
        """Former or parent program names take the normal path: a Dimensions count, the
        phase-1 risk label (10x the title count), the ``optimize_flag_max_hits`` screen and
        the ordinary-phrase/acronym guards. A safe name may stand alone, a risky one is
        gated by the research flags; a name with no hits, above the screen or not counted
        is not searched. Only the program's own names stand alone without a count."""
        safe, risky = list(safe), list(risky)
        keys = {a.casefold() for a in [*safe, *risky]}
        max_hits = self._settings.optimize_flag_max_hits
        not_searched: list[str] = []
        for p in sanitize_terms(list(rmap.former_or_parent_names or []), "alias"):
            if p.casefold() in keys:
                continue
            keys.add(p.casefold())
            if not self._phase1_budget_left():
                not_searched.append(f"{p} (not counted: Dimensions budget)")
                continue
            c = await self._count_alias(p, notes_parts)
            if c is None:
                not_searched.append(f"{p} (Dimensions could not parse it)")
                continue
            risk = _alias_risk_label(c, c_full, has_fn)
            if is_short_acronym(p) or acronym_form(p) or detect_rhetorical_name(p):
                risk = "risky"
            alias_entries.append(AliasCountEntry(alias=p, count=c, risk=risk))
            if c <= 0:
                not_searched.append(f"{p} (no hits)")
            elif max_hits and c > max_hits:
                not_searched.append(f"{p} ({c} Dimensions hits)")
            else:
                (risky if risk == "risky" else safe).append(p)
        if rmap.former_or_parent_names:
            notes_parts.append(
                "resource map: former or parent program names counted like aliases"
                + (f"; not searched {not_searched}" if not_searched else ""))
        return safe, risky

    @classmethod
    def _apply_resource_map(cls, rmap, safe: list[str], risky: list[str], flag_terms: list[str],
                            notes_parts: list[str], *, max_hits: int = 500_000
                            ) -> tuple[list[str], list[str], list[str]]:
        """Broad query for a split dataset (owner decision, round 3): program names and
        specific data resource names are OR-ed; nothing is gated by data names, and the
        full-text judge labels papers that only discuss the program.

        Guards still apply: a bare ordinary phrase ("All of Us") or a short acronym never
        stands alone (risky, gated by the research flags as without a map); a name phase 1
        marked risky stays risky (an ambiguous phrase stands alone only when its probe
        passes); generated data phrases built from an acronym stay risky. A specific data
        name stands alone when it is not a bare short acronym and has at most *max_hits*
        hits; generic data names ("Data Browser") are not searched. ``build_for_clause`` is
        unchanged."""
        safe, risky = list(safe), list(risky)
        keys = {a.casefold() for a in [*safe, *risky]}
        added, guarded = [], []
        for p in sanitize_terms(cls._own_program_names(rmap), "alias"):
            if p.casefold() in keys:
                continue
            if is_short_acronym(p) or acronym_form(p) or detect_rhetorical_name(p):
                guarded.append(p)
            else:
                added.append(p)
            keys.add(p.casefold())
        safe += added
        risky += guarded
        # Generated data phrases: a non-acronym phrase is a data reference by construction
        # (safe candidate); an acronym phrase keeps the acronym guards.
        acr = {p.casefold() for p in generated_data_phrases(list(rmap.program_names))
               if acronym_form(p)}
        phrases = {p.casefold() for p in cls._standalone_data_phrases(rmap)}
        freed = [a for a in risky if a.casefold() in phrases]
        risky = [a for a in risky if a.casefold() not in phrases]
        held = [a for a in safe if a.casefold() in acr]
        safe = [a for a in safe if a.casefold() not in acr] + freed
        risky += held
        specific, screened = [], []
        for d in sanitize_terms(list(rmap.specific_data_names), "alias"):
            hits = rmap.data_name_hits.get(d)
            if is_short_acronym(d):
                screened.append(f"{d} (bare short acronym)")
            elif hits is not None and hits > max_hits:
                screened.append(f"{d} ({hits} Dimensions hits)")
            elif d.casefold() not in {a.casefold() for a in safe}:
                specific.append(d)
        safe += specific
        risky = [a for a in risky if a.casefold() not in {d.casefold() for d in specific}]
        safe_keys = {a.casefold() for a in safe}
        flags = [f for f in flag_terms if f.casefold() not in safe_keys]
        unused = [d for d in rmap.data_resource_names
                  if d.casefold() not in safe_keys]
        notes_parts.append(
            f"resource map: query broad on the program name: program names {added} and "
            f"specific data names {specific} stand alone"
            + (f"; never stand alone (ordinary phrase or acronym) {guarded}" if guarded else "")
            + (f"; acronym data phrases kept risky {held}" if held else "")
            + (f"; specific data names screened {screened}" if screened else "")
            + (f"; data names not searched {unused}" if unused else "")
            + f"; gate = research flags {flags}"
        )
        return safe, risky, flags

    async def _add_data_wording(self, program_ds, notes_parts: list[str]) -> None:
        """Attach the informational data-wording clause and its count (one budgeted call)."""
        from dataset_agent.adapters.resource_map import DATA_WORDING_NOTE, data_wording_clause

        clause = data_wording_clause(self._resource_map,
                                     max_hits=self._settings.optimize_flag_max_hits)
        if not clause:
            return
        program_ds.data_wording_clause = clause
        program_ds.data_wording_note = DATA_WORDING_NOTE
        if self._room():
            ys, ye = self._dataset_years()
            runner = self._make_eval_runner(asyncio.get_running_loop())
            try:
                program_ds.data_wording_count = await asyncio.to_thread(
                    lambda: parse_total_count(runner(build_for_clause_count_dsl(
                        clause, where=f"year in [{ys}:{ye}] and type=\"article\""))))
            except Exception as exc:
                logger.warning("data-wording alternative: count failed: %s", exc)
        notes_parts.append(
            f"data-wording alternative ({DATA_WORDING_NOTE}): "
            f"{program_ds.data_wording_count if program_ds.data_wording_count is not None else '?'}"
            " papers")

    def _dataset_years(self) -> tuple[int, int]:
        """The dataset window: it starts at the first data release year when research
        found one and counts confirmed it (never before the default start)."""
        s = self._settings
        start = s.default_years_start
        rmap = self._resource_map
        if rmap is not None and rmap.first_release_year:
            start = max(start, rmap.first_release_year)
        return start, s.default_years_end

    def _build_where_dsl(self, for_clause: str) -> str:
        ys, ye = self._dataset_years()
        return (
            f'search publications in full_data for "{for_clause}" '
            f"where year in [{ys}:{ye}] and type=\"article\" "
            "return publications[id+title+doi+year+times_cited] sort by times_cited"
        )

    async def _research_resource_map(self, rec: DatasetRecord):
        """Program versus data for a stored or fresh research record (counts are budgeted).
        None is a firm "no split"; raises ``ResourceMapUndecided`` when it could not be
        decided."""
        from dataset_agent.adapters.resource_map import (
            ResourceMapUndecided,
            map_for_record,
            map_options,
        )

        runner = self._make_eval_runner(asyncio.get_running_loop())

        def count_where(clause: str, where: str) -> int | None:
            try:
                return parse_total_count(runner(build_for_clause_count_dsl(clause, where=where)))
            except Exception as exc:
                logger.warning("resource map: count failed: %s", exc)
                return None

        try:
            return await asyncio.to_thread(
                map_for_record, self._agent, rec, count=lambda c: count_where(c, ""),
                count_where=count_where, **map_options(self._settings),
            )
        except ResourceMapUndecided:
            raise
        except Exception as exc:
            logger.warning("resource map: failed: %s", exc)
            raise ResourceMapUndecided(f"the map step failed: {exc}") from exc

    async def _draw_map_again(self, rec, previous: str | None,
                              notes_parts: list[str]) -> tuple[object, str | None]:
        """Draw the resource map (once). Returns (map or None, error or None); a failure is
        noted and never read as a firm "no split"."""
        from dataset_agent.adapters.resource_map import ResourceMapUndecided

        if previous:
            notes_parts.append(f"resource map: research could not decide it ({previous}); "
                               "drawn once more")
        try:
            return await self._research_resource_map(rec), None
        except ResourceMapUndecided as exc:
            self._resource_map_error = str(exc)
            notes_parts.append(
                f"resource map: could not be decided ({exc}); this run has no program split, "
                "which is not a firm 'no split'")
            return None, str(exc)

    async def _stored_or_drawn_map(self, rec, previous: str | None, notes_parts: list[str],
                                   *, refresh: bool) -> tuple[object, str | None]:
        """The map for a record research did not decide: the stored decision (the dataset's
        research record, see ``DatasetResearchUseCase.stored_map_decision``) unless
        *refresh*, else drawn once here and stored when decided. Sets ``_map_decision``."""
        research = self._research
        stored_fn = getattr(research, "stored_map_decision", None)
        name = rec.main_dataset_name
        if not refresh and callable(stored_fn):
            stored = stored_fn(name)
            if isinstance(stored, DatasetRecord):
                self._map_decision = research.reused_decision(stored)
                return (stored.resource_map.model_copy(deep=True)
                        if stored.resource_map is not None else None), None
        drawn, err = await self._draw_map_again(rec, previous, notes_parts)
        if err is not None:
            return None, err
        drawn_fn = getattr(research, "drawn_decision", None)
        if callable(drawn_fn):
            decision = drawn_fn(name, drawn, refreshed=refresh)
            if isinstance(decision, ResourceMapDecision):
                self._map_decision = decision
                try:
                    if research.remember_map(rec, drawn, decision) is None:
                        self._map_decision = decision.model_copy(update={"stored_in": None})
                except Exception as exc:  # the run goes on; the next run draws again
                    logger.warning("resource map: could not be stored: %s", exc)
                    notes_parts.append(f"resource map: could not be stored ({exc})")
                    self._map_decision = decision.model_copy(update={"stored_in": None})
        return drawn, None

    @staticmethod
    def _request_map_decision(request: OptimizeRequest) -> ResourceMapDecision:
        """The decision of a map (or a firm "no split": ``resource_map_checked`` with no
        map) passed in *request*: research's own decision when it passed one along, else
        origin "request"."""
        return request.resource_map_decision or ResourceMapDecision(
            origin="request", split=request.resource_map is not None)

    def _map_decision_note(self) -> str | None:
        d = self._map_decision
        if d is None:
            return None
        what = "a program/data split" if d.split else "no split"
        when = f"drawn {d.decided_at}" if d.decided_at else "drawing date unknown"
        where = f" in {d.stored_in}" if d.stored_in else ""
        if d.origin == "reused":
            return (f"resource map: reused the stored decision ({what}, {when}{where}); "
                    "refresh it with refresh_resource_map=true (CLI --refresh-map)")
        if d.origin == "request":
            return f"resource map: passed in the request ({what}); it wins over a stored map"
        stored = f"; stored{where} for later runs" if d.stored_in else "; not stored"
        return (f"resource map: drawn in this run ({what}, {when}"
                f"{', refresh asked' if d.refreshed else ''}){stored}")

    async def _discover_from_literature(self, evidence: Evidence, cand: QueryCandidate,
                                        notes_parts: list[str], iteration: int
                                        ) -> QueryCandidate | None:
        """Split datasets only: capitalised names that co-occur with the program name in
        passages of papers judged "used" become data-name candidates. Each passes the map
        checks (co-occurrence and a high program share of its own hits); a specific one
        joins the safe group (nothing is gated by data names). Returns the new candidate,
        or None when no specific name was added."""
        from dataset_agent.adapters.resource_map import (
            check_data_names,
            literature_candidates,
            merge_names,
        )

        rmap = self._resource_map
        s = self._settings
        if rmap is None or not s.resource_map_discovery:
            return None
        passages = {
            v.publication_id: [str(c.get("text") or "") for c in v.contexts
                               if c.get("part") in ("title", "abstract", "body")]
            for v in evidence.verdicts if v.used
        }
        cands = literature_candidates(passages, list(rmap.program_names),
                                      list(rmap.data_resource_names),
                                      min_papers=s.resource_map_literature_min_papers)
        if not cands:
            return None
        runner = self._make_eval_runner(asyncio.get_running_loop())

        def count(clause: str) -> int | None:
            if not self._room(keep=int(s.refine_impact_budget_reserve)):
                return None
            try:
                return parse_total_count(runner(build_for_clause_count_dsl(clause)))
            except Exception as exc:
                logger.warning("literature discovery: count failed: %s", exc)
                return None

        found = {r["name"].casefold(): r for r in rmap.found_names}
        for c in cands:
            ids = [pid for pid, ps in passages.items()
                   if any(_phrase_in(c, x, plural_ok=False) for x in ps)]
            found.setdefault(c.casefold(), {"name": c, "source": "literature",
                                            "where": "used papers: " + ", ".join(ids[:5])})
        hits, alone = await asyncio.to_thread(
            check_data_names, cands, list(rmap.program_names), count=count,
            min_cooccurrence=s.resource_map_min_cooccurrence,
            min_share=s.resource_map_specific_min_share, found=found, source="literature")
        rmap.found_names = list(found.values())
        added = merge_names(rmap, hits, alone, source="literature",
                            specific_min_share=s.resource_map_specific_min_share,
                            max_hits=s.optimize_flag_max_hits)
        rejected = [c for c in cands if c not in added]
        notes_parts.append(f"gate_iter{iteration}: literature discovery: candidates {cands}; "
                           f"added {added}" + (f"; rejected {rejected}" if rejected else ""))
        if not added:
            return None
        specific = [d for d in added if d in rmap.specific_data_names and not is_short_acronym(d)]
        if not specific:
            return None
        safe = dedupe_strings_ci_preserve_order([*cand.safe, *specific])
        import dataclasses

        return dataclasses.replace(cand, safe=safe)

    async def _run_research(self, request: OptimizeRequest) -> DatasetRecord:
        req = ResearchRequest(
            dataset_name=request.dataset_name,
            dataset_url=request.dataset_url,
            sample_size=min(50, self._settings.fp_sample_size),
            refresh_resource_map=request.refresh_resource_map,
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

        # Reset before anything counts: the map redraw below is part of this run's
        # Dimensions total and budget.
        self._dims_calls = 0
        self._cap_extra = 0
        self._release_reserve = 0
        self._resource_map = None
        self._resource_map_error = None
        self._map_decision = None
        self._scope_calls = 0
        self._dataset_description = ""
        if request.entity_type == "ai_model":
            return await self._execute_ai_model(request, notes_parts, run_id, t0)

        desc_for_signal3 = ""
        rec: DatasetRecord | None = None
        if request.dataset_names:
            # A research run's resource map comes with the request (full pipeline). When
            # research could not decide it, it is drawn once more here.
            # A map passed in the request wins over a stored one. Research passes its own
            # decision along (drawn or reused); a map without one was passed by the caller.
            if request.resource_map is not None or request.resource_map_checked:
                self._map_decision = self._request_map_decision(request)
            if (request.resource_map is None and request.resource_map_error
                    and self._settings.resource_map_enabled):
                from types import SimpleNamespace

                shim = SimpleNamespace(
                    main_dataset_name=request.dataset_name, home_url=request.dataset_url,
                    description=request.dataset_description or "",
                    dataset_names=list(request.dataset_names), flag_terms=list(request.flag_terms))
                drawn, _err = await self._stored_or_drawn_map(
                    shim, request.resource_map_error, notes_parts,
                    refresh=request.refresh_resource_map)
                if drawn is not None:
                    request = request.model_copy(update={"resource_map": drawn,
                                                         "resource_map_error": None})
            if request.resource_map is not None:
                # A copy: literature discovery adds names during the run, and the caller's
                # (stored) map must stay the decision that was drawn.
                self._resource_map = request.resource_map.model_copy(deep=True)
                self._dataset_description = request.dataset_description or ""
                desc_for_signal3 = resource_map_description(
                    (request.dataset_description or "")[:2000], request.resource_map)
                notes_parts.append(
                    "resource map: program "
                    f"{request.resource_map.program_names} separate from data "
                    f"{request.resource_map.data_resource_names}"
                )
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
            # One map decision per run: research already made it when it ran the step. A
            # map research could not decide is drawn once more.
            # Research reuses the stored decision unless asked to refresh; a record it did
            # not decide (or a stored research output from before the map was stored)
            # reads the stored decision here, else draws once and stores it.
            if request.resource_map is not None or request.resource_map_checked:
                # A map (or a firm "no split") passed in the request wins over the one
                # research reused or drew; it applies to this run and is not stored.
                rec.resource_map = (request.resource_map.model_copy(deep=True)
                                    if request.resource_map is not None else None)
                rec.resource_map_checked, rec.resource_map_error = True, None
                self._map_decision = self._request_map_decision(request)
                rec.resource_map_decision = self._map_decision
            elif rec.resource_map_checked:
                self._map_decision = rec.resource_map_decision
            elif self._settings.resource_map_enabled and rec.resource_map is None:
                rec.resource_map, err = await self._stored_or_drawn_map(
                    rec, rec.resource_map_error, notes_parts,
                    refresh=request.refresh_resource_map)
                rec.resource_map_error = err
                rec.resource_map_checked = err is None
                rec.resource_map_decision = self._map_decision
            # The seed description (read by the full-text validator) names the program
            # and its data resources when research split them.
            desc_for_signal3 = resource_map_description(desc_for_signal3, rec.resource_map)
            self._resource_map = (rec.resource_map.model_copy(deep=True)
                                  if rec.resource_map is not None else None)
            self._dataset_description = rec.description or ""
            if rec.resource_map is not None and rec.resource_map.first_release_year:
                notes_parts.append(
                    f"window starts at the first data release ({rec.resource_map.first_release_year})"
                )
            if rec.resource_map is not None:
                notes_parts.append(
                    "resource map: program "
                    f"{rec.resource_map.program_names} separate from data "
                    f"{rec.resource_map.data_resource_names}"
                )
            aliases = dedupe_strings_ci_preserve_order(list(rec.dataset_names))
            flag_terms = dedupe_strings_ci_preserve_order(list(rec.flag_terms))
            exclude_terms = dedupe_strings_ci_preserve_order(
                list(rec.exclude_terms) + list(request.exclude_terms)
            )

        decision_note = self._map_decision_note()
        if decision_note:
            notes_parts.append(decision_note)
        aliases = sanitize_terms(aliases, "alias")
        # A lone generic word ("dataset", "data") matches every paper; it is never a name.
        generic = [a for a in aliases if not content_tokens(a)]
        if generic:
            aliases = [a for a in aliases if content_tokens(a)]
            notes_parts.append(f"dropped generic alias(es) {generic}")
            logger.warning("optimize: dropped generic alias(es) %s", generic)
        # "A & B" searches as "A B", never as "A and B": a name written either way is
        # searched both ways (each spelling is counted and judged like any alias).
        # Deduplicated by Dimensions search, not by case (the plural is the same search).
        spelled = ampersand_additions(aliases)
        if spelled:
            aliases += spelled
            notes_parts.append(
                f"'&' and 'and' are different Dimensions searches: added {spelled}"
            )
        flag_terms = sanitize_terms(flag_terms, "flag term")
        exclude_terms = sanitize_terms(exclude_terms, "exclude term")
        self._reserve_release_calls()
        rmap = self._resource_map
        if rmap is not None:
            # Generated data phrases are ordinary alias candidates: counted and probed.
            have = {a.casefold() for a in aliases}
            extra = [p for p in sanitize_terms(generated_data_phrases(list(rmap.program_names)),
                                               "alias") if p.casefold() not in have]
            aliases += extra
            if extra:
                notes_parts.append(f"resource map: generated data phrases {extra}")

        if not aliases:
            return QueryOptimizationRecord(
                success=False,
                failed_phase="phase1_alias_counting",
                errors=["No dataset name aliases available after research/inputs."],
                notes="",
            )

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
                    risky.append(a)
                else:
                    safe.append(a)

            # Aliases that only name a listed organization neither show the dataset is cited
            # nor keep grounding from looking for its real names. The rule applies to a
            # title with no hits: a dataset named after its program has the same shape.
            org_aliases = (
                [e.alias for e in alias_entries if is_organization_alias(e.alias, flag_terms, request.dataset_name)]
                if rec is not None and c_full <= 0
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

            # A title with hits already names the dataset: history there added a different
            # dataset, a name with no hits, or nothing.
            if rec is not None and self._settings.optimize_name_history_enabled and c_full <= 0:
                for h_alias, h_count in await self._name_history(
                    request, rec, alias_entries, flag_terms, org_keys, notes_parts
                ):
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

            # "WR dataset" still matches other meanings of "WR", as research
            # lists it: flag-gated. Without flag terms a risky alias
            # would never be searched, so it stays safe then.
            if flag_terms:
                for s_alias in [x for x in safe if is_acronym_suffix_form(x)]:
                    risky.append(s_alias)
                    for entry in alias_entries:
                        if entry.alias == s_alias:
                            entry.risk = "risky"
                    note = f"suffix_risky: {s_alias!r} kept risky (short acronym)"
                    if note not in notes_parts:
                        notes_parts.append(note)
            if rmap is not None:
                safe, risky = await self._screen_former_program_names(
                    rmap, safe, risky, alias_entries, c_full=c_full, has_fn=has_fn,
                    notes_parts=notes_parts)
                safe, risky, flag_terms = self._apply_resource_map(
                    rmap, safe, risky, flag_terms, notes_parts,
                    max_hits=self._settings.optimize_flag_max_hits)
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

        With *model* (AI-model runs) the terms were measured already: full-text probes are skipped and
        refinement keeps to the model rules (see ``_refine_model_terms``)."""
        exclusion_terms = dedupe_strings_ci_preserve_order(exclude_terms)

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
        if model is None and current_risky and not current_flags:
            notes_parts.append(f"risky alias(es) {current_risky} not searched: no flag terms")
            current_risky = []

        # Start with the best Phase-2 variant: non-empty result first, then the variant
        # preference order (V4 > V3 > V2 > V1 > V5).
        best_variant = _pick_start_variant(tested)
        current_for_clause = best_variant.for_clause
        current_variant_label = best_variant.label.split("~", 1)[0]

        best_ev: dict | None = None
        phase3_error: str | None = None
        phase3_error_iter: int | None = None
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
        ft_refiner_notes: set[str] = set()
        if ft_enabled:
            current_for_clause = candidate_clause(ft_candidate)
            current_variant_label = "V4"

        iter_limit = max_iters
        iteration = -1
        while iteration + 1 < iter_limit:
            iteration += 1
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
                **(
                    {"entity_context": model.context, "years": model.years}
                    if model is not None else {}
                ),
            )

            if ev.get("error"):
                # A failed evaluation says nothing about precision; never read it as 0% FP.
                phase3_error = str(ev["error"])
                phase3_error_iter = iteration
                reason = str(ev.get("error_reason") or short_dsl_error(phase3_error))
                if phase3_error.startswith("dimensions_query_failed"):
                    notes_parts.append(
                        f"gate_iter{iteration}: Dimensions query failed ({reason}); "
                        "iteration not evaluated"
                        + ("; it cannot be selected." if ft_enabled else
                           "; evaluation failed, result is partial and confidence is "
                           "capped at LOW.")
                    )
                    if ft_enabled:
                        ft_eval_error = (iteration, reason)
                else:
                    notes_parts.append(
                        f"gate_iter{iteration}: evaluation failed ({phase3_error}); "
                        + ("this iteration cannot be selected." if ft_enabled else
                           "result is partial and confidence is capped at LOW.")
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

            if model is not None:
                fp_k = int(ev.get("fp_count") or 0)
                judged = ev.get("judged_count")
                genuine_all = int(ev.get("genuine_count") or 0)
                judged = int(judged) if isinstance(judged, int) else genuine_all + fp_k
                model.abstract_stats[current_for_clause] = (
                    fp_k, judged, max(0, genuine_all - (judged - fp_k)),
                    float(ev.get("coverage_pct") or 0.0),
                )
                # Every rejected abstract is kept, so a LOW verdict can be read by title.
                for h in ev.get("fp_hits") or []:
                    if not isinstance(h, dict):
                        continue
                    alias = str(h.get("matched_alias") or "")
                    title = str(h.get("title") or "")
                    model.decide(
                        title or alias, "abstract_fp", "false_positive",
                        f"abstract check gate_iter{iteration}: {alias!r} read as another meaning",
                        publication_id=str(h.get("publication_id") or ""),
                        year=h.get("year"), title=title, matched_alias=alias,
                        iteration=iteration,
                    )
            fp_unknown = ev.get("fp_rate_pct") is None
            fp_rate = float(ev.get("fp_rate_pct") or 0.0)
            fp_shown: float | None = None if fp_unknown else fp_rate
            if int(ev.get("failed_count") or 0):
                notes_parts.append(
                    f"gate_iter{iteration}: abstract check: {int(ev['failed_count'])} of "
                    f"{int(ev.get('attempted_count') or 0)} classifications failed (unclassified)"
                )
            expected_count = int(ev.get("publications_total") or 0)
            coverage = float(ev.get("coverage_pct") or 0.0)

            label = f"{current_variant_label}~gate{iteration}"
            entry = VariantTestedEntry(
                label=label,
                for_clause=current_for_clause,
                expected_count=expected_count,
                fp_rate_pct=fp_shown,
                top_titles=[],
            )
            tested.append(entry)
            notes_parts.append(
                f"gate_iter{iteration} fp_rate={_pct_text(fp_shown)} coverage={coverage:.1f}% total={expected_count}"
            )

            if best_ev is None or _fp_better(fp_shown, best_ev.get("fp_rate_pct")):
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
                            abstract_fp_rate_pct=fp_shown,
                            error=str(exc),
                        )
                    )
                    notes_parts.append(f"gate_iter{iteration}: full-text probe failed: {exc}")
                    break
                metrics = evidence.metrics(iteration, current_for_clause, fp_shown)
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
                if metrics.data_share is not None:
                    fw = metrics.four_way or {}
                    notes_parts.append(
                        f"gate_iter{iteration}: split dataset, program-only counted genuine; "
                        f"shares: data {metrics.data_share:.2f}, program-only "
                        f"{(metrics.program_only_rate or 0):.2f}, used {(metrics.used_rate or 0):.2f}"
                        + (f", references-only {fw['references_only']['share']:.2f}, "
                           f"unrelated {fw['unrelated']['share']:.2f}" if fw else "")
                    )
                if iteration == 0 and iteration < max_iters - 1:
                    found = await self._discover_from_literature(
                        evidence, ft_candidate, notes_parts, iteration)
                    if found is not None and candidate_clause(found) != current_for_clause:
                        ft_candidate = found
                        current_for_clause = candidate_clause(found)
                        continue
                if (
                    prec is not None
                    and metrics.n_evaluated >= min_ev
                    and prec >= 1.0 - threshold / 100.0
                    and (not self._split_release() or self._meets_split_target(metrics))
                ):
                    notes_parts.append(f"gate_iter{iteration}: full-text precision within threshold.")
                    break
                if prec is not None and (self._meets_split_target(metrics)
                                         if self._split_release()
                                         else self._meets_release_bar(metrics)):
                    if self._split_release():
                        notes_parts.append(
                            f"gate_iter{iteration}: full-text unrelated share meets the "
                            f"split-dataset bar (<= {self._settings.fulltext_split_unrelated_max:.2f} "
                            f"on >= {self._settings.fulltext_recall_min_evidence} papers) "
                            f"and the data floor (>= {self._settings.fulltext_split_data_min:.2f}); "
                            f"data-only precision {(metrics.data_share or 0):.2f}."
                        )
                    else:
                        notes_parts.append(
                            f"gate_iter{iteration}: full-text precision meets the bar "
                            f"(>= {self._settings.fulltext_recall_min_precision:.2f} on >= "
                            f"{self._settings.fulltext_recall_min_evidence} papers)."
                        )
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
                    program_names=(list(self._resource_map.program_names)
                                   if self._resource_map is not None else None),
                    impact_fn=lambda t, fc=current_for_clause: asyncio.run_coroutine_threadsafe(
                        self._exclude_impact(fc, t), loop
                    ).result(),
                )
                ft_diffs[iteration] = result.diffs
                refiner_notes = [f"gate_iter{iteration}: {n}" for n in result.notes]
                notes_parts.extend(refiner_notes)
                ft_refiner_notes.update(refiner_notes)
                if not result.changed:
                    notes_parts.append(f"gate_iter{iteration}: refiner found nothing to change.")
                    break
                ft_candidate = result.candidate
                new_fc = candidate_clause(ft_candidate)
                if not new_fc.strip() or new_fc == current_for_clause:
                    break
                current_for_clause = new_fc
                continue

            if fp_unknown:
                notes_parts.append(f"gate_iter{iteration}: FP rate unknown; stopping.")
                break
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
                regate = refined[1] != current_risky or refined[2] != current_flags
                current_safe, current_risky, current_flags, exc_work = refined
                if regate and iteration == iter_limit - 1 and iter_limit == max_iters:
                    # A re-gated query must not be selected unchecked: one more check.
                    iter_limit += 1
                    notes_parts.append(
                        f"gate_iter{iteration}: re-gated in the last iteration; "
                        "one more abstract check"
                    )
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
                self._selection_key(m)
                for m in ft_iterations
                if m.fulltext_precision is not None
                and m.n_evaluated >= self._settings.fulltext_min_evidence
                and m.iteration in ft_entries
            ]
            if scored:
                top = max(scored)
                best_ft_iter = -top[-1]
                best_variant_entry = ft_entries[best_ft_iter]
                bc = ft_candidates[best_ft_iter]
                safe, risky = list(bc.safe), list(bc.risky)
                flag_terms, exc_work = list(bc.flag_terms), list(bc.exclude_terms)
                met, floor, top_prec, top_data, _ = top
                if self._split_release() and floor:
                    notes_parts.append(
                        f"Selected iteration {best_ft_iter}: it reaches the data floor "
                        f"(confirmed data share {top_data:.2f}) within the unrelated limit; "
                        f"full-text precision {top_prec:.2f}."
                    )
                else:
                    notes_parts.append(
                        f"Selected iteration {best_ft_iter} by full-text precision "
                        f"({top_prec:.2f})"
                        + (f"; confirmed data share {top_data:.2f} breaks precision ties among "
                           "iterations meeting the unrelated limit." if self._split_release()
                           and met else ".")
                    )

        recall_diffs: list[TermDiff] = []
        recall_min_precision: float | None = None
        recall_final: QueryCandidate | None = None
        recall_lower_bound_terms: list[str] = []
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
                (
                    final_c, recall_diffs, rnotes, recall_min_precision, recall_lower_bound_terms,
                ) = await self._recall_check(
                    request, sel_c, main_names=names, known_aliases=known,
                    flag_terms=flags, evidence_against=against,
                    base_count=(
                        best_variant_entry.expected_count if best_variant_entry is not None else None
                    ),
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
                    notes_parts.extend(_big_addition_notes(recall_diffs, count))
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
            if recall_diffs:
                notes_parts.append(
                    f"Selected variant {best.label} (abstract fp_rate={_pct_text(best.fp_rate_pct)} "
                    "was measured before the recall addition)."
                )
            else:
                notes_parts.append(
                    f"Selected gate-evaluated variant {best.label} (fp_rate={_pct_text(best.fp_rate_pct)})."
                )
        else:
            candidates: list[tuple[float, int, int, VariantTestedEntry]] = []
            for ent in tested:
                cnt = ent.expected_count or 0
                if ent.fp_rate_pct is None:
                    continue  # unknown rate: worst, never preferred
                fp = float(ent.fp_rate_pct)
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

        # The release sample: the decision reads a random, year-stratified sample of the
        # selected query's papers for split datasets (``fulltext_release_sampling``); the
        # refinement iterations above stay on top-cited papers, the refiner's input.
        release_sample: dict | None = None
        rel_m: IterationMetrics | None = None
        random_decides = False
        release_need = self._release_min_sample()
        if (ft_enabled and best_ft_iter is not None
                and not (phase3_error is not None and phase3_error_iter == best_ft_iter)):
            rel_cand = recall_final if recall_final is not None else ft_candidates[best_ft_iter]
            sel0 = next(m for m in ft_iterations if m.iteration == best_ft_iter)
            if self._random_release():
                random_decides = True
                rel_m, release_sample = await self._random_release_sample(
                    rel_cand, request, best.for_clause, best_ft_iter, decides=True)
                notes_parts.append(self._release_sample_note(release_sample))
            else:
                release_sample = self._top_cited_sample_info(sel0)
                notes_parts.append(self._release_sample_note(release_sample))
                if self._settings.fulltext_release_shadow:
                    _m, shadow = await self._random_release_sample(
                        rel_cand, request, best.for_clause, best_ft_iter, decides=False)
                    release_sample["shadow"] = shadow
                    notes_parts.append(self._release_sample_note(shadow))

        release_bar: str | None = None
        release_met: bool | None = None
        four_way: dict | None = None
        data_precision: float | None = None
        data_floor: dict | None = None
        data_floor_met: bool | None = None
        if ft_enabled:
            # In full-text mode confidence comes from the selected iteration's full-text
            # precision and sample size only; the abstract FP rate stays in the notes and
            # in IterationMetrics.abstract_fp_rate_pct. MEDIUM is the release bar: precision
            # for most datasets, the unrelated share for a split dataset.
            confidence = ConfidenceLevel.LOW
            release_bar, release_met = self._release_bar_text(), False
            # A failed gate in an iteration other than the selected one (e.g. an extra
            # floor round) says nothing about the selected iteration: it keeps its own
            # confidence, and the failure is noted.
            gate_failed_here = phase3_error is not None and (
                best_ft_iter is None or phase3_error_iter == best_ft_iter)
            if phase3_error is not None and not gate_failed_here:
                notes_parts.append(
                    f"gate iteration {phase3_error_iter} could not be evaluated; the selected "
                    f"iteration {best_ft_iter} keeps its own confidence."
                )
            elif phase3_error is not None and best_ft_iter is None:
                notes_parts.append(
                    f"gate iteration {phase3_error_iter} could not be evaluated and no "
                    "full-text iteration could be selected; confidence LOW."
                )
            sel: IterationMetrics | None = None
            if not gate_failed_here and best_ft_iter is not None:
                # The deciding sample: the random release sample when it decides (None
                # when it failed: fail closed), else the selected iteration's top-cited one.
                sel = (rel_m if random_decides else
                       next(m for m in ft_iterations if m.iteration == best_ft_iter))
            if sel is not None:
                prec = float(sel.fulltext_precision or 0.0)
                if recall_min_precision is not None:
                    # A promoted complement may dominate the result set and was accepted
                    # at its own (possibly lower) bar; never report more than it supports.
                    prec = min(prec, recall_min_precision)
                release_met = self._meets_release_bar(sel, prec, release_need)
                released = self._released(sel, prec, release_need)
                if sel.n_evaluated >= release_need and released:
                    # HIGH needs the release bar too: with optimize_fp_threshold_pct above
                    # 20 the threshold alone would grant HIGH below the 0.80 bar, and a split
                    # dataset released below its data floor is never HIGH.
                    if release_met and prec >= 1.0 - threshold / 100.0:
                        confidence = ConfidenceLevel.HIGH
                    else:
                        confidence = ConfidenceLevel.MEDIUM
                if confidence == ConfidenceLevel.HIGH:
                    # HIGH needs more than a precise sample: a sample cut short of its paper
                    # target for any reason (exhausted pool, batch cap, budget), or a precise
                    # addition the recall check could not make, caps it.
                    capped = False
                    if (
                        sel.fulltext_target is not None
                        and sel.n_fulltext_delivered < sel.fulltext_target
                    ):
                        why = (
                            "the probe pool was exhausted"
                            if sel.fulltext_stop_reason == STOP_POOL_EXHAUSTED
                            else "the probe stopped below its paper target"
                            f" ({sel.fulltext_stop_reason or 'stopped'})"
                        )
                        notes_parts.append(
                            f"confidence capped at MEDIUM: {why} "
                            f"({sel.n_fulltext_delivered} of {sel.fulltext_target} papers)"
                        )
                        capped = True
                    if recall_lower_bound_terms:
                        notes_parts.append(
                            "confidence capped at MEDIUM: the recall check measured a precise "
                            "addition it could not make "
                            f"({', '.join(recall_lower_bound_terms)})"
                        )
                        capped = True
                    if capped:
                        confidence = ConfidenceLevel.MEDIUM
            if random_decides:
                sel_m = rel_m
            else:
                sel_m = (next(m for m in ft_iterations if m.iteration == best_ft_iter)
                         if best_ft_iter is not None else None)
            if sel_m is not None and self._split_release():
                four_way, data_precision = sel_m.four_way, sel_m.data_share
                data_floor = self._data_floor(sel_m, rounds=len(ft_iterations),
                                              need=release_need)
                data_floor_met = data_floor["met"]
            notes_parts.append(self._release_note(
                release_bar, bool(release_met), sel_m, gate_failed_here, data_floor,
                need=release_need))
        else:
            fp_sel = float(best.fp_rate_pct or 0.0)
            if best.fp_rate_pct is None:
                confidence = ConfidenceLevel.LOW
            elif fp_sel < 1.0:
                confidence = ConfidenceLevel.HIGH
            elif fp_sel < 3.0:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            if phase3_error is not None:
                confidence = ConfidenceLevel.LOW
            if self._split_release():
                # The broad program query of a split dataset relies on the full-text judge
                # to separate data papers from program-only ones; its bar is the unrelated
                # limit and the data floor on full text. Without that measurement it is not
                # released, whatever the abstract rate says.
                confidence = ConfidenceLevel.LOW
                notes_parts.append(
                    "split dataset: the full-text bar was not evaluated (full-text refinement "
                    "off), so the broad program query is not released (confidence LOW); "
                    "enable full-text refinement against an extraction target with probe "
                    "support to measure it."
                )

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
                        or n.startswith("confidence capped")
                        or n in ft_refiner_notes
                    ],
                    seed_kw={"extra": {"home_url": request.dataset_url},
                             "years": self._dataset_years()},
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

        program_ds = program_datasets(self._resource_map, clause=best.for_clause)
        if program_ds is not None:
            await self._add_data_wording(program_ds, notes_parts)
            notes_parts.append(program_datasets_summary(program_ds).replace("\n", " | "))
            if proposal is not None:
                proposal.program_datasets = program_ds.model_dump()
                if self._proposal_repo is not None:
                    self._proposal_repo.save(proposal)
        scope_cfg = self._scope_check_config() if self._resource_map is not None else None
        if scope_cfg is not None:
            notes_parts.append(self._judge_config_note(
                "scope check configuration", scope_cfg, "ask(s) per paper"))
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
            release_bar=release_bar,
            release_bar_met=release_met,
            fulltext_four_way=four_way,
            fulltext_data_floor=data_floor,
            data_floor_met=data_floor_met,
            fulltext_data_precision=data_precision,
            release_sample=release_sample,
            proposal=proposal,
            resource_map=self._resource_map,
            program_datasets=program_ds,
            scope_check_calls=self._scope_calls if self._resource_map is not None else None,
            resource_map_error=self._resource_map_error,
            resource_map_decision=self._map_decision,
            scope_check_config=scope_cfg,
            notes=" ".join(notes_parts),
        )
