"""Four-phase Dimensions query optimization orchestration (spec §7)."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Literal

from dataset_agent.adapters.dimensions_dsl import (
    AsyncThrottledDimensionsDsl,
    DimensionsDslError,
    fetch_top_n,
    publication_title,
    run_alias_count,
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
from dataset_agent.adapters.literature import evaluate_for_clause_sync
from dataset_agent.adapters.query_builder import build_for_clause, default_variant_build_order, run_variant
from dataset_agent.adapters.query_heuristics import (
    _SUFFIX_CANDIDATES,
    apply_short_acronym_heuristic,
    check_suffix_necessity,
    detect_rhetorical_name,
    is_short_acronym,
    platform_exclude_suggestions,
    rhetorical_exclusions,
)
from dataset_agent.adapters.text_processing import dedupe_strings_ci_preserve_order
from dataset_agent.domain.models import (
    AliasCountEntry,
    ConfidenceLevel,
    DatasetRecord,
    OptimizeRequest,
    QueryOptimizationAliases,
    QueryOptimizationRecord,
    ResearchRequest,
    VariantTestedEntry,
)
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


class QueryOptimizationUseCase:
    """Runs Phase 1–4: alias counts + risk → variant probes → light FP proxy → selection."""

    def __init__(
        self,
        *,
        dsl_port: DimensionsDslPort,
        research: DatasetResearchUseCase,
        agent: AgentPort,
        settings: Settings,
    ) -> None:
        self._dsl = dsl_port
        self._research = research
        self._agent = agent
        self._settings = settings
        self._dims_calls = 0

    def _budget_left(self) -> bool:
        cap = int(self._settings.optimize_max_dimensions_calls)
        if cap <= 0:
            return True
        return self._dims_calls < cap

    def _scope_budget_left(self) -> bool:
        cap = int(self._settings.optimize_max_dimensions_calls)
        if cap <= 0:
            return True
        return self._dims_calls + 2 <= cap

    async def _count(self, alias: str, *, search_in: str = "full_data") -> int:
        if not self._budget_left():
            return 0
        self._dims_calls += 1
        return await run_alias_count(self._dsl, alias, search_in=search_in)

    async def _try_suffix_promote(
        self,
        alias: str,
        bare_count: int,
        *,
        notes_parts: list[str],
        suffix_checked: int,
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
        suffixed = await check_suffix_necessity(
            self._dsl,
            alias,
            bare_count=bare_count,
            ratio_threshold=ratio,
        )
        self._dims_calls += len(_SUFFIX_CANDIDATES)
        suffix_checked += 1
        if suffixed:
            notes_parts.append(f"suffix_promoted: {alias!r} -> {suffixed!r}")
        return suffixed, suffix_checked

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
            st0, rk0 = apply_short_acronym_heuristic(aliases, [])
            c_full = await self._count(request.dataset_name)
            alias_entries: list[AliasCountEntry] = []
            risky: list[str] = []
            safe: list[str] = []
            has_fn = bool((request.dataset_name or "").strip())
            suffix_checked = 0

            for a in rk0:
                if not self._budget_left():
                    notes_parts.append("Stopped short-acronym counting early (Dimensions call budget).")
                    break
                c = await self._count(a)
                alias_entries.append(AliasCountEntry(alias=a, count=c, risk="risky"))
                if c <= 0:
                    risky.append(a)
                    continue
                suffixed, suffix_checked = await self._try_suffix_promote(
                    a, c, notes_parts=notes_parts, suffix_checked=suffix_checked
                )
                if suffixed:
                    safe.append(suffixed)
                else:
                    risky.append(a)

            for a in st0:
                if not self._budget_left():
                    notes_parts.append("Stopped alias counting early (Dimensions call budget).")
                    break
                c = await self._count(a)
                risk = _alias_risk_label(c, c_full, has_fn)
                alias_entries.append(AliasCountEntry(alias=a, count=c, risk=risk))
                if c <= 0:
                    continue
                if risk == "risky":
                    suffixed, suffix_checked = await self._try_suffix_promote(
                        a, c, notes_parts=notes_parts, suffix_checked=suffix_checked
                    )
                    if suffixed:
                        safe.append(suffixed)
                    else:
                        risky.append(a)
                else:
                    safe.append(a)

            risky = dedupe_strings_ci_preserve_order(risky)
            r_low = {r.lower() for r in risky}
            safe = dedupe_strings_ci_preserve_order([s for s in safe if s.lower() not in r_low])
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

        ex_extra: list[str] = []
        if detect_rhetorical_name(request.dataset_name):
            ex_extra.extend(rhetorical_exclusions(request.dataset_name))
            notes_parts.append("Applied rhetorical NOT heuristics.")
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
        aliases = dedupe_strings_ci_preserve_order([a for a in alias_pool if a])

        exc_work = dedupe_strings_ci_preserve_order(list(exclusion_terms))
        current_safe = list(safe)
        current_risky = list(risky)
        current_flags = list(flag_terms)

        # Start with the best Phase-2 variant
        best_variant = tested[0]
        current_for_clause = best_variant.for_clause

        best_ev: dict | None = None
        best_variant_entry: VariantTestedEntry | None = None

        max_iters = max(1, self._settings.optimize_max_iterations)
        threshold = float(self._settings.optimize_fp_threshold_pct)

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
            )

            fp_rate = float(ev.get("fp_rate_pct") or 0.0)
            expected_count = int(ev.get("publications_total") or 0)
            coverage = float(ev.get("coverage_pct") or 0.0)

            label = f"V4~gate{iteration}"
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

            if fp_rate <= threshold:
                notes_parts.append(f"gate_iter{iteration}: FP rate within threshold.")
                break

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

        logger.info(
            "optimize_phase run_id=%s phase=phase3_gate dims=%s elapsed_s=%.2f",
            run_id,
            self._dims_calls,
            time.perf_counter() - t_phase3,
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
            notes=" ".join(notes_parts),
        )
