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

        fp_keywords: dict[str, list[str]] = {}
        sorted_risky: list[tuple[str, int]] = []
        if risky:
            sorted_risky = sorted(
                [
                    (
                        r,
                        next(
                            (e.count for e in alias_entries if e.alias.lower() == r.lower()),
                            0,
                        ),
                    )
                    for r in risky
                ],
                key=lambda x: -x[1],
            )
        fp_cap = int(self._settings.optimize_fp_domains_max_aliases)
        if fp_cap > 0 and sorted_risky:
            for alias_text, _ in sorted_risky[:fp_cap]:
                domains = await asyncio.to_thread(
                    identify_fp_domains,
                    self._agent,
                    request.dataset_name,
                    alias_text,
                )
                for domain, kws in domains.items():
                    fp_keywords.setdefault(domain, []).extend(kws)
            for k in fp_keywords:
                fp_keywords[k] = dedupe_strings_ci_preserve_order(fp_keywords[k])
            notes_parts.append(
                "fp_domains: "
                f"{sum(len(v) for v in fp_keywords.values())} keywords across {len(fp_keywords)} domains"
            )
        if not fp_keywords:
            fp_keywords = {"risky": [r.lower() for r in risky if len(r) > 2][:12]}

        exc_work = dedupe_strings_ci_preserve_order(list(exclusion_terms))
        web_cap = int(self._settings.optimize_web_disambig_max_aliases)
        if web_cap > 0 and sorted_risky:
            for alias_text, _ in sorted_risky[:web_cap]:
                web_domains = await asyncio.to_thread(
                    web_search_alias_meanings,
                    self._agent,
                    alias_text,
                )
                for domain, cands in web_domains.items():
                    fp_keywords.setdefault(domain, []).extend(cands)
                    exc_work = dedupe_strings_ci_preserve_order(exc_work + cands[:5])
            for k in fp_keywords:
                fp_keywords[k] = dedupe_strings_ci_preserve_order(fp_keywords[k])
            notes_parts.append(
                f"web_disambig: enriched fp_keywords from web search ({web_cap} aliases)"
            )

        ranked = sorted(
            enumerate(tested),
            key=lambda it: (-(it[1].expected_count or 0), _VARIANT_ORDER.get(it[1].label, 99)),
        )
        top_idx = [i for i, _ in ranked[:2]]
        primary_i = ranked[0][0]
        scope_ratios: dict[str, float] = {}
        noise_by_idx: dict[int, list[str]] = {}
        pubs_primary: list[dict] = []
        t_phase3 = time.perf_counter()

        for i in top_idx:
            ent = tested[i]
            fc = ent.for_clause
            if self._scope_budget_left():
                try:
                    scope_ratios[ent.label] = await run_scope_comparison(self._dsl, fc)
                    self._dims_calls += 2
                except DimensionsDslError:
                    scope_ratios[ent.label] = 0.0
            else:
                notes_parts.append("Skipped scope comparison (Dimensions call budget).")
                scope_ratios[ent.label] = 0.0

            titles = list(ent.top_titles)
            if (
                i == primary_i
                and risky
                and flag_terms
                and _fp_keywords_active(fp_keywords)
                and self._budget_left()
            ):
                sample_goal = min(max(60, int(self._settings.fp_sample_size)), 5000)
                pubs, n_fp_calls = await fetch_top_n(
                    self._dsl,
                    fc,
                    max_results=sample_goal,
                    search_in="full_data",
                    page_size=1000,
                    fields="basics+title+abstract+times_cited",
                )
                self._dims_calls += n_fp_calls
                if n_fp_calls:
                    notes_parts.append(f"fp_sample_dims_calls={n_fp_calls}")
                pubs_primary = list(pubs)
                titles = [t for p in pubs if (t := publication_title(p)).strip()]

            fp_pct, noise = _fp_rate_and_noise_terms(titles, fp_keywords)
            fp_final = fp_pct
            if int(self._settings.optimize_signal3_max_titles) > 0 and titles:
                try:
                    s3 = await self._signal3_fp_rate_pct(request, desc_for_signal3, titles)
                except Exception as e:
                    logger.warning("optimize signal3 batch failed: %s", e)
                    s3 = None
                if s3 is not None:
                    fp_final = max(fp_pct, s3)
                    notes_parts.append(
                        f"signal3_{ent.label}={s3:.1f} keyword_fp_pct={fp_pct:.1f} "
                        f"n={min(len(titles), int(self._settings.optimize_signal3_max_titles))}",
                    )
            noise_by_idx[i] = noise
            tested[i] = ent.model_copy(
                update={
                    "fp_rate_pct": fp_final,
                    "top_titles": titles[:10],
                },
            )

        abs_cap = int(self._settings.optimize_abstract_exclude_max_hits)
        cls_cap = int(self._settings.optimize_abstract_classify_max_calls)
        if pubs_primary and risky and abs_cap > 0:
            try:
                raw_hits = scan_abstracts_for_aliases(pubs_primary, risky)
                hits = raw_hits[:abs_cap]
                if hits:
                    pub_titles: dict[str, str] = {}
                    for p in pubs_primary:
                        if not isinstance(p, dict):
                            continue
                        pid = str(p.get("id") or "").strip()
                        if pid:
                            pub_titles[pid] = publication_title(p)

                    if cls_cap <= 0:
                        fp_hits = hits
                    else:
                        fp_hits = []
                        n_cls = min(len(hits), cls_cap)
                        for h in hits[:n_cls]:
                            if not isinstance(h, dict):
                                continue
                            pid = str(h.get("publication_id") or "").strip()
                            title = pub_titles.get(pid, "")
                            alias = str(h.get("matched_alias") or "").strip()
                            snippet = str(h.get("snippet") or "")

                            def _classify_one() -> bool:
                                return classify_alias_mention(
                                    self._agent,
                                    dataset_name=request.dataset_name,
                                    alias=alias,
                                    snippet=snippet,
                                    title=title,
                                )

                            try:
                                genuine = await asyncio.to_thread(_classify_one)
                            except Exception as e:
                                logger.warning("classify_alias_mention failed: %s", e)
                                genuine = True
                            if not genuine:
                                fp_hits.append(h)
                        if not fp_hits:
                            notes_parts.append(
                                "abstract_fp_scan: no FP-confirmed abstract hits in classified sample",
                            )

                    if fp_hits:

                        def _derive_excludes() -> list[str]:
                            return derive_exclude_terms_from_fps(
                                self._agent,
                                dataset_name=request.dataset_name,
                                fp_hits=fp_hits,
                            )

                        suggested = await asyncio.to_thread(_derive_excludes)
                        if suggested:
                            exc_work = dedupe_strings_ci_preserve_order(exc_work + suggested[:25])
                            notes_parts.append(
                                "abstract_fp_derived_excludes="
                                f"{len(suggested)} scan_hits={len(raw_hits)} fp_confirmed={len(fp_hits)}",
                            )
            except Exception as e:
                logger.warning("optimize abstract exclude derivation failed: %s", e)

        for iter_no in range(2):
            worst_idx: int | None = None
            worst_fp = -1.0
            worst_cnt = -1
            for idx, ent in enumerate(tested):
                fpv = float(ent.fp_rate_pct or 0.0)
                if fpv <= 3.0:
                    continue
                cnt = ent.expected_count or 0
                if cnt <= 0:
                    continue
                if worst_idx is None or fpv > worst_fp or (fpv == worst_fp and cnt > worst_cnt):
                    worst_idx, worst_fp, worst_cnt = idx, fpv, cnt
            if worst_idx is None:
                break
            i_pick = worst_idx
            exc_lower = {e.lower() for e in exc_work}
            extras: list[str] = []
            for kw in noise_by_idx.get(i_pick, []):
                k = kw.strip()
                if len(k) < 4:
                    continue
                if k.lower() in exc_lower:
                    continue
                extras.append(k)
                if len(extras) >= 5:
                    break
            if not extras:
                notes_parts.append("fp_refine: no new exclude candidates from keyword hits.")
                break
            exc_work = dedupe_strings_ci_preserve_order(exc_work + extras)
            if not (risky and flag_terms):
                break
            new_fc = build_for_clause(
                safe=safe,
                variant="V4",
                risky=risky,
                flag_terms=flag_terms,
                exclusion_terms=exc_work,
            )
            if not new_fc.strip():
                break
            if not self._budget_left():
                notes_parts.append("fp_refine: skipped (Dimensions call budget).")
                break
            label_r = f"V4~{iter_no + 1}"
            try:
                vr = await run_variant(self._dsl, label_r, new_fc, limit=20)
                self._dims_calls += 1
            except DimensionsDslError as e:
                errors.append(f"{label_r}: {e}")
                break
            tit = list(vr.get("top_titles") or [])
            fp2, noise2 = _fp_rate_and_noise_terms(tit, fp_keywords)
            tested.append(
                VariantTestedEntry(
                    label=label_r,
                    for_clause=new_fc,
                    expected_count=vr.get("expected_count"),
                    fp_rate_pct=fp2,
                    top_titles=tit[:10],
                )
            )
            noise_by_idx[len(tested) - 1] = noise2
            notes_parts.append(
                f"fp_refine_iter{iter_no + 1} added_excludes={extras!s} new_fp_pct={fp2:.1f}",
            )

        logger.info(
            "optimize_phase run_id=%s phase=phase3_fp dims=%s elapsed_s=%.2f",
            run_id,
            self._dims_calls,
            time.perf_counter() - t_phase3,
        )

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

        scope_r = scope_ratios.get(best.label, 0.0)
        fp_sel = float(best.fp_rate_pct or 0.0)
        if fp_sel < 1.0 and scope_r < 5.0:
            confidence = ConfidenceLevel.HIGH
        elif fp_sel < 3.0 and scope_r < 10.0:
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
