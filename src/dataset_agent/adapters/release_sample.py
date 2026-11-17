"""The release sample: a random, year-stratified draw of a query's papers with full text.

The release decision for a dataset reads this sample, not the top-cited papers the
refinement probes pull (those stay the refiner's input: its false positives are what it
learns from). The design follows the extraction probe API:

* a probe pulls its pool from Dimensions sorted by citations and capped at 1,000 papers
  (``probe_pull_cap``); a raw ``dimensions_query`` may not contain ``skip`` or ``limit``,
  so there is no seeded offset into the full result set;
* so the sample is stratified by year: one Dimensions year facet of the query gives the
  per-year counts, the sample size is allocated to years in proportion to them (largest
  remainder, so the allocation sums to the target exactly), and one probe per year pulls
  that year with a cap of ``count + 50`` (at most 1,000). When a year has at most 950
  papers the pool is the whole year and the draw inside it is uniformly random; above
  that the pool is the year's 1,000 most cited. A pull that reaches its cap may have
  left papers out, and the record says so (``pool_covers_year`` false); a pull below
  its cap is the whole year, whatever the facet counted. The facet counts the probe
  seed's population (its year window and, when set, the US-affiliation filter; the
  extraction seed search applies no publication-type filter). This cap
  deliberately replaces ``probe_pull_cap`` for these probes: one probe per year with an
  allocation, each pulling up to ``count + 50`` records (at most 1,000), so a query of
  11 years with about 400 papers each pulls about 4,950 records;
* a *large* year (facet count above ``LARGE_YEAR_COUNT``, 950) loses that whole-year
  pool: the pull cap is the 1,000 ceiling itself and no headroom covers the year. Large
  years are sliced by month instead: the year's own month counts (twelve Dimensions
  calls per year, on the release sample's month-count budget, not the optimizer budget)
  allocate the year's quota across its months (largest remainder, ``allocate`` again),
  and one capped probe per month pulls that month through the extraction probe's
  inclusive ``date_range`` window. Every month's pull keeps the ``count + 50`` headroom,
  ``pool_covers_month`` reports a capped month pool exactly as ``pool_covers_year`` does
  for years, and the year record marks every incompleteness (a capped month, a missing
  month count, the month budget spent mid-year). A year whose month counts cannot be
  completed falls back to its whole-year probe, and the run record marks its coverage
  ``degraded_whole_year`` -- never a silently uniform sample;
* inside a year or month, papers are drawn in a seeded random order (the pool sorted by
  id, then shuffled with a generator seeded from the run seed and the year -- and the
  month for a slice), sent for open full text in that order until the allocation has
  delivered full text; the first delivered papers in draw order are kept, so an
  over-delivering batch never skews the strata, and the pooled months of a sliced year
  dedupe by publication id. The run seed is recorded, so a run can be reproduced while
  the pool is the same.
"""

from __future__ import annotations

import calendar
import random
import secrets
from datetime import date
from typing import Any

#: The extraction caps one probe pull at 1,000 papers.
MAX_PULL = 1000
#: Headroom over the facet count: the extraction's own search may admit a few more
#: papers than the facet counted.
PULL_HEADROOM = 50
#: A facet count above this makes ``pull_cap`` reach its 1,000 ceiling: the pull can no
#: longer keep the ``count + 50`` headroom, so it cannot pool the year's whole population
#: and the draw leans on the year's most-cited 1,000. Such a year is *large* (one capped
#: pull cannot cover it, with no margin left) and the sample slices it by month; at 950
#: the cap is still exactly count + 50, so one whole-year pull covers the year fully.
LARGE_YEAR_COUNT = MAX_PULL - PULL_HEADROOM


def new_seed() -> int:
    """A fresh run seed (recorded in the output so the draw can be repeated)."""
    return secrets.randbelow(2**31 - 1) + 1


def allocate(counts: dict[int, int], n: int) -> dict[int, int]:
    """Split *n* papers over years in proportion to *counts* (largest remainder).

    Years with no papers get nothing; the result sums to *n* (or to the total count when
    the query has fewer papers). Ties in the remainder go to the larger year count, then
    to the earlier year. The same rule slices a year's quota across its months: called
    with one year's per-month counts and that year's allocation, the keys are months and
    the result sums to the year's quota exactly."""
    years = {int(y): int(c) for y, c in counts.items() if int(c) > 0}
    total = sum(years.values())
    n = min(max(0, int(n)), total)
    if not n:
        return {}
    quotas = {y: n * c / total for y, c in years.items()}
    alloc = {y: int(q) for y, q in quotas.items()}
    rest = n - sum(alloc.values())
    order = sorted(years, key=lambda y: (-(quotas[y] - alloc[y]), -years[y], y))
    for y in order[:rest]:
        alloc[y] += 1
    return {y: k for y, k in sorted(alloc.items()) if k}


def year_seed(seed: int, year: int) -> int:
    """The generator seed of one year: a year's draw does not depend on the other years."""
    return int(seed) * 10_000 + int(year)


def month_seed(seed: int, year: int, month: int) -> int:
    """The generator seed of one month: a month's draw does not depend on the other
    months or on the year draw of the same year."""
    return int(seed) * 1_000_000 + int(year) * 100 + int(month)


def draw_order(pool: list[str], seed: int, year: int, *,
               month: int | None = None) -> list[str]:
    """The pool's papers in the seeded random order of *year* (independent of input
    order); with *month*, the seeded random order of that month of the year."""
    gen_seed = month_seed(seed, year, month) if month is not None else year_seed(seed, year)
    ids = sorted({str(p) for p in pool})
    random.Random(gen_seed).shuffle(ids)
    return ids


def month_window(year: int, month: int, *,
                 within: tuple[int, int] | None = None) -> tuple[date, date] | None:
    """The inclusive bounds of one month of *year* (February follows the year: a leap
    year's February ends on the 29th); these go to the extraction probe's ``date_range``
    as ISO dates, inclusive on both ends like Dimensions' ``date >=``/``date <=``.
    *within* is the query's own year window (``(start_year, end_year)``): the bounds are
    clamped to it, and a month none of which is inside comes back as ``None`` (nothing
    of it can be probed). Whole-year windows never split a month, so the clamp is exact
    for every month of a sampled year."""
    first = date(int(year), int(month), 1)
    last = date(int(year), int(month), calendar.monthrange(int(year), int(month))[1])
    if within:
        lo, hi = date(int(within[0]), 1, 1), date(int(within[1]), 12, 31)
        first, last = max(first, lo), min(last, hi)
        if first > last:
            return None
    return first, last


def pull_cap(count: int) -> int:
    """The probe pull cap for a year with *count* papers in the facet."""
    return max(1, min(MAX_PULL, int(count) + PULL_HEADROOM))


def is_large_year(count: int) -> bool:
    """True when a year with *count* facet papers is large: above ``LARGE_YEAR_COUNT``
    the probe's pull cap has lost its ``count + 50`` headroom (one capped pull cannot
    cover the year, even for the smallest allocation), so the year is sliced by month and
    each month's pull keeps its own headroom."""
    return int(count) > LARGE_YEAR_COUNT


def year_counts_from_result(result: Any) -> dict[int, int]:
    """Per-year counts from a Dimensions ``return year`` facet result."""
    rows: Any = None
    if result is None:
        return {}
    if hasattr(result, "get"):
        rows = result.get("year")
    if rows is None:
        rows = getattr(result, "year", None)
    if rows is None:
        data = getattr(result, "json", None)
        rows = data.get("year") if isinstance(data, dict) else None
    out: dict[int, int] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        try:
            out[int(r["id"])] = int(r["count"])
        except (KeyError, TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def batch_size_for(need: int, batch_cap: int) -> int:
    """Papers to send for *need* more full texts: about half of the papers sent deliver
    open full text, so twice the need, within the batch cap."""
    return max(1, min(int(batch_cap), 2 * int(need)))
