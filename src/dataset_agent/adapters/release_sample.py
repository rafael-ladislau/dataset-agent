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
* inside a year, papers are drawn in a seeded random order (the pool sorted by id, then
  shuffled with a generator seeded from the run seed and the year), sent for open full
  text in that order until the year's allocation has delivered full text; the first
  delivered papers in draw order are kept, so an over-delivering batch never skews the
  strata. The run seed is recorded, so a run can be reproduced while the pool is the same.
"""

from __future__ import annotations

import random
import secrets
from typing import Any

#: The extraction caps one probe pull at 1,000 papers.
MAX_PULL = 1000
#: Headroom over the facet count: the extraction's own search may admit a few more
#: papers than the facet counted.
PULL_HEADROOM = 50


def new_seed() -> int:
    """A fresh run seed (recorded in the output so the draw can be repeated)."""
    return secrets.randbelow(2**31 - 1) + 1


def allocate(counts: dict[int, int], n: int) -> dict[int, int]:
    """Split *n* papers over years in proportion to *counts* (largest remainder).

    Years with no papers get nothing; the result sums to *n* (or to the total count when
    the query has fewer papers). Ties in the remainder go to the larger year count, then
    to the earlier year."""
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


def draw_order(pool: list[str], seed: int, year: int) -> list[str]:
    """The pool's papers in the seeded random order of *year* (independent of input order)."""
    ids = sorted({str(p) for p in pool})
    random.Random(year_seed(seed, year)).shuffle(ids)
    return ids


def pull_cap(count: int) -> int:
    """The probe pull cap for a year with *count* papers in the facet."""
    return max(1, min(MAX_PULL, int(count) + PULL_HEADROOM))


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
