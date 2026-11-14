"""Run the resource-map research step on stored research records (step-5 gate).

Usage: PYTHONPATH=src resource_map_check.py <out.json> <research.json> [<research.json> ...]
Pages are fetched live and saved next to the output (``<out>.pages/<n>.txt``); with
PAGES_DIR set, saved pages are read instead (reproducible reruns). Counts go through the
agent's Dimensions client; set DSL_CACHE_DIR to reuse the replay cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

from dataset_agent.adapters.dimensions_dsl import (
    AsyncThrottledDimensionsDsl,
    build_for_clause_count_dsl,
    parse_total_count,
)
from dataset_agent.adapters.resource_map import gather_pages, map_for_record
from dataset_agent.domain.models import DatasetRecord


def _cache(dir_):
    orig = AsyncThrottledDimensionsDsl.execute_dsl

    async def cached(self, dsl):
        f = Path(dir_) / (hashlib.sha256(dsl.encode()).hexdigest() + ".pkl")
        if f.exists():
            return pickle.loads(f.read_bytes())
        res = await orig(self, dsl)
        f.write_bytes(pickle.dumps(res))
        return res

    AsyncThrottledDimensionsDsl.execute_dsl = cached


def main(out_path: str, paths: list[str]) -> None:
    from dataset_agent.bootstrap import _build_agent
    from dataset_agent.settings import Settings

    s = Settings()
    if os.environ.get("DSL_CACHE_DIR"):
        _cache(os.environ["DSL_CACHE_DIR"])
    dsl = AsyncThrottledDimensionsDsl(s)
    agent = _build_agent(s)
    pages_dir = Path(os.environ.get("PAGES_DIR") or (out_path + ".pages"))
    pages_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()

    def count(clause: str) -> int | None:
        dsl_text = build_for_clause_count_dsl(clause)
        return parse_total_count(loop.run_until_complete(dsl.execute_dsl(dsl_text)))

    out = {}
    for p in paths:
        rec = DatasetRecord.model_validate(json.loads(Path(p).read_text()))
        f = pages_dir / (hashlib.sha256(rec.main_dataset_name.encode()).hexdigest()[:12] + ".txt")
        if f.exists():
            pages = f.read_text()
        else:
            pages = gather_pages(rec.main_dataset_name, rec.home_url)
            f.write_text(pages)
        m = map_for_record(agent, rec, count=count, pages=pages,
                           min_cooccurrence=s.resource_map_min_cooccurrence)
        out[rec.main_dataset_name] = m.model_dump() if m else None
        print(rec.main_dataset_name, "->", "map" if m else "no map", file=sys.stderr)
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
