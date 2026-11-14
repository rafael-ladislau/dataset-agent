"""Pure parts of scripts/probe_ai_model_seed.py (no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dataset_agent.adapters.extraction_client import ExtractionError, TopupResult
from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "probe_ai_model_seed.py"
FIXTURE = ROOT / "tests" / "fixtures" / "llm-results-model-task-v2.json"
spec = importlib.util.spec_from_file_location("probe_ai_model_seed", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _settings(tmp_path):
    return Settings(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out",
                    default_years_start=2018, default_years_end=2026, _env_file=None)


def _body():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_build_seed_defaults_every_start_field(tmp_path):
    seed = probe.build_seed({"model_name": "PhaseNet", "description": "Seismic phase picker."},
                            _settings(tmp_path))
    assert seed == {
        "model_name": "PhaseNet", "model_family": "PhaseNet", "description": "Seismic phase picker.",
        "hugging_face_id": None, "model_aliases": [], "flag_terms": [], "exclude_terms": [],
        "years_range": {"start_year": 2018, "end_year": 2026}, "dimensions_limit": None,
        "filter_us_affiliation": False,
    }


def test_build_seed_keeps_given_fields(tmp_path):
    raw = {"model_name": "GraphCast", "description": "Weather model.", "model_family": "DeepMind",
           "years_range": {"start_year": 2022, "end_year": 2025}, "exclude_terms": ["graph cast"]}
    seed = probe.build_seed(raw, _settings(tmp_path))
    assert seed["model_family"] == "DeepMind" and seed["exclude_terms"] == ["graph cast"]
    assert seed["years_range"] == {"start_year": 2022, "end_year": 2025}


@pytest.mark.parametrize("raw, match", [
    ({"description": "x"}, "model_name"),
    ({"model_name": "PhaseNet"}, "description"),
    ({"model_name": "PhaseNet", "description": "x", "seed_type": "ai_model"}, "seed_type"),
    ({"model_name": "PhaseNet", "description": "x", "probe": True}, "probe"),
    ({"model_name": "PhaseNet", "description": "x", "aliases": []}, "aliases"),
])
def test_build_seed_rejects_what_the_server_would_refuse(tmp_path, raw, match):
    with pytest.raises(ValueError, match=match):
        probe.build_seed(raw, _settings(tmp_path))


def test_wilson_interval():
    lo, hi = probe.wilson_interval(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943
    assert probe.wilson_interval(0, 0) == (None, None)
    lo, hi = probe.wilson_interval(0, 5)
    assert lo == 0.0 and round(hi, 3) == 0.434
    assert probe.wilson_interval(2, 2)[1] == 1.0


def test_summarize_fixture():
    ev = build_evidence(_body())
    out = probe.summarize(ev, quote_chars=20)
    assert (out["n_sampled"], out["n_delivered"], out["n_evaluated"]) == (4, 3, 2)
    assert out["precision"] == 1.0 and out["used_rate"] == 0.5
    assert out["n_cited_only"] == 1 and out["cited_rate"] == 0.5
    assert out["n_error_rows"] == 1 and out["error_row_share"] == round(1 / 3, 4)
    assert out["wilson_95"] == [round(probe.wilson_interval(2, 2)[0], 4), 1.0]
    assert out["n_overridden"] == 0
    first = out["papers"][0]
    assert first == {"publication_id": "pub.1001", "title": "Aftershock detection with deep phase picking",
                     "verdict": "used", "usage_type": "inference", "match_type": "llm_validation",
                     "cited_only": False, "text_quote": "We picked P and S ar…"}
    assert out["papers"][1]["cited_only"] is True


def test_summarize_empty_evidence_has_null_rates():
    out = probe.summarize(build_evidence({"seed_type": "ai_model", "publications": []}))
    assert out["n_evaluated"] == 0 and out["precision"] is None and out["wilson_95"] == [None, None]
    assert out["error_row_share"] is None


class _FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls: list[tuple] = []
        self.cleanup_log = []

    def run_probe_topup(self, seed, **kw):
        self.calls.append((seed, kw))
        self.cleanup_log.append({"task_id": "T9", "ok": True, "items": [{"status": "queued"}],
                                 "error": None})
        if self.error:
            raise self.error
        return self.result


def test_run_probe_reports_probe_and_cleanup():
    body = _body()
    topup = TopupResult(results=body, batches=2, attempted=4, delivered=3, target=30,
                        stop_reason="pool_exhausted", task_id="T9")
    client = _FakeClient(result=topup)
    seed = {"model_name": "PhaseNet", "description": "d"}
    out = probe.run_probe(client, seed, target=30, batch_size=25, max_batches=4, pull_cap=150)
    assert client.calls[0][1] == dict(target=30, batch_size=25, max_batches=4, pull_cap=150,
                                      entity="ai_model")
    assert out["probe_task_id"] == "T9" and out["n_evaluated"] == 2
    assert out["stop_reason"] == "pool_exhausted" and out["batches"] == 2
    assert out["cleanup"] == {"ok": True, "items": [{"status": "queued"}], "error": None}
    assert out["seed"] == seed and out["error"] is None


def test_run_probe_reports_an_extraction_error_with_cleanup():
    client = _FakeClient(error=ExtractionError("probe T9 timed out after 300s"))
    out = probe.run_probe(client, {"model_name": "X", "description": "d"}, target=30,
                          batch_size=25, max_batches=4, pull_cap=150)
    assert "timed out" in out["error"] and out["probe_task_id"] == "T9"
    assert out["cleanup"]["ok"] is True and "n_evaluated" not in out


def test_parse_args_defaults(tmp_path):
    args = probe.parse_args(["seed.json"], _settings(tmp_path))
    assert args.seed == "seed.json" and args.target == 30 and args.max_batches == 4
    assert args.poll_timeout == 900.0 and args.extraction == "dev"
