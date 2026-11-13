"""Former and successor names (renames, merges) added in optimize phase 1."""

from __future__ import annotations

from dataset_agent.adapters.name_history import HISTORY_TOOL, history_phrases
from tests.test_name_grounding import TITLE, _record, _run

NOTA_WEB = (
    "- Network of the Americas | EarthScope\n  The Network of the Americas (NOTA), formerly "
    "the Plate Boundary Observatory, spans 20 countries. UNAVCO merged with IRIS to form "
    "EarthScope Consortium in 2023."
)


def _history_calls(agent) -> list:
    return [c for c in agent.get_structured.call_args_list if c.args[1] is HISTORY_TOOL]


def _llm(history: dict, names: list[str] | None = None):
    def reply(prompt, tool, *a, **k):
        return history if tool is HISTORY_TOOL else {"names": names or []}
    return reply


def test_history_phrases() -> None:
    text = (
        "NOTA, formerly the Plate Boundary Observatory, and later renamed to Network of the "
        "Americas. The Data Portal was formerly called portal."
    )
    assert history_phrases(text) == ["Plate Boundary Observatory", "Network of the Americas"]


def test_history_phrases_ignore_results_about_something_else() -> None:
    web = (
        "- Network of the Americas | EarthScope\n  NOTA, formerly the Plate Boundary "
        "Observatory.\n- American Broadcasting Company - Wikipedia\n  ABC, formerly the NBC "
        "Blue Network, merged with United Paramount Theatres."
    )
    assert history_phrases(web, ["Network of the Americas"]) == ["Plate Boundary Observatory"]


def test_former_name_from_web_phrase_when_llm_is_silent(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Network of the Americas"], [])
    counts = {"Network of the Americas": 1011, "Plate Boundary Observatory": 2115}
    out, agent = _run(
        tmp_path, monkeypatch, counts, aliases=[], suggested=[], llm_names=[], record=rec,
        web=NOTA_WEB,
    )
    assert len(_history_calls(agent)) == 1
    pool = out.aliases.safe + out.aliases.risky
    assert "Plate Boundary Observatory" in pool
    assert "'Plate Boundary Observatory' (named as a former/successor in web results; 2115 hits)" in out.notes


def test_llm_history_adds_names_and_renamed_organizations(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Network of the Americas"], [])
    rec.flag_terms = ["UNAVCO"]
    history = {
        "dataset_names": [
            {"name": "Plate Boundary Observatory", "relation": "former", "year": 2018},
            {"name": "PBO", "relation": "former"},
            {"name": "Tiny Old Network", "relation": "former"},
        ],
        "organizations": [
            {"name": "EarthScope Consortium", "of": "UNAVCO", "relation": "successor"},
            {"name": "National Science Foundation", "of": "UNAVCO", "relation": "former"},
            {"name": "IRIS Consortium", "of": "Seismology Group", "relation": "merged_into"},
            {"name": "Incorporated Research Institutions for Seismology", "of": "UNAVCO",
             "relation": "merged_from"},
        ],
    }
    counts = {"Network of the Americas": 1011, "Plate Boundary Observatory": 2115,
              "Tiny Old Network": 2}
    out, agent = _run(
        tmp_path, monkeypatch, counts, aliases=[], suggested=[], llm_names=[], record=rec,
        web="no phrases here", llm=_llm(history),
    )
    pool = out.aliases.safe + out.aliases.risky
    assert "Plate Boundary Observatory" in pool
    assert "PBO" not in pool and "Tiny Old Network" not in pool
    assert "'Plate Boundary Observatory' (former, 2018; 2115 hits)" in out.notes
    assert out.flag_terms == ['UNAVCO'] and 'added organization name' not in out.notes


def test_name_history_can_be_disabled(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Network of the Americas"], [])
    _out, agent = _run(
        tmp_path, monkeypatch, {"Network of the Americas": 1011}, aliases=[], suggested=[],
        llm_names=[], record=rec, web=NOTA_WEB, optimize_name_history_enabled=False,
    )
    assert _history_calls(agent) == []
