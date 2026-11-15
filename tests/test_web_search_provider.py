"""The configured web search provider decides; a Tavily key alone does not switch it."""

from dataset_agent.adapters import tools


def _spy(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_tavily_search", lambda q, api_key="": calls.append("tavily") or "t")
    monkeypatch.setattr(tools, "_ddg_search", lambda q: calls.append("duckduckgo") or "d")
    return calls


def test_duckduckgo_stays_duckduckgo_when_a_tavily_key_is_set(monkeypatch):
    calls = _spy(monkeypatch)
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setenv("TAVILY_API_KEY", "key-not-used")
    tools.web_search("example query")
    assert calls == ["duckduckgo"]


def test_tavily_is_used_when_configured(monkeypatch):
    calls = _spy(monkeypatch)
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "key")
    tools.web_search("example query")
    assert calls == ["tavily"]
