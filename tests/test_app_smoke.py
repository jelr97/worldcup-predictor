from streamlit.testing.v1 import AppTest


def test_app_renders_without_api_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file("../app.py")
    at.run(timeout=60)
    assert not at.exception
