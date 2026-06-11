from streamlit.testing.v1 import AppTest


def test_app_renders_without_api_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file("../app.py")
    at.run(timeout=60)
    assert not at.exception


def test_pin_gate_blocks_until_correct(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file("../app.py")
    at.secrets["APP_PIN"] = "2026"
    at.run(timeout=60)
    assert not at.exception
    assert not at.subheader  # no match cards before the PIN
    at.sidebar.text_input[0].set_value("2026")
    at.run(timeout=60)
    assert not at.exception
    assert at.subheader  # picks render after the PIN
