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
    # Before unlocking: no cards (the main page has only title + lock + pin input)
    # Cards rendered via st.markdown — assert none contain "POOL 1"
    assert not any("POOL 1" in (m.value or "") for m in at.markdown)
    # Enter the correct PIN via the main page text_input (not sidebar)
    at.text_input[0].set_value("2026")
    at.run(timeout=60)
    assert not at.exception
    # After unlocking via rerun, session state is set; run again to see cards
    # AppTest may need another run after the rerun triggered inside the app
    at.run(timeout=60)
    assert not at.exception
    # Cards should now be present — any markdown element contains "POOL 1"
    assert any("POOL 1" in (m.value or "") for m in at.markdown)
