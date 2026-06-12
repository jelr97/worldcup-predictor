import math
from datetime import datetime, timedelta, timezone

import streamlit as st

from config import get_api_key, load_config
from data.elo import load_ratings
from data.experts import load_experts
from data.fixtures import load_fixtures, upcoming
from data.flags import flag
from data.odds_api import OddsClient
from model.predict import match_event, predict_upcoming
from ui import render_card

st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="centered")
cfg = load_config()

st.title("⚽ World Cup 2026 — Pool Predictor")


def _secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml configured — normal for local runs
        return default


# ── PIN gate ──────────────────────────────────────────────────────────────────
# When APP_PIN is set (cloud deployment), block everything until the correct
# code is entered on the main page.  No sidebar is used at any point.
_pin = _secret("APP_PIN")
if _pin:
    if not st.session_state.get("pin_ok"):
        st.markdown("## 🔒")
        entered = st.text_input("PIN", type="password")
        if entered and entered != _pin:
            st.error("Wrong PIN")
        if entered == _pin:
            st.session_state["pin_ok"] = True
            st.rerun()
        st.stop()

# ── API key ───────────────────────────────────────────────────────────────────
_odds_key = (get_api_key()
             or _secret("ODDS_API_KEY")
             or st.text_input("The Odds API key", type="password"))
api_key = _odds_key if _odds_key else None

# ── Control row: window radio + refresh button ────────────────────────────────
_default_hours = cfg["display"]["upcoming_window_hours"]
_window_map = {"Today": 24, "2 days": 48, "4 days": 96}
_default_label = {24: "Today", 48: "2 days", 96: "4 days"}.get(_default_hours, "2 days")

col_radio, col_btn = st.columns([4, 1])
with col_radio:
    window_label = st.radio(
        "Window",
        options=list(_window_map.keys()),
        index=list(_window_map.keys()).index(_default_label),
        horizontal=True,
        label_visibility="collapsed",
    )
with col_btn:
    force = st.button("🔄 Refresh")

window = _window_map[window_label]

# ── Load fixtures, Elo, and experts ──────────────────────────────────────────
fixtures = load_fixtures()
window_fx = upcoming(fixtures, window_hours=window)
elo = load_ratings()
experts = load_experts()

# ── Fetch odds ────────────────────────────────────────────────────────────────
events, age = None, None
extras = {}
client = None
if api_key:
    client = OddsClient(api_key, cfg["odds"]["sport_key"],
                        tuple(cfg["odds"]["regions"]))
    # Never auto-fetch: cached odds of any age are shown; the network is only
    # hit when 🔄 Refresh is pressed (and even then, not for data fetched
    # minutes ago — see OddsClient force floors).
    events, age = client.get_main_odds(math.inf, force=force)
    if events:
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=cfg["odds"]["extra_markets_window_hours"])
        for f in window_fx:
            e, _ = match_event(f, events)
            if e and f["kickoff_utc"] <= horizon:
                extras[e["id"]], _ = client.get_event_extras(
                    e["id"], math.inf, force=force)
    if events is None:
        st.info("No market odds loaded yet — tap 🔄 Refresh. "
                "Until then, picks below are model-only (Elo).")

if age and age > cfg["odds"]["cache_max_age_hours"]:
    st.warning(f"⚠️ Odds are {age:.0f} hours old — tap 🔄 Refresh for current prices.")

if not window_fx:
    st.info("No matches in the selected window.")

# ── Render match cards ────────────────────────────────────────────────────────
for p in predict_upcoming(window_fx, events, extras, elo, cfg, odds_age=age,
                          experts=experts):
    f = p.fixture
    st.markdown(
        render_card(p, flag(f["home"]), flag(f["away"])),
        unsafe_allow_html=True,
    )
    with st.expander("Top 5 scores by expected points"):
        st.dataframe(
            [{"Score": r["score"], "P(exact)": f'{r["p_exact"]:.1%}',
              "Expected points": round(r["ep"], 3)} for r in p.ep_table],
            hide_index=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
if client is not None:
    parts = []
    if client.quota_remaining is not None:
        parts.append(f"quota: {client.quota_remaining}")
    if age is not None:
        parts.append(f"odds {age:.1f}h old")
    if parts:
        st.caption(" · ".join(parts))
