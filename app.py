from datetime import datetime, timedelta, timezone

import streamlit as st

from config import get_api_key, load_config
from data.elo import load_ratings
from data.fixtures import load_fixtures, upcoming
from data.odds_api import OddsClient
from model.predict import match_event, predict_upcoming

st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="wide")
cfg = load_config()

st.title("⚽ World Cup 2026 — Pool Predictor")


def _secrets_api_key():
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:  # no secrets.toml configured — normal for local runs
        return ""


with st.sidebar:
    api_key = (get_api_key() or _secrets_api_key()
               or st.text_input("The Odds API key", type="password"))
    window = st.slider("Show matches in next (hours)", 24, 96,
                       cfg["display"]["upcoming_window_hours"], step=24)
    force = st.button("🔄 Refresh odds now")

fixtures = load_fixtures()
window_fx = upcoming(fixtures, window_hours=window)
elo = load_ratings()

events, age = None, None
extras = {}
if api_key:
    client = OddsClient(api_key, cfg["odds"]["sport_key"],
                        tuple(cfg["odds"]["regions"]))
    events, age = client.get_main_odds(cfg["odds"]["cache_max_age_hours"], force=force)
    if events:
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=cfg["odds"]["extra_markets_window_hours"])
        for f in window_fx:
            e, _ = match_event(f, events)
            if e and f["kickoff_utc"] <= horizon:
                extras[e["id"]], _ = client.get_event_extras(
                    e["id"], cfg["odds"]["cache_max_age_hours"], force=force)
    if client.quota_remaining:
        st.sidebar.caption(f"API quota remaining: {client.quota_remaining}")
    if events is None:
        st.error("Could not fetch odds and no cache exists — Elo-only predictions.")
else:
    st.sidebar.warning("No API key — Elo-only predictions.")

if age and age > cfg["odds"]["cache_max_age_hours"]:
    st.warning(f"⚠️ Odds are {age:.0f} hours old (API unreachable — using cache).")

if not window_fx:
    st.info("No matches in the selected window.")

for p in predict_upcoming(window_fx, events, extras, elo, cfg, odds_age=age):
    f = p.fixture
    st.subheader(f'{f["home"]} vs {f["away"]}')
    st.caption(f'{f["stage"]} · {f["kickoff_et"].strftime("%a %b %d, %I:%M %p ET")}'
               f' · {f.get("venue", "")}')
    if p.source == "none":
        st.error(p.note)
        st.divider()
        continue
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f'{f["home"]} win', f'{p.probs["home"]:.0%}')
    c2.metric("Draw", f'{p.probs["draw"]:.0%}')
    c3.metric(f'{f["away"]} win', f'{p.probs["away"]:.0%}')
    c4.metric("Pool 1 pick", p.pool1["score"])
    c5.metric("Pool 2 pick", p.pool2["score"])
    tags = []
    if p.source == "market":
        tags.append(f"{p.books_count} bookmakers")
    if p.source == "elo":
        tags.append("🟡 " + p.note)
    if p.elo_disagrees:
        tags.append("⚠️ market and Elo disagree — double-check this one")
    if tags:
        st.caption(" · ".join(tags))
    with st.expander("Top 5 scores by expected points"):
        st.dataframe(
            [{"Score": r["score"], "P(exact)": f'{r["p_exact"]:.1%}',
              "Expected points": round(r["ep"], 3)} for r in p.ep_table],
            hide_index=True)
    st.divider()
