# World Cup 2026 Pool Predictor

Streamlit app that converts live bookmaker odds into exact-score picks for two
prediction pools. For each upcoming match it de-vigs and aggregates bookmaker
odds (Pinnacle-weighted), solves a Dixon-Coles Poisson model to match the
market, and recommends the scorelines that maximize expected pool points:
the #1 score for Pool 1, the #2 for Pool 2.

## Setup

1. `pip install -r requirements.txt`
2. Get a free key at https://the-odds-api.com (500 credits/month) and set it:
   `setx ODDS_API_KEY your_key` (new shells) — or paste it in the app sidebar.
3. **Check `config.yaml` → `pool.scoring`** — it encodes the pools' three-tier scoring (exact / goal-difference-or-draw / winner) with higher stakes in later rounds. Edit if your pool's values change.

## Daily use

```
streamlit run app.py
```

Check the day's matches, hit "Refresh odds now" near kickoff for the freshest
prices, submit Pool 1 / Pool 2 picks. Odds are cached in `data/cache/` so
re-opening the app costs no API quota.

## Phone access (Streamlit Community Cloud)

The app is deployed from this repo at share.streamlit.io. The API key lives in
the app's **Settings → Secrets** as `ODDS_API_KEY = "..."` (never committed).
Keep viewing restricted (Settings → Sharing) — protects the API quota and the
picks. On iPhone: open the app URL in Safari → Share → **Add to Home Screen**.

## Maintenance scripts

- `python scripts/build_fixtures.py` — rebuild the match schedule
  (re-run when knockout pairings are decided).
- `python scripts/snapshot_elo.py` — refresh Elo ratings (every few days).

## Tests

```
python -m pytest
```
