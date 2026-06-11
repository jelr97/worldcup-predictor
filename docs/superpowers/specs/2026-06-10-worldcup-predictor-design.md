# World Cup Predictor — Design

**Date:** 2026-06-10
**Repo:** `worldcup-predictor` (private, personal GitHub)
**Deadline:** first picks needed before the opening match on June 11, 2026

## Goal

A Streamlit app that recommends exact-score predictions for FIFA World Cup 2026
matches, optimized to win two prediction pools. Both pools use the same scoring
rules: full points for the exact score, partial points for the correct outcome
(win/draw/win). Picks are submitted before each match, so the app runs as a
daily companion using the freshest betting odds.

For each match the app shows two picks from one expected-points ranking:
the **#1 score for Pool 1** and the **#2 score for Pool 2**. Submitting two
different scores also diversifies: the same scoreline can never beat the user
in both pools.

## Approach

Odds-anchored Poisson (chosen over a pure historical model): bookmaker odds
are the sharpest public forecast available, so the model converts market
prices into a full scoreline distribution rather than fitting team strength
from noisy international history. An Elo signal is included in v1 as fallback
and sanity check. Phase 2 grows this into a hybrid with tournament simulation.

## Architecture

```
worldcup-predictor/
├── app.py                 # Streamlit dashboard
├── data/
│   ├── fixtures.json      # all 104 WC2026 matches (teams, group, kickoff ET), bundled
│   ├── fixtures.py        # load schedule, filter to upcoming matches
│   ├── odds_api.py        # The Odds API client + on-disk cache (data/cache/)
│   └── elo.py             # World Football Elo ratings fetch + cache
├── model/
│   ├── implied.py         # bookmaker odds → de-vigged probabilities
│   ├── poisson.py         # market probs → (λ_home, λ_away) → scoreline matrix
│   └── picks.py           # expected-points optimizer per pool scoring config
├── config.yaml            # pool scoring points, knockout rule flag, settings
├── tests/                 # pytest suite
├── requirements.txt       # streamlit, pandas, numpy, scipy, requests, pyyaml
└── README.md
```

Conventions follow the user's other projects: Streamlit with
`st.set_page_config()` and `st.session_state`, config-driven, no secrets in
the repo.

## Data flow

1. App loads `fixtures.json` (static — the 2026 schedule ships in the repo;
   built once at implementation time from the official published schedule and
   spot-checked against FIFA's site).
2. Fetches odds from The Odds API for all upcoming World Cup matches in one
   request. Markets: 1X2 (`h2h`), totals, both-teams-to-score, alternate
   totals (the last two via per-event calls, only for matches within 24h of
   kickoff). Regions: UK + EU — adding US would roughly double credit burn
   and push past the free tier's 500/month; Pinnacle quotes in the EU
   region, so the sharp-book signal is kept.
3. Response cached to `data/cache/` with timestamp; the app reuses the cache
   on reload. A "Refresh odds" button forces a new fetch.
4. Elo ratings loaded from a bundled snapshot (data/elo_snapshot.csv), refreshed manually via scripts/snapshot_elo.py.
5. Model runs per match; dashboard shows: kickoff time (ET), de-vigged
   win/draw/win probabilities, recommended Pool 1 and Pool 2 scores, and the
   top-5 candidate scores with expected points.

API key via `ODDS_API_KEY` environment variable or sidebar input.

## Model

**1. De-vig (`implied.py`).** Decimal odds → implied probabilities (1/odds),
normalized to sum to 1 (proportional method), per market. Across bookmakers:
when Pinnacle quotes the match, final probability = 50% Pinnacle + 50% median
of the other books; otherwise the plain median.

**2. Goal rates (`poisson.py`).** Solve for (λ_home, λ_away) such that the
Dixon-Coles-adjusted Poisson scoreline matrix best reproduces the de-vigged
market probabilities — least squares (scipy) over all available constraints:
P(home), P(draw), P(away), P(over/under lines), P(BTTS). The Dixon-Coles
correction (fixed ρ) repairs plain-Poisson's underestimation of low-scoring
draws. Matrix truncated at 10 goals per side. Missing markets simply drop out
of the constraint set; 1X2 alone is sufficient (with a ~2.5 total-goals prior).

**3. Picks (`picks.py`).** For each candidate score (i, j):
`EP = P(exact i-j) × pts_exact + (P(outcome of i-j) − P(exact i-j)) × pts_outcome`,
point values from `config.yaml`. Rank all scores by EP; #1 → Pool 1,
#2 → Pool 2. Note the ranking depends on the exact-to-outcome points ratio,
so the user must enter the pools' real point values in `config.yaml`
(default until then: 3 exact / 1 outcome, a common polla scheme).

**Knockouts:** bookmaker 1X2 prices the 90-minute result (draw included),
which is also what the pools score by default. The config validates this ('90min'); post-extra-time scoring is rejected with a clear error and would need a small extension if a pool turns out to differ.

**Elo (v1, `elo.py`):** two uses — (a) fallback when a match is missing from
the odds feed: the Elo win expectancy is split into win/draw/loss
probabilities (draw share shrinking as the matchup gets more lopsided) and
fed through the same Poisson solver with a 2.5 total-goals prior to give λs;
pick labeled "model-only (no market odds)"; (b) sanity check: flag matches where Elo and market
disagree sharply (possible stale odds).

## Error handling

- Odds API unreachable → last cached snapshot with a visible
  "odds as of N hours ago" warning.
- Match absent from feed → Elo fallback, clearly labeled.
- Market missing for a match → solve with remaining constraints.
- No API key → sidebar input prompt.
- The app must always produce a pick for every upcoming match.

## Testing

pytest suite covering the math:

- De-vig: probabilities sum to 1; median/Pinnacle aggregation.
- Round-trip: known (λ_home, λ_away) → matrix → market probs → re-solved λs
  match the originals.
- Pick optimizer: hand-computed toy distribution yields the known optimal
  pick and correct #1/#2 ordering.
- Fixtures integrity: 104 matches, 48 teams, valid groups and dates.

## Phase 2 (during group stage)

- Monte Carlo tournament simulator: group standings including best-third
  qualification rules, full bracket, P(advance) and P(champion) per team.
- Results tracker: actual scores vs picks, points earned per pool.

## Out of scope

- Correct-score odds markets (not on The Odds API; scraping too fragile).
- Scraping Opta or other forecast sites.
- Automated submission to the pools (manual entry by the user).
