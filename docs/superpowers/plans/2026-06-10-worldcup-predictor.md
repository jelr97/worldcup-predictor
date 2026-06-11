# World Cup Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app that converts live bookmaker odds into exact-score recommendations (one pick per pool, two pools) for every upcoming World Cup 2026 match.

**Architecture:** Bookmaker odds (The Odds API, cached on disk) are de-vigged and aggregated, a Dixon-Coles-adjusted Poisson model is solved to reproduce the market probabilities, and an expected-points optimizer ranks every scoreline under the pools' scoring rules. Elo ratings (bundled snapshot) serve as fallback and sanity check. `app.py` is a thin Streamlit UI over `model/predict.py`.

**Tech Stack:** Python 3.14, Streamlit, numpy/scipy, requests, pyyaml, pytest. Windows 11, Git Bash shell, repo at `~/worldcup-predictor`.

**Spec:** `docs/superpowers/specs/2026-06-10-worldcup-predictor-design.md`

**Working directory for all commands:** `~/worldcup-predictor` (repo root). All test runs are `python -m pytest` from the root.

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`, `config.yaml`, `config.py`, `.gitignore`, `conftest.py`, `data/__init__.py`, `model/__init__.py`, `tests/test_config.py`

- [ ] **Step 1: Write requirements.txt**

```
streamlit>=1.40
pandas>=2.0
numpy>=1.26
scipy>=1.12
requests>=2.31
pyyaml>=6.0
tzdata>=2024.1
pytest>=8.0
```

(`tzdata` is required on Windows — `zoneinfo` has no system timezone database there.)

- [ ] **Step 2: Write .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
data/cache/
.env
```

- [ ] **Step 3: Write config.yaml**

```yaml
pool:
  # IMPORTANT: user must replace with the pools' real point values.
  # The ranking depends on the exact-to-outcome ratio.
  pts_exact: 3      # total points for predicting the exact score
  pts_outcome: 1    # points for right outcome, wrong score
knockout_scoring: 90min   # only '90min' supported in v1
odds:
  sport_key: soccer_fifa_world_cup   # verify in Task 14 against /v4/sports
  regions: [uk, eu]                  # us too would double extras cost; free tier = 500/mo
  cache_max_age_hours: 12
  extra_markets_window_hours: 24     # fetch btts/alt-totals only this close to kickoff
elo:
  disagreement_threshold: 0.15
display:
  upcoming_window_hours: 48
```

- [ ] **Step 4: Write the failing test** — `tests/test_config.py`

```python
import pytest
from config import load_config


def test_load_config_defaults():
    cfg = load_config()
    assert cfg["pool"]["pts_exact"] > cfg["pool"]["pts_outcome"] > 0
    assert cfg["odds"]["sport_key"]
    assert cfg["knockout_scoring"] == "90min"


def test_non_90min_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("pool: {pts_exact: 3, pts_outcome: 1}\nknockout_scoring: after_et\n")
    with pytest.raises(NotImplementedError):
        load_config(p)
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pip install -r requirements.txt` then `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 6: Write config.py (repo root)**

```python
import os
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_config(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("knockout_scoring", "90min") != "90min":
        raise NotImplementedError(
            "Only 90-minute scoring is supported in v1. "
            "If a pool scores the post-extra-time result, this needs extension."
        )
    return cfg


def get_api_key():
    return os.environ.get("ODDS_API_KEY", "")
```

Also create empty files: `conftest.py` (root — makes the repo root importable for pytest), `data/__init__.py`, `model/__init__.py`.

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding (config, requirements, test setup)"
```

---

### Task 2: Team-name normalization

The three data sources (fixtures feed, The Odds API, Elo ratings) spell teams differently ("USA" / "United States", "Korea Republic" / "South Korea"). One module owns matching.

**Files:**
- Create: `data/team_names.py`
- Test: `tests/test_team_names.py`

- [ ] **Step 1: Write the failing test**

```python
from data.team_names import normalize, same_team


def test_aliases():
    assert same_team("USA", "United States")
    assert same_team("Korea Republic", "South Korea")
    assert same_team("Türkiye", "Turkey")
    assert same_team("Côte d'Ivoire", "Ivory Coast")


def test_accents_and_case():
    assert normalize("  MÉXICO ") == "mexico"


def test_different_teams():
    assert not same_team("Brazil", "Argentina")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_team_names.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write data/team_names.py**

```python
"""Canonical team-name matching across data sources (fixtures, odds API, Elo)."""
import unicodedata

# post-accent-strip, lowercase source name -> canonical name
ALIASES = {
    "usa": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "cote d'ivoire": "ivory coast",
    "turkiye": "turkey",
    "czechia": "czech republic",
    "bosnia and herzegovina": "bosnia",
    "bosnia-herzegovina": "bosnia",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    return ALIASES.get(s, s)


def same_team(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)
```

(Extend `ALIASES` during live verification, Task 14, when real source names are visible.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_team_names.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: team-name normalization across data sources"
```

---

### Task 3: Fixtures — loader, build script, real fixtures.json

**Files:**
- Create: `data/fixtures.py`, `scripts/build_fixtures.py`, `data/fixtures.json` (generated)
- Test: `tests/test_fixtures.py`, `tests/test_fixtures_integrity.py`

- [ ] **Step 1: Write the failing loader test** — `tests/test_fixtures.py`

```python
import json
from datetime import datetime, timezone

from data.fixtures import load_fixtures, upcoming

SAMPLE = [
    {"match_id": 2, "stage": "Group B", "home": "Canada", "away": "Qatar",
     "kickoff_utc": "2026-06-12T01:00:00Z", "venue": "Toronto"},
    {"match_id": 1, "stage": "Group A", "home": "Mexico", "away": "South Africa",
     "kickoff_utc": "2026-06-11T19:00:00Z", "venue": "Mexico City"},
]


def _write(tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return p


def test_load_sorts_and_converts_tz(tmp_path):
    fx = load_fixtures(_write(tmp_path))
    assert [f["match_id"] for f in fx] == [1, 2]
    # 19:00 UTC on Jun 11 is 15:00 EDT
    assert fx[0]["kickoff_et"].hour == 15


def test_upcoming_window(tmp_path):
    fx = load_fixtures(_write(tmp_path))
    now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    assert len(upcoming(fx, now=now, window_hours=12)) == 1
    assert len(upcoming(fx, now=now, window_hours=48)) == 2
    assert len(upcoming(fx, now=datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc),
                        window_hours=48)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write data/fixtures.py**

```python
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"
ET = ZoneInfo("America/New_York")


def load_fixtures(path=FIXTURES_PATH):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in raw:
        f = dict(f)
        f["kickoff_utc"] = datetime.fromisoformat(f["kickoff_utc"].replace("Z", "+00:00"))
        f["kickoff_et"] = f["kickoff_utc"].astimezone(ET)
        out.append(f)
    return sorted(out, key=lambda f: f["kickoff_utc"])


def upcoming(fixtures, now=None, window_hours=48):
    """Matches from 3h ago (today's in-play still visible) to now + window."""
    now = now or datetime.now(timezone.utc)
    end = now + timedelta(hours=window_hours)
    return [f for f in fixtures
            if now - timedelta(hours=3) <= f["kickoff_utc"] <= end]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fixtures.py -v`
Expected: 2 passed

- [ ] **Step 5: Write scripts/build_fixtures.py**

```python
"""One-time: build data/fixtures.json from fixturedownload.com."""
import json
import sys
from pathlib import Path

import requests

FEED = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures.json"


def main():
    r = requests.get(FEED, timeout=30)
    if r.status_code != 200:
        sys.exit(
            f"Feed returned {r.status_code}. Open https://fixturedownload.com, "
            "find the FIFA World Cup 2026 page, and update FEED with the exact "
            "json feed URL shown there."
        )
    fixtures = []
    for e in r.json():
        fixtures.append({
            "match_id": e["MatchNumber"],
            "stage": e.get("Group") or f"Knockout R{e['RoundNumber']}",
            "home": e["HomeTeam"],
            "away": e["AwayTeam"],
            "kickoff_utc": e["DateUtc"].replace(" ", "T").replace("Z", "") + "Z",
            "venue": e.get("Location", ""),
        })
    teams = {f["home"] for f in fixtures} | {f["away"] for f in fixtures}
    print(f"{len(fixtures)} matches, {len(teams)} team slots "
          "(knockouts may be placeholders like '1A')")
    OUT.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the build script (network)**

Run: `python scripts/build_fixtures.py`
Expected: `104 matches, ... team slots` and `Wrote .../data/fixtures.json`.
If the feed 404s: follow the error message — find the 2026 feed slug on fixturedownload.com and update `FEED`. If fixturedownload has no 2026 feed at all, fall back to building `fixtures.json` by hand for the group stage from the official FIFA schedule (72 entries; tedious but unblocked) — or temporarily run with only the next few days of matches entered manually; the schema stays the same.

- [ ] **Step 7: Write the integrity test** — `tests/test_fixtures_integrity.py`

```python
from datetime import datetime, timezone

import pytest

from data.fixtures import FIXTURES_PATH, load_fixtures


@pytest.mark.skipif(not FIXTURES_PATH.exists(), reason="fixtures.json not built yet")
def test_real_fixtures_integrity():
    fx = load_fixtures()
    assert len(fx) == 104
    groups = [f for f in fx if f["stage"].startswith("Group")]
    assert len(groups) == 72
    start = datetime(2026, 6, 10, tzinfo=timezone.utc)
    end = datetime(2026, 7, 21, tzinfo=timezone.utc)
    assert all(start <= f["kickoff_utc"] <= end for f in fx)
```

- [ ] **Step 8: Run all tests**

Run: `python -m pytest -v`
Expected: all pass (integrity test passes, or is the only failure — in which case fix the build per Step 6 notes before proceeding)

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat: WC2026 fixtures (loader, build script, bundled schedule)"
```

---

### Task 4: De-vig and bookmaker aggregation (`model/implied.py`)

**Files:**
- Create: `model/implied.py`
- Test: `tests/test_implied.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from model.implied import aggregate, devig


def test_devig_sums_to_one():
    p = devig({"home": 1.65, "draw": 3.9, "away": 5.8})
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["home"] == pytest.approx(0.586, abs=0.005)


def test_devig_rejects_bad_odds():
    assert devig({}) is None
    assert devig({"home": 1.0, "away": 2.0}) is None
    assert devig({"home": None, "away": 2.0}) is None


def test_aggregate_median():
    books = {
        "a": {"home": 0.50, "draw": 0.30, "away": 0.20},
        "b": {"home": 0.60, "draw": 0.25, "away": 0.15},
        "c": {"home": 0.55, "draw": 0.28, "away": 0.17},
    }
    agg = aggregate(books)
    assert agg["home"] == pytest.approx(0.55, abs=0.01)
    assert sum(agg.values()) == pytest.approx(1.0)


def test_aggregate_pinnacle_weighted():
    books = {
        "pinnacle": {"home": 0.60, "draw": 0.25, "away": 0.15},
        "b": {"home": 0.50, "draw": 0.30, "away": 0.20},
    }
    agg = aggregate(books)
    # 0.5*pinnacle + 0.5*median(others) = 0.5*0.60 + 0.5*0.50 = 0.55
    assert agg["home"] == pytest.approx(0.55, abs=0.01)


def test_aggregate_empty():
    assert aggregate({}) is None
    assert aggregate({"a": None}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_implied.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write model/implied.py**

```python
"""Bookmaker odds -> de-vigged probabilities, aggregated across books."""
import statistics


def devig(odds):
    """{'home': 2.1, 'draw': 3.3, 'away': 3.6} -> probs summing to 1, or None."""
    if not odds or any(o is None or o <= 1.0 for o in odds.values()):
        return None
    inv = {k: 1.0 / v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def aggregate(per_book):
    """per_book: {bookmaker_key: {outcome: prob}} -> consensus probs.

    Median across books; when Pinnacle quotes the match:
    0.5 * pinnacle + 0.5 * median(other books). Renormalized.
    """
    per_book = {b: p for b, p in per_book.items() if p}
    if not per_book:
        return None
    keys = list(next(iter(per_book.values())).keys())
    pinn = per_book.get("pinnacle")
    others = {b: p for b, p in per_book.items() if b != "pinnacle"}
    med = ({k: statistics.median(p[k] for p in others.values()) for k in keys}
           if others else None)
    if pinn and med:
        probs = {k: 0.5 * pinn[k] + 0.5 * med[k] for k in keys}
    else:
        probs = pinn or med
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_implied.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: de-vig and Pinnacle-weighted bookmaker aggregation"
```

---

### Task 5: Dixon-Coles scoreline matrix (`model/poisson.py`, part 1)

**Files:**
- Create: `model/poisson.py`
- Test: `tests/test_poisson.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from model.poisson import outcome_probs, prob_btts, prob_over, score_matrix


def test_matrix_is_distribution():
    m = score_matrix(1.4, 1.1)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_dixon_coles_boosts_low_draws():
    dc = score_matrix(1.4, 1.1, rho=-0.10)
    plain = score_matrix(1.4, 1.1, rho=0.0)
    assert dc[0, 0] > plain[0, 0]
    assert dc[1, 1] > plain[1, 1]
    assert dc[1, 0] < plain[1, 0]


def test_outcome_probs():
    m = score_matrix(2.0, 0.8)
    oc = outcome_probs(m)
    assert sum(oc.values()) == pytest.approx(1.0)
    assert oc["home"] > oc["away"]


def test_over_and_btts():
    m = score_matrix(1.4, 1.1)
    g = np.add.outer(np.arange(11), np.arange(11))
    assert prob_over(m, 2.5) == pytest.approx(float(m[g >= 3].sum()))
    assert prob_btts(m) == pytest.approx(float(m[1:, 1:].sum()))
    assert 0 < prob_over(m, 2.5) < 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_poisson.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write model/poisson.py (matrix + derived probabilities)**

```python
"""Dixon-Coles-adjusted Poisson scoreline model."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as _poisson

RHO = -0.10   # fixed low-score correlation; typical fitted range -0.05..-0.15
MAX_GOALS = 10


def score_matrix(lam_home, lam_away, rho=RHO):
    """P(home goals=i, away goals=j) for i,j in 0..MAX_GOALS. Rows = home."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(_poisson.pmf(g, lam_home), _poisson.pmf(g, lam_away))
    # Dixon-Coles tau adjustment on the four low-score cells
    m[0, 0] *= 1 - lam_home * lam_away * rho
    m[0, 1] *= 1 + lam_home * rho
    m[1, 0] *= 1 + lam_away * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def outcome_probs(m):
    return {
        "home": float(np.tril(m, -1).sum()),   # i > j
        "draw": float(np.trace(m)),
        "away": float(np.triu(m, 1).sum()),    # j > i
    }


def prob_over(m, line):
    g = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return float(m[g > line].sum())


def prob_btts(m):
    return float(m[1:, 1:].sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_poisson.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Dixon-Coles scoreline matrix and derived probabilities"
```

---

### Task 6: Solving goal rates from market probabilities (`model/poisson.py`, part 2)

**Files:**
- Modify: `model/poisson.py` (append `solve_rates`)
- Test: `tests/test_poisson.py` (append)

- [ ] **Step 1: Write the failing round-trip tests** (append to `tests/test_poisson.py`)

```python
from model.poisson import solve_rates


def _constraints_from(lh, la):
    m = score_matrix(lh, la)
    return {
        "1x2": outcome_probs(m),
        "totals": [(2.5, prob_over(m, 2.5))],
        "btts": prob_btts(m),
    }


def test_round_trip_full_constraints():
    lh, la = solve_rates(_constraints_from(1.8, 0.9))
    assert lh == pytest.approx(1.8, abs=0.05)
    assert la == pytest.approx(0.9, abs=0.05)


def test_round_trip_1x2_only():
    m = score_matrix(1.5, 1.0)
    lh, la = solve_rates({"1x2": outcome_probs(m), "totals": [], "btts": None})
    assert lh == pytest.approx(1.5, abs=0.15)
    assert la == pytest.approx(1.0, abs=0.15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_poisson.py -v`
Expected: 4 passed, 2 FAIL — `ImportError: cannot import name 'solve_rates'`

- [ ] **Step 3: Append solve_rates to model/poisson.py**

```python
def solve_rates(constraints, total_prior=2.5, rho=RHO):
    """Find (lam_home, lam_away) whose matrix best reproduces market probs.

    constraints: {'1x2': {'home','draw','away'} (required),
                  'totals': [(line, p_over), ...],
                  'btts': p_yes or None}
    Extra keys are ignored. When no totals are quoted, a soft prior keeps
    total goals near total_prior.
    """
    one_x2 = constraints["1x2"]
    totals = constraints.get("totals") or []
    btts = constraints.get("btts")

    def loss(x):
        lh, la = np.exp(x)
        m = score_matrix(lh, la, rho)
        oc = outcome_probs(m)
        err = sum((oc[k] - one_x2[k]) ** 2 for k in ("home", "draw", "away"))
        for line, p in totals:
            err += (prob_over(m, line) - p) ** 2
        if btts is not None:
            err += (prob_btts(m) - btts) ** 2
        if not totals:
            err += 0.01 * ((lh + la) - total_prior) ** 2
        return err

    edge = one_x2["home"] - one_x2["away"]
    share = min(0.8, max(0.2, 0.5 + 0.4 * edge))
    x0 = np.log([total_prior * share, total_prior * (1 - share)])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000})
    lh, la = np.exp(res.x)
    return float(lh), float(la)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_poisson.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: solve Poisson goal rates from market probabilities"
```

---

### Task 7: Expected-points pick optimizer (`model/picks.py`)

**Files:**
- Create: `model/picks.py`
- Test: `tests/test_picks.py`

- [ ] **Step 1: Write the failing test**

The toy matrix is hand-computed: outcome probs are home 0.35, draw 0.45, away 0.20. With 3 pts exact / 1 pt outcome: EP(0-0) = 0.2·3 + (0.45−0.2)·1 = 0.85; EP(1-1) = 0.85 (tie → fewer total goals wins → 0-0 first); EP(1-0) = 0.15·3 + 0.20·1 = 0.65.

```python
import numpy as np
import pytest

from model.picks import expected_points, outcome_of, top_picks

TOY = np.array([
    [0.20, 0.10, 0.05],
    [0.15, 0.20, 0.05],
    [0.10, 0.10, 0.05],
])  # rows = home goals; sums to 1.0


def test_outcome_of():
    assert outcome_of(2, 1) == "home"
    assert outcome_of(0, 0) == "draw"
    assert outcome_of(0, 3) == "away"


def test_expected_points_toy():
    rows = expected_points(TOY, pts_exact=3, pts_outcome=1)
    by_score = {r["score"]: r["ep"] for r in rows}
    assert by_score["0-0"] == pytest.approx(0.85)
    assert by_score["1-1"] == pytest.approx(0.85)
    assert by_score["1-0"] == pytest.approx(0.65)


def test_top_picks_tiebreak_and_pools():
    p = top_picks(TOY, pts_exact=3, pts_outcome=1, n=5)
    assert p["pool1"]["score"] == "0-0"   # EP tie with 1-1, fewer goals first
    assert p["pool2"]["score"] == "1-1"
    assert len(p["table"]) == 5
    assert p["table"][0] is p["pool1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_picks.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write model/picks.py**

```python
"""Rank scorelines by expected pool points; pick #1 for Pool 1, #2 for Pool 2."""
from model.poisson import outcome_probs


def outcome_of(i, j):
    return "home" if i > j else "away" if j > i else "draw"


def expected_points(matrix, pts_exact, pts_outcome):
    oc = outcome_probs(matrix)
    rows = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j])
            ep = p * pts_exact + (oc[outcome_of(i, j)] - p) * pts_outcome
            rows.append({"score": f"{i}-{j}", "home_goals": i, "away_goals": j,
                         "p_exact": p, "ep": ep})
    # ties: higher exact probability, then fewer total goals (deterministic)
    rows.sort(key=lambda r: (-round(r["ep"], 9), -round(r["p_exact"], 9),
                             r["home_goals"] + r["away_goals"]))
    return rows


def top_picks(matrix, pts_exact, pts_outcome, n=5):
    rows = expected_points(matrix, pts_exact, pts_outcome)
    return {"pool1": rows[0], "pool2": rows[1], "table": rows[:n]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_picks.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: expected-points optimizer with Pool 1 / Pool 2 picks"
```

---

### Task 8: The Odds API client with disk cache (`data/odds_api.py`)

**Files:**
- Create: `data/odds_api.py`
- Test: `tests/test_odds_api.py`

No live network in tests — `requests.get` is monkeypatched.

- [ ] **Step 1: Write the failing test**

```python
import requests

from data.odds_api import OddsClient


def make_client(tmp_path):
    return OddsClient("k", "soccer_fifa_world_cup", cache_dir=tmp_path)


def test_cache_round_trip(tmp_path):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "e1"}])
    data, age = c.load_cache("main")
    assert data == [{"id": "e1"}]
    assert age < 0.01


def test_fresh_cache_skips_network(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "e1"}])
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")))
    data, age = c.get_main_odds(max_age_hours=12)
    assert data == [{"id": "e1"}]


def test_network_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    c.save_cache("main", [{"id": "old"}])
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    data, age = c.get_main_odds(max_age_hours=0)  # cache counts as stale
    assert data == [{"id": "old"}]


def test_no_cache_no_network_returns_none(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    data, age = c.get_main_odds()
    assert data is None and age is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds_api.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write data/odds_api.py**

```python
"""The Odds API v4 client with on-disk JSON cache and quota tracking."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"
DEFAULT_CACHE = Path(__file__).parent / "cache"


class OddsClient:
    def __init__(self, api_key, sport_key, regions=("uk", "eu"), cache_dir=DEFAULT_CACHE):
        self.api_key = api_key
        self.sport_key = sport_key
        self.regions = ",".join(regions)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quota_remaining = None

    def _get(self, path, **params):
        params = {"apiKey": self.api_key, "regions": self.regions,
                  "oddsFormat": "decimal", **params}
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        self.quota_remaining = r.headers.get("x-requests-remaining")
        return r.json()

    def save_cache(self, name, data):
        payload = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                   "quota_remaining": self.quota_remaining, "data": data}
        (self.cache_dir / f"{name}.json").write_text(json.dumps(payload),
                                                     encoding="utf-8")

    def load_cache(self, name):
        p = self.cache_dir / f"{name}.json"
        if not p.exists():
            return None, None
        payload = json.loads(p.read_text(encoding="utf-8"))
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(payload["fetched_at"])).total_seconds() / 3600
        self.quota_remaining = self.quota_remaining or payload.get("quota_remaining")
        return payload["data"], age_h

    def _cached_or_fetch(self, name, path, markets, max_age_hours, force):
        data, age = self.load_cache(name)
        if data is not None and not force and age <= max_age_hours:
            return data, age
        try:
            fresh = self._get(path, markets=markets)
            self.save_cache(name, fresh)
            return fresh, 0.0
        except requests.RequestException:
            return (data, age) if data is not None else (None, None)

    def get_main_odds(self, max_age_hours=12, force=False):
        """All upcoming WC events with h2h + totals. Returns (events, age_hours)."""
        return self._cached_or_fetch(
            "main", f"/sports/{self.sport_key}/odds", "h2h,totals",
            max_age_hours, force)

    def get_event_extras(self, event_id, max_age_hours=12, force=False):
        """BTTS + alternate totals for one event (extra API credits)."""
        return self._cached_or_fetch(
            f"event_{event_id}",
            f"/sports/{self.sport_key}/events/{event_id}/odds",
            "btts,alternate_totals", max_age_hours, force)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds_api.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Odds API client with disk cache and graceful fallback"
```

---

### Task 9: Market constraints builder (`model/markets.py`)

Converts a raw Odds-API event (plus optional per-event extras) into the constraints dict `solve_rates` consumes.

**Files:**
- Create: `model/markets.py`, `tests/sample_odds_event.json`, `tests/sample_odds_extras.json`, `tests/conftest.py`
- Test: `tests/test_markets.py`

- [ ] **Step 1: Create the canned API samples**

`tests/sample_odds_event.json` (shape matches The Odds API v4 `/odds` response items):

```json
{
  "id": "abc123",
  "sport_key": "soccer_fifa_world_cup",
  "commence_time": "2026-06-11T19:00:00Z",
  "home_team": "Mexico",
  "away_team": "South Africa",
  "bookmakers": [
    {"key": "pinnacle", "title": "Pinnacle", "markets": [
      {"key": "h2h", "outcomes": [
        {"name": "Mexico", "price": 1.65},
        {"name": "Draw", "price": 3.9},
        {"name": "South Africa", "price": 5.8}]},
      {"key": "totals", "outcomes": [
        {"name": "Over", "point": 2.5, "price": 2.10},
        {"name": "Under", "point": 2.5, "price": 1.78}]}]},
    {"key": "bet365", "title": "Bet365", "markets": [
      {"key": "h2h", "outcomes": [
        {"name": "Mexico", "price": 1.62},
        {"name": "Draw", "price": 4.0},
        {"name": "South Africa", "price": 6.0}]},
      {"key": "totals", "outcomes": [
        {"name": "Over", "point": 2.5, "price": 2.05},
        {"name": "Under", "point": 2.5, "price": 1.80}]}]}
  ]
}
```

`tests/sample_odds_extras.json` (shape matches the per-event `/events/{id}/odds` response):

```json
{
  "id": "abc123",
  "home_team": "Mexico",
  "away_team": "South Africa",
  "bookmakers": [
    {"key": "pinnacle", "title": "Pinnacle", "markets": [
      {"key": "btts", "outcomes": [
        {"name": "Yes", "price": 2.4},
        {"name": "No", "price": 1.55}]},
      {"key": "alternate_totals", "outcomes": [
        {"name": "Over", "point": 1.5, "price": 1.45},
        {"name": "Under", "point": 1.5, "price": 2.75},
        {"name": "Over", "point": 3.5, "price": 3.6},
        {"name": "Under", "point": 3.5, "price": 1.28}]}]}
  ]
}
```

`tests/conftest.py`:

```python
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture
def sample_event():
    return json.loads((HERE / "sample_odds_event.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_extras():
    return json.loads((HERE / "sample_odds_extras.json").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Write the failing test** — `tests/test_markets.py`

```python
import pytest

from model.markets import build_constraints


def test_h2h_aggregated(sample_event):
    c = build_constraints(sample_event)
    # pinnacle devig home ~.586, bet365 ~.597 -> 50/50 blend ~.591
    assert c["1x2"]["home"] == pytest.approx(0.591, abs=0.01)
    assert sum(c["1x2"].values()) == pytest.approx(1.0)
    assert c["books_count"] == 2


def test_totals_collected(sample_event):
    c = build_constraints(sample_event)
    lines = dict(c["totals"])
    assert 2.5 in lines
    assert 0.3 < lines[2.5] < 0.6


def test_extras_add_btts_and_lines(sample_event, sample_extras):
    c = build_constraints(sample_event, sample_extras)
    assert c["btts"] == pytest.approx(0.392, abs=0.01)
    assert {ln for ln, _ in c["totals"]} == {1.5, 2.5, 3.5}


def test_no_h2h_returns_none(sample_event):
    for b in sample_event["bookmakers"]:
        b["markets"] = [m for m in b["markets"] if m["key"] != "h2h"]
    assert build_constraints(sample_event) is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_markets.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write model/markets.py**

```python
"""Build solve_rates constraints from raw Odds-API event JSON."""
from data.team_names import same_team
from model import implied


def _h2h_odds(market, home, away):
    out = {}
    for o in market.get("outcomes", []):
        if o["name"] == "Draw":
            out["draw"] = o["price"]
        elif same_team(o["name"], home):
            out["home"] = o["price"]
        elif same_team(o["name"], away):
            out["away"] = o["price"]
    return out if len(out) == 3 else None


def _totals_odds(market):
    """-> {line: {'over': odds, 'under': odds}} for complete over/under pairs."""
    lines = {}
    for o in market.get("outcomes", []):
        line = o.get("point")
        side = o["name"].lower()
        if line is not None and side in ("over", "under"):
            lines.setdefault(line, {})[side] = o["price"]
    return {ln: v for ln, v in lines.items() if len(v) == 2}


def _btts_odds(market):
    out = {o["name"].lower(): o["price"] for o in market.get("outcomes", [])
           if o["name"].lower() in ("yes", "no")}
    return out if len(out) == 2 else None


def build_constraints(event, extras=None):
    """Returns a constraints dict for poisson.solve_rates, or None without 1X2.

    event: item from /sports/{sport}/odds (h2h, totals)
    extras: response from /sports/{sport}/events/{id}/odds (btts, alternate_totals)
    """
    home, away = event["home_team"], event["away_team"]
    h2h_books, totals_books, btts_books = {}, {}, {}
    for src in [event] + ([extras] if extras else []):
        for book in src.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "h2h":
                    odds = _h2h_odds(mkt, home, away)
                    if odds:
                        h2h_books[book["key"]] = implied.devig(odds)
                elif mkt["key"] in ("totals", "alternate_totals"):
                    for line, ou in _totals_odds(mkt).items():
                        d = implied.devig(ou)
                        if d:
                            totals_books.setdefault(line, {})[book["key"]] = d
                elif mkt["key"] == "btts":
                    d = implied.devig(_btts_odds(mkt) or {})
                    if d:
                        btts_books[book["key"]] = d
    one_x2 = implied.aggregate(h2h_books)
    if not one_x2:
        return None
    totals = []
    for line, books in sorted(totals_books.items()):
        agg = implied.aggregate(books)
        if agg:
            totals.append((line, agg["over"]))
    btts_agg = implied.aggregate(btts_books)
    return {"1x2": one_x2, "totals": totals,
            "btts": btts_agg["yes"] if btts_agg else None,
            "books_count": len(h2h_books)}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_markets.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: build model constraints from raw odds API events"
```

---

### Task 10: Elo ratings (`data/elo.py` + snapshot script)

**Files:**
- Create: `data/elo.py`, `scripts/snapshot_elo.py`, `data/elo_snapshot.csv` (generated)
- Test: `tests/test_elo.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from data.elo import elo_1x2, load_ratings, rating_for, win_expectancy


def test_load_ratings(tmp_path):
    p = tmp_path / "elo.csv"
    p.write_text("team,rating\nBrazil,2100\nUSA,1790\n", encoding="utf-8")
    r = load_ratings(p)
    assert r["brazil"] == 2100
    assert rating_for(r, "United States") == 1790  # via alias


def test_missing_snapshot_is_empty(tmp_path):
    assert load_ratings(tmp_path / "nope.csv") == {}


def test_win_expectancy():
    assert win_expectancy(1800, 1800) == pytest.approx(0.5)
    assert win_expectancy(2000, 1600) > 0.9


def test_elo_1x2_balanced():
    p = elo_1x2(0.5)
    assert p["draw"] == pytest.approx(0.27, abs=0.01)
    assert p["home"] == pytest.approx(p["away"])
    assert sum(p.values()) == pytest.approx(1.0)


def test_elo_1x2_favorite():
    p = elo_1x2(0.9)
    assert p["home"] > 0.8
    assert sum(p.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_elo.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write data/elo.py**

```python
"""World Football Elo ratings (bundled snapshot, refreshed via scripts/snapshot_elo.py)."""
import csv
from pathlib import Path

from data.team_names import normalize

SNAPSHOT = Path(__file__).parent / "elo_snapshot.csv"


def load_ratings(path=SNAPSHOT):
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {normalize(r["team"]): float(r["rating"]) for r in csv.DictReader(f)}


def rating_for(ratings, team):
    return ratings.get(normalize(team))


def win_expectancy(r_home, r_away):
    """Elo expectancy (draws count half). No home advantage: WC is ~neutral."""
    return 1.0 / (1.0 + 10 ** (-(r_home - r_away) / 400.0))


def elo_1x2(we):
    """Split Elo expectancy into 1X2: we = p_home + p_draw/2.

    Draw probability shrinks linearly as the matchup gets lopsided
    (balanced ~27%, floor 5%).
    """
    p_draw = max(0.05, 0.27 - 0.22 * abs(2 * we - 1))
    p_home = max(0.01, we - p_draw / 2)
    p_away = max(0.01, 1 - p_home - p_draw)
    total = p_home + p_draw + p_away
    return {"home": p_home / total, "draw": p_draw / total, "away": p_away / total}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_elo.py -v`
Expected: 5 passed

- [ ] **Step 5: Write scripts/snapshot_elo.py**

The eloratings.net TSV layout is not officially documented, so the parser is heuristic (name field + 4-digit rating in plausible range) with a loud failure and a manual fallback.

```python
"""Refresh data/elo_snapshot.csv from eloratings.net. Re-run every few days
during the tournament (ratings move after each match day)."""
import csv
import re
import sys
from pathlib import Path

import requests

URL = "https://www.eloratings.net/World.tsv"
OUT = Path(__file__).resolve().parents[1] / "data" / "elo_snapshot.csv"


def main():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        sys.exit(
            f"{URL} returned {r.status_code}.\n"
            "Manual fallback: open https://www.eloratings.net, copy the ranking "
            "table (at least the 48 WC teams) into data/elo_snapshot.csv with "
            "header 'team,rating'."
        )
    rows = []
    for line in r.text.splitlines():
        fields = line.split("\t")
        name = next((f for f in fields if re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{2,}", f)), None)
        rating = next((int(f) for f in fields
                       if re.fullmatch(r"\d{4}", f) and 1000 <= int(f) <= 2400), None)
        if name and rating:
            rows.append({"team": name, "rating": rating})
    if len(rows) < 50:
        sys.exit(f"Only parsed {len(rows)} rows - source format changed. "
                 "Inspect the response, adjust parsing, or use the manual "
                 "fallback in the error above.")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "rating"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} ratings; top 5: {[r['team'] for r in rows[:5]]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the snapshot script (network)**

Run: `python scripts/snapshot_elo.py`
Expected: `Wrote 200+ ratings; top 5: ['Argentina', 'Spain', ...]` (top teams should look right — eyeball them). If it fails, use the manual fallback in the error message (only the 48 WC teams are strictly needed); do not block the rest of the build on this.

- [ ] **Step 7: Run all tests, then commit**

Run: `python -m pytest -v`
Expected: all pass

```bash
git add -A && git commit -m "feat: Elo ratings snapshot, expectancy and 1X2 split"
```

---

### Task 11: Prediction orchestrator (`model/predict.py`)

Ties fixtures + odds + Elo + model together; this is the only module `app.py` calls for predictions.

**Files:**
- Create: `model/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from model.predict import match_event, predict_match

CFG = {
    "pool": {"pts_exact": 3, "pts_outcome": 1},
    "elo": {"disagreement_threshold": 0.15},
}
FIXTURE = {"match_id": 1, "stage": "Group A", "home": "Mexico",
           "away": "South Africa", "venue": "Mexico City"}
ELO = {"mexico": 1800, "south africa": 1620}


def test_match_event_direct_and_swapped(sample_event):
    e, swapped = match_event(FIXTURE, [sample_event])
    assert e is sample_event and swapped is False
    flipped = dict(FIXTURE, home="South Africa", away="Mexico")
    e, swapped = match_event(flipped, [sample_event])
    assert e is sample_event and swapped is True
    assert match_event(dict(FIXTURE, home="Brazil"), [sample_event]) == (None, False)


def test_market_prediction(sample_event, sample_extras):
    p = predict_match(FIXTURE, sample_event, sample_extras, False, ELO, CFG)
    assert p.source == "market"
    assert p.books_count == 2
    assert p.probs["home"] > p.probs["away"]
    assert p.pool1["score"] != p.pool2["score"]
    assert len(p.ep_table) == 5
    assert p.elo_disagrees is False  # Elo also makes Mexico favorite here


def test_swapped_orientation(sample_event):
    flipped = dict(FIXTURE, home="South Africa", away="Mexico")
    p = predict_match(flipped, sample_event, None, True, ELO, CFG)
    assert p.probs["away"] > p.probs["home"]  # Mexico is now the away side


def test_elo_fallback():
    p = predict_match(FIXTURE, None, None, False, ELO, CFG)
    assert p.source == "elo"
    assert "model-only" in p.note
    assert p.pool1 is not None


def test_no_data_at_all():
    p = predict_match(FIXTURE, None, None, False, {}, CFG)
    assert p.source == "none"
    assert p.pool1 is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predict.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write model/predict.py**

```python
"""Per-match prediction pipeline: odds -> constraints -> rates -> picks."""
from dataclasses import dataclass, field

from data import elo as elo_mod
from data.team_names import same_team
from model import markets, picks, poisson


@dataclass
class MatchPrediction:
    fixture: dict
    source: str                    # 'market' | 'elo' | 'none'
    probs: dict | None = None      # consensus 1X2 in fixture orientation
    lam_home: float | None = None
    lam_away: float | None = None
    pool1: dict | None = None
    pool2: dict | None = None
    ep_table: list = field(default_factory=list)
    books_count: int = 0
    odds_age_hours: float | None = None
    elo_disagrees: bool = False
    note: str = ""


def match_event(fixture, events):
    """Find the odds event for a fixture. Returns (event, swapped)."""
    for e in events or []:
        if same_team(fixture["home"], e["home_team"]) and \
           same_team(fixture["away"], e["away_team"]):
            return e, False
        if same_team(fixture["home"], e["away_team"]) and \
           same_team(fixture["away"], e["home_team"]):
            return e, True
    return None, False


def _finish(pred, constraints, cfg):
    lh, la = poisson.solve_rates(constraints)
    m = poisson.score_matrix(lh, la)
    p = picks.top_picks(m, cfg["pool"]["pts_exact"], cfg["pool"]["pts_outcome"])
    pred.lam_home, pred.lam_away = lh, la
    pred.probs = constraints["1x2"]
    pred.pool1, pred.pool2, pred.ep_table = p["pool1"], p["pool2"], p["table"]
    return pred


def _elo_1x2_for(fixture, elo_ratings):
    rh = elo_mod.rating_for(elo_ratings, fixture["home"])
    ra = elo_mod.rating_for(elo_ratings, fixture["away"])
    if rh is None or ra is None:
        return None
    return elo_mod.elo_1x2(elo_mod.win_expectancy(rh, ra))


def predict_match(fixture, event, extras, swapped, elo_ratings, cfg, odds_age=None):
    pred = MatchPrediction(fixture=fixture, source="none", odds_age_hours=odds_age)
    constraints = markets.build_constraints(event, extras) if event else None
    if constraints:
        if swapped:  # totals/btts are symmetric; only 1X2 needs flipping
            constraints["1x2"] = {"home": constraints["1x2"]["away"],
                                  "draw": constraints["1x2"]["draw"],
                                  "away": constraints["1x2"]["home"]}
        pred.source = "market"
        pred.books_count = constraints.get("books_count", 0)
        _finish(pred, constraints, cfg)
        elo_probs = _elo_1x2_for(fixture, elo_ratings)
        if elo_probs:
            thr = cfg["elo"]["disagreement_threshold"]
            pred.elo_disagrees = abs(elo_probs["home"] - pred.probs["home"]) > thr
        return pred
    elo_probs = _elo_1x2_for(fixture, elo_ratings)
    if elo_probs:
        pred.source = "elo"
        pred.note = "model-only (no market odds)"
        return _finish(pred, {"1x2": elo_probs, "totals": [], "btts": None}, cfg)
    pred.note = "no odds and no Elo rating - no pick"
    return pred


def predict_upcoming(fixtures_window, events, extras_by_event_id, elo_ratings,
                     cfg, odds_age=None):
    preds = []
    for f in fixtures_window:
        e, swapped = match_event(f, events)
        extras = (extras_by_event_id or {}).get(e["id"]) if e else None
        preds.append(predict_match(f, e, extras, swapped, elo_ratings, cfg, odds_age))
    return preds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predict.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: prediction orchestrator with Elo fallback and disagreement flag"
```

---

### Task 12: Streamlit app + README

**Files:**
- Create: `app.py`, `README.md`
- Test: `tests/test_app_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
from streamlit.testing.v1 import AppTest


def test_app_renders_without_api_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: FAIL — app.py does not exist

- [ ] **Step 3: Write app.py**

```python
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

with st.sidebar:
    api_key = get_api_key() or st.text_input("The Odds API key", type="password")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: 1 passed (renders Elo-only or no-pick states without exceptions)

- [ ] **Step 5: Manual check**

Run: `streamlit run app.py`
Expected: page loads; with no API key, matches show Elo-only picks (or "no pick" if a team is missing from the Elo snapshot). Stop with Ctrl+C.

- [ ] **Step 6: Write README.md**

```markdown
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
3. **Edit `config.yaml` → `pool:` with your pools' real point values.**
   The recommendation depends on the exact-to-outcome points ratio.

## Daily use

```
streamlit run app.py
```

Check the day's matches, hit "Refresh odds now" near kickoff for the freshest
prices, submit Pool 1 / Pool 2 picks. Odds are cached in `data/cache/` so
re-opening the app costs no API quota.

## Maintenance scripts

- `python scripts/build_fixtures.py` — rebuild the match schedule
  (re-run when knockout pairings are decided).
- `python scripts/snapshot_elo.py` — refresh Elo ratings (every few days).

## Tests

```
python -m pytest
```
```

- [ ] **Step 7: Run the full suite, then commit**

Run: `python -m pytest -v`
Expected: all tests pass

```bash
git add -A && git commit -m "feat: Streamlit dashboard and README"
```

---

### Task 13: Create the private GitHub repo and push

**Files:** none (tooling + remote)

- [ ] **Step 1: Install GitHub CLI**

Run: `winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements`
Expected: installed. Open a **new** shell so `gh` is on PATH; verify with `gh --version`.

- [ ] **Step 2: Authenticate (USER ACTION — interactive)**

Run: `gh auth login --hostname github.com --web`
The user must complete the browser device-code flow with their **personal** GitHub account. Verify: `gh auth status` shows logged in.

- [ ] **Step 3: Create the private repo and push**

```bash
cd ~/worldcup-predictor
gh repo create worldcup-predictor --private --source . --remote origin --push
```

Expected: repo created at `github.com/<user>/worldcup-predictor`, branch pushed.
Verify: `gh repo view worldcup-predictor` shows the README and `git log --oneline` matches the remote.

- [ ] **Step 4: Commit nothing — just verify**

Run: `git status`
Expected: clean working tree, branch tracking `origin`.

---

### Task 14: Live verification (real key, real odds, tomorrow's picks)

**Files:**
- Possibly modify: `config.yaml` (sport key), `data/team_names.py` (aliases)

- [ ] **Step 1: Get an Odds API key (USER ACTION)**

User signs up at https://the-odds-api.com/ (free tier). Set it for current and future shells:

```bash
export ODDS_API_KEY=<key>
setx ODDS_API_KEY <key>
```

- [ ] **Step 2: Verify the World Cup sport key**

```bash
curl -s "https://api.the-odds-api.com/v4/sports/?apiKey=$ODDS_API_KEY" | python -m json.tool | grep -i -B2 -A3 "world cup"
```

Expected: an entry like `"key": "soccer_fifa_world_cup"`. If the 2026 key differs, update `odds.sport_key` in `config.yaml`.

- [ ] **Step 3: Run the app against live data**

Run: `streamlit run app.py`
Check, for each of tomorrow's matches (the opener kicks off June 11):
- probabilities look sane vs any odds site (favorite matches the market);
- Pool 1 / Pool 2 picks are plausible scorelines (typically 1-0, 2-1, 2-0 territory);
- bookmaker count ≥ 1 and quota caption is visible;
- no match shows "no pick". If a match misses odds because of a name mismatch (check terminal logs / compare `data/fixtures.json` names vs the API's), add an alias to `data/team_names.py` `ALIASES` and add that pair to `tests/test_team_names.py`.

- [ ] **Step 4: Remind the user to set real pool points**

The user must put the two pools' actual point values into `config.yaml` (`pts_exact`, `pts_outcome`). Until then the default 3/1 is used. (Same rules in both pools per spec, so a single profile is correct.)

- [ ] **Step 5: Final commit and push**

```bash
python -m pytest -v   # everything green
git add -A && git commit -m "chore: live-verified odds pipeline, aliases and sport key"
git push
```

---

## Phase 2 backlog (not in this plan)

Per spec, after v1 is live: Monte Carlo tournament simulator (12 groups, best-thirds advancement, bracket, P(advance)/P(champion)) and a results tracker (actual scores vs picks, points per pool). Plan separately during the group stage.
