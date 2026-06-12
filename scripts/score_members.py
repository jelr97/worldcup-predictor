"""Member scoring CLI.

Usage: python scripts/score_members.py

Two phases:
  1. SNAPSHOT: for each upcoming match within 48h, compute each member's
     Pool 1 pick from current data and append to data/pick_log.csv
     (idempotent on match_id+member).
  2. SCORE: fetch actual results from the fixturedownload feed, score every
     logged pick, and print a per-member summary table.

Members:
  market   - Pool 1 pick from cached Odds-API constraints (skipped if
             cache > 24h old or missing).
  davo     - Davo's exact-score expert prediction (Pool 1 pick via experts
             matrix; retro-scored for all finished matches without snapshots).
  maldini  - Same for Maldini.
  experts  - 50/50 Davo+Maldini mixture matrix (Pool 1 pick).
  blend    - Ensemble with current config weights.
"""

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import requests

from config import load_config
from data.elo import load_ratings
from data.experts import load_experts, picks_for
from data.fixtures import load_fixtures
from data.team_names import same_team
from model.experts import experts_matrix
from model.markets import build_constraints
from model.picks import top_picks
from model.poisson import score_matrix, solve_rates
from model.predict import match_event, predict_match, stage_points
from model.scoring import score_pick

PICK_LOG = REPO / "data" / "pick_log.csv"
MEMBER_SCORES = REPO / "data" / "member_scores.csv"
CACHE_DIR = REPO / "data" / "cache"
FEED_URL = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
MAX_CACHE_AGE_HOURS = 24
SNAPSHOT_WINDOW_HOURS = 48

PICK_LOG_FIELDS = ["match_id", "member", "pick", "computed_at"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_pick_log() -> dict[tuple, str]:
    """Load pick_log.csv -> {(match_id, member): pick}."""
    if not PICK_LOG.exists():
        return {}
    with PICK_LOG.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {(int(row["match_id"]), row["member"]): row["pick"]
                for row in reader}


def _append_picks(new_rows: list[dict]) -> None:
    """Append rows to pick_log.csv; create with header if absent."""
    write_header = not PICK_LOG.exists()
    with PICK_LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PICK_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)


def _load_cache_events() -> tuple[list | None, float | None]:
    """Load main odds cache. Returns (events, age_hours) or (None, None)."""
    cache_file = CACHE_DIR / "main.json"
    if not cache_file.exists():
        return None, None
    import json
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    age_h = ((datetime.now(timezone.utc)
               - datetime.fromisoformat(payload["fetched_at"]))
              .total_seconds() / 3600)
    return payload["data"], age_h


def _fetch_results() -> dict[int, tuple[int, int]] | None:
    """Fetch final scores from fixturedownload feed.

    Returns {match_id: (home_score, away_score)} for finished matches,
    or None if the feed is unreachable.
    """
    try:
        r = requests.get(FEED_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"  [warn] fixturedownload feed unreachable: {exc}")
        return None
    results = {}
    for item in data:
        h = item.get("HomeTeamScore")
        a = item.get("AwayTeamScore")
        if h is not None and a is not None:
            results[int(item["MatchNumber"])] = (int(h), int(a))
    return results


def _experts_pool1_pick(raw_picks, pts) -> str | None:
    """Return Pool 1 pick (score string) from the experts mixture matrix."""
    if raw_picks is None:
        return None
    m = experts_matrix(raw_picks)
    p = top_picks(m, pts)
    return p["pool1"]["score"]


def _davo_pick(raw_picks, pts) -> str | None:
    """Return Pool 1 pick from Davo's matrix only."""
    if raw_picks is None:
        return None
    from model.poisson import score_matrix as _sm
    from model.experts import pick_to_rates
    lh, la = pick_to_rates(raw_picks["davo"])
    m = _sm(lh, la)
    p = top_picks(m, pts)
    return p["pool1"]["score"]


def _maldini_pick(raw_picks, pts) -> str | None:
    """Return Pool 1 pick from Maldini's matrix only."""
    if raw_picks is None:
        return None
    from model.poisson import score_matrix as _sm
    from model.experts import pick_to_rates
    lh, la = pick_to_rates(raw_picks["maldini"])
    m = _sm(lh, la)
    p = top_picks(m, pts)
    return p["pool1"]["score"]


def _market_pick(fixture, events, cfg) -> str | None:
    """Return Pool 1 pick from market odds only (no experts, no Elo)."""
    event, swapped = match_event(fixture, events)
    if event is None:
        return None
    constraints = build_constraints(event)
    if constraints is None:
        return None
    if swapped:
        constraints["1x2"] = {
            "home": constraints["1x2"]["away"],
            "draw": constraints["1x2"]["draw"],
            "away": constraints["1x2"]["home"],
        }
        constraints["spreads"] = [
            (-line, p) for line, p in (constraints.get("spreads") or [])
        ]
    lh, la = solve_rates(constraints)
    m = score_matrix(lh, la)
    pts = stage_points(fixture["stage"], cfg)
    p = top_picks(m, pts)
    return p["pool1"]["score"]


def _blend_pick(fixture, events, experts_data, elo_ratings, cfg) -> str | None:
    """Return Pool 1 pick from the full blended predict_match pipeline."""
    event, swapped = match_event(fixture, events)
    pred = predict_match(fixture, event, None, swapped, elo_ratings, cfg,
                         experts=experts_data)
    if pred.pool1 is None:
        return None
    return pred.pool1["score"]


# ── phase 1: snapshot upcoming picks ─────────────────────────────────────────

def snapshot_upcoming(fixtures, cfg, experts_data, events, cache_age):
    """Compute picks for upcoming fixtures and append to pick_log.csv."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=SNAPSHOT_WINDOW_HOURS)
    upcoming_fx = [f for f in fixtures
                   if now <= f["kickoff_utc"] <= cutoff]
    if not upcoming_fx:
        print("No upcoming fixtures within 48h window.")
        return

    existing = _load_pick_log()
    elo_ratings = load_ratings()
    new_rows = []
    computed_at = now.isoformat()

    market_ok = (events is not None
                 and cache_age is not None
                 and cache_age <= MAX_CACHE_AGE_HOURS)
    if not market_ok:
        reason = "missing" if events is None else f"{cache_age:.1f}h old (>24h)"
        print(f"  [note] market/blend picks skipped: odds cache is {reason}")

    for f in upcoming_fx:
        mid = int(f["match_id"])
        pts = stage_points(f["stage"], cfg)
        raw_picks = picks_for(experts_data, f["home"], f["away"])

        member_picks = {
            "davo": _davo_pick(raw_picks, pts),
            "maldini": _maldini_pick(raw_picks, pts),
            "experts": _experts_pool1_pick(raw_picks, pts),
        }
        if market_ok:
            member_picks["market"] = _market_pick(f, events, cfg)
            member_picks["blend"] = _blend_pick(f, events, experts_data, elo_ratings, cfg)

        for member, pick in member_picks.items():
            if pick is None:
                continue
            if (mid, member) in existing:
                continue   # idempotent: already logged
            new_rows.append({
                "match_id": mid,
                "member": member,
                "pick": pick,
                "computed_at": computed_at,
            })

    if new_rows:
        _append_picks(new_rows)
        print(f"  Appended {len(new_rows)} new rows to pick_log.csv")
    else:
        print("  No new rows to append (all already logged or no picks available).")


# ── phase 2: score backward ────────────────────────────────────────────────────

def score_backward(fixtures, cfg, experts_data, results: dict | None):
    """Score all logged + retro-computable picks against actual results."""
    if results is None:
        print("  Skipping scoring: feed unreachable.")
        return

    # Build fixture lookup by match_id
    fx_by_id = {int(f["match_id"]): f for f in fixtures}

    # Load pick_log (snapshot-based picks)
    logged = _load_pick_log()  # {(match_id, member): pick}

    # For static members (davo/maldini/experts): retro-score ALL finished matches
    # even if no snapshot exists — derive pick on the fly.
    retro_members = {"davo", "maldini", "experts"}

    # Accumulate per-member stats
    stats: dict[str, dict] = {}

    def _ensure(member):
        if member not in stats:
            stats[member] = {
                "matched": 0, "exact": 0, "gd": 0, "winner": 0, "total": 0
            }

    for match_id, (home_score, away_score) in sorted(results.items()):
        actual = f"{home_score}-{away_score}"
        f = fx_by_id.get(match_id)
        if f is None:
            continue
        pts = stage_points(f["stage"], cfg)
        raw_picks = picks_for(experts_data, f["home"], f["away"])

        # Determine retro picks for static members (fall back to on-the-fly)
        retro_picks = {}
        if raw_picks is not None:
            retro_picks["davo"] = _davo_pick(raw_picks, pts)
            retro_picks["maldini"] = _maldini_pick(raw_picks, pts)
            retro_picks["experts"] = _experts_pool1_pick(raw_picks, pts)

        # Collect all member picks for this match
        all_picks: dict[str, str] = {}
        # 1. Logged (snapshot-based)
        for member in ("market", "blend", "davo", "maldini", "experts"):
            if (match_id, member) in logged:
                all_picks[member] = logged[(match_id, member)]
        # 2. Retro for static members if not logged
        for member in retro_members:
            if member not in all_picks and member in retro_picks:
                pick = retro_picks[member]
                if pick is not None:
                    all_picks[member] = pick

        # Score each pick
        for member, pick in all_picks.items():
            _ensure(member)
            earned = score_pick(pick, actual, pts)
            stats[member]["matched"] += 1
            stats[member]["total"] += earned
            if earned == pts["exact"]:
                stats[member]["exact"] += 1
            elif earned == pts["gd"]:
                stats[member]["gd"] += 1
            elif earned == pts["winner"]:
                stats[member]["winner"] += 1

    return stats


# ── output ────────────────────────────────────────────────────────────────────

def print_table(stats: dict):
    """Print a formatted per-member summary table."""
    MEMBER_ORDER = ["market", "blend", "davo", "maldini", "experts"]
    # Include any members not in the canonical order (defensive)
    all_members = MEMBER_ORDER + [m for m in stats if m not in MEMBER_ORDER]

    header = f"{'Member':<12} {'Scored':>6} {'Exact':>6} {'GD':>6} {'Winner':>7} {'Total':>7}"
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for member in all_members:
        if member not in stats:
            continue
        s = stats[member]
        rows.append({
            "member": member,
            "matched": s["matched"],
            "exact_hits": s["exact"],
            "gd_hits": s["gd"],
            "winner_hits": s["winner"],
            "total_points": s["total"],
        })
        print(f"{member:<12} {s['matched']:>6} {s['exact']:>6} {s['gd']:>6} {s['winner']:>7} {s['total']:>7}")

    print()
    return rows


def write_member_scores(rows: list[dict]):
    """Write member scores to data/member_scores.csv."""
    if not rows:
        return
    fields = ["member", "matched", "exact_hits", "gd_hits", "winner_hits", "total_points"]
    with MEMBER_SCORES.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {MEMBER_SCORES}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    fixtures = load_fixtures()
    experts_data = load_experts()

    print("=== Phase 1: Snapshot upcoming picks ===")
    events, cache_age = _load_cache_events()
    snapshot_upcoming(fixtures, cfg, experts_data, events, cache_age)

    print()
    print("=== Phase 2: Score finished matches ===")
    results = _fetch_results()
    if results:
        print(f"  Fetched {len(results)} finished match results from feed.")
    stats = score_backward(fixtures, cfg, experts_data, results)

    if stats:
        rows = print_table(stats)
        write_member_scores(rows)
    else:
        print("  No scored picks yet.")


if __name__ == "__main__":
    main()
