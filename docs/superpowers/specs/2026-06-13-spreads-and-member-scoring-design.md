# Spreads constraint, member scoring, visible expert picks — Design

**Date:** 2026-06-13
**Goal served:** sharper scoreline probabilities (spreads pin the goal-difference
tier the pools pay for) and an evidence loop to tune ensemble weights.

## 1. Handicap (spreads) market into the solver

- `data/odds_api.py`: main fetch markets become `h2h,totals,spreads`
  (cost 4 → 6 credits per refresh; per-event extras unchanged).
- `model/markets.py`: parse `spreads` markets — outcomes carry the team name
  and a `point` (home e.g. −1.5, away +1.5). Keep **half-integer lines only**
  (push semantics, same rule as totals). Per book: devig the 2-outcome pair;
  aggregate across books (Pinnacle-weighted, as elsewhere). Constraint shape:
  `"spreads": [(home_line, p_home_covers), ...]` where home covers ⇔
  goal difference (home−away) > −home_line.
- `model/poisson.py`: `prob_home_covers(m, line)` = mass where (i−j) > −line;
  `solve_rates` gains spreads terms in the least-squares loss (same weight as
  other constraints; missing spreads simply absent).
- Orientation: when an odds event is home/away-swapped vs the fixture, spread
  lines flip sign along with the 1X2 swap (home_line_fixture = away_point of
  the event). Handled where the 1X2 swap happens today.
- Tests: sample spreads market parsing (incl. integer-line skip and swapped
  orientation); solver round-trip with a spreads constraint; sample event JSON
  extended with a realistic spreads block.

## 2. Member scoring (evidence for ensemble weights)

A self-contained CLI, no UI: `python scripts/score_members.py`.

- **Snapshot forward**: for each upcoming group match (within 48h), compute
  each member's hypothetical Pool 1 pick from current data — `market`
  (cached odds; skipped with a note if cache older than 24h), `davo`,
  `maldini`, `experts` (50/50), `blend` (current config weights) — and append
  to `data/pick_log.csv`
  (`match_id,member,pick,computed_at`; one row per member; idempotent — a
  match+member already logged is not overwritten, preserving the pre-match
  snapshot).
- **Score backward**: fetch final scores from the fixturedownload feed (the
  fixtures source; free, no Odds-API quota — finished matches carry
  HomeTeamScore/AwayTeamScore). For each finished match, score every logged
  member pick with the real pool rules (stage-aware 5/3/2, 8/5/3, 11/7/5 via
  `model.predict.stage_points`). Experts/davo/maldini are static, so they are
  retro-scored even for matches played before this feature existed; market
  and blend can only be scored from their snapshots forward.
- Output: per-member table (matches scored, exact hits, gd hits, winner hits,
  total points) printed + written to `data/member_scores.csv`. Both CSVs are
  committed (public repo is fine per user).
- This is deliberately not in the Streamlit app — it is a tuning instrument.
  After ~2–3 matchdays the user adjusts `ensemble:` weights in config.yaml
  based on the table.

## 3. Expert picks visible on cards

- `ui.py`: replace the caption mention with a dedicated chip row directly
  under the POOL badges: two outlined chips `DAVO 2-1` / `MALDINI 2-0`
  (13px, dark outline, no fill — visually subordinate to the pool badges).
  Names static, scores escaped. Only rendered when expert picks exist.
- Tests updated accordingly (chips present when picks exist, absent otherwise).

## Error handling

- Spreads absent from a book/match → constraint absent (existing pattern).
- score_members: feed unreachable → scores skipped with message, snapshots
  still appended; missing pick_log → created; never crashes on partial data.
- App behavior unchanged when scripts never run.

## Testing

Everything via pytest as usual; existing 167 tests stay green. New: spreads
parsing/solving (≥5), prob_home_covers properties (complement symmetry on
half lines), score_members scoring table against hand-computed points for the
three tiers across stages, pick_log idempotence, retro-scoring of static
members without snapshots.
