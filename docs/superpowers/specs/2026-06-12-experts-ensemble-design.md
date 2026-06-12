# Experts Ensemble — Design

**Date:** 2026-06-12
**Goal served:** maximize expected pool points in two pools (exact 5/8/11, goal-difference 3/5/7, winner 2/3/5 by stage). The ensemble changes only the probability matrix; the expected-points optimizer remains the decision layer.

## What's being added

Blend two probability sources per match into one scoreline distribution:

1. **market** — the existing pipeline (de-vigged Pinnacle-weighted odds → Dixon-Coles Poisson).
2. **experts** — Davo's and Maldini's exact-score predictions for all 72 group
   matches, from the user's `Poia.xlsx`.

Form was considered and explicitly skipped (the market already prices it; Elo
remains the emergency fallback only).

## Data & import

- Source of truth: the user's Excel (`~/Downloads/Poia.xlsx`, sheet `Sheet1`,
  columns Group | Team_A | Team_B | Davo(h,a) | Maldini(h,a) | Promedio(h,a)).
  Team names in Spanish. The **Promedio column is ignored** — it is
  inconsistently rounded in the source (e.g., Bélgica–Egipto lists 1-0 as the
  "average" of 3-1 and 0-1); proper averaging happens in the model.
- `scripts/import_experts.py [xlsx-path]` (re-runnable; default
  `~/Downloads/Poia.xlsx`): validates 72 rows, 12 groups × 6, integer scores
  0–15, and that every team name resolves via `team_names.normalize()` to a
  group-stage fixture team — failing loudly with the list of unmatched names —
  then writes canonical `data/experts.csv`
  (`group,team_a,team_b,davo_a,davo_b,maldini_a,maldini_b`, UTF-8).
- ~40 Spanish aliases added to `data/team_names.ALIASES` (post-accent-strip
  keys: "alemania"→"germany", "paises bajos"→"netherlands", "chequia"→"czech
  republic", "curazao"→"curacao", "rd congo"→"dr congo", …).
- `data/experts.py`: `load_experts()` → `{}` when the CSV is absent;
  `picks_for(experts, home, away)` resolves a fixture in either orientation
  (swapping the score tuples when the row is reversed).
- Updates flow: user edits Excel → re-run importer → commit/push →
  Streamlit Cloud redeploys. Knockout picks can be added later the same way.
- `openpyxl>=3.1` joins requirements.txt (importer only; the app reads CSV).
- `predict_upcoming(...)` gains an `experts` parameter; `app.py` calls
  `load_experts()` once and passes it through.

## Model

- `model/experts.py`:
  - `pick_to_rates((a, b))` → `(min(a + 0.4, 6.0), min(b + 0.4, 6.0))` —
    the +0.4 makes the predicted score the modal scoreline of its Poisson
    while keeping realistic mass on neighbors; the cap tames 7-0 style picks.
  - `experts_matrix(picks)` = 0.5 · DC-matrix(Davo) + 0.5 · DC-matrix(Maldini).
    Averaging distributions (not scores) preserves disagreement: Davo 2-0 +
    Maldini 0-2 keeps both scorelines likelier than 1-1.
- `model/predict.py`: per match, collect available members —
  market matrix (existing path, including home/away swap of 1X2) and experts
  matrix. Final matrix = Σ wᵢ·Mᵢ / Σ wᵢ with weights from `config.yaml`:

  ```yaml
  ensemble:
    market: 1.0
    experts: 1.0
  ```

  Missing members drop out and weights renormalize (knockouts → market-only;
  no odds → experts-only; neither → Elo fallback, unchanged; nothing → none).
- Displayed probabilities AND picks both derive from the **final blended
  matrix** (always consistent). `MatchPrediction` gains `members: list[str]`
  and `expert_picks: dict | None` (e.g. `{"davo": "2-1", "maldini": "2-0"}`);
  `source` becomes the joined member label ("market+experts", "market",
  "experts", "elo", "none").
- The Elo-disagrees stale-odds flag keeps comparing Elo vs the **market**
  member's 1X2 (its purpose is detecting stale odds, not judging the blend).

## UI

One added caption segment per card when experts are present:
`Davo 2-1 · Maldini 2-0` (names escaped), plus the members label.
No other layout changes.

## Robustness & error handling

- Importer: loud validation failures with named teams; never writes a partial CSV.
- App: missing/corrupt experts.csv → experts member silently absent
  (market-only), never a crash.
- Per-match: unmatched fixture → no experts member for that match only.
- Equal weights are the user's explicit starting choice; config-tunable.
  Phase 2's results tracker can later score each member's hypothetical pool
  points to inform reweighting.

## Testing

- Importer: valid file round-trip; bad team name → exit with name listed;
  wrong row count → failure.
- Loader: orientation swap returns swapped score tuples; missing file → {}.
- `pick_to_rates`: predicted score is the modal cell of its matrix; cap holds.
- Mixture: Davo 2-0 + Maldini 0-2 → P(2-0) and P(0-2) each > P(1-1).
- Ensemble: weights renormalize when a member is missing; market-only equals
  pre-ensemble behavior when experts absent.
- Full coverage: all 72 expert rows match all 72 group fixtures (either
  orientation); count of swapped rows reported.
- End-to-end: blended prediction yields valid Pool 1 ≠ Pool 2 picks; existing
  116 tests stay green (updated where signatures changed).
