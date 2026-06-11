"""Tests for data.flags — every group-stage team must have a flag."""
import json
from pathlib import Path

import pytest

from data.flags import flag

FIXTURES_PATH = Path(__file__).parent.parent / "data" / "fixtures.json"


def _group_stage_teams():
    """Collect distinct real team names (group-stage only, no placeholders)."""
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        fixtures = json.load(f)
    teams = set()
    for m in fixtures:
        for key in ("home", "away"):
            name = m[key]
            # Skip knockout placeholders: start with a digit, or are "To be announced"
            if name[0].isdigit() or name.lower().startswith("to be"):
                continue
            teams.add(name)
    return teams


@pytest.mark.parametrize("team", sorted(_group_stage_teams()))
def test_group_stage_team_has_flag(team):
    assert flag(team) != "", f"No flag for team: {team!r}"


def test_placeholder_returns_empty():
    assert flag("To be announced") == ""


def test_knockout_code_returns_empty():
    assert flag("1A") == ""
    assert flag("2B") == ""
