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
