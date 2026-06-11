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
