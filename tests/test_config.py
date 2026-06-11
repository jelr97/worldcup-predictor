import pytest
from config import load_config


def test_load_config_defaults():
    cfg = load_config()
    for tier in ("group", "r32_r16", "qf_plus"):
        s = cfg["pool"]["scoring"][tier]
        assert s["exact"] > s["gd"] > s["winner"] > 0
    assert cfg["odds"]["sport_key"]
    assert cfg["knockout_scoring"] == "90min"


def test_non_90min_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("pool: {pts_exact: 3, pts_outcome: 1}\nknockout_scoring: after_et\n")
    with pytest.raises(NotImplementedError):
        load_config(p)
