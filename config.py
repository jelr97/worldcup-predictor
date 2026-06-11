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
