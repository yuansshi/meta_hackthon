"""Settings: config/models.yaml plus keys from server/.env. Everything tunable lives in the YAML."""

from __future__ import annotations

import os
from pathlib import Path

import truststore
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # server/
WEB_DIR = ROOT.parent / "web"
DATA_DIR = Path(os.environ.get("EVERYMUSE_DATA_DIR", ROOT / "data"))
# EveryMuse was called Earshot until Nov 4, 2026: move the data folder kept under the old name, once.
_OLD_DATA_DIR = DATA_DIR.parent / "earshot"
if not DATA_DIR.exists() and _OLD_DATA_DIR.is_dir():
    _OLD_DATA_DIR.rename(DATA_DIR)

truststore.inject_into_ssl()  # corporate networks re-sign TLS; trust the OS keychain like the browser does
load_dotenv(ROOT / ".env")


class MissingKey(RuntimeError):
    pass


def load_config(path: Path | None = None) -> dict:
    cfg = yaml.safe_load((path or ROOT / "config" / "models.yaml").read_text())
    cfg["brain"] = os.environ.get("EVERYMUSE_BRAIN") or cfg["brain"]
    return cfg


def model_cfg(cfg: dict, key: str) -> dict:
    if key not in cfg["models"]:
        raise ValueError(f"unknown model {key!r}; choose from {sorted(cfg['models'])}")
    m = {**cfg["models"][key], "key": key}
    return {**cfg["providers"][m["provider"]], **m}


def api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "")
    if not key:
        raise MissingKey(f"{env_name} is not set: copy server/.env.example to server/.env and paste your key")
    return key
