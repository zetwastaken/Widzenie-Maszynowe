from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
  if getattr(sys, "frozen", False):
    return Path(sys._MEIPASS)
  return Path(__file__).resolve().parent


def asset_path(name: str) -> Path:
  return project_root() / name
