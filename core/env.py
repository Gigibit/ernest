"""Utility helpers for loading environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> Dict[str, str]:
    """Load ``KEY=VALUE`` pairs from ``path`` into :mod:`os.environ`.

    Lines starting with ``#`` or empty lines are ignored. Values wrapped in single
    or double quotes are unwrapped. If ``override`` is ``False`` (the default),
    existing environment variables are left untouched.
    """

    env_override = os.environ.get("ENV_FILE") or os.environ.get("ERNEST_ENV_FILE")
    env_path = Path(env_override or path)
    if not env_path.exists():
        return {}

    loaded: Dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    token_value = next((loaded[key] for key in HF_TOKEN_KEYS if key in loaded), None)
    if token_value:
        for key in HF_TOKEN_KEYS:
            if override or key not in os.environ:
                os.environ[key] = token_value
            loaded.setdefault(key, token_value)

    return loaded
