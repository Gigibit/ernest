"""Visitor statistics tracking for the Ernest web UI."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class StatsStore:
    """JSON-backed store that aggregates visitor statistics by IP address."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"visitors": {}}
        if self._path.exists():
            self._load()
        else:
            self._save()

    def _load(self) -> None:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            self._data = {"visitors": {}}
            self._save()
            return
        self._data = {
            "visitors": payload.get("visitors", {}),
        }

    def _save(self) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(self._path)

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def record(self, ip_address: str | None, path: str, logged_in: bool) -> None:
        """Record a visit for ``ip_address`` and update aggregate metrics."""

        sanitized_ip = (ip_address or "unknown").strip() or "unknown"
        timestamp = self._now()
        with self._lock:
            visitors: Dict[str, Dict[str, Any]] = self._data.setdefault("visitors", {})
            entry = visitors.get(sanitized_ip)
            if not entry:
                entry = {
                    "ip": sanitized_ip,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "hits": 1,
                    "last_path": path,
                    "logged_in": bool(logged_in),
                    "last_logged_in_at": timestamp if logged_in else None,
                }
            else:
                entry.setdefault("logged_in", False)
                entry.setdefault("last_logged_in_at", None)
                entry["last_seen"] = timestamp
                entry["hits"] = int(entry.get("hits", 0) or 0) + 1
                entry["last_path"] = path
                if logged_in:
                    entry["logged_in"] = True
                    entry["last_logged_in_at"] = timestamp
            visitors[sanitized_ip] = entry
            self._save()

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return the visitor statistics ordered by most recent activity."""

        with self._lock:
            entries = [dict(item) for item in self._data.get("visitors", {}).values()]
        entries.sort(key=lambda item: item.get("last_seen") or "", reverse=True)
        return entries
