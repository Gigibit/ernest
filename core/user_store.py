"""Persistence helpers for web/API authentication and project tracking."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class ProjectRecord:
    """Representation of a tracked migration project."""

    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    original_filename: str
    target_framework: str
    target_language: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "original_filename": self.original_filename,
            "target_framework": self.target_framework,
            "target_language": self.target_language,
        }
        payload.update(self.metadata)
        return payload


class UserStore:
    """Simple JSON-backed store for users, tokens, and migration projects."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "users": {},
            "passphrases": {},
            "tokens": {},
            "projects": {},
        }
        if self._path.exists():
            self._load()
        else:
            self._save()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as handle:
            self._data = json.load(handle)
        self._data.setdefault("users", {})
        self._data.setdefault("passphrases", {})
        self._data.setdefault("tokens", {})
        self._data.setdefault("projects", {})

    def _save(self) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(self._path)

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def authenticate(self, passphrase: str) -> Dict[str, Any]:
        """Return user metadata for ``passphrase``, creating it when missing."""

        pass_hash = sha256(passphrase.encode("utf-8")).hexdigest()
        with self._lock:
            if pass_hash in self._data["passphrases"]:
                user_id = self._data["passphrases"][pass_hash]["user_id"]
                user_record = self._data["users"][user_id]
                return {
                    "user_id": user_id,
                    "token": user_record["token"],
                    "created": False,
                }

            user_id = uuid4().hex
            token = token_hex(32)
            timestamp = self._now()
            self._data["users"][user_id] = {
                "user_id": user_id,
                "token": token,
                "created_at": timestamp,
            }
            self._data["passphrases"][pass_hash] = {
                "user_id": user_id,
                "created_at": timestamp,
            }
            self._data["tokens"][token] = {
                "user_id": user_id,
                "created_at": timestamp,
            }
            self._data["projects"].setdefault(user_id, [])
            self._save()
            return {"user_id": user_id, "token": token, "created": True}

    def resolve_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        with self._lock:
            record = self._data["tokens"].get(token)
            return record["user_id"] if record else None

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            user = self._data["users"].get(user_id)
            if not user:
                return None
            return dict(user)

    def list_projects(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            projects = self._data["projects"].get(user_id, [])
            return [dict(project) for project in projects]

    def create_project(
        self,
        user_id: str,
        *,
        name: str,
        original_filename: str,
        target_framework: str,
        target_language: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = self._now()
        project = ProjectRecord(
            id=uuid4().hex,
            name=name,
            status="processing",
            created_at=timestamp,
            updated_at=timestamp,
            original_filename=original_filename,
            target_framework=target_framework,
            target_language=target_language,
            metadata=metadata or {},
        )
        with self._lock:
            self._data["projects"].setdefault(user_id, []).append(project.to_dict())
            self._save()
        return project.to_dict()

    def update_project(self, user_id: str, project_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            projects = self._data["projects"].get(user_id)
            if not projects:
                return None
            for project in projects:
                if project["id"] == project_id:
                    project.update(updates)
                    project["updated_at"] = self._now()
                    self._save()
                    return dict(project)
            return None

    def get_project(self, user_id: str, project_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            projects = self._data["projects"].get(user_id)
            if not projects:
                return None
            for project in projects:
                if project["id"] == project_id:
                    return dict(project)
            return None

