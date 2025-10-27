# migration/recovery_manager.py
import json
from pathlib import Path

class RecoveryManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {"completed": [], "failed": [], "skipped": []}
        if self.state_path.exists():
            self.state.update(json.loads(self.state_path.read_text()))

    def mark_completed(self, f: str):
        self.state["completed"].append(f); self._save()

    def mark_failed(self, f: str):
        self.state["failed"].append(f); self._save()

    def mark_skipped(self, f: str):
        self.state["skipped"].append(f); self._save()

    def _save(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))
