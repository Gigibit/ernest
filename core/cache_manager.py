# core/cache_manager.py
import sqlite3
import json
from pathlib import Path

class CacheManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS llm_cache (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    def get(self, key: str):
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM llm_cache WHERE k=?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value):
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO llm_cache (k, v) VALUES (?, ?)", (key, json.dumps(value)))
        self.conn.commit()

    def close(self):
        self.conn.close()
