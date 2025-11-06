from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.stats_store import StatsStore


def test_stats_store_records_visits(tmp_path):
    store_path = tmp_path / "stats.json"
    store = StatsStore(store_path)

    store.record("1.2.3.4", "/login", False)
    store.record("1.2.3.4", "/dashboard", True)
    store.record("5.6.7.8", "/", False)

    visitors = store.snapshot()

    by_ip = {entry["ip"]: entry for entry in visitors}
    first = by_ip["1.2.3.4"]
    assert first["hits"] == 2
    assert first["logged_in"] is True
    assert first["last_path"] == "/dashboard"
    assert first["last_logged_in_at"] is not None

    second = by_ip["5.6.7.8"]
    assert second["hits"] == 1
    assert second["logged_in"] is False
    assert second["last_logged_in_at"] is None
