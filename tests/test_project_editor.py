import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from christophe import create_app
from core.stats_store import StatsStore
from core.user_store import UserStore


@pytest.fixture()
def app(tmp_path, monkeypatch):
    user_store_path = tmp_path / "users.json"
    stats_store_path = tmp_path / "stats.json"
    output_root = tmp_path / "output"
    user_store = UserStore(user_store_path)
    stats_store = StatsStore(stats_store_path)
    app = create_app(output_root=output_root, user_store=user_store, stats_store=stats_store)
    app.config["TESTING"] = True
    yield app


@pytest.fixture()
def sample_project(app, tmp_path):
    user_store: UserStore = app.config["ERNEST_USER_STORE"]
    auth = user_store.authenticate("demo-passphrase")
    user_id = auth["user_id"]
    token = auth["token"]

    project_root = tmp_path / "generated"
    project_root.mkdir(parents=True)
    (project_root / "hello.txt").write_text("hello world\n", encoding="utf-8")

    archive_root = tmp_path / "archives"
    archive_root.mkdir(parents=True)
    archive_path = shutil.make_archive(str(archive_root / "bundle"), "zip", root_dir=project_root)

    metadata = {
        "completed_at": "2024-01-01T00:00:00Z",
        "archive_path": archive_path,
        "download_name": Path(archive_path).name,
        "output_path": str(project_root),
        "error": None,
    }
    project = user_store.create_project(
        user_id,
        name="Demo",
        original_filename="demo.zip",
        target_framework="spring",
        target_language="java",
        status="completed",
        metadata=metadata,
    )
    return {"user_id": user_id, "token": token, "project": project}


def login(client, user_id):
    with client.session_transaction() as session:
        session["user_id"] = user_id


def test_project_editor_page_renders(app, sample_project):
    client = app.test_client()
    login(client, sample_project["user_id"])

    project_id = sample_project["project"]["id"]
    response = client.get(f"/projects/{project_id}/editor")

    assert response.status_code == 200
    assert b"live editor" in response.data


def test_project_editor_manifest_lists_files(app, sample_project):
    client = app.test_client()
    login(client, sample_project["user_id"])

    project_id = sample_project["project"]["id"]
    response = client.get(f"/projects/{project_id}/editor/manifest")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert any(item["path"] == "hello.txt" for item in payload.get("files", []))


def test_project_editor_fetches_file_content(app, sample_project):
    client = app.test_client()
    login(client, sample_project["user_id"])

    project_id = sample_project["project"]["id"]
    response = client.get(f"/projects/{project_id}/editor/files/hello.txt")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["path"] == "hello.txt"
    assert "hello world" in payload["content"]


def test_api_projects_includes_editor_url(app, sample_project):
    client = app.test_client()
    token = sample_project["token"]

    response = client.get(
        "/api/projects",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    project = payload["projects"][0]
    assert "editor_url" in project
