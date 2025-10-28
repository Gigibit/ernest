# christophe.py
"""CLI and web front-end for the migration orchestrator."""
"""
o gioia, ch'io conobbi, esser amato amando!
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from analysis.dependency_agent import DependencyAnalysisAgent
from analysis.heuristic_agent import HeuristicAnalysisAgent
from analysis.semantic_graph import SemanticGraphBuilder
from core.cache_manager import CacheManager
from core.cost_model import estimate_h100_receipt
from core.file_utils import secure_unzip
from core.llm_service import LLMService
from core.user_store import UserStore
from migration.dependency_resolver import DependencyResolver
from migration.recovery_manager import RecoveryManager
from migration.resource_migrator import ResourceMigrator
from migration.source_migrator import SourceMigrator
from planning.planning_agent import PlanningAgent
from scaffolding.scaffolding_agent import ScaffoldingAgent


DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "classify": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 512, "temp": 0.1},
    "analyze": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 512, "temp": 0.1},
    "translate": {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "max": 4096, "temp": 0.0},
    "adapt": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1024, "temp": 0.0},
    "scaffold": {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "max": 2048, "temp": 0.1},
    "dependency": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1536, "temp": 0.0},
}


def print_section(title: str, content: str) -> None:
    line = "-" * (len(title) + 8)
    print(f"\n--- {title.upper()} ---\n{content}\n{line}")


def build_services(profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> tuple[LLMService, CacheManager]:
    """Initialise the shared LLM and cache services."""

    llm = LLMService(profiles or DEFAULT_PROFILES)
    cache = CacheManager(Path(".cache/migration_cache.db"))
    return llm, cache


def run_migration(
    zip_path: Path,
    target_framework: str,
    target_lang: str = "java",
    *,
    src_lang: Optional[str] = None,
    src_framework: Optional[str] = None,
    reuse_cache: bool = False,
    output_root: Optional[Path] = None,
    llm: Optional[LLMService] = None,
    cache: Optional[CacheManager] = None,
    page_size: Optional[int] = None,
    refine_passes: int = 0,
    safe_mode: bool = True,
) -> Dict[str, Any]:
    """Execute the full migration pipeline for ``zip_path``.

    When ``page_size`` is provided the translation of each source artefact is
    paginated to avoid overloading the model context window.  ``refine_passes``
    controls how many iterative polishing rounds are executed for every page.
    """

    output_root = output_root or Path("output_project")
    llm_service = llm or LLMService(DEFAULT_PROFILES)
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))
    close_cache = cache is None

    if hasattr(llm_service, "reset_usage"):
        llm_service.reset_usage()

    logging.info("Starting migration for %%s", zip_path)

    with tempfile.TemporaryDirectory(prefix="christophe_") as tmp:
        temp_dir = Path(tmp)
        secure_unzip(zip_path, temp_dir)

        heuristic = HeuristicAnalysisAgent(str(temp_dir), llm_service, cache_manager)
        classification = heuristic.classify_files()

        detected_stack = (
            {"language": src_lang, "framework": src_framework}
            if src_lang or src_framework
            else heuristic.detect_stack(classification.get("source"))
        )
        if not detected_stack:
            raise RuntimeError("Impossibile determinare lo stack sorgente.")

        graph = SemanticGraphBuilder()
        graph.add_nodes(classification.get("source", []))
        plan = PlanningAgent().create_plan({}, classification.get("source", []))

        scaffold_agent = ScaffoldingAgent(llm_service, cache_manager)
        project_name = Path(zip_path.stem).name.replace("-", "_") or "migrated_project"
        target_path = scaffold_agent.generate(
            output_root, project_name, target_framework, target_lang
        )

        dependency_agent = DependencyAnalysisAgent(temp_dir, llm_service, cache_manager)
        dependency_snapshot = dependency_agent.extract_dependencies(
            target_language=target_lang,
            target_framework=target_framework,
        )

        recovery_path = target_path / "migration_state.json"
        recovery = RecoveryManager(recovery_path)
        src_migrator = SourceMigrator(llm_service, cache_manager, recovery)
        res_migrator = ResourceMigrator(llm_service, cache_manager)
        dependency_resolver = DependencyResolver(
            llm_service,
            cache_manager,
            download_root=target_path / "third_party",
        )
        dependency_resolution = dependency_resolver.resolve(
            dependency_snapshot,
            target_language=target_lang,
            target_framework=target_framework,
            perform_downloads=True,
        )

        for src in plan:
            source_file = temp_dir / src
            if not source_file.exists():
                recovery.mark_skipped(src)
                continue

            destination = target_path / "src" / Path(src).with_suffix(".java").name
            src_migrator.translate_legacy_backend(
                source_file,
                destination,
                page_size=page_size,
                refine_passes=refine_passes,
                safe_mode=safe_mode,
            )

        for res in classification.get("resource", []):
            resource_path = temp_dir / res
            if resource_path.exists():
                res_migrator.process(resource_path, target_path / "resources")

    token_usage = (
        llm_service.get_usage_summary()
        if hasattr(llm_service, "get_usage_summary")
        else {}
    )
    cost_estimate = estimate_h100_receipt(token_usage) if token_usage else {}

    if close_cache:
        cache_manager.close()

    logging.info("Migration completed for %%s", zip_path)

    return {
        "classification": classification,
        "detected_stack": detected_stack,
        "plan": plan,
        "target_path": target_path,
        "output_root": output_root,
        "project_name": project_name,
        "recovery_path": recovery_path,
        "dependencies": dependency_snapshot,
        "dependency_resolution": dependency_resolution,
        "token_usage": token_usage,
        "cost_estimate": cost_estimate,
        "safe_mode": safe_mode,
    }


def create_app(
    llm: Optional[LLMService] = None,
    cache: Optional[CacheManager] = None,
    *,
    output_root: Optional[Path] = None,
) -> "Flask":
    """Create a Flask application exposing the migration pipeline with auth."""

    from flask import (
        Flask,
        abort,
        g,
        redirect,
        render_template_string,
        request,
        send_file,
        session,
        url_for,
    )
    from werkzeug.utils import secure_filename

    app = Flask(__name__)
    app.secret_key = os.environ.get("CHRISTOPHE_WEB_SECRET", "christophe-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

    llm_service = llm or LLMService(DEFAULT_PROFILES)
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))

    base_output_root = output_root or Path("output_project")
    web_output_root = base_output_root / "web"
    api_output_root = base_output_root / "api"
    web_output_root.mkdir(parents=True, exist_ok=True)
    api_output_root.mkdir(parents=True, exist_ok=True)

    store_path = Path(os.environ.get("CHRISTOPHE_USER_STORE", ".cache/users.json"))
    user_store = UserStore(store_path)

    def _timestamp() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _extract_token(req: Any) -> Optional[str]:
        header = req.headers.get("Authorization", "") if hasattr(req, "headers") else ""
        if header.lower().startswith("bearer "):
            return header.split(" ", 1)[1].strip()
        return (
            req.headers.get("X-Auth-Token")
            if hasattr(req, "headers") and req.headers.get("X-Auth-Token")
            else None
        ) or req.values.get("auth_token") or req.values.get("token") or req.values.get("access_token")

    def _finalise_success(user_id: str, project_id: str, migration: Dict[str, Any]) -> Dict[str, Any]:
        archive_file = shutil.make_archive(
            str(migration["target_path"]),
            "zip",
            root_dir=migration["target_path"],
        )
        archive_path = Path(archive_file)
        metadata = {
            "status": "completed",
            "completed_at": _timestamp(),
            "output_path": str(migration["target_path"]),
            "archive_path": str(archive_path),
            "download_name": archive_path.name,
            "token_usage": migration.get("token_usage", {}),
            "cost_estimate": migration.get("cost_estimate", {}),
            "safe_mode": migration.get("safe_mode"),
            "project_name": migration.get("project_name"),
            "detected_stack": migration.get("detected_stack"),
            "plan": migration.get("plan"),
        }
        return user_store.update_project(user_id, project_id, **metadata) or metadata

    def _finalise_failure(user_id: str, project_id: str, message: str) -> None:
        user_store.update_project(
            user_id,
            project_id,
            status="failed",
            error=message,
        )

    def _group_projects(user_id: str) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        pending: list[Dict[str, Any]] = []
        completed: list[Dict[str, Any]] = []
        for project in user_store.list_projects(user_id):
            entry = dict(project)
            if entry.get("status") == "completed" and entry.get("archive_path"):
                entry["download_url"] = url_for("download_project", project_id=entry["id"])
                completed.append(entry)
            else:
                pending.append(entry)
        pending.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        completed.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return pending, completed

    PAGE_TEMPLATE = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Christophe Migration Portal</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6fb; margin: 0; padding: 0; }
            .container { max-width: 1080px; margin: 40px auto; background: #fff; padding: 32px; border-radius: 16px; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.12); }
            header.top { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }
            header.top h1 { margin: 0; color: #0f172a; }
            header.top .subtitle { margin: 6px 0 0; color: #475569; }
            .account { text-align: right; color: #1e293b; }
            .account code { background: rgba(99, 102, 241, 0.1); padding: 4px 8px; border-radius: 6px; display: inline-block; }
            .logout { display: inline-block; margin-top: 8px; color: #ef4444; text-decoration: none; font-weight: 600; }
            form { margin-top: 32px; display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; align-items: flex-start; }
            label { display: block; font-weight: 600; margin-bottom: 6px; color: #334155; }
            input[type="text"], input[type="number"] { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #cbd5f5; font-size: 15px; }
            .drop-zone { grid-column: 1 / -1; border: 2px dashed #6366f1; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; background: rgba(99, 102, 241, 0.05); transition: background 0.2s, border-color 0.2s; }
            .drop-zone.dragover { background: rgba(99, 102, 241, 0.15); border-color: #4f46e5; }
            .drop-zone p { margin: 0; color: #4338ca; font-weight: 600; }
            button { grid-column: 1 / -1; background: linear-gradient(135deg, #4f46e5, #6366f1); color: white; border: none; border-radius: 12px; padding: 14px 18px; font-size: 16px; font-weight: 600; cursor: pointer; box-shadow: 0 14px 35px rgba(79, 70, 229, 0.3); transition: transform 0.15s ease, box-shadow 0.2s ease; }
            button:hover { transform: translateY(-1px); box-shadow: 0 18px 40px rgba(79, 70, 229, 0.35); }
            .messages { grid-column: 1 / -1; }
            .error { background: #fee2e2; color: #b91c1c; padding: 12px 16px; border-radius: 10px; margin-bottom: 12px; }
            .projects-section { margin-top: 36px; }
            .projects-section h2 { margin-bottom: 12px; color: #312e81; }
            .projects-list { display: grid; gap: 16px; }
            .project-card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; background: #f8fafc; }
            .project-card.completed { background: #eef2ff; border-color: #c7d2fe; }
            .project-card.failed { background: #fef2f2; border-color: #fecaca; }
            .project-card .title { font-weight: 600; color: #1e1b4b; }
            .project-card .meta { margin-top: 6px; font-size: 13px; color: #475569; }
            .project-card .error-text { margin-top: 10px; color: #b91c1c; font-weight: 600; }
            .project-card .download { display: inline-block; margin-top: 12px; padding: 8px 14px; border-radius: 8px; background: #4f46e5; color: #fff; text-decoration: none; font-weight: 600; }
            .project-card .download:hover { background: #4338ca; }
            .toggle { grid-column: 1 / -1; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 18px; }
            .checkbox-label { display: flex; align-items: center; gap: 12px; font-weight: 600; color: #1e293b; }
            .checkbox-label input { width: 20px; height: 20px; }
            .hint { margin-top: 8px; font-size: 13px; color: #475569; }
            .empty { color: #64748b; font-style: italic; }
            .result { background: #eef2ff; color: #1e1b4b; padding: 20px 24px; border-radius: 12px; margin-top: 32px; }
            .result pre { background: rgba(15, 23, 42, 0.85); color: #e2e8f0; padding: 16px; border-radius: 10px; overflow-x: auto; }
            footer { margin-top: 48px; text-align: center; color: #64748b; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <header class="top">
                <div>
                    <h1>Christophe Migration Portal</h1>
                    <p class="subtitle">Upload a ZIP archive and orchestrate safe, paginated migrations.</p>
                </div>
                <div class="account">
                    <div>Workspace <code>{{ user_id }}</code></div>
                    <div>API token</div>
                    <code>{{ user_token }}</code>
                    <div><a class="logout" href="{{ url_for('logout') }}">Log out</a></div>
                </div>
            </header>
            <form method="post" enctype="multipart/form-data">
                <div>
                    <label for="target_framework">Target framework</label>
                    <input id="target_framework" name="target_framework" type="text" value="{{ defaults.target_framework }}" placeholder="e.g. modern service platform">
                </div>
                <div>
                    <label for="target_lang">Target language</label>
                    <input id="target_lang" name="target_lang" type="text" value="{{ defaults.target_lang }}" placeholder="e.g. Java">
                </div>
                <div>
                    <label for="src_lang">Source language (optional)</label>
                    <input id="src_lang" name="src_lang" type="text" value="{{ defaults.src_lang }}" placeholder="e.g. legacy batch language">
                </div>
                <div>
                    <label for="src_framework">Source framework (optional)</label>
                    <input id="src_framework" name="src_framework" type="text" value="{{ defaults.src_framework }}" placeholder="e.g. on-prem enterprise platform">
                </div>
                <div>
                    <label for="page_size">Pagination size (chunks per pass)</label>
                    <input id="page_size" name="page_size" type="number" min="0" step="1" value="{{ defaults.page_size }}" placeholder="e.g. 5">
                </div>
                <div>
                    <label for="refine_passes">Refinement passes per page</label>
                    <input id="refine_passes" name="refine_passes" type="number" min="0" step="1" value="{{ defaults.refine_passes }}" placeholder="e.g. 1">
                </div>
                <div class="toggle">
                    <label class="checkbox-label" for="safe_mode">
                        <input id="safe_mode" name="safe_mode" type="checkbox" value="1" {% if defaults.safe_mode %}checked{% endif %}>
                        Safe mode (fallback guardrails)
                    </label>
                    <div class="hint">Automatically retries risky chunks with strict prompts to avoid conflicts and incomplete files.</div>
                </div>
                <div class="drop-zone" id="drop-zone">
                    <p id="drop-zone-text">Drag &amp; drop your source ZIP or click to browse</p>
                    <input id="source_zip" name="source_zip" type="file" accept=".zip" hidden required>
                </div>
                <div class="messages">
                {% if errors %}
                    <div class="error">
                        <ul>
                        {% for err in errors %}
                            <li>{{ err }}</li>
                        {% endfor %}
                        </ul>
                    </div>
                {% endif %}
                </div>
                <button type="submit">Run migration</button>
            </form>
            <section class="projects-section">
                <h2>Active uploads</h2>
                {% if pending_projects %}
                <div class="projects-list">
                    {% for project in pending_projects %}
                    <div class="project-card {% if project.status == 'failed' %}failed{% endif %}">
                        <div class="title">{{ project.name }}</div>
                        <div class="meta">Status: {{ project.status }} • Updated: {{ project.updated_at }}</div>
                        <div class="meta">Original: {{ project.original_filename }}</div>
                        {% if project.error %}
                        <div class="error-text">{{ project.error }}</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <p class="empty">No active migrations.</p>
                {% endif %}
            </section>
            <section class="projects-section">
                <h2>Completed migrations</h2>
                {% if completed_projects %}
                <div class="projects-list">
                    {% for project in completed_projects %}
                    <div class="project-card completed">
                        <div class="title">{{ project.name }}</div>
                        <div class="meta">Finished: {{ project.completed_at or project.updated_at }}</div>
                        <div class="meta">Output: {{ project.download_name }}</div>
                        {% if project.cost_estimate %}
                        <pre class="meta">{{ project.cost_estimate | tojson(indent=2) }}</pre>
                        {% endif %}
                        <a class="download" href="{{ project.download_url }}">Download archive</a>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <p class="empty">No completed migrations yet.</p>
                {% endif %}
            </section>
            {% if result %}
            <div class="result">
                <h2>Latest migration</h2>
                <p><strong>Project:</strong> {{ result.project_name }}</p>
                <p><strong>Output directory:</strong> {{ result.output_path }}</p>
                {% if result.download_url %}<p><strong>Archive:</strong> <a href="{{ result.download_url }}">Download</a></p>{% endif %}
                <p><strong>Detected stack:</strong></p>
                <pre>{{ result.detected_stack | tojson(indent=2) }}</pre>
                <p><strong>Safe mode:</strong> {{ 'enabled' if result.safe_mode else 'disabled' }}</p>
                <p><strong>Migration plan:</strong></p>
                <ul>
                {% for step in result.plan %}
                    <li>{{ step }}</li>
                {% endfor %}
                </ul>
                <p><strong>Classification summary:</strong></p>
                <pre>{{ result.classification | tojson(indent=2) }}</pre>
                {% if result.dependencies %}
                <p><strong>Dependency manifests:</strong></p>
                <pre>{{ result.dependencies | tojson(indent=2) }}</pre>
                {% endif %}
                {% if result.dependency_resolution %}
                <p><strong>Dependency handling:</strong></p>
                <pre>{{ result.dependency_resolution | tojson(indent=2) }}</pre>
                {% endif %}
                {% if result.token_usage %}
                <p><strong>Token usage:</strong></p>
                <pre>{{ result.token_usage | tojson(indent=2) }}</pre>
                {% endif %}
                {% if result.cost_estimate %}
                <p><strong>H100 cost estimate:</strong></p>
                <pre>{{ result.cost_estimate | tojson(indent=2) }}</pre>
                {% endif %}
            </div>
            {% endif %}
            <footer>
                Powered by Christophe — upload, track, and download your migrated projects securely.
            </footer>
        </div>
        <script>
            const dropZone = document.getElementById('drop-zone');
            const dropZoneText = document.getElementById('drop-zone-text');
            const fileInput = document.getElementById('source_zip');

            if (dropZone) {
                dropZone.addEventListener('click', () => fileInput.click());

                fileInput.addEventListener('change', () => {
                    if (fileInput.files.length > 0) {
                        dropZoneText.textContent = fileInput.files[0].name;
                    }
                });

                ['dragenter', 'dragover'].forEach(eventName => {
                    dropZone.addEventListener(eventName, (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        dropZone.classList.add('dragover');
                    });
                });

                ['dragleave', 'dragend'].forEach(eventName => {
                    dropZone.addEventListener(eventName, (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        dropZone.classList.remove('dragover');
                    });
                });

                dropZone.addEventListener('drop', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    dropZone.classList.remove('dragover');
                    const files = event.dataTransfer.files;
                    if (files.length > 0) {
                        const file = files[0];
                        if (!file.name.toLowerCase().endsWith('.zip')) {
                            alert('Please drop a .zip archive');
                            return;
                        }
                        fileInput.files = files;
                        dropZoneText.textContent = file.name;
                    }
                });
            }
        </script>
    </body>
    </html>
    """

    LOGIN_TEMPLATE = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Authenticate - Christophe</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
            .login-container { background: #fff; padding: 36px 40px; border-radius: 16px; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.25); max-width: 420px; width: 100%; }
            h1 { margin-top: 0; color: #1e1b4b; }
            p { color: #475569; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #334155; }
            input[type="password"] { width: 100%; padding: 12px 14px; border-radius: 10px; border: 1px solid #cbd5f5; font-size: 16px; }
            button { margin-top: 18px; width: 100%; padding: 14px; border: none; border-radius: 12px; background: linear-gradient(135deg, #4f46e5, #6366f1); color: #fff; font-size: 16px; font-weight: 600; cursor: pointer; }
            .error { background: #fee2e2; color: #b91c1c; padding: 12px 16px; border-radius: 10px; margin-bottom: 12px; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h1>Christophe Portal</h1>
            <p>Use your passphrase to create or access your personal migration workspace.</p>
            {% if errors %}
            <div class="error">
                <ul>
                {% for err in errors %}
                    <li>{{ err }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            <form method="post">
                <label for="passphrase">Passphrase</label>
                <input id="passphrase" name="passphrase" type="password" required autocomplete="current-password">
                <button type="submit">Continue</button>
            </form>
        </div>
    </body>
    </html>
    """

    @app.before_request
    def load_user() -> None:
        user_id = session.get("user_id")
        g.user = None
        g.auth_token = None
        if not user_id:
            return
        user = user_store.get_user(user_id)
        if not user:
            session.clear()
            return
        g.user = user
        g.auth_token = user.get("token")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("auth"))

    @app.route("/auth", methods=["GET", "POST"])
    @app.route("/login", methods=["GET", "POST"])
    def auth():
        errors: list[str] = []
        if request.method == "POST":
            passphrase = (request.form.get("passphrase") or "").strip()
            if not passphrase:
                errors.append("Passphrase is required.")
            else:
                auth_result = user_store.authenticate(passphrase)
                session["user_id"] = auth_result["user_id"]
                session["token"] = auth_result["token"]
                return redirect(url_for("index"))
        return render_template_string(LOGIN_TEMPLATE, errors=errors)

    @app.route("/", methods=["GET", "POST"])
    def index() -> Any:
        if not g.get("user"):
            return redirect(url_for("auth"))

        errors: list[str] = []
        result_payload: Optional[Dict[str, Any]] = None

        safe_mode_flag = request.form.get("safe_mode")
        safe_mode_enabled = True if request.method != "POST" else bool(safe_mode_flag)

        defaults = {
            "target_framework": request.form.get("target_framework", ""),
            "target_lang": request.form.get("target_lang", "java"),
            "src_lang": request.form.get("src_lang", ""),
            "src_framework": request.form.get("src_framework", ""),
            "page_size": request.form.get("page_size", ""),
            "refine_passes": request.form.get("refine_passes", "0"),
            "safe_mode": safe_mode_enabled,
        }

        user_id = session["user_id"]

        if request.method == "POST":
            target_framework = defaults["target_framework"].strip()
            target_lang = defaults["target_lang"].strip() or "java"
            src_lang = defaults["src_lang"].strip() or None
            src_framework = defaults["src_framework"].strip() or None
            page_size_value = defaults["page_size"].strip()
            refine_passes_value = defaults["refine_passes"].strip()
            page_size_int: Optional[int] = None
            refine_passes_int = 0
            uploaded = request.files.get("source_zip")

            if not target_framework:
                errors.append("Target framework is required.")
            if not uploaded or uploaded.filename == "":
                errors.append("You must upload a ZIP archive.")
            elif not uploaded.filename.lower().endswith(".zip"):
                errors.append("Uploaded file must have .zip extension.")

            if page_size_value:
                try:
                    page_size_int = int(page_size_value)
                    if page_size_int < 0:
                        raise ValueError
                    if page_size_int == 0:
                        page_size_int = None
                except ValueError:
                    errors.append("Pagination size must be a non-negative integer.")
            if refine_passes_value:
                try:
                    refine_passes_int = int(refine_passes_value)
                    if refine_passes_int < 0:
                        raise ValueError
                except ValueError:
                    errors.append("Refinement passes must be a non-negative integer.")

            project_record: Optional[Dict[str, Any]] = None

            if not errors and uploaded:
                filename = secure_filename(uploaded.filename)
                suffix = Path(filename).suffix or ".zip"
                project_record = user_store.create_project(
                    user_id,
                    name=Path(filename).stem,
                    original_filename=uploaded.filename,
                    target_framework=target_framework,
                    target_language=target_lang,
                )
                temp_zip_path: Optional[Path] = None
                try:
                    user_output_root = web_output_root / user_id
                    user_output_root.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        uploaded.save(tmp_file)
                        temp_zip_path = Path(tmp_file.name)
                    migration = run_migration(
                        temp_zip_path,
                        target_framework,
                        target_lang,
                        src_lang=src_lang,
                        src_framework=src_framework,
                        reuse_cache=True,
                        output_root=user_output_root,
                        llm=llm_service,
                        cache=cache_manager,
                        page_size=page_size_int,
                        refine_passes=refine_passes_int,
                        safe_mode=safe_mode_enabled,
                    )
                    final_record = _finalise_success(user_id, project_record["id"], migration)
                    safe_mode_result = migration.get("safe_mode", safe_mode_enabled)
                    result_payload = {
                        "project_id": final_record.get("id"),
                        "project_name": migration["project_name"],
                        "output_path": str(migration["target_path"]),
                        "detected_stack": migration["detected_stack"],
                        "plan": migration["plan"],
                        "classification": migration["classification"],
                        "dependencies": migration.get("dependencies", {}),
                        "dependency_resolution": migration.get("dependency_resolution", {}),
                        "token_usage": migration.get("token_usage", {}),
                        "cost_estimate": migration.get("cost_estimate", {}),
                        "safe_mode": safe_mode_result,
                        "download_url": url_for("download_project", project_id=final_record.get("id")),
                    }
                except Exception as exc:  # noqa: BLE001
                    app.logger.exception("Migration failed")
                    errors.append(str(exc))
                    if project_record is not None:
                        _finalise_failure(user_id, project_record["id"], str(exc))
                finally:
                    if temp_zip_path is not None:
                        temp_zip_path.unlink(missing_ok=True)

        pending_projects, completed_projects = _group_projects(user_id)

        return render_template_string(
            PAGE_TEMPLATE,
            errors=errors,
            result=result_payload,
            defaults=defaults,
            pending_projects=pending_projects,
            completed_projects=completed_projects,
            user_id=user_id,
            user_token=g.get("auth_token", ""),
        )

    @app.route("/projects/<project_id>/download")
    def download_project(project_id: str):
        if not g.get("user"):
            return redirect(url_for("auth"))
        project = user_store.get_project(session["user_id"], project_id)
        if not project or project.get("status") != "completed":
            abort(404)
        archive_path = project.get("archive_path")
        if not archive_path or not Path(archive_path).exists():
            abort(404)
        download_name = project.get("download_name") or Path(archive_path).name
        return send_file(archive_path, as_attachment=True, download_name=download_name)

    @app.route("/api/auth", methods=["POST"])
    def api_auth():
        payload = request.get_json(silent=True) or {}
        passphrase = (
            (payload.get("passphrase") if isinstance(payload, dict) else None)
            or request.form.get("passphrase")
            or request.values.get("passphrase")
        )
        if not passphrase:
            return ({"error": "passphrase is required"}, 400)
        auth_result = user_store.authenticate(passphrase.strip())
        return (
            {
                "user_id": auth_result["user_id"],
                "token": auth_result["token"],
                "created": auth_result["created"],
            },
            200,
        )

    @app.route("/api/projects", methods=["GET"])
    def api_projects():
        token = _extract_token(request)
        user_id = user_store.resolve_token(token)
        if not user_id:
            return ({"error": "invalid or missing token"}, 401)
        projects_payload = []
        for project in user_store.list_projects(user_id):
            entry = dict(project)
            if entry.get("status") == "completed" and entry.get("archive_path"):
                entry["download_url"] = url_for(
                    "api_download_project",
                    project_id=entry["id"],
                    _external=True,
                )
            projects_payload.append(entry)
        return ({"projects": projects_payload}, 200)

    @app.route("/api/projects/<project_id>/download", methods=["GET"])
    def api_download_project(project_id: str):
        token = _extract_token(request)
        user_id = user_store.resolve_token(token)
        if not user_id:
            return ({"error": "invalid or missing token"}, 401)
        project = user_store.get_project(user_id, project_id)
        if not project or project.get("status") != "completed":
            return ({"error": "project not available"}, 404)
        archive_path = project.get("archive_path")
        if not archive_path or not Path(archive_path).exists():
            return ({"error": "archive not found"}, 404)
        download_name = project.get("download_name") or Path(archive_path).name
        return send_file(archive_path, as_attachment=True, download_name=download_name)

    @app.route("/api/migrate", methods=["POST"])
    def api_migrate():
        token = _extract_token(request)
        user_id = user_store.resolve_token(token)
        if not user_id:
            return ({"error": "invalid or missing token"}, 401)

        uploaded = (
            request.files.get("archive")
            or request.files.get("file")
            or request.files.get("source_zip")
        )
        target_framework = (
            request.form.get("target_framework")
            or request.form.get("targetFramework")
            or request.values.get("target_framework")
        )
        target_lang = (
            request.form.get("target_lang")
            or request.form.get("targetLanguage")
            or request.values.get("target_lang")
            or "java"
        )
        src_lang = (
            request.form.get("src_lang")
            or request.form.get("source_language")
            or request.values.get("src_lang")
        )
        src_framework = (
            request.form.get("src_framework")
            or request.values.get("src_framework")
        )
        safe_mode_value = (
            request.form.get("safe_mode")
            or request.values.get("safe_mode")
            or request.args.get("safe_mode")
        )
        safe_mode_enabled = True
        if safe_mode_value is not None:
            safe_mode_enabled = str(safe_mode_value).lower() not in {"0", "false", "off", "no"}

        if uploaded is None or uploaded.filename == "":
            return ({"error": "A ZIP archive must be provided."}, 400)
        if not uploaded.filename.lower().endswith(".zip"):
            return ({"error": "The uploaded file must have a .zip extension."}, 400)
        if not target_framework:
            return ({"error": "target_framework is required."}, 400)

        project_record = user_store.create_project(
            user_id,
            name=Path(uploaded.filename).stem,
            original_filename=uploaded.filename,
            target_framework=target_framework,
            target_language=target_lang,
        )

        suffix = Path(uploaded.filename).suffix or ".zip"
        temp_path: Optional[Path] = None
        try:
            user_output_root = api_output_root / user_id
            user_output_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                uploaded.save(tmp_file)
                temp_path = Path(tmp_file.name)

            migration = run_migration(
                temp_path,
                target_framework,
                target_lang,
                src_lang=src_lang,
                src_framework=src_framework,
                reuse_cache=True,
                output_root=user_output_root,
                llm=llm_service,
                cache=cache_manager,
                safe_mode=safe_mode_enabled,
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("API migration failed")
            _finalise_failure(user_id, project_record["id"], str(exc))
            return ({"error": str(exc)}, 500)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        final_record = _finalise_success(user_id, project_record["id"], migration)
        response = {
            "project_id": final_record.get("id"),
            "project_name": migration["project_name"],
            "output_path": str(migration["target_path"]),
            "detected_stack": migration["detected_stack"],
            "classification": migration["classification"],
            "plan": migration["plan"],
            "dependencies": migration.get("dependencies", {}),
            "dependency_resolution": migration.get("dependency_resolution", {}),
            "token_usage": migration.get("token_usage", {}),
            "cost_estimate": migration.get("cost_estimate", {}),
            "safe_mode": migration.get("safe_mode", safe_mode_enabled),
            "status": final_record.get("status"),
            "download_url": url_for(
                "api_download_project",
                project_id=final_record.get("id"),
                _external=True,
            ) if final_record.get("archive_path") else None,
        }
        return (response, 200)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AI-driven project migration orchestrator.")
    parser.add_argument("zip_path", type=Path, nargs="?", help="Source project ZIP file")
    parser.add_argument("--target-framework", default=None)
    parser.add_argument("--target-lang", default="java")
    parser.add_argument("--src-lang", default=None)
    parser.add_argument("--src-framework", default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Launch the Flask web interface instead of running the CLI workflow.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the web interface when --serve is used.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the web interface when --serve is used.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode when serving the web UI.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="Number of source chunks to translate per LLM call page. Use 0 to disable pagination.",
    )
    parser.add_argument(
        "--refine-passes",
        type=int,
        default=0,
        help="How many refinement passes to run for each translated page.",
    )
    parser.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fallback guardrails to reissue risky chunks with stricter prompts (enabled by default).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.serve:
        if args.zip_path is not None:
            parser.error("zip_path is not compatible with --serve")
        llm, cache = build_services()
        app = create_app(llm=llm, cache=cache)
        try:
            app.run(host=args.host, port=args.port, debug=args.debug)
        finally:
            cache.close()
        return

    if args.zip_path is None:
        parser.error("zip_path is required unless --serve is specified")
    if not args.target_framework:
        parser.error("--target-framework is required when running the CLI workflow")
    if args.page_size is not None and args.page_size < 0:
        parser.error("--page-size must be >= 0")
    if args.refine_passes < 0:
        parser.error("--refine-passes must be >= 0")

    llm, cache = build_services()
    try:
        page_size = args.page_size if args.page_size not in (None, 0) else None
        result = run_migration(
            args.zip_path,
            args.target_framework,
            args.target_lang,
            src_lang=args.src_lang,
            src_framework=args.src_framework,
            reuse_cache=args.reuse_cache,
            llm=llm,
            cache=cache,
            page_size=page_size,
            refine_passes=args.refine_passes,
            safe_mode=args.safe_mode,
        )
    finally:
        cache.close()

    print_section("Classificazione File", json.dumps(result["classification"], indent=2))
    print_section("Stack Rilevato", json.dumps(result["detected_stack"], indent=2))
    print_section("Piano di Migrazione", "\n".join(result["plan"]))
    safe_mode_message = "Abilitato" if result.get("safe_mode", True) else "Disabilitato"
    print_section("Safe Mode", safe_mode_message)
    print_section("Progetto Migrato", f"Output generato in: {result['target_path']}")
    dependencies = result.get("dependencies")
    if dependencies and (
        dependencies.get("manifests") or dependencies.get("dependencies")
    ):
        print_section(
            "Dipendenze Rilevate",
            json.dumps(dependencies, indent=2, ensure_ascii=False),
        )
    resolution = result.get("dependency_resolution")
    if resolution:
        print_section(
            "Gestione Dipendenze",
            json.dumps(resolution, indent=2, ensure_ascii=False),
        )
    if result.get("token_usage"):
        print_section("Consumo Token", json.dumps(result["token_usage"], indent=2))
    if result.get("cost_estimate"):
        print_section("Stima Costi H100", json.dumps(result["cost_estimate"], indent=2))


if __name__ == "__main__":
    main()
