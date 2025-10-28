from __future__ import annotations
# christophe.py
"""CLI and web front-end for the migration orchestrator."""
"""
o gioia, ch'io conobbi, esser amato amando!
"""


import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

    logging.info("Starting migration for %s", zip_path)

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

    logging.info("Migration completed for %s", zip_path)

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
        render_template,
        request,
        send_file,
        session,
        url_for,
    )
    from werkzeug.utils import secure_filename

    template_dir = Path(__file__).resolve().with_name("templates")
    app = Flask(__name__, template_folder=str(template_dir))
    app.secret_key = os.environ.get("CHRISTOPHE_WEB_SECRET", "christophe-dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

    llm_service = llm or LLMService(DEFAULT_PROFILES)
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))
    executor = ThreadPoolExecutor(
        max_workers=int(os.environ.get("CHRISTOPHE_WORKERS", "2"))
    )

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

    def _schedule_api_migration(
        *,
        user_id: str,
        project_id: str,
        archive_path: Path,
        target_framework: str,
        target_lang: str,
        src_lang: Optional[str],
        src_framework: Optional[str],
        page_size: Optional[int],
        refine_passes: int,
        safe_mode: bool,
    ) -> None:
        user_output_root = api_output_root / user_id / project_id
        user_output_root.mkdir(parents=True, exist_ok=True)

        def _runner() -> None:
            try:
                user_store.update_project(
                    user_id,
                    project_id,
                    status="processing",
                    error=None,
                    started_at=_timestamp(),
                )
                migration = run_migration(
                    archive_path,
                    target_framework,
                    target_lang,
                    src_lang=src_lang,
                    src_framework=src_framework,
                    reuse_cache=True,
                    output_root=user_output_root,
                    llm=llm_service,
                    cache=cache_manager,
                    page_size=page_size,
                    refine_passes=refine_passes,
                    safe_mode=safe_mode,
                )
            except Exception as exc:  # noqa: BLE001
                app.logger.exception("API migration failed for project %s", project_id)
                _finalise_failure(user_id, project_id, str(exc))
            else:
                _finalise_success(user_id, project_id, migration)
            finally:
                archive_path.unlink(missing_ok=True)

        executor.submit(_runner)

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
        return render_template("login.html", errors=errors)

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

        return render_template(
            "dashboard.html",
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

        page_size_raw = (
            request.form.get("page_size")
            or request.values.get("page_size")
            or request.args.get("page_size")
        )
        refine_passes_raw = (
            request.form.get("refine_passes")
            or request.values.get("refine_passes")
            or request.args.get("refine_passes")
        )
        metadata = {
            "safe_mode": safe_mode_enabled,
            "page_size": page_size_raw,
            "refine_passes": refine_passes_raw,
        }
        page_size_value: Optional[int] = None
        refine_passes_value = 0
        try:
            if metadata["page_size"]:
                page_size_value = max(int(str(metadata["page_size"]).strip()), 0)
        except (ValueError, TypeError):
            metadata["page_size_error"] = "invalid"
        try:
            if metadata["refine_passes"]:
                refine_passes_value = max(int(str(metadata["refine_passes"]).strip()), 0)
        except (ValueError, TypeError):
            metadata["refine_passes_error"] = "invalid"

        project_record = user_store.create_project(
            user_id,
            name=Path(uploaded.filename).stem,
            original_filename=uploaded.filename,
            target_framework=target_framework,
            target_language=target_lang,
            status="queued",
            metadata=metadata,
        )

        suffix = Path(uploaded.filename).suffix or ".zip"
        incoming_dir = api_output_root / user_id / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        archive_target = incoming_dir / f"{project_record['id']}{suffix}"
        uploaded.save(str(archive_target))

        _schedule_api_migration(
            user_id=user_id,
            project_id=project_record["id"],
            archive_path=archive_target,
            target_framework=target_framework,
            target_lang=target_lang,
            src_lang=src_lang,
            src_framework=src_framework,
            page_size=page_size_value,
            refine_passes=refine_passes_value,
            safe_mode=safe_mode_enabled,
        )

        response = {
            "project_id": project_record["id"],
            "status": "queued",
            "detail": "Migration accepted for processing.",
            "projects_url": url_for("api_projects", _external=True),
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
