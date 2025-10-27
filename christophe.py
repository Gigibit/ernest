# christophe.py
"""CLI and web front-end for the migration orchestrator."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from analysis.heuristic_agent import HeuristicAnalysisAgent
from analysis.semantic_graph import SemanticGraphBuilder
from core.cache_manager import CacheManager
from core.file_utils import secure_unzip
from core.llm_service import LLMService
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
) -> Dict[str, Any]:
    """Execute the full migration pipeline for ``zip_path``."""

    output_root = output_root or Path("output_project")
    llm_service = llm or LLMService(DEFAULT_PROFILES)
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))
    close_cache = cache is None

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

        recovery_path = target_path / "migration_state.json"
        recovery = RecoveryManager(recovery_path)
        src_migrator = SourceMigrator(llm_service, cache_manager, recovery)
        res_migrator = ResourceMigrator(llm_service, cache_manager)

        for src in plan:
            source_file = temp_dir / src
            if not source_file.exists():
                recovery.mark_skipped(src)
                continue

            destination = target_path / "src" / Path(src).with_suffix(".java").name
            src_migrator.translate_cobol(source_file, destination)

        for res in classification.get("resource", []):
            resource_path = temp_dir / res
            if resource_path.exists():
                res_migrator.process(resource_path, target_path / "resources")

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
    }


def create_app(
    llm: Optional[LLMService] = None,
    cache: Optional[CacheManager] = None,
    *,
    output_root: Optional[Path] = None,
) -> "Flask":
    """Create a Flask application exposing the migration pipeline."""

    from flask import Flask, render_template_string, request
    from werkzeug.utils import secure_filename

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

    llm_service = llm or LLMService(DEFAULT_PROFILES)
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))
    web_output_root = output_root or Path("output_project/web")

    PAGE_TEMPLATE = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Christophe Migration Portal</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6fb; margin: 0; padding: 0; }
            .container { max-width: 960px; margin: 40px auto; background: #fff; padding: 32px; border-radius: 16px; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.12); }
            h1 { margin-top: 0; color: #0f172a; }
            form { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; align-items: flex-start; }
            label { display: block; font-weight: 600; margin-bottom: 6px; color: #334155; }
            input[type="text"] { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #cbd5f5; font-size: 15px; }
            .drop-zone { grid-column: 1 / -1; border: 2px dashed #6366f1; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; background: rgba(99, 102, 241, 0.05); transition: background 0.2s, border-color 0.2s; }
            .drop-zone.dragover { background: rgba(99, 102, 241, 0.15); border-color: #4f46e5; }
            .drop-zone p { margin: 0; color: #4338ca; font-weight: 600; }
            button { grid-column: 1 / -1; background: linear-gradient(135deg, #4f46e5, #6366f1); color: white; border: none; border-radius: 12px; padding: 14px 18px; font-size: 16px; font-weight: 600; cursor: pointer; box-shadow: 0 14px 35px rgba(79, 70, 229, 0.3); transition: transform 0.15s ease, box-shadow 0.2s ease; }
            button:hover { transform: translateY(-1px); box-shadow: 0 18px 40px rgba(79, 70, 229, 0.35); }
            .messages { grid-column: 1 / -1; }
            .error { background: #fee2e2; color: #b91c1c; padding: 12px 16px; border-radius: 10px; margin-bottom: 12px; }
            .result { background: #eef2ff; color: #1e1b4b; padding: 20px 24px; border-radius: 12px; margin-top: 28px; }
            .result pre { background: rgba(15, 23, 42, 0.85); color: #e2e8f0; padding: 16px; border-radius: 10px; overflow-x: auto; }
            ul { padding-left: 20px; }
            footer { margin-top: 48px; text-align: center; color: #64748b; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Christophe Migration Portal</h1>
            <form method="post" enctype="multipart/form-data">
                <div>
                    <label for="target_framework">Target framework</label>
                    <input id="target_framework" name="target_framework" type="text" value="{{ defaults.target_framework }}" placeholder="e.g. Spring Boot">
                </div>
                <div>
                    <label for="target_lang">Target language</label>
                    <input id="target_lang" name="target_lang" type="text" value="{{ defaults.target_lang }}" placeholder="e.g. Java">
                </div>
                <div>
                    <label for="src_lang">Source language (optional)</label>
                    <input id="src_lang" name="src_lang" type="text" value="{{ defaults.src_lang }}" placeholder="e.g. COBOL">
                </div>
                <div>
                    <label for="src_framework">Source framework (optional)</label>
                    <input id="src_framework" name="src_framework" type="text" value="{{ defaults.src_framework }}" placeholder="e.g. SAP ECC">
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
            {% if result %}
            <div class="result">
                <h2>Migration result</h2>
                <p><strong>Project:</strong> {{ result.project_name }}</p>
                <p><strong>Output directory:</strong> {{ result.output_path }}</p>
                <p><strong>Detected stack:</strong></p>
                <pre>{{ result.detected_stack | tojson(indent=2) }}</pre>
                <p><strong>Migration plan:</strong></p>
                <ul>
                {% for step in result.plan %}
                    <li>{{ step }}</li>
                {% endfor %}
                </ul>
                <p><strong>Classification summary:</strong></p>
                <pre>{{ result.classification | tojson(indent=2) }}</pre>
            </div>
            {% endif %}
            <footer>
                Powered by Christophe — upload a ZIP archive and select the destination stack to begin.
            </footer>
        </div>
        <script>
            const dropZone = document.getElementById('drop-zone');
            const dropZoneText = document.getElementById('drop-zone-text');
            const fileInput = document.getElementById('source_zip');

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
        </script>
    </body>
    </html>
    """

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        errors: list[str] = []
        result_payload: Optional[Dict[str, Any]] = None

        defaults = {
            "target_framework": request.form.get("target_framework", ""),
            "target_lang": request.form.get("target_lang", "java"),
            "src_lang": request.form.get("src_lang", ""),
            "src_framework": request.form.get("src_framework", ""),
        }

        if request.method == "POST":
            target_framework = defaults["target_framework"].strip()
            target_lang = defaults["target_lang"].strip() or "java"
            src_lang = defaults["src_lang"].strip() or None
            src_framework = defaults["src_framework"].strip() or None
            uploaded = request.files.get("source_zip")

            if not target_framework:
                errors.append("Target framework is required.")
            if not uploaded or uploaded.filename == "":
                errors.append("You must upload a ZIP archive.")
            elif not uploaded.filename.lower().endswith(".zip"):
                errors.append("Uploaded file must have .zip extension.")

            if not errors and uploaded:
                filename = secure_filename(uploaded.filename)
                suffix = Path(filename).suffix or ".zip"
                temp_zip_path: Optional[Path] = None
                try:
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
                        output_root=web_output_root,
                        llm=llm_service,
                        cache=cache_manager,
                    )
                    result_payload = {
                        "project_name": migration["project_name"],
                        "output_path": str(migration["target_path"]),
                        "detected_stack": migration["detected_stack"],
                        "plan": migration["plan"],
                        "classification": migration["classification"],
                    }
                except Exception as exc:  # noqa: BLE001 - bubble error to UI and log stack
                    app.logger.exception("Migration failed")
                    errors.append(str(exc))
                finally:
                    if temp_zip_path is not None:
                        temp_zip_path.unlink(missing_ok=True)

        return render_template_string(
            PAGE_TEMPLATE,
            errors=errors,
            result=result_payload,
            defaults=defaults,
        )

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

    llm, cache = build_services()
    try:
        result = run_migration(
            args.zip_path,
            args.target_framework,
            args.target_lang,
            src_lang=args.src_lang,
            src_framework=args.src_framework,
            reuse_cache=args.reuse_cache,
            llm=llm,
            cache=cache,
        )
    finally:
        cache.close()

    print_section("Classificazione File", json.dumps(result["classification"], indent=2))
    print_section("Stack Rilevato", json.dumps(result["detected_stack"], indent=2))
    print_section("Piano di Migrazione", "\n".join(result["plan"]))
    print_section("Progetto Migrato", f"Output generato in: {result['target_path']}")


if __name__ == "__main__":
    main()
