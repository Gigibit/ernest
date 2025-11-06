from __future__ import annotations
# christophe.py
"""CLI and web front-end for the migration orchestrator."""
"""oh gioia ch'io conobbi, essere amato, amando!"""

import base64
import binascii
import json
import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from analysis.compatibility_agent import CompatibilitySearchAgent
from analysis.dependency_agent import DependencyAnalysisAgent
from analysis.heuristic_agent import HeuristicAnalysisAgent
from analysis.semantic_graph import SemanticGraphBuilder
from core.cache_manager import CacheManager
from core.cost_model import (
    DEFAULT_MARKUP_RATE,
    DEFAULT_RESOURCE_CONTEXT,
    DEFAULT_RESOURCE_COST,
    estimate_h100_receipt,
)
from core.file_utils import secure_unzip
from core.llm_service import LLMService
from core.stats_store import StatsStore
from core.user_store import UserStore
from migration.dependency_resolver import DependencyResolver
from migration.recovery_manager import RecoveryManager
from migration.resource_migrator import ResourceMigrator
from migration.source_migrator import SourceMigrator
from editor import LiveEditorAgent
from planning.architecture_agent import ArchitecturePlanner
from planning.containerization_agent import ContainerizationAgent
from planning.scaffolding_blueprint_agent import ScaffoldingBlueprintAgent
from planning.planning_agent import PlanningAgent
from scaffolding.scaffolding_agent import ScaffoldingAgent


ENV_PREFIX = "ERNEST"
LEGACY_PREFIX = "CHRISTOPHE"


MAX_EDITOR_PREVIEW_BYTES = 512 * 1024  # 512 KB per file preview
EDITOR_MANIFEST_LIMIT = 2000


DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "classify": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 512, "temp": 0.1},
    "analyze": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 512, "temp": 0.1},
    "architecture": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1024, "temp": 0.1},
    "container": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1024, "temp": 0.0},
    "translate": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 4096, "temp": 0.0},
    "adapt": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1024, "temp": 0.0},
    "scaffold": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 2048, "temp": 0.1},
    "dependency": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1536, "temp": 0.0},
    "compatibility": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1536, "temp": 0.0},
    "blueprint": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1536, "temp": 0.0},
    "editor": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 2048, "temp": 0.0},
}


def _resolved_profiles() -> Dict[str, Dict[str, Any]]:
    """Return the default profile set with optional environment overrides."""

    resolved = {name: dict(cfg) for name, cfg in DEFAULT_PROFILES.items()}
    for name, cfg in resolved.items():
        prefix = f"MIGRATION_PROFILE_{name.upper()}"
        override_id = os.environ.get(f"{prefix}_ID")
        override_max = os.environ.get(f"{prefix}_MAX_TOKENS")
        override_temp = os.environ.get(f"{prefix}_TEMP")

        if override_id:
            cfg["id"] = override_id
        if override_max:
            try:
                cfg["max"] = int(override_max)
            except ValueError:
                logging.warning(
                    "Unable to parse %s_MAX_TOKENS=%s as integer; keeping %s",
                    prefix,
                    override_max,
                    cfg["max"],
                )
        if override_temp:
            try:
                cfg["temp"] = float(override_temp)
            except ValueError:
                logging.warning(
                    "Unable to parse %s_TEMP=%s as float; keeping %s",
                    prefix,
                    override_temp,
                    cfg["temp"],
                )
    return resolved


def _coerce_float(value: Optional[str], default: float, env_name: str) -> float:
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return float(stripped)
    except ValueError:
        logging.warning(
            "Unable to parse %s=%s as float; using default %.2f",
            env_name,
            value,
            default,
        )
        return default


def _coerce_markup(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default

    percent_hint = stripped.endswith("%")
    numeric_portion = stripped[:-1].strip() if percent_hint else stripped
    try:
        rate = float(numeric_portion)
    except ValueError:
        logging.warning(
            "Unable to parse markup value %s; using default %.2f",
            value,
            default,
        )
        return default

    if percent_hint:
        rate /= 100.0

    if rate < 0:
        logging.warning(
            "Markup %.3f is negative; clamping to default %.2f",
            rate,
            default,
        )
        return default

    if rate > 1.0:
        logging.warning(
            "Markup %.3f interpreted as a multiplier (%.1f%%). Append '%%' to use percentage semantics.",
            rate,
            rate * 100.0,
        )

    return rate


def _resolve_env(
    primary: str,
    *,
    legacy: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    value = os.environ.get(primary)
    if value is not None:
        return value, primary
    if legacy:
        legacy_value = os.environ.get(legacy)
        if legacy_value is not None:
            return legacy_value, legacy
    return None, primary


def _cost_configuration() -> Dict[str, Any]:
    resource_value, resource_env = _resolve_env(
        f"{ENV_PREFIX}_RESOURCE_COST",
        legacy=f"{LEGACY_PREFIX}_RESOURCE_COST",
    )
    resource_cost = _coerce_float(
        resource_value,
        DEFAULT_RESOURCE_COST,
        resource_env,
    )

    markup_value, _ = _resolve_env(
        f"{ENV_PREFIX}_COST_MARKUP",
        legacy=f"{LEGACY_PREFIX}_COST_MARKUP",
    )
    if markup_value is None:
        markup_value, _ = _resolve_env(
            f"{ENV_PREFIX}_MARKUP_RATE",
            legacy=f"{LEGACY_PREFIX}_MARKUP_RATE",
        )
    markup_rate = _coerce_markup(markup_value, DEFAULT_MARKUP_RATE)

    raw_time, _ = _resolve_env(
        f"{ENV_PREFIX}_RESOURCE_TIME_LEFT",
        legacy=f"{LEGACY_PREFIX}_RESOURCE_TIME_LEFT",
    )
    if raw_time is None:
        time_remaining = DEFAULT_RESOURCE_CONTEXT
    else:
        stripped_time = raw_time.strip()
        time_remaining = stripped_time or DEFAULT_RESOURCE_CONTEXT

    raw_note, _ = _resolve_env(
        f"{ENV_PREFIX}_RESOURCE_CONTEXT",
        legacy=f"{LEGACY_PREFIX}_RESOURCE_CONTEXT",
    )
    if raw_note is None:
        raw_note, _ = _resolve_env(
            f"{ENV_PREFIX}_RESOURCE_NOTE",
            legacy=f"{LEGACY_PREFIX}_RESOURCE_NOTE",
        )
    note = raw_note.strip() if raw_note and raw_note.strip() else None

    return {
        "resource_cost": resource_cost,
        "markup_rate": markup_rate,
        "resource_time_left": time_remaining,
        "resource_note": note,
    }


def _collect_project_stats(base_path: Path, sources: Sequence[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total_files": len(sources),
        "total_lines": 0,
        "max_file_lines": 0,
        "line_counts": {},
    }

    for rel_path in sources:
        try:
            path = (Path(base_path) / rel_path).resolve()
            path.relative_to(Path(base_path).resolve())
        except Exception:  # noqa: BLE001
            continue
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:  # noqa: BLE001
            continue
        count = len(lines)
        stats["line_counts"][rel_path] = count
        stats["total_lines"] += count
        stats["max_file_lines"] = max(stats["max_file_lines"], count)

    if stats["total_files"]:
        stats["average_file_lines"] = int(
            stats["total_lines"] / max(stats["total_files"], 1)
        )
    else:
        stats["average_file_lines"] = 0

    return stats


def _stub_content(
    *,
    target_language: str,
    class_name: str,
    package: str | None,
    reason: str,
    expected_contract: str | None,
    notes: str | None,
) -> str:
    normalized = (target_language or "").lower()
    advisory_lines = [reason]
    if expected_contract and expected_contract.strip():
        advisory_lines.append(expected_contract.strip())
    if notes and notes.strip():
        advisory_lines.append(notes.strip())
    advisory = "\n".join(f"// {line}" for line in advisory_lines if line)

    if normalized in {"java", "kotlin", "scala"}:
        package_line = f"package {package};\n\n" if package else ""
        return (
            f"{package_line}public class {class_name} {{\n"
            f"    {advisory or '// TODO: implement migrated behaviour'}\n"
            "}\n"
        )
    if normalized in {"c#", "csharp", "dotnet"}:
        namespace_line = f"namespace {package} {{\n" if package else ""
        closing = "}\n" if package else ""
        inner_indent = "    " if package else ""
        return (
            (namespace_line)
            + f"{inner_indent}public class {class_name}\n"
            + f"{inner_indent}{{\n"
            + ("\n".join(f"{inner_indent}    {line}" for line in advisory.splitlines())
               if advisory else f"{inner_indent}    // TODO: implement migrated behaviour")
            + f"\n{inner_indent}}}\n"
            + closing
        )
    if normalized in {"typescript", "javascript"}:
        return (
            f"export class {class_name} {{\n"
            + ("\n".join(f"  {line}" for line in advisory.splitlines())
               if advisory else "  // TODO: implement migrated behaviour")
            + "\n}\n"
        )
    if normalized in {"go"}:
        pkg = package or "migrated"
        lines = [f"package {pkg}", "", f"// {class_name} TODO scaffold"]
        if advisory:
            lines.extend(advisory.splitlines())
        lines.append(f"type {class_name} struct {{}}")
        return "\n".join(lines) + "\n"
    if normalized in {"rust"}:
        lines = [f"pub struct {class_name} {{}}", ""]
        if advisory:
            lines.append(advisory.replace("//", "//"))
        lines.append(
            f"impl {class_name} {{\n    pub fn new() -> Self {{\n        // TODO: initialise scaffolded type\n        Self {{}}\n    }}\n}}\n"
        )
        return "\n".join(lines)

    # Default plain text placeholder
    lines = [
        f"Stub for {class_name}",
        "",
        "This file was generated automatically. Provide the equivalent behaviour",
        "for the migrated system following the guidance below:",
        "",
    ]
    if advisory:
        lines.extend(line.replace("// ", "- ") for line in advisory.splitlines())
    else:
        lines.append("- TODO: implement migrated behaviour")
    return "\n".join(lines) + "\n"


def _create_stub_file(
    base_path: Path,
    entry: Mapping[str, Any],
    *,
    target_language: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source": entry.get("source"),
        "target_path": entry.get("target_path"),
        "created": False,
    }

    rel_path = entry.get("target_path")
    class_name = entry.get("class_name") or Path(str(entry.get("source") or "stub")).stem
    package = entry.get("package")

    if not isinstance(rel_path, str) or not rel_path.strip():
        result["error"] = "missing target path"
        return result

    target_file = base_path / rel_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists():
        result["skipped_reason"] = "already exists"
        return result

    content = _stub_content(
        target_language=target_language,
        class_name=class_name,
        package=package if isinstance(package, str) else None,
        reason=str(entry.get("reason") or "TODO"),
        expected_contract=entry.get("expected_contract")
        if isinstance(entry.get("expected_contract"), str)
        else None,
        notes=entry.get("notes") if isinstance(entry.get("notes"), str) else None,
    )
    target_file.write_text(content, encoding="utf-8")
    result.update({
        "created": True,
        "written_bytes": len(content.encode("utf-8")),
    })
    return result


def print_section(title: str, content: str) -> None:
    line = "-" * (len(title) + 8)
    print(f"\n--- {title.upper()} ---\n{content}\n{line}")


def build_services(profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> tuple[LLMService, CacheManager]:
    """Initialise the shared LLM and cache services."""

    llm = LLMService(profiles or _resolved_profiles())
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
    refine_passes: Optional[int] = None,
    safe_mode: bool = True,
) -> Dict[str, Any]:
    """Execute the full migration pipeline for ``zip_path``.

    When ``page_size`` is provided the translation of each source artefact is
    paginated to avoid overloading the model context window.  ``refine_passes``
    overrides the automatic refinement heuristics; when ``None`` the migrator
    chooses how many polishing rounds to perform per page.
    """

    output_root = output_root or Path("output_project")
    llm_service = llm or LLMService(_resolved_profiles())
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))
    close_cache = cache is None

    if hasattr(llm_service, "reset_usage"):
        llm_service.reset_usage()

    logging.info("Starting migration for %s", zip_path)

    architecture_map: Dict[str, Dict[str, str]] = {}
    compatibility_report: Dict[str, Any] = {}
    scaffolding_blueprint: Dict[str, Any] = {}
    stub_results: list[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="ernest_") as tmp:
        temp_dir = Path(tmp)
        logging.info("Unpacking archive %s into %s", zip_path, temp_dir)
        secure_unzip(zip_path, temp_dir)

        logging.info("Running heuristic classification for %s", temp_dir)
        
        heuristic = HeuristicAnalysisAgent(str(temp_dir), llm_service, cache_manager)
        classification = heuristic.classify_files()

        source_files = classification.get("source", []) or []
        resource_files = classification.get("resource", []) or []
        other_files = classification.get("other", []) or []
        logging.info(
            "Classification completed: %d source, %d resource, %d other files",
            len(source_files),
            len(resource_files),
            len(other_files),
        )

        detected_stack = (
            {"language": src_lang, "framework": src_framework}
            if src_lang or src_framework
            else heuristic.detect_stack(classification.get("source"))
        )
        if not detected_stack:
            raise RuntimeError("Impossibile determinare lo stack sorgente.")
        logging.info(
            "Detected source stack: language=%s framework=%s",
            detected_stack.get("language"),
            detected_stack.get("framework"),
        )

        graph = SemanticGraphBuilder()
        graph.add_nodes(classification.get("source", []))
        plan = PlanningAgent().create_plan({}, classification.get("source", []))
        logging.info("Planning produced %d source artefacts", len(plan))

        project_stats = _collect_project_stats(temp_dir, plan)
        logging.info(
            "Project statistics: files=%d total_lines=%d max_file_lines=%d",
            project_stats.get("total_files", 0),
            project_stats.get("total_lines", 0),
            project_stats.get("max_file_lines", 0),
        )

        project_name = Path(zip_path.stem).name.replace("-", "_") or "migrated_project"
        architecture_planner = ArchitecturePlanner(temp_dir, llm_service, cache_manager)
        architecture_map = architecture_planner.propose(
            plan,
            target_language=target_lang,
            target_framework=target_framework,
            project_name=project_name,
        )
        logging.info(
            "Architecture planner prepared %d mappings", len(architecture_map)
        )

        scaffold_agent = ScaffoldingAgent(llm_service, cache_manager)
        target_path = scaffold_agent.generate(
            output_root, project_name, target_framework, target_lang
        )
        logging.info("Scaffold generated at %s", target_path)

        dependency_agent = DependencyAnalysisAgent(temp_dir, llm_service, cache_manager)
        dependency_snapshot = dependency_agent.extract_dependencies(
            target_language=target_lang,
            target_framework=target_framework,
        )
        snapshot_dependencies = dependency_snapshot.get("dependencies", []) or []
        logging.info(
            "Dependency snapshot captured %d dependencies across %d manifests",
            len(snapshot_dependencies),
            len(dependency_snapshot.get("manifests", []) or []),
        )

        blueprint_agent = ScaffoldingBlueprintAgent(
            temp_dir, llm_service, cache_manager
        )
        scaffolding_blueprint = blueprint_agent.blueprint(
            classification.get("source", []) or [],
            architecture_map,
            target_language=target_lang,
            target_framework=target_framework,
            dependencies=snapshot_dependencies,
        )
        logging.info(
            "Blueprint prepared %d scaffolding entries",
            len(scaffolding_blueprint.get("entries", [])),
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
            target_project=target_path,
        )
        logging.info(
            "Dependency resolver produced %d planned entries and %d downloads",
            len(dependency_resolution.get("plan", {}).get("dependencies", []) or []),
            len(dependency_resolution.get("downloads", []) or []),
        )
        rendered_manifests = dependency_resolution.get("rendered_manifests", []) or []
        if rendered_manifests:
            logging.info(
                "Updated %d target manifests with dependency plan",
                len(rendered_manifests),
            )

        container_agent = ContainerizationAgent(llm_service, cache_manager)
        container_plan = container_agent.generate(
            target_language=target_lang,
            target_framework=target_framework,
            detected_stack=detected_stack,
            dependencies=snapshot_dependencies,
            architecture_map=architecture_map,
        )
        container_written = container_agent.persist(container_plan, target_path)
        if container_written:
            container_plan.written_files = container_written

        compatibility_agent = CompatibilitySearchAgent(
            temp_dir, llm_service, cache_manager
        )
        compatibility_report = compatibility_agent.suggest(
            source_files=classification.get("source", []) or [],
            dependencies=snapshot_dependencies,
            target_language=target_lang,
            target_framework=target_framework,
        )
        logging.info(
            "Compatibility agent produced %d guidance entries",
            len(compatibility_report.get("entries", [])),
        )

        logging.info("Beginning source translation for %d files", len(plan))
        for index, src in enumerate(plan, start=1):
            source_file = temp_dir / src
            if not source_file.exists():
                recovery.mark_skipped(src)
                logging.warning("Source file %s missing from archive; marked skipped", src)
                continue

            arch_entry = architecture_map.get(src, {})
            destination_rel = arch_entry.get("target_path") if arch_entry else None
            if destination_rel:
                destination = target_path / destination_rel
            else:
                destination = target_path / "src" / Path(src).with_suffix(".java").name
            logging.info(
                "[%d/%d] Translating source artefact %s -> %s",
                index,
                len(plan),
                source_file,
                destination,
            )
            src_migrator.translate_legacy_backend(
                source_file,
                destination,
                target_language=target_lang,
                target_framework=target_framework,
                target_package=arch_entry.get("package") if arch_entry else None,
                architecture_notes=arch_entry.get("notes") if arch_entry else None,
                page_size=page_size,
                refine_passes=refine_passes,
                safe_mode=safe_mode,
                project_stats=project_stats,
            )

        logging.info(
            "Source translation complete; processing %d resource files",
            len(resource_files),
        )
        for res in classification.get("resource", []):
            resource_path = temp_dir / res
            if resource_path.exists():
                logging.info("Adapting resource %s", resource_path)
                res_migrator.process(resource_path, target_path / "resources")
            else:
                logging.warning("Resource file %s missing from archive; skipping", res)

        for entry in scaffolding_blueprint.get("entries", []):
            if not isinstance(entry, Mapping):
                continue
            if not entry.get("requires_stub"):
                continue
            stub_info = _create_stub_file(
                target_path,
                entry,
                target_language=target_lang,
            )
            stub_info["requires_stub"] = True
            stub_results.append(stub_info)
        if stub_results:
            logging.info(
                "Generated %d scaffolding stubs for manual completion",
                sum(1 for item in stub_results if item.get("created")),
            )

    token_usage = (
        llm_service.get_usage_summary()
        if hasattr(llm_service, "get_usage_summary")
        else {}
    )
    cost_config = _cost_configuration()
    cost_estimate = (
        estimate_h100_receipt(
            token_usage,
            resource_cost=cost_config["resource_cost"],
            markup_rate=cost_config["markup_rate"],
            resource_time_remaining=cost_config["resource_time_left"],
            resource_notes=cost_config["resource_note"],
        )
        if token_usage
        else {}
    )

    if token_usage:
        logging.info("Token usage summary: %s", token_usage)
    if cost_estimate:
        logging.info("Estimated compute receipt: %s", cost_estimate)

    if close_cache:
        cache_manager.close()

    logging.info("Migration completed for %s", zip_path)

    pagination_report = {
        "mode": "manual" if page_size is not None else "auto",
        "requested_page_size": page_size,
        "decisions": dict(src_migrator.pagination_log),
        "project_stats": project_stats,
        "auto_configuration": getattr(src_migrator, "auto_pagination", {}),
    }

    refinement_report = {
        "mode": "manual" if refine_passes is not None else "auto",
        "requested_refine_passes": refine_passes,
        "decisions": dict(getattr(src_migrator, "refinement_log", {})),
        "auto_configuration": getattr(src_migrator, "auto_refine", {}),
    }

    return {
        "classification": classification,
        "detected_stack": detected_stack,
        "plan": plan,
        "architecture": architecture_map,
        "scaffolding_blueprint": scaffolding_blueprint,
        "scaffolding_stubs": stub_results,
        "target_path": target_path,
        "output_root": output_root,
        "project_name": project_name,
        "recovery_path": recovery_path,
        "dependencies": dependency_snapshot,
        "dependency_resolution": dependency_resolution,
        "compatibility": compatibility_report,
        "containerization": container_plan.to_payload(),
        "token_usage": token_usage,
        "cost_estimate": cost_estimate,
        "safe_mode": safe_mode,
        "pagination": pagination_report,
        "refinement": refinement_report,
        "source_archive": str(zip_path.resolve()),
    }


def create_app(
    llm: Optional[LLMService] = None,
    cache: Optional[CacheManager] = None,
    *,
    output_root: Optional[Path] = None,
    user_store: Optional[UserStore] = None,
    stats_store: Optional[StatsStore] = None,
) -> "Flask":
    """Create a Flask application exposing the migration pipeline with auth."""

    from flask import (
        Flask,
        Response,
        abort,
        g,
        jsonify,
        redirect,
        render_template,
        request,
        send_file,
        session,
        url_for,
    )
    from werkzeug.utils import secure_filename

    template_dir = Path(__file__).resolve().with_name("templates")
    static_dir = template_dir.with_name("static")

    raw_root_prefix, _ = _resolve_env(
        f"{ENV_PREFIX}_WEB_ROOT_PATH",
        legacy=f"{LEGACY_PREFIX}_WEB_ROOT_PATH",
    )
    root_prefix = (raw_root_prefix or "").strip()
    if root_prefix in {"", "/"}:
        root_prefix = ""
    elif not root_prefix.startswith("/"):
        root_prefix = f"/{root_prefix.lstrip('/')}"
    else:
        root_prefix = f"/{root_prefix.strip('/')}"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
        static_url_path="/static",
    )

    if root_prefix:
        app.config["APPLICATION_ROOT"] = root_prefix

        class _PrefixMiddleware:
            def __init__(self, app: Callable[..., Any], prefix: str) -> None:
                self.app = app
                self.prefix = prefix

            def __call__(
                self, environ: Dict[str, Any], start_response: Callable[..., Any]
            ) -> Iterable[bytes]:
                path_info = environ.get("PATH_INFO", "") or "/"
                if path_info.startswith(self.prefix):
                    trimmed = path_info[len(self.prefix) :]
                    existing_script = environ.get("SCRIPT_NAME", "")
                    if existing_script.endswith(self.prefix):
                        environ["SCRIPT_NAME"] = existing_script
                    else:
                        combined = f"{existing_script.rstrip('/')}{self.prefix}"
                        environ["SCRIPT_NAME"] = combined or self.prefix
                    environ["PATH_INFO"] = trimmed or "/"
                    return self.app(environ, start_response)

                start_response(
                    "404 NOT FOUND",
                    [("Content-Type", "text/plain; charset=utf-8")],
                )
                return [b"Not Found"]

        app.wsgi_app = _PrefixMiddleware(app.wsgi_app, root_prefix)
    secret_value, _ = _resolve_env(
        f"{ENV_PREFIX}_WEB_SECRET",
        legacy=f"{LEGACY_PREFIX}_WEB_SECRET",
    )
    app.secret_key = secret_value or "ernest-dev-secret"
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

    brand_name = "ERNEST"
    brand_tagline = "Migration Studio"

    brand_context = {
        "brand_name": brand_name,
        "brand_tagline": brand_tagline,
    }

    @app.context_processor
    def inject_brand() -> Dict[str, Any]:
        return dict(brand_context)

    def _interpret_flag(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
        return None

    donate_toggle, _ = _resolve_env(
        f"{ENV_PREFIX}_SHOW_DONATE_BUTTON",
        legacy=f"{LEGACY_PREFIX}_SHOW_DONATE_BUTTON",
    )
    donate_override = _interpret_flag(donate_toggle)
    show_donate_button = donate_override if donate_override is not None else False
    brand_context["show_donate_button"] = show_donate_button

    raw_whitelist, _ = _resolve_env(
        f"{ENV_PREFIX}_PASSPHRASE_WHITELIST",
        legacy=f"{LEGACY_PREFIX}_PASSPHRASE_WHITELIST",
    )
    raw_whitelist = raw_whitelist or ""
    passphrase_whitelist = {
        candidate.strip()
        for candidate in re.split(r"[,\n;]", raw_whitelist)
        if candidate and candidate.strip()
    }
    whitelist_toggle, _ = _resolve_env(
        f"{ENV_PREFIX}_PASSPHRASE_WHITELIST_ENABLED",
        legacy=f"{LEGACY_PREFIX}_PASSPHRASE_WHITELIST_ENABLED",
    )
    whitelist_override = _interpret_flag(whitelist_toggle)
    whitelist_enabled = bool(passphrase_whitelist)
    if whitelist_override is not None:
        whitelist_enabled = bool(passphrase_whitelist) and whitelist_override

    app.config["PASSPHRASE_WHITELIST_ENABLED"] = whitelist_enabled
    app.config["PASSPHRASE_WHITELIST"] = passphrase_whitelist

    def _is_passphrase_allowed(candidate: str) -> bool:
        if not whitelist_enabled:
            return True
        return candidate in passphrase_whitelist

    llm_service = llm or LLMService(_resolved_profiles())
    cache_manager = cache or CacheManager(Path(".cache/migration_cache.db"))
    app.config["ERNEST_LLM_SERVICE"] = llm_service
    app.config["ERNEST_CACHE_MANAGER"] = cache_manager
    app.config["ERNEST_LIVE_EDITOR_AGENT"] = LiveEditorAgent(llm_service)
    worker_value, worker_env = _resolve_env(
        f"{ENV_PREFIX}_WORKERS",
        legacy=f"{LEGACY_PREFIX}_WORKERS",
    )
    try:
        max_workers = int(worker_value) if worker_value else 2
    except ValueError:
        logging.warning(
            "Unable to parse %s=%s as integer; using 2",
            worker_env,
            worker_value,
        )
        max_workers = 2
    executor = ThreadPoolExecutor(max_workers=max_workers)

    base_output_root = output_root or Path("output_project")
    web_output_root = base_output_root / "web"
    api_output_root = base_output_root / "api"
    web_output_root.mkdir(parents=True, exist_ok=True)
    api_output_root.mkdir(parents=True, exist_ok=True)
    app.config.setdefault("ERNEST_OUTPUT_ROOT", base_output_root)
    app.config.setdefault("ERNEST_WEB_OUTPUT_ROOT", web_output_root)
    app.config.setdefault("ERNEST_API_OUTPUT_ROOT", api_output_root)

    store_value, _ = _resolve_env(
        f"{ENV_PREFIX}_USER_STORE",
        legacy=f"{LEGACY_PREFIX}_USER_STORE",
    )
    store_path = Path(store_value or ".cache/users.json")
    user_store = user_store or UserStore(store_path)
    app.config.setdefault("ERNEST_USER_STORE", user_store)

    stats_value, _ = _resolve_env(
        f"{ENV_PREFIX}_STATS_STORE",
        legacy=f"{LEGACY_PREFIX}_STATS_STORE",
    )
    stats_path = Path(stats_value or ".cache/stats.json")
    stats_store = stats_store or StatsStore(stats_path)
    app.config.setdefault("ERNEST_STATS_STORE", stats_store)

    def _timestamp() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _is_text_file(candidate: Path) -> bool:
        try:
            with candidate.open("rb") as handle:
                sample = handle.read(4096)
        except OSError:
            return False
        if not sample:
            return True
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    def _scan_project_files(base_path: Path) -> Tuple[list[Dict[str, Any]], bool]:
        manifest: list[Dict[str, Any]] = []
        truncated = False
        root = base_path.resolve()
        try:
            candidates = sorted(root.rglob("*"), key=lambda item: item.as_posix())
        except OSError:
            return manifest, truncated
        for entry in candidates:
            if not entry.is_file():
                continue
            try:
                stat_info = entry.stat()
            except OSError:
                continue
            rel_path = entry.relative_to(root).as_posix()
            is_text = _is_text_file(entry)
            size = int(stat_info.st_size)
            manifest.append(
                {
                    "path": rel_path,
                    "size": size,
                    "is_text": is_text,
                    "previewable": bool(is_text and size <= MAX_EDITOR_PREVIEW_BYTES),
                }
            )
            if len(manifest) >= EDITOR_MANIFEST_LIMIT:
                truncated = True
                break
        return manifest, truncated

    def _resolve_project_output(
        user_id: str, project_id: str
    ) -> Tuple[Dict[str, Any], Path]:
        project = user_store.get_project(user_id, project_id)
        if not project or project.get("status") != "completed":
            abort(404)
        output_path = project.get("output_path")
        if not output_path:
            abort(404)
        output_dir = Path(output_path)
        if not output_dir.exists() or not output_dir.is_dir():
            abort(404)
        return project, output_dir

    def _resolve_editor_file(base: Path, requested_path: str) -> Path:
        candidate = (base / requested_path).resolve()
        base_resolved = base.resolve()
        if base_resolved not in candidate.parents:
            abort(404)
        if not candidate.exists() or not candidate.is_file():
            abort(404)
        return candidate

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
        target_path = Path(migration["target_path"])
        archive_stem = f"ernst_{project_id}_archive"
        source_archive = migration.get("source_archive")

        final_path: Optional[Path] = None
        with tempfile.TemporaryDirectory(prefix="ernest_bundle_") as bundle_tmp:
            bundle_root = Path(bundle_tmp) / archive_stem
            input_dir = bundle_root / "input_project"
            output_dir = bundle_root / "output_project"
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            shutil.copytree(target_path, output_dir)

            if source_archive:
                source_path = Path(source_archive)
                if source_path.exists():
                    input_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        secure_unzip(source_path, input_dir)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning(
                            "Unable to include source archive %s: %s",
                            source_path,
                            exc,
                        )
                else:
                    logging.warning(
                        "Source archive %s missing; creating empty input snapshot",
                        source_archive,
                    )
                    input_dir.mkdir(parents=True, exist_ok=True)
            else:
                input_dir.mkdir(parents=True, exist_ok=True)

            archive_file = shutil.make_archive(
                str(bundle_root),
                "zip",
                root_dir=bundle_root.parent,
                base_dir=bundle_root.name,
            )
            final_path = target_path.parent / Path(archive_file).name
            Path(archive_file).replace(final_path)

        archive_path = final_path or target_path.with_suffix(".zip")
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
            "architecture": migration.get("architecture"),
            "scaffolding_blueprint": migration.get("scaffolding_blueprint"),
            "scaffolding_stubs": migration.get("scaffolding_stubs"),
            "compatibility": migration.get("compatibility"),
            "containerization": migration.get("containerization"),
            "pagination": migration.get("pagination"),
            "refinement": migration.get("refinement"),
            "error": None,
        }
        record = user_store.update_project(user_id, project_id, **metadata) or metadata
        app.logger.info(
            "Project %s completed for user %s", project_id, user_id
        )
        return record

    def _finalise_failure(user_id: str, project_id: str, message: str) -> None:
        user_store.update_project(
            user_id,
            project_id,
            status="failed",
            error=message,
            completed_at=_timestamp(),
        )
        app.logger.warning(
            "Project %s failed for user %s: %s", project_id, user_id, message
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
        safe_mode: bool,
    ) -> None:
        user_output_root = api_output_root / user_id / project_id
        user_output_root.mkdir(parents=True, exist_ok=True)

        def _runner() -> None:
            try:
                app.logger.info(
                    "Project %s for user %s moved to processing", project_id, user_id
                )
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
        app.logger.info(
            "Project %s for user %s queued for background processing", project_id, user_id
        )

    def _schedule_web_migration(
        *,
        user_id: str,
        project_id: str,
        archive_path: Path,
        target_framework: str,
        target_lang: str,
        src_lang: Optional[str],
        src_framework: Optional[str],
        safe_mode: bool,
    ) -> None:
        user_output_root = web_output_root / user_id / project_id
        user_output_root.mkdir(parents=True, exist_ok=True)

        def _runner() -> None:
            try:
                app.logger.info(
                    "Project %s for user %s moved to processing (web)", project_id, user_id
                )
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
                    safe_mode=safe_mode,
                )
            except Exception as exc:  # noqa: BLE001
                app.logger.exception("Web migration failed for project %s", project_id)
                _finalise_failure(user_id, project_id, str(exc))
            else:
                _finalise_success(user_id, project_id, migration)
            finally:
                archive_path.unlink(missing_ok=True)

        executor.submit(_runner)
        app.logger.info(
            "Project %s for user %s queued for background processing (web)",
            project_id,
            user_id,
        )

    def _group_projects(user_id: str) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        def _timestamp_key(record: Dict[str, Any], *, completed: bool) -> tuple[int, str]:
            candidates = []
            if completed:
                candidates.append(record.get("completed_at"))
            candidates.extend(
                [
                    record.get("updated_at"),
                    record.get("started_at"),
                    record.get("created_at"),
                    record.get("queued_at"),
                ]
            )
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    parsed = datetime.fromisoformat(str(candidate).replace("Z", "+00:00"))
                except ValueError:
                    continue
                return (-int(parsed.timestamp()), str(candidate))
            return (0, "")

        pending: list[Dict[str, Any]] = []
        completed: list[Dict[str, Any]] = []
        for project in user_store.list_projects(user_id):
            entry = dict(project)
            if entry.get("status") == "completed" and entry.get("archive_path"):
                entry["download_url"] = url_for(
                    "api_download_project", project_id=entry["id"]
                )
                output_path = entry.get("output_path")
                if output_path and Path(output_path).exists():
                    entry["editor_url"] = url_for(
                        "project_editor", project_id=entry["id"]
                    )
                completed.append(entry)
            else:
                pending.append(entry)

        pending.sort(key=lambda item: _timestamp_key(item, completed=False))
        completed.sort(key=lambda item: _timestamp_key(item, completed=True))
        return pending, completed



    def _client_ip() -> str:
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            candidate = forwarded_for.split(",", 1)[0].strip()
            if candidate:
                return candidate
        remote_addr = request.remote_addr
        if remote_addr:
            return remote_addr
        return "unknown"

    def _log_visit(logged_in: bool) -> None:
        try:
            stats_store.record(_client_ip(), request.path or "/", logged_in)
        except Exception:  # noqa: BLE001
            app.logger.exception("Unable to record visit statistics")

    def _check_stats_auth() -> bool:
        header = request.headers.get("Authorization", "")
        if not header.lower().startswith("basic "):
            return False
        try:
            encoded = header.split(" ", 1)[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (IndexError, binascii.Error, UnicodeDecodeError):
            return False
        username, _, password = decoded.partition(":")
        return username == "raspberry3" and password == "cecinestpasunpipe"

    @app.before_request
    def load_user() -> None:
        user_id = session.get("user_id")
        g.user = None
        g.auth_token = None
        if not user_id:
            _log_visit(False)
            return
        user = user_store.get_user(user_id)
        if not user:
            session.clear()
            _log_visit(False)
            return
        g.user = user
        g.auth_token = user.get("token")
        _log_visit(True)

    @app.route("/stats/visitors", methods=["GET"])
    def stats_visitors():
        if not _check_stats_auth():
            response = Response("Authentication required", 401)
            response.headers["WWW-Authenticate"] = 'Basic realm="Ernest Stats"'
            return response
        return jsonify({"visitors": stats_store.snapshot()})

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
                if not _is_passphrase_allowed(passphrase):
                    app.logger.warning("Rejected login attempt with non-whitelisted passphrase")
                    errors.append(
                        "Passphrase is not authorised for this preview."
                    )
                else:
                    auth_result = user_store.authenticate(passphrase)
                    session["user_id"] = auth_result["user_id"]
                    session["token"] = auth_result["token"]
                    return redirect(url_for("index"))
        return render_template(
            "login.html",
            errors=errors,
            whitelist_active=whitelist_enabled,
        )

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
            "safe_mode": safe_mode_enabled,
        }

        user_id = session["user_id"]

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


            project_record: Optional[Dict[str, Any]] = None

            if not errors and uploaded:
                filename = secure_filename(uploaded.filename)
                if not filename:
                    errors.append("Nome file non valido per l'upload.")
                else:
                    queued_at = _timestamp()
                    metadata = {
                        "queued_at": queued_at,
                        "safe_mode": safe_mode_enabled,
                        "error": None,
                    }
                    project_record = user_store.create_project(
                        user_id,
                        name=Path(filename).stem,
                        original_filename=uploaded.filename,
                        target_framework=target_framework,
                        target_language=target_lang,
                        status="queued",
                        metadata=metadata,
                    )
                    try:
                        incoming_dir = web_output_root / user_id / project_record["id"] / "incoming"
                        incoming_dir.mkdir(parents=True, exist_ok=True)
                        archive_target = incoming_dir / filename
                        uploaded.save(str(archive_target))
                        user_store.update_project(
                            user_id,
                            project_record["id"],
                            status="queued",
                            error=None,
                            queued_at=project_record.get("queued_at", queued_at),
                            safe_mode=safe_mode_enabled,
                        )
                        _schedule_web_migration(
                            user_id=user_id,
                            project_id=project_record["id"],
                            archive_path=archive_target,
                            target_framework=target_framework,
                            target_lang=target_lang,
                            src_lang=src_lang,
                            src_framework=src_framework,
                            safe_mode=safe_mode_enabled,
                        )
                        result_payload = {
                            "project_id": project_record.get("id"),
                            "project_name": project_record.get("name"),
                            "status": "queued",
                            "queued_at": project_record.get("queued_at", queued_at),
                            "message": "Richiesta accettata. Ci vorrà un po', verrai notificato quando è pronto.",
                        }
                    except Exception as exc:  # noqa: BLE001
                        app.logger.exception("Unable to queue migration")
                        errors.append(str(exc))
                        if project_record is not None:
                            _finalise_failure(user_id, project_record["id"], str(exc))

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

    @app.route("/projects/<project_id>/editor")
    def project_editor(project_id: str):
        if not g.get("user"):
            return redirect(url_for("auth"))
        project, _ = _resolve_project_output(session["user_id"], project_id)
        manifest_url = url_for("project_editor_manifest", project_id=project_id)
        file_url_template = url_for(
            "project_editor_file", project_id=project_id, requested="__PATH__"
        )
        apply_url = url_for("project_editor_apply", project_id=project_id)
        return render_template(
            "editor.html",
            project=project,
            manifest_url=manifest_url,
            file_url_template=file_url_template,
            apply_url=apply_url,
        )

    @app.route("/projects/<project_id>/editor/manifest", methods=["GET"])
    def project_editor_manifest(project_id: str):
        if not g.get("user"):
            return redirect(url_for("auth"))
        _, output_dir = _resolve_project_output(session["user_id"], project_id)
        manifest, truncated = _scan_project_files(output_dir)
        return jsonify({"files": manifest, "truncated": truncated})

    @app.route(
        "/projects/<project_id>/editor/files/<path:requested>", methods=["GET"]
    )
    def project_editor_file(project_id: str, requested: str):
        if not g.get("user"):
            return redirect(url_for("auth"))
        _, output_dir = _resolve_project_output(session["user_id"], project_id)
        file_path = _resolve_editor_file(output_dir, requested)
        size = file_path.stat().st_size
        if size > MAX_EDITOR_PREVIEW_BYTES:
            return jsonify({"error": "File too large for live preview."}), 413
        if not _is_text_file(file_path):
            return jsonify({"error": "Binary files cannot be previewed."}), 415
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        return jsonify({"path": requested, "size": int(size), "content": content})

    @app.route("/projects/<project_id>/editor/apply", methods=["POST"])
    def project_editor_apply(project_id: str):
        if not g.get("user"):
            return redirect(url_for("auth"))
        project, output_dir = _resolve_project_output(session["user_id"], project_id)
        payload = request.get_json(silent=True) or {}
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        focus = payload.get("focus") if isinstance(payload, dict) else None
        agent: LiveEditorAgent = app.config["ERNEST_LIVE_EDITOR_AGENT"]
        try:
            result = agent.apply_prompt(
                output_dir,
                prompt or "",
                focus_path=focus if isinstance(focus, str) else None,
                project_name=(
                    project.get("name")
                    or project.get("project_name")
                    or project.get("id")
                ),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            app.logger.exception("Failed to apply live editor prompt")
            return jsonify({"error": "Unable to apply modifications."}), 500
        return jsonify(result)

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
        sanitized = passphrase.strip()
        if not sanitized:
            return ({"error": "passphrase is required"}, 400)
        if not _is_passphrase_allowed(sanitized):
            app.logger.warning(
                "Rejected API token request with non-whitelisted passphrase"
            )
            return ({"error": "passphrase not authorised"}, 403)
        auth_result = user_store.authenticate(sanitized)
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
                output_path = entry.get("output_path")
                if output_path and Path(output_path).exists():
                    entry["editor_url"] = url_for(
                        "project_editor",
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

        metadata = {
            "safe_mode": safe_mode_enabled,
            "queued_at": _timestamp(),
            "error": None,
        }

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
    if result.get("compatibility"):
        print_section(
            "Alternative Consigliate",
            json.dumps(result["compatibility"], indent=2, ensure_ascii=False),
        )
    if result.get("token_usage"):
        print_section("Consumo Token", json.dumps(result["token_usage"], indent=2))
    if result.get("cost_estimate"):
        print_section("Stima Costi H100", json.dumps(result["cost_estimate"], indent=2))
    if result.get("pagination"):
        print_section(
            "Strategia di Paginazione",
            json.dumps(result["pagination"], indent=2, ensure_ascii=False),
        )


if __name__ == "__main__":
    main()
