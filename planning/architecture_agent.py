"""LLM-assisted architecture planner for migrated projects."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from core.cache_manager import CacheManager
from core.file_utils import read_head
from core.llm_service import LLMService


class ArchitecturePlanner:
    """Suggest target project layout and packaging for migrated sources."""

    def __init__(
        self,
        project_root: Path,
        llm: LLMService,
        cache: CacheManager,
    ) -> None:
        self.project_root = Path(project_root)
        self.llm = llm
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def propose(
        self,
        sources: Sequence[str],
        *,
        target_language: str,
        target_framework: str,
        project_name: str,
        sample_size: int = 20,
    ) -> Dict[str, Dict[str, str]]:
        """Return mapping of source files to target paths and packages."""

        if not sources:
            return {}

        ordered = list(dict.fromkeys(sources))
        sample = ordered[:sample_size]
        prompt = self._build_prompt(
            sample,
            target_language=target_language,
            target_framework=target_framework,
        )
        cache_key_payload = {
            "sources": sample,
            "target_language": target_language,
            "target_framework": target_framework,
        }
        cache_key = self.llm.prompt_hash(
            "architecture", json.dumps(cache_key_payload, sort_keys=True)
        )
        cached = self.cache.get(cache_key)
        if cached:
            logging.info("Loaded architecture plan from cache")
            return self._merge_with_fallback(
                cached, ordered, target_language=target_language, project_name=project_name
            )

        logging.info(
            "Requesting architecture proposal for %d sources targeting %s/%s",
            len(sample),
            target_language,
            target_framework,
        )
        response = self.llm.invoke("architecture", prompt, max_new_tokens=1024)
        plan = self._parse_plan(response)
        self.cache.set(cache_key, plan)
        return self._merge_with_fallback(
            plan, ordered, target_language=target_language, project_name=project_name
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        sample: Sequence[str],
        *,
        target_language: str,
        target_framework: str,
    ) -> str:
        formatted_snippets = []
        for path in sample:
            snippet = read_head(self.project_root / path, max_chars=1200).strip()
            if snippet:
                formatted_snippets.append(f"### {path}\n{snippet}")
            else:
                formatted_snippets.append(f"### {path}\n<no preview available>")
        snippets_block = "\n\n".join(formatted_snippets)
        return textwrap.dedent(
            f"""
            You are designing the target layout for a migration project.
            The legacy files below will be rewritten.
            Suggest canonical destination paths and (when applicable) package/module names
            for a {target_language} project using {target_framework} conventions.

            Respond ONLY with JSON array items shaped as:
            {{
              "source": "relative/path.cbl",
              "target_path": "src/main/java/com/example/Module/Foo.java",
              "package": "com.example.module",  # use null if irrelevant
              "notes": "short reasoning for structure"  # optional
            }}

            Ensure target paths align with idiomatic build layouts for the chosen language/framework
            (e.g. src/main/java for Spring, app/ for mobile, etc.).
            Do not invent files outside the provided sources.

            --- FILE PREVIEWS ---
            {snippets_block}
            """
        ).strip()

    # ------------------------------------------------------------------
    # Parsing and fallback helpers
    # ------------------------------------------------------------------
    def _parse_plan(self, response: str) -> Sequence[Mapping[str, str]]:
        if not response:
            return []
        match = re.search(r"\[[\s\S]*\]", response)
        if not match:
            logging.warning("Architecture response lacked JSON array; returning empty plan")
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse architecture payload: %s", exc)
            return []
        if not isinstance(payload, list):
            logging.warning("Architecture payload not a list; ignoring")
            return []
        return [item for item in payload if isinstance(item, Mapping)]

    def _merge_with_fallback(
        self,
        plan: Sequence[Mapping[str, str]],
        sources: Sequence[str],
        *,
        target_language: str,
        project_name: str,
    ) -> Dict[str, Dict[str, str]]:
        fallback = self._fallback_plan(sources, target_language, project_name)
        mapping: Dict[str, Dict[str, str]] = {k: dict(v) for k, v in fallback.items()}
        for entry in plan:
            source = entry.get("source") or entry.get("path")
            if not source or source not in mapping:
                continue
            target_path = (entry.get("target_path") or entry.get("destination") or "").strip()
            package = entry.get("package")
            notes = entry.get("notes") or entry.get("reason")
            if target_path:
                cleaned = target_path.lstrip("/\\")
                mapping[source]["target_path"] = cleaned
            if isinstance(package, str) and package.strip():
                mapping[source]["package"] = package.strip()
            if isinstance(notes, str) and notes.strip():
                mapping[source]["notes"] = notes.strip()
        return mapping

    def _fallback_plan(
        self,
        sources: Sequence[str],
        target_language: str,
        project_name: str,
    ) -> Dict[str, Dict[str, str]]:
        default_extension = self._default_extension(target_language)
        java_like_exts = {".java", ".kt", ".kts", ".scala"}
        package_friendly_exts = java_like_exts | {".cs"}
        base_package = (
            self._default_package(project_name)
            if default_extension in package_friendly_exts
            else self._default_package(project_name) if "sap" in (target_language or "").lower() else None
        )
        mapping: Dict[str, Dict[str, str]] = {}
        for source in sources:
            rel = Path(source)
            source_ext = rel.suffix
            extension = default_extension
            if (not extension or extension == ".txt") and source_ext:
                extension = source_ext
            if not extension:
                extension = ""
            if extension and not extension.startswith("."):
                extension = f".{extension}"

            if extension in java_like_exts:
                package = self._join_package(base_package, rel.parent.parts)
                file_name = f"{self._class_name(rel.stem)}{extension}"
                language_folder = {
                    ".java": "java",
                    ".kt": "kotlin",
                    ".kts": "kotlin",
                    ".scala": "scala",
                }[extension]
                path_parts = ["src", "main", language_folder]
                if package:
                    path_parts.extend(package.split("."))
                target_path = str(Path(*path_parts) / file_name)
            else:
                package = self._join_package(base_package, rel.parent.parts) if extension in package_friendly_exts else None
                if extension:
                    file_name = f"{self._class_name(rel.stem)}{extension}"
                else:
                    file_name = rel.name
                target_root = Path("src")
                if rel.parent.parts:
                    target_root = target_root / Path(*rel.parent.parts)
                target_path = str(target_root / file_name)
            mapping[source] = {
                "target_path": target_path,
                "package": package,
            }
        return mapping

    @staticmethod
    def _default_extension(target_language: str) -> str:
        normalized = (target_language or "").lower()
        mapping = {
            "java": ".java",
            "kotlin": ".kt",
            "scala": ".scala",
            "c#": ".cs",
            "csharp": ".cs",
            "f#": ".fs",
            "typescript": ".ts",
            "javascript": ".js",
            "go": ".go",
            "rust": ".rs",
            "python": ".py",
            "ruby": ".rb",
            "php": ".php",
            "cobol": ".cbl",
            "abap": ".abap",
            "sap": ".abap",
            "hana": ".abap",
            "sap hana": ".abap",
        }
        for key, ext in mapping.items():
            if key in normalized:
                return ext
        return ".txt"

    @staticmethod
    def _default_package(project_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", project_name.lower()).strip("_")
        if not slug:
            slug = "app"
        segments = [seg for seg in slug.split("_") if seg]
        return ".".join(["com", "migrated"] + segments)

    @staticmethod
    def _join_package(base: str | None, extra: Iterable[str]) -> str | None:
        segments = []
        if base:
            segments.extend(part for part in base.split(".") if part)
        for part in extra:
            cleaned = re.sub(r"[^a-z0-9]", "_", part.lower()).strip("_")
            if not cleaned:
                continue
            if cleaned[0].isdigit():
                cleaned = f"_{cleaned}"
            segments.append(cleaned)
        return ".".join(segments) if segments else base

    @staticmethod
    def _class_name(stem: str) -> str:
        parts = re.split(r"[^a-zA-Z0-9]+", stem)
        cleaned = [part for part in parts if part]
        if not cleaned:
            return "MigratedModule"
        return "".join(part.capitalize() for part in cleaned)
