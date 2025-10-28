"""Agents to analyse dependency manifests and surface metadata."""

from __future__ import annotations

import json
import logging
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from core.cache_manager import CacheManager
from core.file_utils import read_head
from core.llm_service import LLMService


class DependencyAnalysisAgent:
    """Extract dependency metadata from a project directory."""

    def __init__(self, project_root: Path, llm: LLMService, cache: CacheManager) -> None:
        self.project_root = Path(project_root)
        self.llm = llm
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover_manifests(self) -> List[str]:
        """Return relative paths of files that likely describe dependencies."""

        listing, truncated = self._collect_file_listing()

        if not listing:
            return []

        prompt = textwrap.dedent(
            """
            You review project directory trees to identify files that declare dependencies.
            The list below contains relative file paths from the project root, one per line.
            Return ONLY JSON of the form {"manifests": ["path", ...]} with the files that most likely define dependencies for build, package, or module managers.
            If no dependency manifests are present, respond with {"manifests": []}.
            Do not invent paths that are not present in the listing.
            """
        ).strip()

        if truncated:
            prompt += "\nThe listing may be truncated; prioritise conventional manifest names when unsure."

        prompt += "\n--- FILE LISTING ---\n" + "\n".join(listing)

        cache_key = self.llm.prompt_hash("dependency", prompt)
        cached = self.cache.get(cache_key)
        if isinstance(cached, list):
            manifests = cached
        elif isinstance(cached, Mapping):
            manifests = cached.get("manifests")
        else:
            response = self.llm.invoke("dependency", prompt, max_new_tokens=400)
            result = self._extract_json(response)
            manifests = result.get("manifests") if isinstance(result, Mapping) else []
            self.cache.set(cache_key, manifests)

        resolved: List[str] = []
        for manifest in manifests or []:
            if not isinstance(manifest, str):
                continue
            path = (self.project_root / manifest).resolve()
            try:
                path.relative_to(self.project_root.resolve())
            except ValueError:
                continue
            if path.is_file():
                resolved.append(str(path.relative_to(self.project_root)))

        return sorted(set(resolved))

    def extract_dependencies(
        self,
        *,
        target_language: str,
        target_framework: str,
    ) -> Dict[str, List[Dict[str, object]]]:
        """Parse all discovered manifests using an LLM-assisted prompt."""

        manifests = self.discover_manifests()
        aggregated: Dict[str, List[Dict[str, object]]] = {"manifests": [], "dependencies": []}

        for manifest in manifests:
            manifest_path = self.project_root / manifest
            snippet = read_head(manifest_path, max_chars=6000)
            if not snippet.strip():
                continue

            prompt = textwrap.dedent(
                f"""
                You are analysing dependency manifests for a legacy project being migrated.
                Summarise the dependencies declared in the file below.
                Respond ONLY with JSON using this schema:
                {{
                  "manifest": "relative/path",
                  "items": [
                    {{
                      "name": "package name",
                      "version": "declared version or null",
                      "scope": "runtime | development | plugin | optional | unknown",
                      "notes": "short description"
                    }}
                  ]
                }}
                Keep names as they appear, do not invent packages, and keep the list compact.
                --- FILE ({manifest}) ---
                {snippet}
                """
            )
            cache_key = self.llm.prompt_hash("dependency", prompt)
            cached = self.cache.get(cache_key)
            if cached is not None:
                record = cached
            else:
                response = self.llm.invoke("dependency", prompt, max_new_tokens=700)
                record = self._extract_json(response)
                self.cache.set(cache_key, record)

            if not isinstance(record, Mapping):
                continue
            manifest_name = str(record.get("manifest") or manifest)
            items = list(record.get("items") or [])
            fallback_items = self._fallback_dependencies(manifest_path)
            if fallback_items:
                existing_names = {
                    (item.get("name") or "").lower()
                    for item in items
                    if isinstance(item, Mapping)
                }
                for fallback in fallback_items:
                    name = fallback.get("name")
                    if not name:
                        continue
                    if name.lower() in existing_names:
                        continue
                    items.append(fallback)
            aggregated["manifests"].append(
                {
                    "file": manifest_name,
                    "item_count": len(items),
                }
            )
            for item in items:
                if isinstance(item, Mapping):
                    entry = {
                        "manifest": manifest_name,
                        "name": item.get("name"),
                        "version": item.get("version"),
                        "scope": item.get("scope"),
                        "notes": item.get("notes"),
                    }
                    aggregated["dependencies"].append(entry)

        aggregated["context"] = {
            "target_language": target_language,
            "target_framework": target_framework,
        }
        return aggregated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_file_listing(self, limit: int = 800) -> Tuple[List[str], bool]:
        """Return a bounded listing of project files for LLM manifest discovery."""

        listing: List[str] = []
        for path in sorted(self.project_root.rglob("*")):
            if not path.is_file():
                continue
            listing.append(str(path.relative_to(self.project_root)))
            if len(listing) >= limit:
                return listing, True
        return listing, False

    def _fallback_dependencies(self, manifest_path: Path) -> List[Dict[str, object]]:
        """Best-effort manifest parsing when LLM output is empty or incomplete."""

        name = manifest_path.name.lower()
        try:
            raw_text = manifest_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        if not raw_text.strip():
            return []

        if "require" in name or name.endswith(".txt"):
            return self._parse_requirements_like(raw_text)
        if name == "pyproject.toml":
            return self._parse_pyproject(raw_text)
        if name.endswith("package.json"):
            return self._parse_package_json(raw_text)
        if name == "pom.xml":
            return self._parse_maven_pom(raw_text)
        if name in {"build.gradle", "build.gradle.kts"}:
            return self._parse_gradle(raw_text)
        return []

    @staticmethod
    def _parse_requirements_like(text: str) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("-"):
                # Skip pip directives such as -r other.txt
                continue
            version = None
            package = stripped
            for marker in ("==", "~=", ">=", "<=", ">", "<", "!="):
                if marker in stripped:
                    pkg, version_part = stripped.split(marker, 1)
                    package = pkg.strip()
                    version = f"{marker}{version_part.strip()}" if version_part.strip() else marker
                    break
            if not package:
                continue
            items.append(
                {
                    "name": package,
                    "version": version,
                    "scope": "runtime",
                    "notes": "parsed from requirements file",
                }
            )
        return items

    @staticmethod
    def _parse_pyproject(text: str) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        try:
            import tomllib  # type: ignore
        except Exception:  # pragma: no cover - tomllib missing
            return items

        try:
            data = tomllib.loads(text)
        except Exception:
            return items

        def _append_from(section: str, scope: str) -> None:
            deps = data
            for part in section.split(":"):
                if isinstance(deps, dict):
                    deps = deps.get(part)
                else:
                    deps = None
                if deps is None:
                    break
            if isinstance(deps, dict):
                for pkg, ver in deps.items():
                    items.append(
                        {
                            "name": str(pkg),
                            "version": str(ver),
                            "scope": scope,
                            "notes": "parsed from pyproject.toml",
                        }
                    )
            elif isinstance(deps, list):
                for entry in deps:
                    if isinstance(entry, str):
                        parts = entry.split()
                        pkg = parts[0]
                        ver = " ".join(parts[1:]) if len(parts) > 1 else None
                        items.append(
                            {
                                "name": pkg,
                                "version": ver,
                                "scope": scope,
                                "notes": "parsed from pyproject.toml",
                            }
                        )

        _append_from("project:dependencies", "runtime")
        _append_from("project:optional-dependencies", "optional")
        _append_from("tool:poetry:dependencies", "runtime")
        _append_from("tool:poetry:dev-dependencies", "development")
        return items

    @staticmethod
    def _parse_package_json(text: str) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return items

        for section, scope in (
            ("dependencies", "runtime"),
            ("devDependencies", "development"),
            ("peerDependencies", "optional"),
        ):
            deps = payload.get(section)
            if isinstance(deps, dict):
                for pkg, ver in deps.items():
                    items.append(
                        {
                            "name": str(pkg),
                            "version": str(ver),
                            "scope": scope,
                            "notes": "parsed from package.json",
                        }
                    )
        return items

    @staticmethod
    def _parse_maven_pom(text: str) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return items

        def strip_ns(tag: str) -> str:
            return tag.split('}', 1)[1] if '}' in tag else tag

        for dependency in root.findall('.//{*}dependency'):
            record: Dict[str, object] = {}
            for child in dependency:
                tag = strip_ns(child.tag).lower()
                record[tag] = (child.text or '').strip()
            group = record.get('groupid') or ''
            artifact = record.get('artifactid') or ''
            version = record.get('version') or None
            scope = record.get('scope') or None
            if not (group or artifact):
                continue
            name = f"{group}:{artifact}" if group and artifact else artifact or group
            items.append(
                {
                    "name": name,
                    "version": version,
                    "scope": scope or 'runtime',
                    "notes": "parsed from pom.xml",
                }
            )
        return items

    @staticmethod
    def _parse_gradle(text: str) -> List[Dict[str, object]]:
        pattern = re.compile(
            r"^(?P<configuration>api|implementation|compileOnly|runtimeOnly|testImplementation|testCompile|kapt)\s*\(?(?:['\"])(?P<coordinate>[^'\"\)]+)['\"]\)?",
            re.MULTILINE,
        )
        items: List[Dict[str, object]] = []
        for match in pattern.finditer(text):
            coordinate = match.group('coordinate')
            configuration = match.group('configuration')
            parts = coordinate.split(':')
            if len(parts) >= 2:
                group, artifact, *rest = parts
                name = f"{group}:{artifact}" if group and artifact else coordinate
                version = rest[0] if rest else None
            else:
                name = coordinate
                version = None
            items.append(
                {
                    "name": name,
                    "version": version,
                    "scope": configuration,
                    "notes": "parsed from Gradle build file",
                }
            )
        return items

    @staticmethod
    def _extract_json(text: str) -> Mapping[str, object]:
        import re

        if text is None:
            return {}
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            logging.error("No JSON payload detected in dependency response")
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logging.exception("Failed to decode dependency JSON")
            return {}
