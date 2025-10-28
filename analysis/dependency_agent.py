"""Agents to analyse dependency manifests and surface metadata."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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

            record = self._request_manifest_summary(manifest, snippet)
            if not isinstance(record, Mapping):
                continue

            manifest_name = str(record.get("manifest") or manifest)
            raw_items = record.get("items")
            items: List[Mapping[str, object]] = []
            if isinstance(raw_items, list):
                for entry in raw_items:
                    if isinstance(entry, Mapping):
                        items.append(entry)

            if not items:
                extractor_items = self._invoke_python_extractor(manifest, snippet)
                if extractor_items:
                    items.extend(extractor_items)
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

    def _request_manifest_summary(self, manifest: str, snippet: str) -> Optional[Mapping[str, object]]:
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
            If no dependencies are present, respond with an empty list for "items" but still return valid JSON.
            Never add commentary or Markdown.
            --- FILE ({manifest}) ---
            {snippet}
            """
        )

        cache_key = self.llm.prompt_hash("dependency", prompt)
        cached = self.cache.get(cache_key)
        if isinstance(cached, Mapping):
            return cached

        attempts = 0
        follow_up = prompt
        record: Optional[Mapping[str, object]] = None
        while attempts < 3 and not record:
            response = self.llm.invoke("dependency", follow_up, max_new_tokens=700)
            candidate = self._extract_json(response)
            if isinstance(candidate, Mapping) and "items" in candidate:
                record = candidate
                break
            attempts += 1
            follow_up = textwrap.dedent(
                f"""
                The previous response was not valid JSON.
                Reply again for manifest {manifest} using ONLY valid JSON with the requested schema.
                --- FILE ({manifest}) ---
                {snippet}
                """
            )

        if record:
            self.cache.set(cache_key, record)
        else:
            logging.error("Dependency manifest summary failed for %s", manifest)
        return record

    def _invoke_python_extractor(self, manifest: str, snippet: str) -> List[Mapping[str, object]]:
        plan = self._request_python_extractor(manifest, snippet)
        if not plan:
            return []

        script = plan.get("python")
        if not isinstance(script, str) or not script.strip():
            return []

        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                input=snippet,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except Exception:  # pragma: no cover - defensive
            logging.exception("Python extractor execution failed for %s", manifest)
            return []

        if completed.returncode != 0:
            logging.error(
                "Python extractor returned non-zero exit status for %s: %s",
                manifest,
                completed.stderr.strip(),
            )
            return []

        stdout = completed.stdout.strip()
        if not stdout:
            return []

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            logging.error("Python extractor emitted invalid JSON for %s", manifest)
            return []

        extracted: List[Mapping[str, object]] = []
        if isinstance(parsed, list):
            for entry in parsed:
                if isinstance(entry, Mapping) and entry.get("name"):
                    extracted.append(entry)
        return extracted

    def _request_python_extractor(self, manifest: str, snippet: str) -> Optional[Mapping[str, object]]:
        prompt = textwrap.dedent(
            f"""
            You design lightweight Python 3 scripts that parse arbitrary dependency manifests.
            Provide ONLY JSON with this schema:
            {{
              "python": "script"
            }}
            The script must read the manifest content from STDIN and print a JSON array to STDOUT.
            Each JSON element must include keys: name, version (or null), scope (or null), and notes.
            Use only Python's standard library. Do not read or write files; rely solely on STDIN and STDOUT.
            Keep the script concise and deterministic.
            Manifest name: {manifest}
            --- MANIFEST CONTENT START ---
            {snippet}
            --- MANIFEST CONTENT END ---
            """
        )

        cache_key = self.llm.prompt_hash("dependency_extractor", prompt)
        cached = self.cache.get(cache_key)
        if isinstance(cached, Mapping):
            return cached

        response = self.llm.invoke("dependency", prompt, max_new_tokens=800)
        plan = self._extract_json(response)
        if isinstance(plan, Mapping) and plan.get("python"):
            self.cache.set(cache_key, plan)
            return plan

        logging.error("Failed to obtain Python extractor for %s", manifest)
        return None

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
