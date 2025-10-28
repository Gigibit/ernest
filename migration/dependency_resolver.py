"""Dependency resolution helpers for migration workflows."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping

from core.cache_manager import CacheManager
from core.llm_service import LLMService


class DependencyResolver:
    """Plan downloads, alternatives, and translation fallbacks for dependencies."""

    def __init__(
        self,
        llm: LLMService,
        cache: CacheManager,
        *,
        download_root: Path,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.download_root = download_root
        self.download_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(
        self,
        dependency_snapshot: Mapping[str, Any],
        *,
        target_language: str,
        target_framework: str,
        perform_downloads: bool = True,
        target_project: Path | None = None,
    ) -> Dict[str, Any]:
        """Plan and optionally download the project's dependencies."""

        logging.info(
            "Resolving %d dependencies for %s/%s",
            len(dependency_snapshot.get("dependencies", []) or []),
            target_language,
            target_framework,
        )
        plan = self._build_plan(
            dependency_snapshot,
            target_language=target_language,
            target_framework=target_framework,
        )
        downloads = []
        if perform_downloads:
            downloads = self._run_wget(plan)
        rendered = []
        if target_project is not None:
            rendered = self._materialize_manifests(
                target_project,
                dependency_snapshot,
                plan,
                target_language=target_language,
                target_framework=target_framework,
            )
        return {"plan": plan, "downloads": downloads, "rendered_manifests": rendered}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_plan(
        self,
        dependency_snapshot: Mapping[str, Any],
        *,
        target_language: str,
        target_framework: str,
    ) -> Dict[str, Any]:
        dependencies = dependency_snapshot.get("dependencies", [])
        manifests = dependency_snapshot.get("manifests", [])

        if not dependencies:
            return {
                "dependencies": [],
                "manifests": manifests,
                "notes": "No dependency manifests detected in the uploaded archive.",
            }

        payload = {
            "target_language": target_language,
            "target_framework": target_framework,
            "dependencies": dependencies,
        }
        payload_json = json.dumps(payload)
        prompt = textwrap.dedent(
            f"""
            You analyse dependency lists for migration programmes.
            Given the JSON payload below, produce a detailed plan with the following structure:
            {{
              "manifests": [...],
              "dependencies": [
                {{
                  "name": "string",
                  "current_version": "string | null",
                  "source_manifest": "string",
                  "download": {{
                    "url": "https://... or null",
                    "notes": "how to retrieve or build it"
                  }},
                  "alternatives": [
                    {{
                      "name": "string",
                      "language": "target language",
                      "framework": "target framework or ecosystem",
                      "rationale": "why it fits",
                      "adoption_risk": "low | medium | high"
                    }}
                  ],
                  "translation_required": true | false,
                  "translation_schedule": {{
                    "estimated_days": number,
                    "milestones": ["phase description"]
                  }}
                }}
              ],
              "summary": "concise explanation"
            }}
            Use wget-friendly URLs when possible. If a package is built from source,
            provide instructions instead of a URL. When translation is required,
            describe realistic milestones even if the effort spans multiple days.
            Respond ONLY with JSON.
            --- INPUT PAYLOAD ---
            {payload_json}
            """
        )

        cache_key = self.llm.prompt_hash("dependency", prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logging.info("Loaded dependency plan from cache (%d entries)", len(cached.get("dependencies", []) or []))
            return cached

        logging.info("Requesting dependency plan from LLM")
        response = self.llm.invoke("dependency", prompt, max_new_tokens=1200)
        plan = self._extract_json(response)
        self.cache.set(cache_key, plan)
        logging.info(
            "Dependency plan generated with %d entries",
            len(plan.get("dependencies", []) or []),
        )
        return plan

    def _run_wget(self, plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
        downloads: List[Dict[str, Any]] = []
        dependencies = plan.get("dependencies", [])
        if not isinstance(dependencies, list):
            return downloads

        logging.info("Executing download plan for %d dependencies", len(dependencies))
        for item in dependencies:
            if not isinstance(item, Mapping):
                continue
            download_info = item.get("download") or {}
            url = download_info.get("url") if isinstance(download_info, Mapping) else None
            if not url:
                continue

            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(item.get("name") or "dependency"))
            target_dir = self.download_root / safe_name
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / "downloaded"

            cmd = [
                "wget",
                "--quiet",
                "--show-progress",
                "--progress=dot:giga",
                "-O",
                str(output_path),
                str(url),
            ]
            logging.info("Downloading dependency %s from %s", item.get("name"), url)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                downloads.append(
                    {
                        "name": item.get("name"),
                        "url": url,
                        "returncode": result.returncode,
                        "stderr": (result.stderr or "").strip(),
                        "stdout": (result.stdout or "").strip(),
                        "output_path": str(output_path),
                    }
                )
            except FileNotFoundError:
                logging.error("wget not available when downloading %s", item.get("name"))
                downloads.append(
                    {
                        "name": item.get("name"),
                        "url": url,
                        "returncode": -1,
                        "stderr": "wget executable not found",
                        "stdout": "",
                        "output_path": str(output_path),
                    }
                )
            except subprocess.TimeoutExpired:
                logging.error("Timeout downloading %s from %s", item.get("name"), url)
                downloads.append(
                    {
                        "name": item.get("name"),
                        "url": url,
                        "returncode": -2,
                        "stderr": "download timed out",
                        "stdout": "",
                        "output_path": str(output_path),
                    }
                )
        logging.info("Completed dependency downloads: %d attempted", len(downloads))
        return downloads

    def _materialize_manifests(
        self,
        project_root: Path,
        snapshot: Mapping[str, Any],
        plan: Mapping[str, Any],
        *,
        target_language: str,
        target_framework: str,
    ) -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        project_root = Path(project_root)
        candidate_names = [
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "package.json",
            "requirements.txt",
        ]
        snapshot_json = json.dumps(snapshot, indent=2, sort_keys=True)
        plan_json = json.dumps(plan, indent=2, sort_keys=True)

        for name in candidate_names:
            path = project_root / name
            if not path.exists():
                continue
            try:
                original = path.read_text(encoding="utf-8")
            except OSError:
                logging.warning("Unable to read manifest %s", path)
                continue

            prompt = textwrap.dedent(
                f"""
                You update dependency manifests after modernisation programmes.
                Target stack: {target_language} / {target_framework}.

                Existing manifest ({name}):
                ---
                {original}
                ---

                Legacy dependency snapshot:
                {snapshot_json}

                Resolved migration plan:
                {plan_json}

                Produce an updated {name} for the target stack that:
                - Preserves structural elements already present.
                - Adds dependencies or TODO placeholders reflecting the plan and alternatives.
                - Keeps valid syntax for {name} and avoids commentary or markdown fences.
                Respond ONLY with the revised manifest contents.
                """
            ).strip()

            cache_key = self.llm.prompt_hash("dependency", prompt)
            cached = self.cache.get(cache_key)
            if cached is not None:
                revised = self._clean_manifest(cached)
            else:
                logging.info("Updating target manifest %s using dependency plan", path)
                response = self.llm.invoke("dependency", prompt, max_new_tokens=1500)
                revised = self._clean_manifest(response)
                self.cache.set(cache_key, revised)

            if not revised:
                logging.warning("Dependency manifest update for %s produced empty output", path)
                continue

            try:
                path.write_text(revised, encoding="utf-8")
            except OSError:
                logging.exception("Failed to write revised manifest %s", path)
                continue

            manifests.append(
                {
                    "file": str(path.relative_to(project_root)),
                    "updated": True,
                }
            )

        return manifests

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        if text is None:
            return {}
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            logging.error("No JSON detected in dependency plan response")
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logging.exception("Failed to decode dependency plan JSON")
            return {}

    @staticmethod
    def _clean_manifest(text: str) -> str:
        if text is None:
            return ""
        stripped = text.strip()
        matches = re.findall(r"```(?:[\w+-]*)\n([\s\S]*?)\n```", stripped)
        if matches:
            stripped = max(matches, key=len).strip()
        stripped = re.sub(r"^```(?:[\w+-]*)\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()
