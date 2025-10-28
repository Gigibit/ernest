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
    ) -> Dict[str, Any]:
        """Plan and optionally download the project's dependencies."""

        plan = self._build_plan(
            dependency_snapshot,
            target_language=target_language,
            target_framework=target_framework,
        )
        downloads = []
        if perform_downloads:
            downloads = self._run_wget(plan)
        return {"plan": plan, "downloads": downloads}

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
            return cached

        response = self.llm.invoke("dependency", prompt, max_new_tokens=1200)
        plan = self._extract_json(response)
        self.cache.set(cache_key, plan)
        return plan

    def _run_wget(self, plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
        downloads: List[Dict[str, Any]] = []
        dependencies = plan.get("dependencies", [])
        if not isinstance(dependencies, list):
            return downloads

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
            logging.info("Downloading dependency %%s from %%s", item.get("name"), url)
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
        return downloads

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
