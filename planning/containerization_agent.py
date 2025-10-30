"""LLM-backed helper that prepares Docker assets for backend migrations."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from core.cache_manager import CacheManager
from core.llm_service import LLMService


_BACKEND_KEYWORDS = {
    "backend",
    "service",
    "microservice",
    "api",
    "rest",
    "graphql",
    "server",
    "sap",
    "hana",
    "abap",
    "spring",
    "quarkus",
    "asp",
    "dotnet",
    "node",
    "express",
    "django",
    "flask",
    "fastapi",
}


@dataclass
class ContainerisationResult:
    """Typed structure returned by :class:`ContainerizationAgent`."""

    enabled: bool
    reason: str | None = None
    dockerfile: str | None = None
    compose: str | None = None
    dockerignore: str | None = None
    notes: str | None = None
    written_files: Dict[str, str] | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "enabled": self.enabled,
            "reason": self.reason,
            "dockerfile": self.dockerfile,
            "compose": self.compose,
            "dockerignore": self.dockerignore,
            "notes": self.notes,
            "written_files": self.written_files or {},
        }
        return {k: v for k, v in payload.items() if v is not None and v != {}}


class ContainerizationAgent:
    """Coordinate Docker artefact synthesis for backend-oriented migrations."""

    def __init__(self, llm: LLMService, cache: CacheManager) -> None:
        self.llm = llm
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        target_language: str,
        target_framework: str,
        detected_stack: Mapping[str, Any] | None,
        dependencies: Sequence[Mapping[str, Any]] | Sequence[str],
        architecture_map: Mapping[str, Mapping[str, Any]],
    ) -> ContainerisationResult:
        """Return a containerisation blueprint for backend targets."""

        if not self._should_containerize(
            target_language, target_framework, detected_stack, dependencies
        ):
            return ContainerisationResult(
                enabled=False,
                reason="Target stack does not look like a backend service",
            )

        prompt = self._build_prompt(
            target_language=target_language,
            target_framework=target_framework,
            detected_stack=detected_stack or {},
            dependencies=dependencies,
            architecture_map=architecture_map,
        )

        cache_payload = {
            "target_language": target_language,
            "target_framework": target_framework,
            "detected_stack": detected_stack or {},
            "dependencies": self._normalise_dependencies(dependencies),
            "architecture_summary": self._summarise_architecture(architecture_map),
        }
        cache_key = self.llm.prompt_hash(
            "container", json.dumps(cache_payload, sort_keys=True)
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            logging.info("Loaded containerisation plan from cache")
            return self._from_dict(cached)

        logging.info(
            "Requesting containerisation assets for %s/%s", target_language, target_framework
        )
        response = self.llm.invoke("container", prompt, max_new_tokens=1024)
        plan = self._parse_response(response)
        if plan is None:
            logging.warning("Containerisation response was empty; skipping Docker assets")
            return ContainerisationResult(
                enabled=False,
                reason="LLM response missing container plan",
            )

        result = self._from_dict(plan)
        self.cache.set(cache_key, result.to_payload())
        return result

    def persist(self, result: ContainerisationResult, project_root: Path) -> Dict[str, str]:
        """Write generated Docker assets to the migrated project tree."""

        if not result.enabled:
            return {}

        written: Dict[str, str] = {}
        if result.dockerfile:
            dockerfile_path = project_root / "Dockerfile"
            if dockerfile_path.exists():
                logging.info("Skipping Dockerfile write; file already exists at %s", dockerfile_path)
            else:
                dockerfile_path.write_text(self._clean_block(result.dockerfile), encoding="utf-8")
                written["dockerfile"] = str(dockerfile_path)
        if result.compose:
            compose_path = project_root / "docker-compose.yml"
            if compose_path.exists():
                logging.info("Skipping docker-compose.yml; file already exists at %s", compose_path)
            else:
                compose_path.write_text(self._clean_block(result.compose), encoding="utf-8")
                written["compose"] = str(compose_path)
        if result.dockerignore:
            ignore_path = project_root / ".dockerignore"
            if ignore_path.exists():
                logging.info("Skipping .dockerignore; file already exists at %s", ignore_path)
            else:
                ignore_path.write_text(self._clean_block(result.dockerignore), encoding="utf-8")
                written["dockerignore"] = str(ignore_path)

        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_containerize(
        self,
        target_language: str,
        target_framework: str,
        detected_stack: Mapping[str, Any] | None,
        dependencies: Sequence[Mapping[str, Any]] | Sequence[str],
    ) -> bool:
        haystack = " ".join(
            filter(
                None,
                [
                    target_language,
                    target_framework,
                    str((detected_stack or {}).get("language")),
                    str((detected_stack or {}).get("framework")),
                ],
            )
        ).lower()
        if any(keyword in haystack for keyword in _BACKEND_KEYWORDS):
            return True

        # Fall back to dependency heuristics
        normalised = self._normalise_dependencies(dependencies)
        if not normalised:
            return False
        dependency_terms = " ".join(normalised).lower()
        return any(keyword in dependency_terms for keyword in _BACKEND_KEYWORDS)

    def _build_prompt(
        self,
        *,
        target_language: str,
        target_framework: str,
        detected_stack: Mapping[str, Any],
        dependencies: Sequence[Mapping[str, Any]] | Sequence[str],
        architecture_map: Mapping[str, Mapping[str, Any]],
    ) -> str:
        dependency_block = "\n".join(
            f"- {name}" for name in self._normalise_dependencies(dependencies)[:20]
        )
        if not dependency_block:
            dependency_block = "- <nessuna dipendenza esplicita rilevata>"

        architecture_summary = self._summarise_architecture(architecture_map)
        architecture_block = "\n".join(f"- {entry}" for entry in architecture_summary)
        if not architecture_block:
            architecture_block = "- struttura target non disponibile"

        return textwrap.dedent(
            f"""
            You coordinate enterprise migrations landing on container-ready runtimes.
            Given the inputs below, produce a JSON object describing container assets for the migrated backend.

            - Honour {target_framework or 'the chosen'} conventions for {target_language or 'the target language'}.
            - Optimise for production usage (multi-stage builds, health checks, environment variables).
            - Highlight any stack-specific operational nuances surfaced by the inputs.
            - Always include security hardening notes.

            --- STACK CONTEXT ---
            detected_language: {detected_stack.get('language') or 'unknown'}
            detected_framework: {detected_stack.get('framework') or 'unknown'}

            --- KEY DEPENDENCIES ---
            {dependency_block}

            --- TARGET ARCHITECTURE HINTS ---
            {architecture_block}

            Respond ONLY with a JSON object shaped as:
            {{
              "enabled": true,
              "dockerfile": "<Dockerfile contents>",
              "compose": "<docker-compose.yml>" | null,
              "dockerignore": "<.dockerignore>" | null,
              "notes": "short bullet list with operational notes"
            }}
            """
        ).strip()

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any] | None:
        if not response:
            return None
        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse containerisation payload: %s", exc)
            return None

    @staticmethod
    def _normalise_dependencies(
        dependencies: Sequence[Mapping[str, Any]] | Sequence[str],
    ) -> list[str]:
        names: list[str] = []
        for entry in dependencies:
            if isinstance(entry, Mapping):
                candidate = (
                    entry.get("name")
                    or entry.get("package")
                    or entry.get("artifact")
                    or entry.get("id")
                )
                if candidate:
                    names.append(str(candidate))
            elif isinstance(entry, str):
                names.append(entry)
        deduped: list[str] = []
        seen = set()
        for name in names:
            lowered = name.lower()
            if lowered in seen:
                continue
            deduped.append(name)
            seen.add(lowered)
        return deduped

    @staticmethod
    def _summarise_architecture(
        architecture_map: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        destinations: list[str] = []
        for entry in architecture_map.values():
            target = entry.get("target_path") if isinstance(entry, Mapping) else None
            if isinstance(target, str) and target:
                destinations.append(target)
        summary: list[str] = []
        seen = set()
        for target in destinations[:40]:
            major = str(Path(target).parts[:3]) if target else target
            if major in seen:
                continue
            summary.append(target)
            seen.add(major)
        return summary

    @staticmethod
    def _clean_block(text: str) -> str:
        stripped = text.strip()
        fence_blocks = re.findall(r"```(?:[\w+-]*)\n([\s\S]*?)\n```", stripped)
        if fence_blocks:
            stripped = max(fence_blocks, key=len).strip()
        stripped = re.sub(r"^```(?:[\w+-]*)\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _from_dict(self, payload: Mapping[str, Any]) -> ContainerisationResult:
        return ContainerisationResult(
            enabled=bool(payload.get("enabled", True)),
            reason=payload.get("reason"),
            dockerfile=payload.get("dockerfile") or payload.get("Dockerfile"),
            compose=payload.get("compose") or payload.get("docker_compose"),
            dockerignore=payload.get("dockerignore") or payload.get("docker_ignore"),
            notes=payload.get("notes"),
            written_files=payload.get("written_files") if isinstance(payload.get("written_files"), Mapping) else None,
        )

