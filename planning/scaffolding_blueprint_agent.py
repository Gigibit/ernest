"""LLM-backed scaffolding blueprint generator."""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from core.cache_manager import CacheManager
from core.file_utils import read_head
from core.llm_service import LLMService


class ScaffoldingBlueprintAgent:
    """Prepare stubs/contract guidance for the destination project."""

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
    def blueprint(
        self,
        sources: Sequence[str],
        architecture_map: Mapping[str, Mapping[str, str]],
        *,
        target_language: str,
        target_framework: str,
        dependencies: Sequence[Mapping[str, Any]] | None = None,
        sample_size: int = 15,
    ) -> Dict[str, Any]:
        if not sources:
            return {"entries": [], "notes": "No source files discovered."}

        ordered_sources = list(dict.fromkeys(sources))
        sample = ordered_sources[:sample_size]
        prompt = self._build_prompt(
            sample,
            architecture_map=architecture_map,
            target_language=target_language,
            target_framework=target_framework,
            dependencies=dependencies or [],
        )
        cache_key_payload = {
            "sources": sample,
            "target_language": target_language,
            "target_framework": target_framework,
            "architecture": list(sorted(architecture_map.keys())),
        }
        cache_key = self.llm.prompt_hash(
            "blueprint", json.dumps(cache_key_payload, sort_keys=True)
        )
        cached = self.cache.get(cache_key)
        if isinstance(cached, Mapping):
            logging.info("Loaded scaffolding blueprint from cache")
            return self._merge_with_fallback(
                cached, ordered_sources, architecture_map, target_language
            )

        logging.info(
            "Requesting scaffolding blueprint for %d sources targeting %s/%s",
            len(sample),
            target_language,
            target_framework,
        )
        response = self.llm.invoke("blueprint", prompt, max_new_tokens=900)
        plan = self._parse_response(response)
        self.cache.set(cache_key, plan)
        return self._merge_with_fallback(
            plan, ordered_sources, architecture_map, target_language
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        sample: Sequence[str],
        *,
        architecture_map: Mapping[str, Mapping[str, str]],
        target_language: str,
        target_framework: str,
        dependencies: Sequence[Mapping[str, Any]],
    ) -> str:
        snippets: list[str] = []
        for source in sample:
            snippet = read_head(self.project_root / source, max_chars=1600).strip()
            header = f"### {source}"
            if snippet:
                snippets.append(f"{header}\n{snippet}")
            else:
                snippets.append(f"{header}\n<no preview available>")

        arch_lines: list[str] = []
        for key in sample:
            info = architecture_map.get(key, {})
            if info:
                arch_lines.append(
                    f"- {key} -> {info.get('target_path')} (package={info.get('package')})"
                )
        dependencies_block = "\n".join(
            f"- {dep.get('name')} ({dep.get('version') or 'unspecified'})"
            for dep in dependencies[:20]
            if isinstance(dep, Mapping)
        )

        arch_block = "\n".join(arch_lines) if arch_lines else "<no existing architecture hints>"
        deps_block = dependencies_block or "<no dependency manifests detected>"

        snippets_block = "\n\n".join(snippets)

        return textwrap.dedent(
            f"""
            You help migration teams identify destination scaffolding when a direct
            translation cannot provide working code immediately.

            Review the legacy source excerpts, the provisional architecture mapping,
            and the known third-party dependencies. Suggest which destination modules
            must be created as TODO scaffolding so developers can later implement the
            required integrations.

            Respond ONLY with JSON object using this structure:
            {{
              "entries": [
                {{
                  "source": "relative/path.py",
                  "class_name": "TargetClass",
                  "target_path": "src/main/java/com/example/TargetClass.java",
                  "package": "com.example",
                  "requires_stub": true,
                  "reason": "short justification of why manual implementation is required",
                  "expected_contract": "describe parameters/return types to honour",
                  "notes": "any guidance for the developer"
                }}
              ],
              "notes": "project level comments"
            }}

            Keep entries concise, reference only provided sources, and prefer the
            supplied architecture mapping paths. If an item does not need a stub,
            set "requires_stub" to false but still include the reasoning.

            --- TARGET CONTEXT ---
            language: {target_language}
            framework: {target_framework}

            --- ARCHITECTURE MAP ---
            {arch_block}

            --- DEPENDENCIES ---
            {deps_block}

            --- SOURCE EXCERPTS ---
            {snippets_block}
            """
        ).strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        if not response:
            return {}
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
        except ValueError:
            logging.warning("Blueprint response missing JSON payload")
            return {}
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            logging.exception("Failed to parse blueprint JSON payload")
            return {}

    def _merge_with_fallback(
        self,
        payload: Mapping[str, Any],
        sources: Sequence[str],
        architecture_map: Mapping[str, Mapping[str, str]],
        target_language: str,
    ) -> Dict[str, Any]:
        entries = []
        existing = set()
        supplied_entries = payload.get("entries") if isinstance(payload, Mapping) else None
        if isinstance(supplied_entries, Iterable):
            for entry in supplied_entries:
                if not isinstance(entry, Mapping):
                    continue
                source = entry.get("source")
                if not source or source in existing:
                    continue
                existing.add(source)
                entries.append(dict(entry))

        for source in sources:
            if source in existing:
                continue
            arch = architecture_map.get(source, {})
            entries.append(
                {
                    "source": source,
                    "class_name": Path(source).stem.title().replace("_", ""),
                    "target_path": arch.get("target_path")
                    or f"src/{Path(source).with_suffix(self._fallback_extension(target_language)).name}",
                    "package": arch.get("package"),
                    "requires_stub": True,
                    "reason": "Automatic translation unavailable; manual port required.",
                    "expected_contract": "Review legacy module behaviour and provide equivalent in target stack.",
                    "notes": "Generated via fallback blueprint.",
                }
            )

        notes = payload.get("notes") if isinstance(payload, Mapping) else None
        if not isinstance(notes, str) or not notes.strip():
            notes = "Review generated scaffolding entries to ensure coverage for critical behaviours."

        return {"entries": entries, "notes": notes}

    @staticmethod
    def _fallback_extension(target_language: str) -> str:
        normalized = (target_language or "").lower()
        if normalized in {"java", "kotlin", "scala"}:
            return ".java"
        if normalized in {"c#", "dotnet", "csharp"}:
            return ".cs"
        if normalized in {"typescript", "javascript"}:
            return ".ts"
        if normalized in {"go"}:
            return ".go"
        if normalized in {"rust"}:
            return ".rs"
        return ".txt"
