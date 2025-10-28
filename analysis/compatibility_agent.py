"""Search for target-language friendly alternatives to legacy imports."""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from core.cache_manager import CacheManager
from core.llm_service import LLMService

try:  # pragma: no cover - optional dependency
    from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
except Exception:  # noqa: BLE001
    DuckDuckGoSearchRun = None


@dataclass
class AlternativeEntry:
    """Single alternative recommendation entry."""

    name: str
    category: str
    recommended: List[str]
    actions: str
    confidence: Optional[str] = None
    research: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "category": self.category,
            "recommended": self.recommended,
            "actions": self.actions,
        }
        if self.confidence:
            payload["confidence"] = self.confidence
        if self.research:
            payload["research"] = self.research
        return payload


class CompatibilitySearchAgent:
    """Derive alternative libraries and APIs suited for the target stack."""

    def __init__(
        self,
        project_root: Path,
        llm: LLMService,
        cache: CacheManager,
        *,
        profile: str = "compatibility",
        search_limit: int = 5,
    ) -> None:
        self.project_root = Path(project_root)
        self.llm = llm
        self.cache = cache
        self.profile = profile
        self.search_limit = search_limit
        self._search_tool = None
        if DuckDuckGoSearchRun is not None:
            try:
                self._search_tool = DuckDuckGoSearchRun()
            except Exception as exc:  # noqa: BLE001
                logging.debug("Unable to initialise DuckDuckGo search tool: %s", exc)
                self._search_tool = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def suggest(
        self,
        source_files: Sequence[str],
        dependencies: Sequence[Mapping[str, object]],
        *,
        target_language: str,
        target_framework: str,
    ) -> Dict[str, object]:
        """Return alternative libraries/methods suited for the target stack."""

        candidates = self._collect_candidates(source_files, dependencies)
        if not candidates:
            return {
                "entries": [],
                "notes": "No imports or dependencies detected that require translation.",
            }

        search_notes = self._perform_search(candidates, target_language, target_framework)

        prompt = self._build_prompt(
            candidates,
            target_language=target_language,
            target_framework=target_framework,
            search_notes=search_notes,
        )
        cache_key = self.llm.prompt_hash(self.profile, prompt)
        cached = self.cache.get(cache_key)
        if isinstance(cached, Mapping):
            return dict(cached)

        logging.info(
            "Requesting compatibility suggestions for %d imports/dependencies",
            len(candidates),
        )
        response = self.llm.invoke(self.profile, prompt, max_new_tokens=900)
        parsed = self._extract_json(response)
        if not parsed:
            parsed = self._fallback_payload(candidates, target_language)
        self.cache.set(cache_key, parsed)
        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_candidates(
        self,
        source_files: Sequence[str],
        dependencies: Sequence[Mapping[str, object]],
        limit: int = 20,
    ) -> List[str]:
        seen: Set[str] = set()
        results: List[str] = []

        for dependency in dependencies:
            name = dependency.get("name") if isinstance(dependency, Mapping) else None
            if isinstance(name, str):
                normalized = name.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    results.append(normalized)
                    if len(results) >= limit:
                        return results

        for rel_path in source_files:
            if len(results) >= limit:
                break
            try:
                path = (self.project_root / rel_path).resolve()
                path.relative_to(self.project_root.resolve())
            except Exception:  # noqa: BLE001
                continue
            if not path.is_file():
                continue
            imports = self._read_imports(path, limit=max(0, limit - len(results)))
            for item in imports:
                if item not in seen:
                    seen.add(item)
                    results.append(item)
                    if len(results) >= limit:
                        break

        return results

    def _read_imports(self, path: Path, *, limit: int) -> List[str]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                snippet = handle.read(8000)
        except Exception:  # noqa: BLE001
            return []

        imports: List[str] = []
        for line in snippet.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            token = self._parse_import_line(stripped)
            if token:
                imports.append(token)
                if len(imports) >= limit:
                    break
        return imports

    @staticmethod
    def _parse_import_line(line: str) -> Optional[str]:
        lowered = line.lower()
        if lowered.startswith("import "):
            return line.split()[1].rstrip(";{}").strip()
        if lowered.startswith("from ") and " import " in lowered:
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].strip()
        if lowered.startswith("using "):
            return line.split()[1].rstrip(";{}").strip()
        if lowered.startswith("#include"):
            return line.split(maxsplit=1)[-1].strip("<>\" ")
        if lowered.startswith("require "):
            return line.split()[1].strip("'\"")
        if lowered.startswith("include "):
            return line.split()[1].strip("'\"<>")
        return None

    def _perform_search(
        self,
        candidates: Iterable[str],
        target_language: str,
        target_framework: str,
    ) -> Dict[str, str]:
        if not self._search_tool:
            return {}

        notes: Dict[str, str] = {}
        for index, name in enumerate(candidates):
            if index >= self.search_limit:
                break
            query = (
                f"{name} {target_language} {target_framework} migration alternative library"
            )
            try:
                result = self._search_tool.run(query)
            except Exception as exc:  # noqa: BLE001
                logging.debug("DuckDuckGo search failed for %s: %s", name, exc)
                continue
            if isinstance(result, str) and result.strip():
                notes[name] = result.strip()
        return notes

    def _build_prompt(
        self,
        candidates: Sequence[str],
        *,
        target_language: str,
        target_framework: str,
        search_notes: Mapping[str, str],
    ) -> str:
        notes_lines = [
            f"- {name}: {snippet}" for name, snippet in search_notes.items()
        ]
        notes_block = (
            "\nRecent web snippets:\n" + "\n".join(notes_lines)
            if notes_lines
            else ""
        )

        prompt = textwrap.dedent(
            """
            You help software teams migrate projects between technology stacks.
            Perform lightweight web research when needed and provide concise guidance.
            Focus on mapping legacy imports, APIs, or dependencies to options idiomatic for the
            requested target language and framework.

            Respond ONLY with JSON using this schema:
            {
              "entries": [
                {
                  "name": "legacy import or dependency",
                  "category": "dependency | api | framework | unknown",
                  "recommended": ["target-side alternative", ...],
                  "actions": "short recommendation text",
                  "confidence": "optional qualitative score",
                  "research": "optional summary of web findings"
                }
              ],
              "notes": "overall guidance for the migration team"
            }

            Target language: {language}
            Target framework: {framework}
            Legacy references to evaluate (JSON array follows):
            {candidates_json}
            {notes}
            """
        ).strip().format(
            language=target_language,
            framework=target_framework,
            candidates_json=json.dumps(list(candidates), ensure_ascii=False, indent=2),
            notes=notes_block,
        )
        return prompt

    @staticmethod
    def _extract_json(text: Optional[str]) -> Dict[str, object]:
        if not text:
            return {}
        import re

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            logging.error("No JSON payload detected in compatibility response")
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logging.exception("Failed to parse compatibility JSON")
            return {}
        if not isinstance(data, Mapping):
            return {}
        entries = []
        for entry in data.get("entries", []):
            if not isinstance(entry, Mapping):
                continue
            alt = AlternativeEntry(
                name=str(entry.get("name", "")),
                category=str(entry.get("category", "unknown")),
                recommended=[
                    str(item)
                    for item in entry.get("recommended", [])
                    if isinstance(item, str)
                ],
                actions=str(entry.get("actions", "")),
                confidence=(
                    str(entry.get("confidence"))
                    if entry.get("confidence") is not None
                    else None
                ),
                research=(
                    str(entry.get("research"))
                    if entry.get("research") is not None
                    else None
                ),
            )
            entries.append(alt.to_dict())
        notes = data.get("notes")
        payload: Dict[str, object] = {"entries": entries}
        if isinstance(notes, str) and notes.strip():
            payload["notes"] = notes.strip()
        return payload

    def _fallback_payload(
        self, candidates: Sequence[str], target_language: str
    ) -> Dict[str, object]:
        entries = [
            AlternativeEntry(
                name=name,
                category="dependency",
                recommended=[
                    f"Research {target_language} equivalent for {name}"
                ],
                actions="Consult target ecosystem documentation for a maintained alternative.",
                confidence="low",
            ).to_dict()
            for name in candidates
        ]
        return {
            "entries": entries,
            "notes": "Fallback guidance generated due to unstructured LLM output. Review manually.",
        }

