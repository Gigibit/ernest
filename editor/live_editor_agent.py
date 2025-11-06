"""AI-assisted live editor that applies prompts to generated projects."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.llm_service import LLMService

LOGGER = logging.getLogger(__name__)


@dataclass
class _FileContext:
    path: str
    size: int
    snippet: str
    previewable: bool


class LiveEditorAgent:
    """Use the migration LLM stack to apply modification prompts."""

    def __init__(
        self,
        llm: LLMService,
        *,
        max_context_files: int = 40,
        max_snippet_chars: int = 600,
        max_snippet_bytes: int = 4096,
    ) -> None:
        self.llm = llm
        self.max_context_files = max_context_files
        self.max_snippet_chars = max_snippet_chars
        self.max_snippet_bytes = max_snippet_bytes

    def apply_prompt(
        self,
        project_root: Path,
        prompt: str,
        *,
        focus_path: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply the requested modification prompt to ``project_root``."""

        sanitized = (prompt or "").strip()
        if not sanitized:
            raise ValueError("A non-empty prompt is required")

        root = project_root.resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError("Project output directory is missing")

        context, truncated = self._gather_context(root)
        llm_prompt = self._build_prompt(
            sanitized,
            context,
            truncated=truncated,
            focus_path=focus_path,
            project_name=project_name,
        )
        LOGGER.debug("Live editor prompt: %s", llm_prompt)

        response = self.llm.invoke("editor", llm_prompt, max_new_tokens=1500)
        parsed = self._parse_response(response)

        changes = self._normalise_changes(parsed.get("changes") or parsed.get("updates") or [])
        applied: List[Dict[str, Any]] = []
        touched_for_refresh: List[str] = []

        for change in changes:
            action, target, content = change["action"], change["path"], change.get("content")
            resolved, relative_path = self._resolve_path(root, target)

            if action == "write":
                if content is None:
                    raise ValueError(f"Missing content for update to {relative_path}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
                applied.append({"path": relative_path, "operation": "write"})
                touched_for_refresh.append(relative_path)
            elif action == "append":
                if content is None:
                    raise ValueError(f"Missing content for append to {relative_path}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with resolved.open("a", encoding="utf-8") as handle:
                    handle.write(content)
                applied.append({"path": relative_path, "operation": "append"})
                touched_for_refresh.append(relative_path)
            elif action == "delete":
                if resolved.exists():
                    resolved.unlink()
                applied.append({"path": relative_path, "operation": "delete"})
                touched_for_refresh.append(relative_path)
            else:
                raise ValueError(f"Unsupported change action: {action}")

        summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else None
        messages = parsed.get("messages") if isinstance(parsed.get("messages"), list) else []

        return {
            "applied": bool(applied),
            "updates": applied,
            "summary": summary,
            "messages": messages,
            "refreshed_paths": touched_for_refresh,
            "raw_response": response,
        }

    def _gather_context(self, root: Path) -> Tuple[List[_FileContext], bool]:
        entries: List[_FileContext] = []
        truncated = False
        try:
            candidates = sorted(root.rglob("*"), key=lambda item: item.as_posix())
        except OSError:
            return entries, truncated

        for entry in candidates:
            if not entry.is_file():
                continue
            if len(entries) >= self.max_context_files:
                truncated = True
                break

            try:
                stat = entry.stat()
            except OSError:
                continue

            rel_path = entry.relative_to(root).as_posix()
            previewable, snippet = self._extract_snippet(entry)
            entries.append(
                _FileContext(
                    path=rel_path,
                    size=int(stat.st_size),
                    snippet=snippet,
                    previewable=previewable,
                )
            )
        return entries, truncated

    def _extract_snippet(self, candidate: Path) -> Tuple[bool, str]:
        try:
            with candidate.open("rb") as handle:
                sample = handle.read(self.max_snippet_bytes)
        except OSError:
            return False, ""

        if b"\x00" in sample:
            return False, ""

        try:
            snippet = sample.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return False, ""

        snippet = snippet[: self.max_snippet_chars].strip()
        return True, snippet

    def _build_prompt(
        self,
        user_prompt: str,
        context: Iterable[_FileContext],
        *,
        truncated: bool,
        focus_path: Optional[str],
        project_name: Optional[str],
    ) -> str:
        lines = [
            "You are Ernest's live editor assistant.",
            "The user will request modifications to a generated migration project.",
            "Return valid JSON describing the changes to apply.",
            "Use newline characters (\\n) for line breaks.",
        ]
        if project_name:
            lines.append(f"Project: {project_name}")
        if focus_path:
            lines.append(f"Focus file: {focus_path}")
        lines.append("\nKnown project files (truncated list):")
        for entry in context:
            descriptor = f"- {entry.path} ({entry.size} bytes)"
            if not entry.previewable:
                descriptor += " [binary preview unavailable]"
            lines.append(descriptor)
            if entry.snippet:
                lines.append("  snippet: \"" + entry.snippet.replace("\n", "\\n") + "\"")
        if truncated:
            lines.append("- … additional files omitted …")

        lines.extend(
            [
                "\nRespond with JSON following this schema:",
                "{",
                "  \"summary\": string (short sentence describing the result),",
                "  \"changes\": [",
                "    {",
                "      \"path\": relative file path (string),",
                "      \"operation\": one of [replace, overwrite, append, delete, create],",
                "      \"content\": new file content when operation requires it (string)",
                "    }",
                "  ],",
                "  \"messages\": optional list of human readable notes",
                "}",
                "Do not include any text outside of the JSON object.",
                "User prompt:",
                user_prompt,
            ]
        )
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        text = (response or "").strip()
        if not text:
            raise ValueError("LLM response was empty")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError("LLM response was not valid JSON") from exc
            raise ValueError("LLM response was not valid JSON")

    def _normalise_changes(self, changes: Any) -> List[Dict[str, Any]]:
        if not isinstance(changes, list):
            raise ValueError("LLM response did not include a change list")
        normalised: List[Dict[str, Any]] = []
        for index, entry in enumerate(changes):
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid change entry at index {index}: {entry!r}")
            path = entry.get("path") or entry.get("file")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Invalid or missing path at index {index}")
            operation = entry.get("operation") or entry.get("action")
            action = self._normalise_action(operation)
            normalised.append({
                "path": path.strip(),
                "action": action,
                "content": entry.get("content"),
            })
        return normalised

    def _normalise_action(self, operation: Any) -> str:
        token = (operation or "write").strip().lower()
        if token in {"replace", "overwrite", "write", "update", "upsert", "create"}:
            return "write"
        if token in {"append", "extend"}:
            return "append"
        if token in {"delete", "remove"}:
            return "delete"
        raise ValueError(f"Unsupported operation requested: {operation}")

    def _resolve_path(self, root: Path, requested: str) -> Tuple[Path, str]:
        candidate = (root / requested).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Illegal path requested: {requested}") from exc
        return candidate, relative.as_posix()
