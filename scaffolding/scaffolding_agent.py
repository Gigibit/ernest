# scaffolding/scaffolding_agent.py
"""High level agent that generates project scaffolds via LLM prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, Mapping


_SERVICE_HINTS = (
    "Include build tooling aligned with the chosen language/runtime",
    "Expose configuration templates with environment placeholders",
    "Provide containerisation or deployment descriptors for reproducibility",
)

_COMPONENT_UI_HINTS = (
    "Set up a modern bundler with TypeScript support",
    "Configure linting and formatting with opinionated defaults",
    "Ship a lightweight unit testing scaffold",
)

_ENTERPRISE_CORE_HINTS = (
    "Definisci una struttura a moduli che separi estensioni dal nucleo standard",
    "Includi script di automazione per build e delivery controllata",
    "Documenta prerequisiti e dipendenze esterne richieste",
)

DEFAULT_FRAMEWORK_HINTS: Dict[str, Iterable[str]] = {
    "modern service stack": _SERVICE_HINTS,
    "component ui stack": _COMPONENT_UI_HINTS,
    "enterprise core modernisation": _ENTERPRISE_CORE_HINTS,
}


class ScaffoldingAgent:
    def __init__(
        self,
        llm,
        cache,
        framework_hints: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.framework_hints: Dict[str, Iterable[str]] = {
            k.lower(): tuple(v)
            for k, v in (framework_hints or DEFAULT_FRAMEWORK_HINTS).items()
        }

    def _build_prompt(self, framework: str, language: str) -> str:
        normalized = framework.lower()
        hints = self.framework_hints.get(normalized)
        if hints is None:
            keyword_map = {
                "modern service stack": ("service", "backend", "api"),
                "component ui stack": ("component", "frontend", "ui"),
                "enterprise core modernisation": ("enterprise", "core", "erp"),
            }
            for key, keywords in keyword_map.items():
                if any(word in normalized for word in keywords):
                    hints = self.framework_hints.get(key)
                    if hints:
                        break
        hints = hints or ()
        hints_block = "".join(f"- {hint}\n" for hint in hints)
        return textwrap.dedent(
            f"""
            You are a project initializer.
            Generate a minimal but functional base structure for a new {language} project using {framework}.
            {"Incorporate the following hints:" if hints else ""}
            {hints_block if hints else ""}
            Return a JSON array, where each element has:
            - "path": relative file path
            - "content": file contents
            Example:
            [
              {{"path": "pom.xml", "content": "<project>...</project>"}},
              {{"path": "src/main/java/com/example/App.java", "content": "public class App {{...}}"}}
            ]
            Respond ONLY with valid JSON.
            """
        ).strip()

    def generate(
        self,
        output_path: Path,
        project_name: str,
        framework: str,
        language: str,
    ) -> Path:
        """Genera la struttura del progetto via LLM e scrive i file scaffolding."""

        prompt = self._build_prompt(framework, language)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_fingerprint = f"{framework}:{language}:{digest}"
        key = self.llm.prompt_hash("scaffold", cache_fingerprint)
        cached = self.cache.get(key)
        if cached is not None:
            logging.info("Loaded scaffold for %s from cache", framework)
            files = cached
        else:
            resp = self.llm.invoke("scaffold", prompt, max_new_tokens=2048)
            match = re.search(r"\[.*\]", resp, re.DOTALL)
            if not match:
                raise ValueError("No JSON scaffold found in response")
            files = json.loads(match.group(0))
            self.cache.set(key, files)

        project_root = output_path / project_name
        for file_spec in files:
            file_path = project_root / file_spec["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_spec["content"], encoding="utf-8")

        logging.info("Generated scaffold for %s in %s", framework, project_root)
        return project_root
