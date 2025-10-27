# migration/resource_migrator.py
"""Adapt configuration and resource files for the target platform."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Dict, Mapping

TEXT_EXT = {".yml", ".yaml", ".json", ".xml", ".sql", ".ini", ".env", ".properties"}
BIN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".db", ".sqlite", ".bin", ".pdf"}


DEFAULT_RESOURCE_PROMPTS: Dict[str, str] = {
    "spring": """
Adapt the following resource to Spring Boot 3.x conventions.
Keep all values, adjust only keys or format if needed.
<<<<<<< HEAD
Return ONLY the adapted content.
=======
Return ONLY the adapted content with no commentary or markdown fences.
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq

--- RESOURCE ({filename}) ---
{content}
""",
    "s4hana": """
Review the following SAP configuration artefact and convert it for an S/4HANA
landscape. Highlight required CDS artifacts, rename deprecated fields and keep
the original semantics intact.
<<<<<<< HEAD
Return ONLY the adapted configuration.
=======
Return ONLY the adapted configuration with no commentary or markdown fences.
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq

--- RESOURCE ({filename}) ---
{content}
""",
    "react": """
Migrate the following front-end configuration or asset for a modern React
tooling stack (Vite + TypeScript + ESLint). Preserve the behaviour while using
current best practices.
<<<<<<< HEAD
Return ONLY the adapted content.
=======
Return ONLY the adapted content with no commentary or markdown fences.
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq

--- RESOURCE ({filename}) ---
{content}
""",
}


class ResourceMigrator:
    def __init__(
        self,
        llm,
        cache,
        resource_prompts: Mapping[str, str] | None = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.resource_prompts: Dict[str, str] = dict(
            resource_prompts or DEFAULT_RESOURCE_PROMPTS
        )

    def process(self, src: Path, dst_dir: Path, profile: str = "spring") -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        target = dst_dir / src.name
        ext = src.suffix.lower()

        if ext in BIN_EXT:
            shutil.copy(src, target)
            return target

        template = self.resource_prompts.get(profile, self.resource_prompts["spring"])
        content = src.read_text(encoding="utf-8", errors="ignore")
        prompt = template.format(filename=src.name, content=content)
        cache_key = self.llm.prompt_hash("adapt", f"{profile}::{prompt}")
        cached = self.cache.get(cache_key)
        if cached is not None:
<<<<<<< HEAD
            target.write_text(cached, encoding="utf-8")
            return target

        out = self.llm.invoke("adapt", prompt, max_new_tokens=1024)
        self.cache.set(cache_key, out)
        target.write_text(out, encoding="utf-8")
=======
            cleaned = self._clean_generation(cached)
            target.write_text(cleaned, encoding="utf-8")
            return target

        out = self.llm.invoke("adapt", prompt, max_new_tokens=1024)
        cleaned = self._clean_generation(out)
        self.cache.set(cache_key, cleaned)
        target.write_text(cleaned, encoding="utf-8")
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq
        return target

    @staticmethod
    def _clean_generation(text: str) -> str:
        """Remove markdown fences and boilerplate from an LLM response."""

        if text is None:
            return ""

        stripped = text.strip()
        fence_blocks = re.findall(r"```(?:[\w+-]*)\n([\s\S]*?)\n```", stripped)
        if fence_blocks:
            stripped = max(fence_blocks, key=len).strip()

        stripped = re.sub(r"^```(?:[\w+-]*)\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = re.sub(
            r"^(?:Here is the (?:updated|translated) (?:file|code)|Updated code|Output|Result)\s*:?\s*",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        return stripped.strip()
