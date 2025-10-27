# migration/resource_migrator.py
"""Adapt configuration and resource files for the target platform."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, Mapping

TEXT_EXT = {".yml", ".yaml", ".json", ".xml", ".sql", ".ini", ".env", ".properties"}
BIN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".db", ".sqlite", ".bin", ".pdf"}


DEFAULT_RESOURCE_PROMPTS: Dict[str, str] = {
    "spring": """
Adapt the following resource to Spring Boot 3.x conventions.
Keep all values, adjust only keys or format if needed.
Return ONLY the adapted content.

--- RESOURCE ({filename}) ---
{content}
""",
    "s4hana": """
Review the following SAP configuration artefact and convert it for an S/4HANA
landscape. Highlight required CDS artifacts, rename deprecated fields and keep
the original semantics intact.
Return ONLY the adapted configuration.

--- RESOURCE ({filename}) ---
{content}
""",
    "react": """
Migrate the following front-end configuration or asset for a modern React
tooling stack (Vite + TypeScript + ESLint). Preserve the behaviour while using
current best practices.
Return ONLY the adapted content.

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
            target.write_text(cached, encoding="utf-8")
            return target

        out = self.llm.invoke("adapt", prompt, max_new_tokens=1024)
        self.cache.set(cache_key, out)
        target.write_text(out, encoding="utf-8")
        return target
