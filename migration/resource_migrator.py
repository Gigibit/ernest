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
    "backend": """
Reconcile the following resource with the conventions of the destination
service platform. Keep behaviours intact, align naming and structure with the
target runtime, and avoid introducing placeholder content.
Return ONLY the adapted artefact with no commentary or markdown fences.

--- RESOURCE ({filename}) ---
{content}
""",
    "enterprise": """
Review the following enterprise configuration or extension and reshape it so it
fits a modular, cloud-oriented core. Indica eventuali personalizzazioni da
isolare, aggiorna nomenclature obsolete e preserva la semantica originale.
Return ONLY the adapted artefact with no commentary or markdown fences.

--- RESOURCE ({filename}) ---
{content}
""",
    "component_ui": """
Porta il seguente asset front-end verso uno stack component-based moderno.
Allinea build tooling, tipizzazione e convenzioni senza alterare il risultato
visivo o funzionale.
Return ONLY the adapted artefact with no commentary or markdown fences.

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

    def process(self, src: Path, dst_dir: Path, profile: str = "backend") -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        target = dst_dir / src.name
        ext = src.suffix.lower()

        if ext in BIN_EXT:
            shutil.copy(src, target)
            return target

        template = self.resource_prompts.get(profile, self.resource_prompts["backend"])
        content = src.read_text(encoding="utf-8", errors="ignore")
        prompt = template.format(filename=src.name, content=content)
        cache_key = self.llm.prompt_hash("adapt", f"{profile}::{prompt}")
        cached = self.cache.get(cache_key)
        if cached is not None:
            cleaned = self._clean_generation(cached)
            target.write_text(cleaned, encoding="utf-8")
            return target

        out = self.llm.invoke("adapt", prompt, max_new_tokens=1024)
        cleaned = self._clean_generation(out)
        self.cache.set(cache_key, cleaned)
        target.write_text(cleaned, encoding="utf-8")
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
