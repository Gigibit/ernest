# migration/resource_migrator.py
from pathlib import Path
import shutil

TEXT_EXT = {".yml", ".yaml", ".json", ".xml", ".sql", ".ini", ".env", ".properties"}
BIN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".db", ".sqlite", ".bin", ".pdf"}

class ResourceMigrator:
    def __init__(self, llm, cache):
        self.llm, self.cache = llm, cache

    def process(self, src: Path, dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)
        target = dst_dir / src.name
        ext = src.suffix.lower()

        if ext in BIN_EXT:
            shutil.copy(src, target)
            return target

        content = src.read_text(encoding="utf-8", errors="ignore")
        prompt = f"""
Adapt the following resource to Spring Boot 3.x conventions.
Keep all values, adjust only keys or format if needed.
Return ONLY adapted content.

--- RESOURCE ({src.name}) ---
{content}
"""
        key = self.llm.prompt_hash("adapt", prompt)
        cached = self.cache.get(key)
        if cached:
            target.write_text(cached, encoding="utf-8"); return target

        out = self.llm.invoke("adapt", prompt, max_new_tokens=1024)
        self.cache.set(key, out)
        target.write_text(out, encoding="utf-8")
        return target
