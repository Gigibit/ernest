# migration/source_migrator.py
from pathlib import Path
import textwrap
import logging

class SourceMigrator:
    def __init__(self, llm, cache, recovery):
        self.llm, self.cache, self.recovery = llm, cache, recovery

    def translate_cobol(self, src: Path, out: Path, data_div="", fd_summary=""):
        try:
            lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
            chunks = ["\n".join(lines[i:i+300]) for i in range(0, len(lines), 300)]
            parts = []

            for idx, ch in enumerate(chunks):
                prompt = textwrap.dedent(f"""
                Convert the following COBOL code to Java (Spring Boot style).
                Keep semantics, use idiomatic Java 21.
                DATA DIVISION:
                {data_div}
                FILE DESCRIPTORS:
                {fd_summary}
                COBOL CHUNK [{idx}]:
                {ch}
                Return ONLY Java code.
                """)
                key = self.llm.prompt_hash("translate", prompt)
                cached = self.cache.get(key)
                if cached:
                    parts.append(cached)
                    continue
                result = self.llm.invoke("translate", prompt, max_new_tokens=2048)
                self.cache.set(key, result)
                parts.append(result)

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n\n".join(parts), encoding="utf-8")
            self.recovery.mark_completed(str(src))
            logging.info(f"Translated {src} -> {out}")
        except Exception as e:
            self.recovery.mark_failed(str(src))
            logging.error(f"Translation failed for {src}:
