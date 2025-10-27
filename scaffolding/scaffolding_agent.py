# scaffolding/scaffolding_agent.py
import textwrap
import json
import logging
from pathlib import Path

class ScaffoldingAgent:
    def __init__(self, llm, cache):
        self.llm = llm
        self.cache = cache

    def generate(self, output_path: Path, project_name: str, framework: str, language: str):
        """Genera la struttura del progetto via LLM e scrive i file scaffolding."""
        prompt = textwrap.dedent(f"""
        You are a project initializer.
        Generate a minimal but functional base structure for a new {language} project using {framework}.
        Return a JSON array, where each element has:
        - "path": relative file path
        - "content": file contents
        Example:
        [
          {{"path": "pom.xml", "content": "<project>...</project>"}},
          {{"path": "src/main/java/com/example/App.java", "content": "public class App {{...}}"}}
        ]
        Respond ONLY with valid JSON.
        """)

        key = self.llm.prompt_hash("scaffold", f"{framework}:{language}")
        cached = self.cache.get(key)
        if cached:
            logging.info("Loaded scaffold from cache")
            files = cached
        else:
            resp = self.llm.invoke("scaffold", prompt, max_new_tokens=2048)
            import re
            m = re.search(r"\[.*\]", resp, re.DOTALL)
            if not m:
                raise ValueError("No JSON scaffold found in response")
            files = json.loads(m.group(0))
            self.cache.set(key, files)

        project_root = output_path / project_name
        for f in files:
            file_path = project_root / f["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f["content"], encoding="utf-8")

        logging.info(f"Generated scaffold for {framework} in {project_root}")
        return project_root
