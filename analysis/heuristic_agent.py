# analysis/heuristic_agent.py
import os
from pathlib import Path
import textwrap
import json
import logging
from typing import List, Dict, Optional
from core.file_utils import read_head, is_text_file
from core.cache_manager import CacheManager
from core.llm_service import LLMService

class HeuristicAnalysisAgent:
    def __init__(self, project_path: str, llm: LLMService, cache: CacheManager):
        self.project_path = Path(project_path)
        self.llm = llm
        self.cache = cache

    def gather_files(self, ext_allow: Optional[List[str]] = None) -> List[str]:
        files = []
        for root, _, filenames in os.walk(self.project_path):
            for fn in filenames:
                p = Path(root) / fn
                rel = p.relative_to(self.project_path)
                if ext_allow and p.suffix.lower() not in ext_allow:
                    continue
                files.append(str(rel))
        files.sort()
        return files

    def classify_files(self, sample_count: int = 50) -> Dict[str, List[str]]:
        all_files = self.gather_files()
        sample = all_files[:sample_count]
        context = "\n".join(f"### {f}\n{read_head(self.project_path / f, max_chars=2000)}" for f in sample)
        prompt = textwrap.dedent(f"""
        You are a project file classifier.
        Classify each file path excerpt below into 'source', 'resource', or 'other'.
        Respond ONLY with JSON object:
        {{
          "source": [...],
          "resource": [...],
          "other": [...]
        }}
        --- FILE EXCERPTS ---
        {context}
        """)
        key = self.llm.prompt_hash('classify', context)
        cached = self.cache.get(key)
        if cached:
            logging.info("Loaded classification from cache")
            return cached
        resp = self.llm.invoke('classify', prompt, max_new_tokens=512)
        obj = self._extract_json(resp)
        self.cache.set(key, obj)
        return obj

    def detect_stack(self, sources: Optional[List[str]] = None) -> Optional[Dict[str, Optional[str]]]:
        if not sources:
            sources = self.gather_files()
        context = "\n".join(f"--- {f}\n{read_head(self.project_path / f, max_chars=1000)}"
                             for f in sources[:15])
        if not context.strip():
            logging.error("No content for stack detection")
            return None
        prompt = textwrap.dedent(f"""
        You are a software architect expert.
        Determine the main programming language and primary framework.
        Respond ONLY with JSON:
        {{
          "language": "string | null",
          "framework": "string | null"
        }}
        --- CODE EXCERPTS ---
        {context}
        """)
        key = self.llm.prompt_hash('analyze', context)
        cached = self.cache.get(key)
        if cached:
            logging.info("Loaded stack detection from cache")
            return cached
        resp = self.llm.invoke('analyze', prompt, max_new_tokens=200)
        obj = self._extract_json(resp)
        self.cache.set(key, obj)
        return obj

    def _extract_json(self, text: str) -> dict:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("No JSON found in LLM response")
        return json.loads(m.group(0))
