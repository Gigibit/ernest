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
        logging.info(
            "Heuristic gather discovered %d files in %s", len(files), self.project_path
        )
        return files

    def classify_files(self, sample_count: int = 50) -> Dict[str, List[str]]:
        all_files = self.gather_files()
        sample = all_files[:sample_count]
        logging.info(
            "Classifying %d files (sample size %d) for heuristic analysis",
            len(all_files),
            len(sample),
        )
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
            augmented = self._augment_classification(cached, all_files)
            if augmented != cached:
                self.cache.set(key, augmented)
            return augmented
        logging.info("Requesting classification from LLM")
        resp = self.llm.invoke('classify', prompt, max_new_tokens=512)
        try:
            obj = self._extract_json(resp)
            logging.info(
                "Classifier identified %d source and %d resource files",
                len(obj.get("source", []) or []),
                len(obj.get("resource", []) or []),
            )
        except ValueError:
            logging.warning("Classifier returned malformed payload, using fallback heuristics")
            obj = self._fallback_classification(all_files)
        obj = self._augment_classification(obj, all_files)
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
        logging.info("Requesting stack detection from LLM")
        resp = self.llm.invoke('analyze', prompt, max_new_tokens=200)
        try:
            obj = self._extract_json(resp)
            logging.info(
                "Stack detection result: language=%s framework=%s",
                obj.get("language"),
                obj.get("framework"),
            )
        except ValueError:
            logging.warning("Stack detector returned malformed payload, defaulting to null stack")
            obj = {"language": None, "framework": None}
        self.cache.set(key, obj)
        return obj

    def _extract_json(self, text: str) -> dict:
        import re

        if not text:
            raise ValueError("No JSON found in LLM response")

        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            parts = stripped.split("```")
            stripped = "\n".join(part for part in parts if part.strip() and not part.strip().startswith("json"))

        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise ValueError("No JSON found in LLM response")

        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            logging.debug("Failed to parse JSON payload: %s", exc)
            raise ValueError("No JSON found in LLM response") from exc

    def _augment_classification(
        self, obj: Dict[str, List[str]], files: List[str]
    ) -> Dict[str, List[str]]:
        """Ensure basic language heuristics are represented in the classification."""
        fallback = self._fallback_classification(files)
        merged: Dict[str, List[str]] = {}
        for bucket in ("source", "resource", "other"):
            combined: List[str] = []
            seen = set()
            for name in (obj.get(bucket) or []) + fallback.get(bucket, []):
                if name not in seen and name in files:
                    combined.append(name)
                    seen.add(name)
            merged[bucket] = combined
        return merged

    def _fallback_classification(self, files: List[str]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {"source": [], "resource": [], "other": []}
        source_ext = {
            ".py",
            ".java",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".c",
            ".cc",
            ".cpp",
            ".cs",
            ".rb",
            ".go",
            ".rs",
            ".php",
            ".scala",
            ".kt",
            ".kts",
            ".swift",
            ".cbl",
            ".cob",
            ".cobol",
        }
        resource_ext = {
            ".json",
            ".xml",
            ".yml",
            ".yaml",
            ".ini",
            ".cfg",
            ".env",
            ".properties",
            ".sql",
            ".csv",
            ".md",
            ".txt",
            ".html",
            ".htm",
            ".css",
        }

        for file_path in files:
            suffix = Path(file_path).suffix.lower()
            if suffix in source_ext:
                result["source"].append(file_path)
            elif suffix in resource_ext or not suffix and is_text_file(self.project_path / file_path):
                result["resource"].append(file_path)
            else:
                result["other"].append(file_path)

        return result
