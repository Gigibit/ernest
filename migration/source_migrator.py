# migration/source_migrator.py
"""High level orchestration for source code translation tasks."""

from __future__ import annotations

import logging
<<<<<<< HEAD
=======
import re
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

from .strategies import DEFAULT_STRATEGIES, TranslationStrategy


class SourceMigrator:
    """Translate source artefacts by delegating to specialised strategies."""

    def __init__(
        self,
        llm,
        cache,
        recovery,
        strategies: Mapping[str, TranslationStrategy] | None = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.recovery = recovery
        self.strategies: Dict[str, TranslationStrategy] = dict(
            strategies or DEFAULT_STRATEGIES
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_strategy(self, key: str, strategy: TranslationStrategy) -> None:
        """Register or override a translation strategy at runtime."""

        self.strategies[key] = strategy

    def translate(
        self,
        strategy_key: str,
        src: Path,
        dst: Path,
        **context,
    ) -> Path:
        """Run the translation pipeline using ``strategy_key``.

        Parameters
        ----------
        strategy_key:
            Identifier of the strategy to use.  See
            :data:`migration.strategies.DEFAULT_STRATEGIES` for the defaults.
        src:
            Path to the source file to migrate.
        dst:
            Destination path where the translated artefact will be written.
        context:
            Optional metadata passed to the prompt (e.g. DATA DIVISION for COBOL,
            Django settings, Angular services).  The ``llm_overrides`` key can be
            used to pass keyword arguments to :meth:`core.llm_service.LLMService.invoke`.
        """

        if strategy_key not in self.strategies:
            raise KeyError(f"Unknown translation strategy: {strategy_key}")

        strategy = self.strategies[strategy_key]
        llm_overrides = dict(context.pop("llm_overrides", {}))
        llm_overrides.setdefault("max_new_tokens", strategy.max_new_tokens)

        try:
            chunks = self._chunk_source(src, strategy.chunk_size)
            translated_parts = self._translate_chunks(
                strategy, chunks, context, llm_overrides
            )

            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("\n\n".join(translated_parts), encoding="utf-8")
            if self.recovery:
                self.recovery.mark_completed(str(src))
            logging.info("Translated %s -> %s using %s", src, dst, strategy.name)
            return dst
        except Exception as exc:  # noqa: BLE001 - we want full trace for recovery
            if self.recovery:
                self.recovery.mark_failed(str(src))
            logging.error("Translation failed for %s: %s", src, exc, exc_info=True)
            raise

    # Convenience wrappers -------------------------------------------------
    def translate_cobol(
        self,
        src: Path,
        dst: Path,
        data_division: str = "",
        fd_summary: str = "",
    ) -> Path:
        return self.translate(
            "cobol_to_java",
            src,
            dst,
            data_division=data_division,
            fd_summary=fd_summary,
        )

    def translate_python_module(
        self,
        src: Path,
        dst: Path,
        project_settings: str | None = None,
    ) -> Path:
        return self.translate(
            "python_to_spring",
            src,
            dst,
            project_settings=project_settings or "",
        )

    def translate_angular_component(
        self,
        src: Path,
        dst: Path,
        shared_services: str | None = None,
    ) -> Path:
        return self.translate(
            "angularjs_to_react",
            src,
            dst,
            shared_services=shared_services or "",
        )

    def translate_abap(
        self,
        src: Path,
        dst: Path,
        ddic_metadata: str | None = None,
        integration_notes: str | None = None,
    ) -> Path:
        return self.translate(
            "abap_to_s4",
            src,
            dst,
            ddic_metadata=ddic_metadata or "",
            integration_notes=integration_notes or "",
        )

    # Internal helpers -----------------------------------------------------
    def _chunk_source(self, src: Path, chunk_size: int) -> Iterable[str]:
        """Split ``src`` into textual chunks limited by ``chunk_size`` lines."""

        content = src.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        if not lines:
            return []
        return [
            "\n".join(lines[i : i + chunk_size])
            for i in range(0, len(lines), chunk_size)
        ]

    def _translate_chunks(
        self,
        strategy: TranslationStrategy,
        chunks: Iterable[str],
        context: Mapping[str, str],
        llm_overrides: MutableMapping[str, object],
    ) -> Iterable[str]:
        translated_parts = []
        for idx, chunk in enumerate(chunks):
            prompt = strategy.build_prompt(chunk, idx, context)
            cache_key = self.llm.prompt_hash(strategy.profile, prompt)
            cached = self.cache.get(cache_key)
            if cached is not None:
<<<<<<< HEAD
                translated_parts.append(cached)
                continue

            response = self.llm.invoke(strategy.profile, prompt, **llm_overrides)
            self.cache.set(cache_key, response)
            translated_parts.append(response)

        return translated_parts
=======
                translated_parts.append(self._clean_generation(cached))
                continue

            response = self.llm.invoke(strategy.profile, prompt, **llm_overrides)
            cleaned = self._clean_generation(response)
            self.cache.set(cache_key, cleaned)
            translated_parts.append(cleaned)

        return translated_parts

    @staticmethod
    def _clean_generation(text: str) -> str:
        """Normalise LLM output by removing fences and boilerplate."""

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
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq

