# migration/source_migrator.py
"""High level orchestration for source code translation tasks."""

from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from .strategies import DEFAULT_STRATEGIES, TranslationStrategy


class SourceMigrator:
    """Translate source artefacts by delegating to specialised strategies."""

    def __init__(
        self,
        llm,
        cache,
        recovery,
        strategies: Mapping[str, TranslationStrategy] | None = None,
        *,
        auto_pagination: Mapping[str, int] | None = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.recovery = recovery
        self.strategies: Dict[str, TranslationStrategy] = dict(
            strategies or DEFAULT_STRATEGIES
        )
        self.auto_pagination: Dict[str, int] = dict(
            auto_pagination or self._load_auto_config()
        )
        self.pagination_log: Dict[str, Dict[str, Any]] = {}

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
        *,
        page_size: int | None = None,
        refine_passes: int = 0,
        safe_mode: bool = True,
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
            Optional metadata passed to the prompt (e.g. legacy structure outlines,
            runtime settings, shared dependencies).  The ``llm_overrides`` key can be
            used to pass keyword arguments to :meth:`core.llm_service.LLMService.invoke`.
        """

        if strategy_key not in self.strategies:
            raise KeyError(f"Unknown translation strategy: {strategy_key}")

        strategy = self.strategies[strategy_key]
        llm_overrides = dict(context.pop("llm_overrides", {}))
        llm_overrides.setdefault("max_new_tokens", strategy.max_new_tokens)

        try:
            logging.info(
                "Preparing translation of %s with strategy %s", src, strategy.name
            )
            chunks = list(self._chunk_source(src, strategy.chunk_size))
            logging.info(
                "Chunked %s into %d segments (size=%d)",
                src,
                len(chunks),
                strategy.chunk_size,
            )
            if not chunks:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text("", encoding="utf-8")
                if self.recovery:
                    self.recovery.mark_completed(str(src))
                logging.info(
                    "Translated %s -> %s using %s (empty source)",
                    src,
                    dst,
                    strategy.name,
                )
                return dst

            project_stats = context.pop("project_stats", None)
            resolved_page_size = self._resolve_page_size(
                strategy,
                src,
                chunks,
                page_size,
                project_stats,
            )

            translated_parts = []
            for page_number, (start, end) in enumerate(
                self._paginate_indices(len(chunks), resolved_page_size)
            ):
                logging.info(
                    "Translating page %d (%d-%d) for %s", page_number + 1, start, end, src
                )
                page_translations = self._translate_page(
                    strategy,
                    chunks,
                    start,
                    end,
                    context,
                    llm_overrides,
                    safe_mode,
                )

                if refine_passes > 0:
                    refined = self._refine_page(
                        strategy,
                        page_translations,
                        page_number,
                        context,
                        llm_overrides,
                        refine_passes,
                    )
                    translated_parts.append(refined)
                else:
                    translated_parts.extend(page_translations)

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
    @staticmethod
    def _format_language_hint(hint: str | None) -> str:
        if not hint:
            return ""
        normalized = hint.strip()
        if not normalized:
            return ""
        if "code" in normalized.lower():
            return normalized
        return f"{normalized} code"

    def translate_legacy_backend(
        self,
        src: Path,
        dst: Path,
        structure_outline: str = "",
        integration_contracts: str = "",
        target_language: str | None = None,
        target_framework: str | None = None,
        target_package: str | None = None,
        architecture_notes: str | None = None,
        *,
        page_size: int | None = None,
        refine_passes: int = 0,
        safe_mode: bool = True,
        project_stats: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.translate(
            "legacy_backend_to_services",
            src,
            dst,
            structure_outline=structure_outline,
            integration_contracts=integration_contracts,
            target_language=self._format_language_hint(target_language),
            target_framework=target_framework or "",
            target_package=target_package or "",
            architecture_notes=architecture_notes or "",
            page_size=page_size,
            refine_passes=refine_passes,
            safe_mode=safe_mode,
            project_stats=project_stats,
        )

    def translate_dynamic_web_module(
        self,
        src: Path,
        dst: Path,
        runtime_configuration: str | None = None,
        *,
        page_size: int | None = None,
        refine_passes: int = 0,
        safe_mode: bool = True,
        project_stats: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.translate(
            "dynamic_web_to_structured_backend",
            src,
            dst,
            runtime_configuration=runtime_configuration or "",
            page_size=page_size,
            refine_passes=refine_passes,
            safe_mode=safe_mode,
            project_stats=project_stats,
        )

    def translate_client_component(
        self,
        src: Path,
        dst: Path,
        shared_dependencies: str | None = None,
        *,
        page_size: int | None = None,
        refine_passes: int = 0,
        safe_mode: bool = True,
        project_stats: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.translate(
            "legacy_frontend_to_component_ui",
            src,
            dst,
            shared_dependencies=shared_dependencies or "",
            page_size=page_size,
            refine_passes=refine_passes,
            safe_mode=safe_mode,
            project_stats=project_stats,
        )

    def translate_enterprise_core(
        self,
        src: Path,
        dst: Path,
        data_model: str | None = None,
        integration_notes: str | None = None,
        *,
        page_size: int | None = None,
        refine_passes: int = 0,
        safe_mode: bool = True,
        project_stats: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.translate(
            "enterprise_core_to_cloud",
            src,
            dst,
            data_model=data_model or "",
            integration_notes=integration_notes or "",
            page_size=page_size,
            refine_passes=refine_passes,
            safe_mode=safe_mode,
            project_stats=project_stats,
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

    def _translate_page(
        self,
        strategy: TranslationStrategy,
        chunks: Sequence[str],
        start: int,
        end: int,
        context: Mapping[str, str],
        llm_overrides: MutableMapping[str, object],
        safe_mode: bool,
    ) -> list[str]:
        translated_parts: list[str] = []
        for idx in range(start, end):
            chunk = chunks[idx]
            logging.debug(
                "Issuing translation prompt for %s chunk %d (fallback=%s)",
                strategy.name,
                idx,
                False,
            )
            translated = self._issue_prompt(
                strategy,
                chunk,
                idx,
                context,
                llm_overrides,
                fallback=False,
            )

            if safe_mode and self._should_retry(chunk, translated, strategy):
                logging.warning(
                    "Fallback translation triggered for chunk %s using strategy %s",
                    idx,
                    strategy.name,
                )
                translated = self._issue_prompt(
                    strategy,
                    chunk,
                    idx,
                    context,
                    llm_overrides,
                    fallback=True,
                )
            else:
                logging.debug(
                    "Chunk %d translated successfully without fallback", idx
                )

            translated_parts.append(translated)

        return translated_parts

    def _refine_page(
        self,
        strategy: TranslationStrategy,
        page_translations: Iterable[str],
        page_number: int,
        context: Mapping[str, str],
        llm_overrides: Mapping[str, object],
        passes: int,
    ) -> str:
        current = "\n\n".join(page_translations)
        profile = strategy.refine_profile or strategy.profile
        overrides = dict(llm_overrides)
        overrides.setdefault(
            "max_new_tokens",
            strategy.refine_max_new_tokens or strategy.max_new_tokens,
        )

        for iteration in range(passes):
            logging.info(
                "Refinement iteration %d/%d for page %d using %s",
                iteration + 1,
                passes,
                page_number + 1,
                strategy.name,
            )
            prompt = strategy.build_refinement_prompt(
                current, page_number, context, iteration
            )
            cache_key = self.llm.prompt_hash(profile, prompt)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logging.debug(
                    "Using cached refinement for page %d iteration %d", page_number + 1, iteration
                )
                current = self._clean_generation(cached)
                continue

            response = self.llm.invoke(profile, prompt, **overrides)
            cleaned = self._clean_generation(response)
            self.cache.set(cache_key, cleaned)
            current = cleaned

        return current

    def _issue_prompt(
        self,
        strategy: TranslationStrategy,
        chunk: str,
        chunk_index: int,
        context: Mapping[str, str],
        llm_overrides: MutableMapping[str, object],
        *,
        fallback: bool,
    ) -> str:
        prompt = strategy.build_prompt(chunk, chunk_index, context, fallback=fallback)
        profile = (
            strategy.fallback_profile
            if fallback and strategy.fallback_profile
            else strategy.profile
        )
        overrides = dict(llm_overrides)
        if fallback and strategy.fallback_max_new_tokens:
            overrides["max_new_tokens"] = strategy.fallback_max_new_tokens

        cache_key = self.llm.prompt_hash(profile, prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logging.debug(
                "Using cached translation for %s chunk %d (fallback=%s)",
                strategy.name,
                chunk_index,
                fallback,
            )
            return self._clean_generation(cached)

        logging.debug(
            "Invoking LLM profile %s for %s chunk %d (fallback=%s)",
            profile,
            strategy.name,
            chunk_index,
            fallback,
        )
        response = self.llm.invoke(profile, prompt, **overrides)
        cleaned = self._clean_generation(response)
        self.cache.set(cache_key, cleaned)
        return cleaned

    def _should_retry(
        self,
        chunk: str,
        translated: str,
        strategy: TranslationStrategy,
    ) -> bool:
        """Detects obviously unsafe translations that warrant a fallback prompt."""

        if not translated.strip():
            return True

        if self._has_disallowed_markers(translated):
            return True

        if self._looks_like_passthrough(chunk, translated):
            return True

        lowered = translated.lower()
        for trigger in strategy.fallback_triggers:
            if trigger and trigger.lower() in lowered:
                return True

        return False

    @staticmethod
    def _has_disallowed_markers(text: str) -> bool:
        patterns = (
            "<<<<<<<",
            ">>>>>>>",
            "====",
            "TODO",
            "FIXME",
            "TBD",
            "NOT IMPLEMENTED",
            "???",
        )
        upper_text = text.upper()
        return any(marker in upper_text for marker in patterns)

    @staticmethod
    def _looks_like_passthrough(original: str, translated: str) -> bool:
        original_norm = re.sub(r"\s+", " ", original.strip().lower())
        translated_norm = re.sub(r"\s+", " ", translated.strip().lower())
        if not original_norm or not translated_norm:
            return False
        if len(translated_norm) < 40:
            return False

        ratio = SequenceMatcher(None, original_norm, translated_norm).ratio()
        return ratio >= 0.85

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

    @staticmethod
    def _paginate_indices(total: int, page_size: int | None) -> Iterable[tuple[int, int]]:
        if total <= 0:
            return []

        if not page_size or page_size <= 0 or page_size >= total:
            return [(0, total)]

        return [
            (start, min(start + page_size, total))
            for start in range(0, total, page_size)
        ]

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_auto_config() -> Dict[str, int]:
        env = os.environ

        def _positive_int(name: str, default: int) -> int:
            raw = env.get(name)
            if raw is None:
                return default
            try:
                value = int(raw)
            except ValueError:
                logging.warning(
                    "Invalid value for %s=%s; falling back to %d", name, raw, default
                )
                return default
            if value <= 0:
                logging.warning(
                    "%s must be > 0, received %d; using %d", name, value, default
                )
                return default
            return value

        return {
            "small_line_threshold": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_SMALL_LINES", 600
            ),
            "medium_line_threshold": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_MEDIUM_LINES", 1500
            ),
            "large_line_threshold": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_LARGE_LINES", 3500
            ),
            "huge_line_threshold": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_HUGE_LINES", 6000
            ),
            "medium_page_chunks": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_MEDIUM_CHUNKS", 4
            ),
            "large_page_chunks": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_LARGE_CHUNKS", 6
            ),
            "huge_page_chunks": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_HUGE_CHUNKS", 8
            ),
            "massive_page_chunks": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_MASSIVE_CHUNKS", 10
            ),
            "project_large_file_count": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_PROJECT_FILES", 40
            ),
            "project_large_total_lines": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_PROJECT_LINES", 60000
            ),
            "project_large_page_chunks": _positive_int(
                "CHRISTOPHE_AUTO_PAGE_PROJECT_CHUNKS", 3
            ),
        }

    @staticmethod
    def _count_lines(chunks: Sequence[str]) -> int:
        if not chunks:
            return 0
        return sum(len(chunk.splitlines()) for chunk in chunks)

    def _resolve_page_size(
        self,
        strategy: TranslationStrategy,
        src: Path,
        chunks: Sequence[str],
        explicit: int | None,
        project_stats: Mapping[str, Any] | None,
    ) -> int | None:
        total_chunks = len(chunks)
        line_count = self._count_lines(chunks)

        if total_chunks <= 1:
            self.pagination_log[str(src)] = {
                "mode": "manual" if explicit is not None else "auto",
                "requested_page_size": explicit,
                "resolved_page_size": None,
                "total_chunks": total_chunks,
                "line_count": line_count,
                "chunk_size": strategy.chunk_size,
            }
            if explicit:
                logging.info(
                    "Pagination override ignored for %s; only one chunk available",
                    src,
                )
            return None

        def _limit(value: int | None) -> int | None:
            if value is None:
                return None
            if value <= 0:
                return None
            bounded = max(1, min(value, total_chunks))
            if bounded >= total_chunks:
                return None
            return bounded

        if explicit is not None:
            resolved = _limit(explicit)
            if resolved is None:
                logging.info(
                    "Manual pagination for %s results in a single pass (chunks=%d)",
                    src,
                    total_chunks,
                )
            else:
                logging.info(
                    "Manual pagination for %s: %d chunks per page (chunks=%d)",
                    src,
                    resolved,
                    total_chunks,
                )
        else:
            resolved = self._automatic_page_size(total_chunks, line_count, project_stats)
            if resolved is None:
                logging.info(
                    "Auto pagination for %s: single pass (chunks=%d, lines=%d)",
                    src,
                    total_chunks,
                    line_count,
                )
            else:
                logging.info(
                    "Auto pagination for %s: %d chunks per page (chunks=%d, lines=%d)",
                    src,
                    resolved,
                    total_chunks,
                    line_count,
                )

        decision = {
            "mode": "manual" if explicit is not None else "auto",
            "requested_page_size": explicit,
            "resolved_page_size": resolved,
            "total_chunks": total_chunks,
            "line_count": line_count,
            "chunk_size": strategy.chunk_size,
        }
        if project_stats:
            decision.update(
                {
                    "project_total_files": project_stats.get("total_files"),
                    "project_total_lines": project_stats.get("total_lines"),
                }
            )
        self.pagination_log[str(src)] = decision
        return resolved

    def _automatic_page_size(
        self,
        total_chunks: int,
        line_count: int,
        project_stats: Mapping[str, Any] | None,
    ) -> int | None:
        config = self.auto_pagination

        thresholds = [
            (config["small_line_threshold"], None),
            (config["medium_line_threshold"], config["medium_page_chunks"]),
            (config["large_line_threshold"], config["large_page_chunks"]),
            (config["huge_line_threshold"], config["huge_page_chunks"]),
        ]

        resolved: int | None = None
        for threshold, chunk_target in thresholds:
            if line_count <= threshold:
                resolved = chunk_target
                break

        if resolved is None:
            resolved = config["massive_page_chunks"]

        if resolved is None:
            return None

        resolved = max(1, min(resolved, total_chunks))
        if resolved >= total_chunks:
            return None

        if project_stats:
            total_files = project_stats.get("total_files", 0) or 0
            total_lines = project_stats.get("total_lines", 0) or 0
            if (
                total_files >= config["project_large_file_count"]
                or total_lines >= config["project_large_total_lines"]
            ):
                adjusted = max(1, min(config["project_large_page_chunks"], resolved))
                resolved = max(1, min(adjusted, resolved))
                if resolved >= total_chunks:
                    return None

        return resolved

