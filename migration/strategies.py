# migration/strategies.py
"""Translation strategies for different migration scenarios.

This module exposes small, declarative strategy objects that describe how
source chunks should be translated.  Each strategy encapsulates the
instructions to provide to the LLM, the preferred chunk size and the target
profile that should be used when invoking the model.  By isolating this
behaviour we can support new migration patterns (e.g. legacy batch workloads
that need service-based refactoring, dynamic web frameworks that must become
strongly typed APIs, or legacy UI stacks that should evolve into component
libraries) without changing the orchestration code in :mod:`source_migrator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping


@dataclass(frozen=True)
class TranslationStrategy:
    """Description of how to translate a source artefact.

    Attributes
    ----------
    name:
        Human readable identifier used for logging.
    source_descriptor:
        Short human-readable description of the legacy environment.
    target_descriptor:
        Short description of the target environment or architecture pattern.
    instructions:
        Optional extra directives appended to the automatically generated
        prompt.
    target_language:
        Short description of the desired output (used in the final prompt
        instruction).
    profile:
        LLM profile name to use when invoking :class:`~core.llm_service.LLMService`.
    chunk_size:
        Number of lines that should be grouped in a single prompt.
    max_new_tokens:
        Default ``max_new_tokens`` override passed to the LLM when the caller
        does not specify a custom value.
    context_labels:
        Optional mapping to format contextual metadata in the prompt (e.g.
        architecture outlines or runtime notes).
    refine_instructions:
        Optional instructions used for refinement passes.  When provided the
        translated output of a page is sent back to the LLM with these
        instructions to polish the result.
    refine_profile:
        Optional profile override used during refinement passes.  Defaults to
        :attr:`profile` when omitted.
    refine_max_new_tokens:
        ``max_new_tokens`` hint used during refinement.  Falls back to
        :attr:`max_new_tokens` when not provided.
    """

    name: str
    source_descriptor: str
    target_descriptor: str
    instructions: str = ""
    target_language: str = "code"
    profile: str = "translate"
    chunk_size: int = 300
    max_new_tokens: int = 2048
    context_labels: Mapping[str, str] = field(default_factory=dict)
    refine_instructions: str | None = None
    refine_profile: str | None = None
    refine_max_new_tokens: int | None = None

    def build_prompt(self, chunk: str, chunk_index: int, context: Mapping[str, str]) -> str:
        """Compose the prompt for a single chunk."""

        guidance = (
            "You are helping a migration effort. "
            f"The legacy context is {self.source_descriptor}. "
            f"Rebuild it so it fits {self.target_descriptor}."
        )
        directives = self.instructions.strip()
        if directives:
            guidance = f"{guidance}\n{directives}"

        return (
            f"{guidance}"
            f"{self._format_context(context)}\n\n"
            f"SOURCE CHUNK [{chunk_index}]:\n{chunk}\n\n"
            f"Return ONLY {self.target_language.strip()} with no commentary,"
            f" TODO markers, or markdown fences. Ensure the output compiles"
            f" cleanly."
        )

    def build_refinement_prompt(
        self,
        page_text: str,
        page_index: int,
        context: Mapping[str, str],
        iteration: int,
    ) -> str:
        """Compose the refinement prompt for a translated page."""

        base_instructions = (
            self.refine_instructions
            or "Review the migrated output below and improve it without"
            " regressing behaviour."
        )

        return (
            f"{base_instructions.strip()}"
            f"{self._format_context(context)}\n\n"
            f"MIGRATED PAGE [{page_index}] PASS {iteration + 1}:\n{page_text}\n\n"
            f"Return ONLY polished {self.target_language.strip()} without"
            f" commentary, TODO markers, or markdown fences."
        )

    def _format_context(self, context: Mapping[str, str]) -> str:
        sections = []
        for key, label in self.context_labels.items():
            value = context.get(key)
            if value:
                sections.append(f"{label}:\n{value}")

        for key, value in context.items():
            if key not in self.context_labels and value:
                sections.append(f"{key}:\n{value}")

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)


DEFAULT_STRATEGIES: Dict[str, TranslationStrategy] = {
    "legacy_backend_to_services": TranslationStrategy(
        name="legacy_backend_to_services",
        source_descriptor="a batch-centric legacy back-end with tightly coupled IO",
        target_descriptor="a modular service-oriented runtime with contemporary language features",
        instructions=(
            "Preserva la logica di dominio, esplicita gli strati applicativi e "
            "sfrutta le costruzioni offerte dal linguaggio di destinazione per "
            "ottenere codice pulito e compilabile."
        ),
        target_language="production-ready service code",
        profile="translate",
        chunk_size=300,
        max_new_tokens=2048,
        context_labels={
            "structure_outline": "SOURCE STRUCTURE",
            "integration_contracts": "INTEGRATION CONTRACTS",
        },
        refine_instructions=(
            "Rivedi il modulo migrato, elimina residui legacy (commenti, TODO, "
            "variabili inutilizzate) e assicurati che sia pronto per il rilascio."
        ),
        refine_profile="translate",
        refine_max_new_tokens=3072,
    ),
    "dynamic_web_to_structured_backend": TranslationStrategy(
        name="dynamic_web_to_structured_backend",
        source_descriptor="un framework web dinamico basato su linguaggi interpretati",
        target_descriptor="un back-end tipizzato con architettura modulare e API esposte",
        instructions=(
            "Mappa modelli e servizi in entità ben tipizzate, trasforma viste "
            "in API pulite e cura gestione errori e sicurezza con pattern moderni."
        ),
        target_language="typed back-end code",
        profile="translate",
        chunk_size=200,
        max_new_tokens=2048,
        context_labels={
            "runtime_configuration": "RUNTIME CONFIGURATION",
        },
        refine_instructions=(
            "Ottimizza annotazioni, gestione dipendenze e naming affinché il "
            "risultato rifletta le best practice della piattaforma target."
        ),
        refine_profile="translate",
        refine_max_new_tokens=3072,
    ),
    "legacy_frontend_to_component_ui": TranslationStrategy(
        name="legacy_frontend_to_component_ui",
        source_descriptor="una UI legacy basata su template mutabili e binding bidirezionale",
        target_descriptor="una libreria component-based con gestione dello stato dichiarativa",
        instructions=(
            "Scomponi le responsabilità in componenti, sostituisci binding "
            "impliciti con stato esplicito e applica tipizzazione coerente."
        ),
        target_language="component-driven front-end code",
        profile="translate",
        chunk_size=150,
        max_new_tokens=1536,
        context_labels={
            "shared_dependencies": "SHARED DEPENDENCIES",
        },
        refine_instructions=(
            "Affina hook, proprietà e tipizzazione rimuovendo residui della UI "
            "sorgente e mantenendo la stessa UX."
        ),
        refine_profile="translate",
        refine_max_new_tokens=2048,
    ),
    "enterprise_core_to_cloud": TranslationStrategy(
        name="enterprise_core_to_cloud",
        source_descriptor="un nucleo enterprise legacy con estensioni pesantemente personalizzate",
        target_descriptor="una piattaforma cloud-native modulare con estensioni pulite",
        instructions=(
            "Modernizza sintassi, separa personalizzazioni da core standard e "
            "indica eventuali adattatori o servizi esterni necessari."
        ),
        target_language="enterprise core code aligned with the target platform",
        profile="translate",
        chunk_size=120,
        max_new_tokens=2048,
        context_labels={
            "data_model": "DOMAIN DATA MODEL",
            "integration_notes": "INTEGRATION NOTES",
        },
        refine_instructions=(
            "Allinea nomenclature, tipi e chiamate a servizi esterni agli "
            "standard della piattaforma di destinazione, eliminando residui legacy."
        ),
        refine_profile="translate",
        refine_max_new_tokens=2560,
    ),
}
