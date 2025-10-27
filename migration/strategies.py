# migration/strategies.py
"""Translation strategies for different migration scenarios.

This module exposes small, declarative strategy objects that describe how
source chunks should be translated.  Each strategy encapsulates the
instructions to provide to the LLM, the preferred chunk size and the target
profile that should be used when invoking the model.  By isolating this
behaviour we can support new migration paths (e.g. SAP ECC → SAP S/4HANA,
Django → Spring Boot, AngularJS → React) without changing the orchestration
code in :mod:`source_migrator`.
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
    instructions:
        Prompt preamble describing the expected migration.
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
        DATA DIVISION for COBOL programs).
    """

    name: str
    instructions: str
    target_language: str
    profile: str = "translate"
    chunk_size: int = 300
    max_new_tokens: int = 2048
    context_labels: Mapping[str, str] = field(default_factory=dict)

    def build_prompt(self, chunk: str, chunk_index: int, context: Mapping[str, str]) -> str:
        """Compose the prompt for a single chunk.

        Parameters
        ----------
        chunk:
            The slice of source code to migrate.
        chunk_index:
            Zero-based index of the chunk; this is included in the prompt so the
            model can keep track of the order when reassembling the pieces.
        context:
            Additional metadata provided by the caller.  Only values that are
            truthy are included in the final prompt.
        """

        sections = []
        for key, label in self.context_labels.items():
            value = context.get(key)
            if value:
                sections.append(f"{label}:\n{value}")

        for key, value in context.items():
            if key not in self.context_labels and value:
                sections.append(f"{key}:\n{value}")

        context_block = "\n\n".join(sections)
        context_snippet = f"\n\n{context_block}" if context_block else ""

        return (
            f"{self.instructions.strip()}"
            f"{context_snippet}\n\n"
            f"SOURCE CHUNK [{chunk_index}]:\n{chunk}\n\n"
<<<<<<< HEAD
            f"Return ONLY {self.target_language.strip()}."
=======
            f"Return ONLY {self.target_language.strip()} with no commentary,"
            f" TODO markers, or markdown fences. Ensure the output compiles"
            f" cleanly."
>>>>>>> codex/evolve-migration-system-for-complex-frameworks-v7nxyq
        )


class CobolToSpringStrategy(TranslationStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="cobol_to_spring",
            instructions=(
                "Convert the following COBOL code to Java using a modern Spring "
                "Boot architecture. Preserve all business rules and ensure the "
                "result leverages Java 21 idioms (records, switch expressions, "
                "optional, etc.)."
            ),
            target_language="Java code",
            profile="translate",
            chunk_size=300,
            max_new_tokens=2048,
            context_labels={
                "data_division": "DATA DIVISION",
                "fd_summary": "FILE DESCRIPTORS",
            },
        )


class PythonDjangoToSpringStrategy(TranslationStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="python_to_spring",
            instructions=(
                "Port the following Django/Python module to an equivalent Java "
                "implementation built with Spring Boot. Map Django ORM models to "
                "Spring Data JPA entities, convert views to REST controllers and "
                "use idiomatic Java 21 style."
            ),
            target_language="Java Spring Boot code",
            profile="translate",
            chunk_size=200,
            max_new_tokens=2048,
            context_labels={
                "project_settings": "DJANGO SETTINGS SUMMARY",
            },
        )


class AngularJsToReactStrategy(TranslationStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="angularjs_to_react",
            instructions=(
                "Rewrite the following AngularJS component/template pair as a "
                "React functional component using hooks. Replace two-way binding "
                "with React state, convert services to hooks/context and favour "
                "modern TypeScript with ES modules."
            ),
            target_language="React (TypeScript) code",
            profile="translate",
            chunk_size=150,
            max_new_tokens=1536,
            context_labels={
                "shared_services": "ANGULARJS SERVICES",
            },
        )


class AbapToS4HanaStrategy(TranslationStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="abap_to_s4",
            instructions=(
                "Modernise the following ABAP logic for SAP S/4HANA. Adopt CDS "
                "views, clean ABAP syntax and steer towards side-by-side "
                "extensions when appropriate. Highlight where Fiori/UI5 wrappers "
                "are required."
            ),
            target_language="ABAP code ready for S/4HANA",
            profile="sap",
            chunk_size=120,
            max_new_tokens=2048,
            context_labels={
                "ddic_metadata": "DDIC METADATA",
                "integration_notes": "INTEGRATION NOTES",
            },
        )


DEFAULT_STRATEGIES: Dict[str, TranslationStrategy] = {
    "cobol_to_java": CobolToSpringStrategy(),
    "python_to_spring": PythonDjangoToSpringStrategy(),
    "angularjs_to_react": AngularJsToReactStrategy(),
    "abap_to_s4": AbapToS4HanaStrategy(),
}

