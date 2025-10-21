import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
from collections import deque
import textwrap
import argparse
import sys
import zipfile
import tempfile
import gc

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    logging.error("Le librerie AI non sono installate. Esegui: pip install transformers torch accelerate")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --- SERVIZIO LLM CENTRALIZZATO (GPU) ---
class LLMService:
    """Gestisce il caricamento e l'interazione con un modello LLM da Hugging Face."""
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_id = model_id
        self.pipeline = None
        self.tokenizer = None

    def load_model(self):
        if self.pipeline:
            logging.info("Modello LLM già caricato.")
            return

        logging.warning(f"Caricamento del modello '{self.model_id}' su GPU (device_map=auto)...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",  # usa GPU automaticamente
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logging.info("Modello LLM caricato con successo su GPU.")
        except Exception as e:
            logging.error(f"Errore durante il caricamento del modello: {e}")
            sys.exit(1)

    def invoke(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not self.pipeline or not self.tokenizer:
            raise RuntimeError("Il modello non è stato caricato.")

        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [self.tokenizer.eos_token_id]

        outputs = self.pipeline(
            prompt_formatted,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        return outputs[0]["generated_text"][len(prompt_formatted):]


# --- INTERFACCIA BASE ---
class IAnalyzer(Protocol):
    def create_dependency_graph(self, project_path: str, all_files: List[str]) -> Dict[str, List[str]]:
        ...


# --- ANALIZZATORE LLM ---
class LLMAnalyzer(IAnalyzer):
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def create_dependency_graph(self, project_path: str, all_files: List[str]) -> Dict[str, List[str]]:
        logging.info("LLMAnalyzer: Analisi delle dipendenze...")
        full_graph = {file: [] for file in all_files}

        for file_path_str in all_files:
            try:
                with open(Path(project_path) / file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if len(content) > 3000:
                    content = content[:3000]

                prompt = textwrap.dedent(f"""
                You are a dependency analysis tool. Analyze the following code file and identify its direct dependencies on other files within the project.
                A dependency is an import, require, COPY statement, or any inclusion of another file.
                Respond ONLY with JSON: {{"dependencies": ["file1.ext", "file2.ext"]}}.
                If none, return {{"dependencies": []}}.

                --- FILE CONTENT ({file_path_str}) ---
                {content}
                --- END FILE CONTENT ---
                """)

                response_text = self.llm_service.invoke(prompt, max_new_tokens=256)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

                if json_match:
                    dependencies = json.loads(json_match.group(0))
                    valid_deps = [dep for dep in dependencies.get("dependencies", []) if dep in all_files]
                    full_graph[file_path_str] = valid_deps
                else:
                    logging.warning(f"Nessun JSON valido trovato per {file_path_str}")

            except Exception as e:
                logging.error(f"Errore durante l'analisi del file {file_path_str}: {e}")
            gc.collect()

        return full_graph


# --- ALTRI AGENTI ---

class AnalysisAgent:
    def __init__(self, llm_service: LLMService):
        self._analyzer = LLMAnalyzer(llm_service)

    def get_analyzer(self) -> IAnalyzer:
        logging.info("AnalysisAgent: uso LLMAnalyzer universale.")
        return self._analyzer


class PlanningAgent:
    def create_migration_plan(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        logging.info("PlanningAgent: Creazione del piano di migrazione...")
        if not dependency_graph:
            return []

        in_degree = {u: 0 for u in dependency_graph}
        for u in dependency_graph:
            for v in dependency_graph[u]:
                if v in in_degree:
                    in_degree[v] += 1

        queue = deque([u for u in in_degree if in_degree[u] == 0])
        plan = []

        reverse_adj = {u: [] for u in dependency_graph}
        for u, deps in dependency_graph.items():
            for v in deps:
                if v in reverse_adj:
                    reverse_adj[v].append(u)

        while queue:
            u = queue.popleft()
            plan.append(u)
            for v in reverse_adj.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(plan) != len(dependency_graph):
            logging.warning("Possibile dipendenza circolare.")
            plan.extend(list(set(dependency_graph.keys()) - set(plan)))

        return plan


class ScaffoldingAgent:
    def generate_project_structure(self, output_path: Path, project_name: str, target_framework: str) -> Path:
        logging.info(f"Creazione struttura per '{target_framework}' in '{output_path}'...")
        if target_framework.lower() == 'spring_boot':
            src_path = output_path / "src" / "main" / "java" / "com" / "example" / project_name
            src_path.mkdir(parents=True, exist_ok=True)
            pom_content = textwrap.dedent(f"""
            <project>
                <modelVersion>4.0.0</modelVersion>
                <groupId>com.example</groupId>
                <artifactId>{project_name}</artifactId>
                <version>0.0.1-SNAPSHOT</version>
                <parent>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-parent</artifactId>
                    <version>3.2.5</version>
                </parent>
                <dependencies>
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-web</artifactId>
                    </dependency>
                </dependencies>
            </project>
            """)
            with open(output_path / "pom.xml", "w") as f:
                f.write(pom_content)
            return src_path
        else:
            output_path.mkdir(exist_ok=True)
            return output_path


class MigrationAgent:
    def __init__(self, llm_service: LLMService, target_lang: str, target_framework: str):
        self.llm_service = llm_service
        self.target_lang = target_lang
        self.target_framework = target_framework

    def translate_and_write_file(self, source_path: Path, target_dir: Path, src_language: str, src_framework: Optional[str]):
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            prompt = textwrap.dedent(f"""
            You are an expert code translator. Convert the following '{src_language}' file (framework '{src_framework or "none"}')
            into '{self.target_lang}' for '{self.target_framework}'.
            Respond ONLY with raw translated code.

            --- SOURCE CODE ({source_path.name}) ---
            {source_code}
            --- END SOURCE CODE ---
            """)

            translated_code = self.llm_service.invoke(prompt, max_new_tokens=2048)
            translated_code = re.sub(r'^```[^\n]*\n', '', translated_code)
            translated_code = re.sub(r'\n```$', '', translated_code.strip())

            new_filename = f"{source_path.stem.capitalize()}.java"
            target_path = target_dir / new_filename

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(translated_code)

            logging.info(f"File tradotto scritto in: {target_path}")
        except Exception as e:
            logging.error(f"Errore nella traduzione di {source_path}: {e}")


# --- ORCHESTRATORE PRINCIPALE ---
class ProjectConverter:
    def __init__(self, project_path: str, output_path: str, llm_service: LLMService):
        self.project_path = Path(project_path)
        self.output_path = Path(output_path)
        self.llm_service = llm_service
        self.analysis_agent = AnalysisAgent(llm_service)
        self.planning_agent = PlanningAgent()
        self.scaffolding_agent = ScaffoldingAgent()

    def run_full_pipeline(self, src_language: str, src_framework: Optional[str], target_framework: str):
        all_files = []
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if not any(file.endswith(ext) for ext in ['.git', '.md', '.txt', '.log']):
                    all_files.append(os.path.relpath(os.path.join(root, file), self.project_path))

        analyzer = self.analysis_agent.get_analyzer()
        dependency_graph = analyzer.create_dependency_graph(str(self.project_path), all_files)
        migration_plan = self.planning_agent.create_migration_plan(dependency_graph)

        print_section("Piano di Migrazione Strategico", "\n".join(f"{i+1}. {s}" for i, s in enumerate(migration_plan)))

        project_name = self.project_path.name.replace("-", "_")
        target_code_path = self.scaffolding_agent.generate_project_structure(self.output_path, project_name, target_framework)

        migration_agent = MigrationAgent(self.llm_service, target_lang="java", target_framework=target_framework)
        for file_to_migrate in migration_plan:
            source_file_path = self.project_path / file_to_migrate
            if source_file_path.exists():
                migration_agent.translate_and_write_file(source_file_path, target_code_path, src_language, src_framework)
            else:
                logging.warning(f"File '{file_to_migrate}' non trovato, saltato.")

        print_section("Progetto Migrato", f"Output generato in: '{self.output_path}'")


# --- FUNZIONI DI UTILITÀ ---
def secure_extract_zip(zip_path: Path, dest_dir: Path):
    logging.info(f"Estrazione sicura di '{zip_path}' in '{dest_dir}'...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target_path = (dest_dir / member.filename).resolve()
            if not str(target_path).startswith(str(dest_dir.resolve())):
                raise ValueError(f"Percorso potenzialmente malevolo: '{member.filename}'")
            if not member.is_dir():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                z.extract(member, dest_dir)
    logging.info("Estrazione completata.")


def print_section(title: str, content: str):
    print(f"\n--- {title.upper()} ---\n{content}\n" + "-" * (len(title) + 8))


def main():
    parser = argparse.ArgumentParser(description="Strumento AI per la conversione di progetti software.")
    parser.add_argument("zip_path", type=Path, help="Percorso del file ZIP del progetto sorgente.")
    parser.add_argument("--target-framework", required=True, help="Framework di destinazione (es. spring_boot).")
    parser.add_argument("--src-language", required=True, help="Linguaggio sorgente (es. python).")
    parser.add_argument("--src-framework", required=False, help="Framework sorgente (opzionale).")

    args = parser.parse_args()
    if not args.zip_path.is_file():
        logging.error(f"Il file specificato non esiste: {args.zip_path}")
        sys.exit(1)

    llm_service = LLMService()
    llm_service.load_model()

    output_dir = Path("output_project")
    output_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="code_converter_") as tmpdir:
        temp_path = Path(tmpdir)
        secure_extract_zip(args.zip_path, temp_path)

        converter = ProjectConverter(temp_path, output_dir, llm_service)
        converter.run_full_pipeline(args.src_language, args.src_framework, args.target_framework)

    logging.info(f"Conversione completata. Output in: {output_dir}")


if __name__ == "__main__":
    main()
