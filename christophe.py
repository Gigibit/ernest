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

# Prova a importare le librerie AI, fornendo un messaggio di errore utile se mancano
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    logging.error("Le librerie AI non sono installate. Esegui: pip install transformers torch accelerate")
    sys.exit(1)

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- SERVIZIO LLM CENTRALIZZATO (MODIFICATO PER CPU) ---

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

        logging.warning(f"Caricamento del modello '{self.model_id}' su CPU... L'operazione richiederà tempo e una quantità significativa di RAM.")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # CORREZIONE: Aggiunto trust_remote_code=True, necessario per Phi-3 con le versioni attuali
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True 
            )
            self.pipeline = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=self.tokenizer
            )
            logging.info("Modello LLM caricato con successo su CPU.")
        except Exception as e:
            logging.error(f"Errore durante il caricamento del modello: {e}")
            logging.error("Assicurati di aver effettuato l'accesso a Hugging Face ('huggingface-cli login').")
            sys.exit(1)

    def invoke(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not self.pipeline or not self.tokenizer:
            raise RuntimeError("Il modello non è stato caricato.")
        
        messages =
        
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
        return outputs["generated_text"][len(prompt_formatted):]

# --- DEFINIZIONE DELLE INTERFACCE E DEGLI ANALIZZATORI ---

class IAnalyzer(Protocol):
    """Interfaccia per tutti gli analizzatori di dipendenze."""
    def create_dependency_graph(self, project_path: str, all_files: List[str]) -> Dict[str, List[str]]:
     ...

class LLMAnalyzer(IAnalyzer):
    """Analizzatore di dipendenze universale basato su LLM."""
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def create_dependency_graph(self, project_path: str, all_files: List[str]) -> Dict[str, List[str]]:
        logging.info("LLMAnalyzer: Avvio analisi delle dipendenze file per file (sarà lento su CPU)...")
        full_graph = {file:  for file in all_files}
        
        for file_path_str in all_files:
            try:
                with open(Path(project_path) / file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if len(content) > 3000: # Riduciamo ulteriormente per sicurezza
                    content = content[:3000]

                # CORREZIONE: Rimosso {all_files} dal prompt per evitare context overflow
                prompt = textwrap.dedent(f"""
                You are a dependency analysis tool. Analyze the following code file and identify its direct dependencies on other files within the project.
                A dependency is an import, a require, a COPY statement, or any other inclusion of another file.

                Your response MUST be a single, valid JSON object like {{"dependencies": ["file1.ext", "file2.ext"]}} and nothing else.
                If there are no dependencies, return {{"dependencies":}}.

                --- FILE CONTENT ({file_path_str}) ---
                {content}
                --- END FILE CONTENT ---

                JSON OUTPUT:
                """)

                response_text = self.llm_service.invoke(prompt, max_new_tokens=256)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    dependencies = json.loads(json_match.group(0))
                    # Filtra le dipendenze per assicurarsi che siano file reali del progetto
                    valid_deps = [dep for dep in dependencies.get("dependencies", ) if dep in all_files]
                    full_graph[file_path_str] = valid_deps
                else:
                    logging.warning(f"Nessun JSON valido trovato per {file_path_str}")

            except Exception as e:
                logging.error(f"Errore durante l'analisi del file {file_path_str}: {e}")
            
            # CORREZIONE: Aggiunto garbage collection per liberare memoria
            gc.collect()
        
        return full_graph

# --- AGENTI DELLA PIPELINE ---

class AnalysisAgent:
    """Motore di analisi che ora utilizza sempre l'analizzatore LLM."""
    def __init__(self, llm_service: LLMService):
        self._analyzer = LLMAnalyzer(llm_service)

    def get_analyzer(self) -> IAnalyzer:
        logging.info("AnalysisAgent: Selezionato analizzatore universale LLM.")
        return self._analyzer

class HeuristicAnalysisAgent:
    """Agente che utilizza un LLM per dedurre lo stack tecnologico del progetto."""
    def __init__(self, project_path: str, llm_service: LLMService):
        self.project_path = Path(project_path)
        self.llm_service = llm_service

    def _curate_context(self, max_files=15, max_lines_per_file=100) -> str:
        logging.info("HeuristicAnalysisAgent: Cura del contesto per l'analisi LLM...")
        context_parts =
        # CORREZIONE: Popolamento delle liste per identificare i file rilevanti
        priority_files =
        exclude_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'target', 'build', 'dist', '.idea', '.vscode'}
        source_extensions =
        
        found_files =
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file in priority_files or any(file.lower().endswith(ext) for ext in source_extensions):
                    found_files.append(Path(root) / file)

        files_to_process = sorted(found_files, key=lambda p: 1 if p.name in priority_files else 0, reverse=True)[:max_files]

        for file_path in files_to_process:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = "".join(f.readlines()[:max_lines_per_file])
                    relative_path = file_path.relative_to(self.project_path)
                    context_parts.append(f"--- File: {relative_path} ---\n{content}\n")
            except Exception: pass
        
        return "\n".join(context_parts)

    def detect_stack(self) -> Optional]:
        context = self._curate_context()
        if not context:
            logging.error("HeuristicAnalysisAgent: Nessun file sorgente utile trovato.")
            return None
        
        prompt = textwrap.dedent(f"""
        You are an expert software architect. Analyze the following file excerpts from a project.
        Identify the primary programming language and main application framework.
        Your response MUST be a single, valid JSON object like {{"language": "string | null", "framework": "string | null"}} and nothing else.

        --- PROJECT CONTEXT ---
        {context}
        --- END CONTEXT ---

        JSON OUTPUT:
        """)
        
        response_text = self.llm_service.invoke(prompt, max_new_tokens=100)
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON found", response_text, 0)
            detected = json.loads(json_match.group(0))
            if not detected.get("language"): return None
            return detected
        except json.JSONDecodeError as e:
            logging.error(f"HeuristicAnalysisAgent: Risposta LLM non valida. Errore: {e}. Risposta: '{response_text}'")
            return None

class PlanningAgent:
    """Agente che crea un piano di migrazione strategico."""
    def create_migration_plan(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        logging.info("PlanningAgent: Creazione del piano di migrazione...")
        if not dependency_graph: return
        
        in_degree = {u: 0 for u in dependency_graph}
        for u in dependency_graph:
            for v in dependency_graph.get(u, ):
                if v in in_degree: in_degree[v] += 1
        
        queue = deque([u for u in in_degree if in_degree[u] == 0])
        plan =
        
        reverse_adj = {u:  for u in dependency_graph}
        for u, deps in dependency_graph.items():
            for v in deps:
                if v in reverse_adj: reverse_adj[v].append(u)

        while queue:
            u = queue.popleft()
            plan.append(u)
            for v in reverse_adj.get(u, ):
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)

        if len(plan)!= len(dependency_graph):
            logging.warning("Rilevata una dipendenza circolare. Il piano potrebbe essere incompleto.")
            plan.extend(list(set(dependency_graph.keys()) - set(plan)))
        
        return plan

class ScaffoldingAgent:
    """Agente che crea la struttura del progetto di destinazione."""
    def generate_project_structure(self, output_path: Path, project_name: str, target_framework: str) -> Path:
        logging.info(f"ScaffoldingAgent: Creazione struttura per '{target_framework}' in '{output_path}'...")
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
            with open(output_path / "pom.xml", "w") as f: f.write(pom_content)
            logging.info("pom.xml generato.")
            return src_path
        else:
            output_path.mkdir(exist_ok=True)
            return output_path

class MigrationAgent:
    """Agente che orchestra la traduzione del codice e la scrittura dei file di output."""
    def __init__(self, llm_service: LLMService, target_lang: str, target_framework: str):
        self.llm_service = llm_service
        self.target_lang = target_lang
        self.target_framework = target_framework

    def translate_and_write_file(self, source_path: Path, target_dir: Path, src_language: str, src_framework: Optional[str]):
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            prompt = textwrap.dedent(f"""
            You are an expert code translator. Convert the following '{src_language}' file, which uses the '{src_framework or "N/A"}' framework, into a '{self.target_lang}' file using the '{self.target_framework}' framework.
            Your response MUST be only the raw translated code and nothing else. Do not include explanations or markdown code blocks.

            --- SOURCE CODE ({source_path.name}) ---
            {source_code}
            --- END SOURCE CODE ---

            TRANSLATED CODE:
            """)
            
            translated_code = self.llm_service.invoke(prompt, max_new_tokens=2048)
            translated_code = re.sub(r'^```[^\n]*\n', '', translated_code)
            translated_code = re.sub(r'\n```$', '', translated_code.strip())

            new_filename = f"{source_path.stem.capitalize()}.java"
            target_path = target_dir / new_filename
            
            with open(target_path, "w", encoding='utf-8') as f:
                f.write(translated_code)
            logging.info(f"File tradotto scritto in: {target_path}")

        except Exception as e:
            logging.error(f"Errore durante la migrazione del file {source_path}: {e}")

class ProjectConverter:
    """Classe orchestratrice che gestisce l'intera pipeline end-to-end."""
    def __init__(self, project_path: str, output_path: str, llm_service: LLMService):
        self.project_path = project_path
        self.output_path = Path(output_path)
        self.llm_service = llm_service
        self.analysis_agent = AnalysisAgent(llm_service)
        self.planning_agent = PlanningAgent()
        self.scaffolding_agent = ScaffoldingAgent()

    def run_full_pipeline(self, src_language: str, src_framework: Optional[str], target_framework: str):
        all_files =
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if not any(file.endswith(ext) for ext in ['.git', '.md', '.txt', '.log']):
                    all_files.append(os.path.relpath(os.path.join(root, file), self.project_path))

        analyzer = self.analysis_agent.get_analyzer()
        dependency_graph = analyzer.create_dependency_graph(self.project_path, all_files)
        migration_plan = self.planning_agent.create_migration_plan(dependency_graph)
        print_section("Piano di Migrazione Strategico", "\n".join(f"{i+1}. Migra: {s}" for i, s in enumerate(migration_plan) if s))

        project_name = Path(self.project_path).name.replace("-", "_")
        target_code_path = self.scaffolding_agent.generate_project_structure(self.output_path, project_name, target_framework)

        migration_agent = MigrationAgent(self.llm_service, target_lang="java", target_framework=target_framework)
        for file_to_migrate in migration_plan:
            source_file_path = Path(self.project_path) / file_to_migrate
            if source_file_path.exists():
                migration_agent.translate_and_write_file(source_file_path, target_code_path, src_language, src_framework)
            else:
                logging.warning(f"File '{file_to_migrate}' dal piano non trovato, saltato.")
        
        print_section("Progetto Migrato", f"La struttura del progetto e i file tradotti sono stati generati in: '{self.output_path}'")

# --- FUNZIONE PRINCIPALE E BLOCCO DI ESECUZIONE ---

def main():
    parser = argparse.ArgumentParser(description="Strumento AI per la conversione di progetti software.")
    parser.add_argument("zip_path", type=Path, help="Percorso del file ZIP del progetto sorgente.")
    parser.add_argument("--target-framework", required=True, help="Il framework di destinazione (es. 'spring_boot').")
    parser.add_argument("--src-language", required=False, help="Specifica esplicitamente il linguaggio sorgente (es. 'cobol').")
    parser.add_argument("--src-framework", required=False, help="Specifica esplicitamente il framework sorgente.")
    
    args = parser.parse_args()

    if not args.zip_path.is_file():
        logging.error(f"Il file specificato non esiste: {args.zip_path}")
        sys.exit(1)

    llm_service = LLMService()
    llm_service.load_model()

    output_dir = Path("output_project")
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="code_converter_") as tmpdir:
            temp_path = Path(tmpdir)
            secure_extract_zip(args.zip_path, temp_path)

            detected_stack = None
            if args.src_language:
                detected_stack = {"language": args.src_language, "framework": args.src_framework}
            else:
                heuristic_agent = HeuristicAnalysisAgent(str(temp_path), llm_service)
                detected_stack = heuristic_agent.detect_stack()

            if not detected_stack or not detected_stack.get("language"):
                logging.error("Impossibile determinare lo stack tecnologico. Usare --src-language.")
                sys.exit(1)
            
            print_section("Stack Tecnologico Rilevato", json.dumps(detected_stack, indent=2))

            converter = ProjectConverter(str(temp_path), str(output_dir), llm_service)
            converter.run_full_pipeline(
                src_language=detected_stack["language"],
                src_framework=detected_stack.get("framework"),
                target_framework=args.target_framework
            )

    except Exception as e:
        logging.error(f"Processo interrotto a causa di un errore: {e}", exc_info=True)
        sys.exit(1)

    logging.info(f"Processo di conversione completato. Controlla la cartella '{output_dir}'.")

if __name__ == '__main__':
    def secure_extract_zip(zip_path: Path, dest_dir: Path):
        logging.info(f"Estrazione sicura di '{zip_path}' in '{dest_dir}'...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                target_path = (dest_dir / member.filename).resolve()
                if not str(target_path).startswith(str(dest_dir.resolve())):
                    raise ValueError(f"Rilevato percorso potenzialmente malevolo: '{member.filename}'")
                if not member.is_dir():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    z.extract(member, dest_dir)
        logging.info("Estrazione completata.")

    def print_section(title: str, content: str):
        print(f"\n--- {title.upper()} ---\n{content}\n" + "-" * (len(title) + 8))
        
    main()