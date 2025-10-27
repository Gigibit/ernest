# christophe.py
import json, logging, tempfile, sys
from pathlib import Path
from core.llm_service import LLMService
from core.cache_manager import CacheManager
from core.file_utils import secure_unzip
from analysis.heuristic_agent import HeuristicAnalysisAgent
from analysis.semantic_graph import SemanticGraphBuilder
from planning.planning_agent import PlanningAgent
from migration.source_migrator import SourceMigrator
from migration.resource_migrator import ResourceMigrator
from migration.recovery_manager import RecoveryManager
from scaffolding.scaffolding_agent import ScaffoldingAgent

def print_section(title, content):
    line = "-" * (len(title) + 8)
    print(f"\n--- {title.upper()} ---\n{content}\n{line}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI-driven project migration orchestrator.")
    parser.add_argument("zip_path", type=Path, help="Source project ZIP file")
    parser.add_argument("--target-framework", required=True)
    parser.add_argument("--target-lang", default="java")
    parser.add_argument("--src-lang", default=None)
    parser.add_argument("--src-framework", default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # LLM setup
    profiles = {
        "classify": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 512, "temp": 0.1},
        "analyze": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 512, "temp": 0.1},
        "translate": {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "max": 4096, "temp": 0.0},
        "adapt": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "max": 1024, "temp": 0.0},
        "scaffold": {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "max": 2048, "temp": 0.1}
    }
    llm = LLMService(profiles)
    cache = CacheManager(Path(".cache/migration_cache.db"))

    with tempfile.TemporaryDirectory(prefix="christophe_") as tmp:
        temp_dir = Path(tmp)
        secure_unzip(args.zip_path, temp_dir)

        # --- ANALYSIS PHASE ---
        heuristic = HeuristicAnalysisAgent(str(temp_dir), llm, cache)
        classification = heuristic.classify_files()
        print_section("Classificazione File", json.dumps(classification, indent=2))

        detected_stack = (
            {"language": args.src_lang, "framework": args.src_framework}
            if args.src_lang else heuristic.detect_stack(classification.get("source"))
        )
        if not detected_stack:
            logging.error("Impossibile determinare lo stack sorgente.")
            sys.exit(1)
        print_section("Stack Rilevato", json.dumps(detected_stack, indent=2))

        # --- DEPENDENCY GRAPH & PLAN ---
        graph = SemanticGraphBuilder()
        graph.add_nodes(classification.get("source", []))
        plan = PlanningAgent().create_plan({}, classification.get("source", []))
        print_section("Piano di Migrazione", "\n".join(plan))

        # --- SCAFFOLDING PHASE ---
        scaffold_agent = ScaffoldingAgent(llm, cache)
        project_name = Path(args.zip_path.stem).name.replace("-", "_")
        output_root = Path("output_project")
        target_path = scaffold_agent.generate(output_root, project_name, args.target_framework, args.target_lang)

        # --- MIGRATION PHASE ---
        recovery = RecoveryManager(Path("migration_state.json"))
        src_migrator = SourceMigrator(llm, cache, recovery)
        res_migrator = ResourceMigrator(llm, cache)

        for src in plan:
            f = temp_dir / src
            if not f.exists():
                recovery.mark_skipped(src)
                continue
            out_path = target_path / "src" / Path(src).name.replace(".cbl", ".java")
            src_migrator.translate_cobol(f, out_path)

        for res in classification.get("resource", []):
            p = temp_dir / res
            if p.exists():
                res_migrator.process(p, target_path / "resources")

        print_section("Progetto Migrato", f"Output generato in: {output_root}")

if __name__ == "__main__":
    main()
