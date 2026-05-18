import sys
import argparse
from pathlib import Path

import papermill as pm
import mlflow

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JSON_REPORTS_DIR,
    JSON_CHUNKS_DIR,
    FAISS_INDEX_DIR,
    PROJECT_ROOT,
)

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DEFAULT_PIPELINE = [
    NOTEBOOKS_DIR / "preprocesing" / "1_pre_json_chunks.ipynb",
    NOTEBOOKS_DIR / "preprocesing" / "2_pre_json_splits.ipynb",
    NOTEBOOKS_DIR / "indexation"  / "3_Indexar.ipynb",
    NOTEBOOKS_DIR / "retrieve"    / "4_retrieve.ipynb",
    NOTEBOOKS_DIR / "evaluation"  / "5_evaluate_RAG.ipynb",
]

EXPERIMENTS = [
    {
        "name": "baseline_BGE",
        "common_params": {
            "experiment_name":      "baseline_test",
            "json_input":           str(JSON_REPORTS_DIR),
            "chunks_jsonl_path":    str(JSON_CHUNKS_DIR / "rag_chunks_all.jsonl"),
            "chunk_size":           1000,
            "chunk_overlap":        100,
            "index_input_path":     str(JSON_CHUNKS_DIR / "rag_chunks_split_langchain.jsonl"),
            "faiss_output_dir":     str(FAISS_INDEX_DIR),
            "embedding_model":      "BAAI/bge-base-en-v1.5",
            "faiss_index_path":     str(FAISS_INDEX_DIR),
            "llm_model":            "Qwen/Qwen1.5-0.5B",
            "retriever_k":          5,
            "temperature":          0.2,
            "max_new_tokens":       512,
            "top_k":                50,
            "test_csv_path":        str(NOTEBOOKS_DIR / "evaluation" / "test_questions.csv"),
            "n_test_rows":          150,
            "similarity_threshold": 0.8,
        },
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG pipeline experiments")
    parser.add_argument("--start", type=int, default=0,
                        help="Index of first notebook to run (0-based)")
    parser.add_argument("--end", type=int, default=len(DEFAULT_PIPELINE) - 1,
                        help="Index of last notebook to run (inclusive)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run only the experiment with this name")
    return parser.parse_args()


def run_experiments(args):
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("rag_experiments")

    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"No experiment named '{args.experiment}'")

    for exp in experiments:
        run_name = exp["name"]
        common = exp["common_params"]

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.log_params({k: v for k, v in common.items()
                               if not isinstance(v, Path)})

            out_dir = PROJECT_ROOT / "output_notebooks" / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, nb in enumerate(DEFAULT_PIPELINE):
                if idx < args.start or idx > args.end:
                    continue

                params = {
                    "experiment_name": run_name,
                    "run_id": run_id,
                }

                if idx == 0:
                    params.update({
                        "input_dir": common["json_input"],
                        "output_file": common["chunks_jsonl_path"],
                    })
                elif idx == 1:
                    params.update({
                        "input_file": common["chunks_jsonl_path"],
                        "output_file": common["index_input_path"],
                        "chunk_size": common["chunk_size"],
                        "chunk_overlap": common["chunk_overlap"],
                    })
                elif idx == 2:
                    params.update({
                        "input_path": common["index_input_path"],
                        "faiss_output_dir": common["faiss_output_dir"],
                        "embedding_model": common["embedding_model"],
                    })
                else:
                    params.update({
                        "faiss_index_path": common["faiss_index_path"],
                        "embedding_model": common["embedding_model"],
                        "llm_model": common["llm_model"],
                        "retriever_k": common["retriever_k"],
                        "temperature": common["temperature"],
                        "max_new_tokens": common["max_new_tokens"],
                        "top_k": common["top_k"],
                        "test_csv_path": common["test_csv_path"],
                        "n_test_rows": common["n_test_rows"],
                        "similarity_threshold": common["similarity_threshold"],
                    })

                print(f"▶ [{idx}/{len(DEFAULT_PIPELINE)-1}] {nb.name}")
                pm.execute_notebook(
                    input_path=str(nb),
                    output_path=str(out_dir / f"{idx:02d}_{nb.name}"),
                    parameters=params,
                )

        print(f"✓ Finished: {run_name}")


if __name__ == "__main__":
    run_experiments(parse_args())
