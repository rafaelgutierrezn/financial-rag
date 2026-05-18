# Financial RAG — Local Inference on a Single Consumer GPU

![Unit Tests](https://github.com/rafaelgutierrezn/financial-rag/actions/workflows/tests.yml/badge.svg)

A **Retrieval-Augmented Generation (RAG)** system for querying SEC 10-K annual reports, designed to run **entirely on a single NVIDIA RTX 3060 (12 GB VRAM)**. No cloud API, no paid endpoints — every component from embedding to generation runs locally.

The project covers the full ML lifecycle: data acquisition, preprocessing, vector indexing, LLM-based retrieval, and automated evaluation, tracked end-to-end with **MLflow**.

---

## Design Constraint: Scarce Resources

The entire stack fits within 12 GB of VRAM by pairing a lightweight embedding model with a sub-1B parameter LLM.

| Component | Model | VRAM (fp16) |
|-----------|-------|-------------|
| Embeddings | `BAAI/bge-base-en-v1.5` | ~0.4 GB |
| LLM | `Qwen/Qwen1.5-0.5B` | ~1.0 GB |
| FAISS index | CPU (system RAM) | — |
| **Total** | | **~1.4 GB** |

The headroom is intentional: it leaves room for the tokeniser, KV-cache, and OS processes to coexist without OOM errors during generation. A larger LLM (e.g. Qwen1.5-7B at ~14 GB) can replace `Qwen1.5-0.5B` if a 24 GB card is available — only the `llm_model` Papermill parameter needs to change.

> **Known tradeoff:** The 0.5B model occasionally produces incomplete or hallucinated answers on complex financial queries. The RAG framework is sound; the bottleneck is generator capacity, not retrieval quality. Exact-match and ROUGE-L metrics in the evaluation notebook quantify this gap honestly.

---

## Architecture

```
SEC EDGAR
    │
    ▼
notebooks/preprocesing/
  0_read_html.ipynb        ← HTML → structured JSON  (pages + tables)
  1_pre_json_chunks.ipynb  ← JSON → fixed-size chunks (JSONL)
  2_pre_json_splits.ipynb  ← JSONL → LangChain recursive splits (overlap)
    │
    ▼
notebooks/indexation/
  3_Indexar.ipynb          ← Chunks → FAISS index (BGE embeddings)
    │
    ▼
notebooks/retrieve/
  4_retrieve.ipynb         ← Query → top-k retrieval → LLM answer
    │
    ▼
notebooks/evaluation/
  5_evaluate_RAG.ipynb     ← ROUGE-L + exact-match + cosine metrics → MLflow
```

---

## Stack

| Component | Library / Model |
|-----------|-----------------|
| Embeddings | `BAAI/bge-base-en-v1.5`, `all-MiniLM-L6-v2` |
| Vector store | FAISS (via LangChain, stored on CPU RAM) |
| LLM | `Qwen/Qwen1.5-0.5B` (local, HuggingFace) |
| Orchestration | LangChain `RetrievalQA` |
| Experiment tracking | MLflow |
| Notebook execution | Papermill |
| Data source | SEC EDGAR (`sec-edgar-downloader`) |
| Config / secrets | `python-dotenv` (`.env`, never committed) |
| Testing / CI | `pytest` + GitHub Actions |

---

## Project Structure

```
RAG_Project_Clean/
├── src/
│   └── utils.py                # Shared helpers: chunking, embedding, metrics
├── tests/
│   └── test_utils.py           # Unit tests (pytest, 34 tests)
├── configs/
│   ├── config.py               # Paths & env vars (reads from .env via dotenv)
│   └── instruct_pipeline.py    # Custom HuggingFace generation pipeline
├── experiments/
│   └── run_experiment.py       # Papermill + MLflow experiment runner
├── notebooks/
│   ├── get_data.ipynb          # SEC EDGAR download
│   ├── scrape_financial_reports.ipynb
│   ├── preprocesing/
│   │   ├── 0_read_html.ipynb
│   │   ├── 1_pre_json_chunks.ipynb
│   │   └── 2_pre_json_splits.ipynb
│   ├── indexation/
│   │   └── 3_Indexar.ipynb
│   ├── retrieve/
│   │   └── 4_retrieve.ipynb
│   └── evaluation/
│       ├── 5_evaluate_RAG.ipynb
│       └── test_questions.csv
├── .github/
│   └── workflows/
│       └── tests.yml           # CI: runs pytest on every push/PR
├── requirements.txt
├── .env.example                # Template — copy to .env and fill in tokens
└── .gitignore
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/rafaelgutierrezn/financial-rag.git
cd RAG_Project_Clean
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch (CUDA 11.8 for RTX 3060)

```bash
pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux / macOS
# Edit .env and set your HUGGINGFACEHUB_API_TOKEN
```

### 5. Prepare data directories

```
data/
├── raw/html_reports/        ← downloaded HTML 10-K filings
└── processed/
    ├── json_reports/
    ├── json_chunks/
    └── index/FAISS/
```

---

## Running the Pipeline

### Full pipeline via Papermill + MLflow

```bash
python experiments/run_experiment.py
```

Executes notebooks 1–5 in sequence, logs all parameters and metrics to MLflow, and saves executed notebooks under `output_notebooks/`.

### Run a single stage

Open any numbered notebook in JupyterLab or VS Code and run it. The top cell contains Papermill-tagged parameters with sensible defaults — no other configuration needed for interactive use.

### View results

```bash
mlflow ui
# open http://localhost:5000
```

### Run tests

```bash
pytest tests/ -v
```

---

## Evaluation Metrics

All metrics are logged to MLflow and printed as a summary table at the end of `5_evaluate_RAG.ipynb`.

| Metric | What it measures |
|--------|-----------------|
| **Correctness** | Cosine similarity — generated answer vs. ground-truth answer |
| **Relevance** | Cosine similarity — generated answer vs. ground-truth evidence |
| **Faithfulness** | Cosine similarity — generated answer vs. retrieved context |
| **ROUGE-L** | Longest common subsequence F1 — generated vs. reference answer |
| **Exact Match** | Normalised string equality (punctuation- and case-insensitive) |
| **Recall@K** | Fraction of queries where a relevant chunk appears in top-K results |

> ROUGE-L and Exact Match are the primary quality signals. Cosine metrics provide complementary signal but are influenced by the same embedding model used for retrieval.

### Test dataset — `test_questions.csv`

`notebooks/evaluation/test_questions.csv` contains 150 question–answer–evidence triples drawn manually from the same SEC 10-K filings used to build the FAISS index. Each row was written by hand:

| Column | Description |
|--------|-------------|
| `question` | A natural-language query about a specific filing (revenue, risk factors, segment data, etc.) |
| `answer` | The expected answer, taken verbatim or closely paraphrased from the source document |
| `evidence` | The exact passage from the 10-K that supports the answer |

Questions cover a mix of factual lookups (single-number answers), multi-sentence summaries, and cross-section reasoning to stress-test both retrieval and generation. The set is small by design — large enough to surface systematic retrieval failures while remaining cheap to evaluate locally without a cloud API.

---

## Key Engineering Decisions

- **`src/utils.py` as a shared module** — all helper functions (`split_into_chunks`, `extract_answer`, `build_embedder`, `rouge_l`, `exact_match`) live in one tested module rather than being redefined per notebook.
- **`build_embedder(model_name)`** — a factory that wraps `embed_query` to prepend the BGE instruction prefix at query time only, fixing the asymmetry between index-time and query-time representations.
- **Two-stage chunking** — stage 1 (1,000 chars, hard split) captures table boundaries; stage 2 (LangChain recursive, 1,500 chars / 200 overlap) adds sentence-aware overlap for better context continuity.
- **FAISS on CPU** — keeps the 12 GB VRAM entirely free for the LLM and embedder. Index search is fast enough for interactive use with ~40K chunks.
- **Papermill + MLflow** — every hyperparameter (chunk size, embedding model, LLM, retriever-k, temperature) is injectable at runtime, making ablation studies a one-line config change.
- **`python-dotenv` for credentials** — `configs/config.py` loads all secrets from `.env` at startup; `.env` is git-ignored and `.env.example` documents the required variables without exposing values.
- **Lightweight CI** — `.github/workflows/tests.yml` installs only the pure-Python subset of dependencies (`numpy`, `scikit-learn`, `rouge-score`, `pytest`) and runs the full unit-test suite on every push and pull request — no GPU or heavy ML deps needed.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Contact

Rafael Gutierrez · [rafaelgutierrez.n@gmail.com](mailto:rafaelgutierrez.n@gmail.com)
