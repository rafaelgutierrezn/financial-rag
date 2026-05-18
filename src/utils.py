"""
Shared utilities for the Financial RAG pipeline.

All text helpers, embedding construction, and evaluation metrics live here
so notebooks stay thin and functions are independently testable.
"""

import re
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def table_to_markdown(table: dict) -> str:
    """Convert a {headers, rows} dict to a Markdown table string."""
    headers = " | ".join(str(h) for h in table["headers"])
    separator = " | ".join(["---"] * len(table["headers"]))
    rows = "\n".join(" | ".join(str(c) for c in row) for row in table["rows"])
    return f"{headers}\n{separator}\n{rows}"


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    """
    Greedily split *text* into chunks of at most *max_chars* characters,
    preferring to break on newline boundaries.
    """
    chunks: List[str] = []
    while len(text) > max_chars:
        split_point = text.rfind("\n", 0, max_chars)
        if split_point == -1:
            split_point = max_chars
        chunks.append(text[:split_point].strip())
        text = text[split_point:].strip()
    if text:
        chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# Answer extraction & normalisation
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Return the text after the first 'Answer:' marker, or the full text."""
    match = re.search(r"(?i)Answer:\s*(.*)", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for exact-match."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: str, reference: str) -> int:
    """Return 1 if normalised strings are identical, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(reference))


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def build_embedder(model_name: str):
    """
    Return a LangChain embedder with correct query-time prefix handling.

    BGE models (bge-*) require a 'Represent: ' prefix on queries at retrieval
    time but NOT on documents at index time.  This wrapper patches embed_query
    so the pipeline is symmetric without manual string manipulation in notebooks.
    """
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings,
        SentenceTransformerEmbeddings,
    )

    if "bge" in model_name.lower():
        class _BGEEmbeddings(HuggingFaceEmbeddings):
            def embed_query(self, text: str) -> List[float]:
                if not text.startswith("Represent: "):
                    text = "Represent: " + text
                return super().embed_query(text)

        return _BGEEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    return SentenceTransformerEmbeddings(model_name=model_name)


def cos_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(
        cosine_similarity(
            np.array(a).reshape(1, -1),
            np.array(b).reshape(1, -1),
        )[0][0]
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 between prediction and reference strings."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(reference, prediction)["rougeL"].fmeasure
