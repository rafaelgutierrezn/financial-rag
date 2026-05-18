"""
Unit tests for src/utils.py

Run with:  pytest tests/
"""

import pytest
from src.utils import (
    split_into_chunks,
    extract_answer,
    normalize_answer,
    exact_match,
    table_to_markdown,
    rouge_l,
)


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    def test_short_text_returns_single_chunk(self):
        text = "This is short."
        assert split_into_chunks(text, max_chars=100) == ["This is short."]

    def test_exact_length_returns_single_chunk(self):
        text = "a" * 50
        assert split_into_chunks(text, max_chars=50) == ["a" * 50]

    def test_splits_on_newline_boundary(self):
        text = "line one\nline two\nline three"
        chunks = split_into_chunks(text, max_chars=15)
        # Every chunk must be at most 15 chars
        assert all(len(c) <= 15 for c in chunks)
        # All content is preserved
        assert " ".join(chunks).replace(" ", "\n").count("line") == 3

    def test_falls_back_to_hard_cut_when_no_newline(self):
        text = "a" * 200
        chunks = split_into_chunks(text, max_chars=50)
        assert len(chunks) == 4
        assert all(len(c) == 50 for c in chunks)

    def test_empty_text_returns_empty_list(self):
        assert split_into_chunks("", max_chars=100) == []

    def test_multiple_splits_reconstruct_original(self):
        text = "word " * 100
        chunks = split_into_chunks(text.strip(), max_chars=50)
        reconstructed = "\n".join(chunks)
        assert "word" in reconstructed


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    def test_extracts_after_answer_prefix(self):
        text = "Some context.\n\nAnswer: The capital is Paris."
        assert extract_answer(text) == "The capital is Paris."

    def test_case_insensitive(self):
        assert extract_answer("ANSWER: yes") == "yes"
        assert extract_answer("answer: no") == "no"

    def test_returns_full_text_when_no_prefix(self):
        text = "The capital is Paris."
        assert extract_answer(text) == "The capital is Paris."

    def test_strips_whitespace(self):
        assert extract_answer("Answer:   $27 million  ") == "$27 million"

    def test_multiline_answer_captured(self):
        text = "Answer: Line one\nLine two"
        result = extract_answer(text)
        assert "Line one" in result
        assert "Line two" in result


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("HELLO") == "hello"

    def test_strips_punctuation(self):
        assert normalize_answer("$27.") == "27"

    def test_collapses_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_usd_equivalence(self):
        assert normalize_answer("$27 million") == normalize_answer("27 million")


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical_strings(self):
        assert exact_match("Paris", "Paris") == 1

    def test_case_insensitive(self):
        assert exact_match("paris", "PARIS") == 1

    def test_different_strings(self):
        assert exact_match("Paris", "London") == 0

    def test_punctuation_normalised(self):
        assert exact_match("$27.", "$27") == 1

    def test_partial_match_is_zero(self):
        assert exact_match("27 million dollars", "27") == 0


# ---------------------------------------------------------------------------
# table_to_markdown
# ---------------------------------------------------------------------------

class TestTableToMarkdown:
    def test_basic_table(self):
        table = {"headers": ["A", "B"], "rows": [["1", "2"], ["3", "4"]]}
        md = table_to_markdown(table)
        assert "A | B" in md
        assert "---" in md
        assert "1 | 2" in md
        assert "3 | 4" in md

    def test_single_column(self):
        table = {"headers": ["Value"], "rows": [["42"]]}
        md = table_to_markdown(table)
        assert "Value" in md
        assert "42" in md

    def test_empty_rows(self):
        table = {"headers": ["X", "Y"], "rows": []}
        md = table_to_markdown(table)
        assert "X | Y" in md
        assert "---" in md


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical_strings_score_one(self):
        assert rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_prediction_score_zero(self):
        assert rouge_l("", "hello world") == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = rouge_l("hello there", "hello world")
        assert 0.0 < score < 1.0

    def test_completely_different_score_low(self):
        score = rouge_l("cat", "completely different text here")
        assert score < 0.5
