from unittest.mock import patch

import numpy as np
import pytest

from ai_playground.merge_ner_results import merge_local_ner_results


def round_scores(results, precision=4):
    """Utility function to round scores in NER results."""
    for res in results:
        res["score"] = round(res["score"], precision)
    return results


@pytest.fixture
def mock_env():
    """Mock environment variable for Hugging Face API token."""
    with patch.dict(
        "os.environ", {"HUGGINGFACEHUB_API_TOKEN": "mock_api_token"}
    ):
        yield


@pytest.fixture
def ner_local_results():
    """Sample local NER results for testing."""
    return [
        {
            "entity": "I-MISC",
            "score": np.float32(0.99846554),
            "index": 2,
            "word": "Swiss",
            "start": 4,
            "end": 9,
        },
        {
            "entity": "I-LOC",
            "score": np.float32(0.9996418),
            "index": 7,
            "word": "New",
            "start": 30,
            "end": 33,
        },
        {
            "entity": "I-LOC",
            "score": np.float32(0.99959344),
            "index": 8,
            "word": "York",
            "start": 34,
            "end": 38,
        },
        {
            "entity": "I-PER",
            "score": np.float32(0.99606055),
            "index": 13,
            "word": "Anna",
            "start": 53,
            "end": 57,
        },
        {
            "entity": "I-PER",
            "score": np.float32(0.99870443),
            "index": 14,
            "word": "Win",
            "start": 58,
            "end": 61,
        },
        {
            "entity": "I-PER",
            "score": np.float32(0.99451065),
            "index": 15,
            "word": "##tour",
            "start": 61,
            "end": 65,
        },
    ]


@pytest.fixture
def original_text():
    """Original text input for accurate reconstruction."""
    return "The Swiss man has traveled to New York. There he met Anna Wintour."


def test_merge_local_ner_results(ner_local_results, original_text):
    """Test merging of local NER results."""
    merged = merge_local_ner_results(ner_local_results, original_text)

    expected_output = [
        {
            "entity_group": "MISC",
            "score": 0.99847,
            "word": "Swiss",
            "start": 4,
            "end": 9,
        },
        {
            "entity_group": "LOC",
            "score": 0.99962,
            "word": "New York",
            "start": 30,
            "end": 38,
        },
        {
            "entity_group": "PER",
            "score": 0.99642,
            "word": "Anna Wintour",
            "start": 53,
            "end": 65,
        },
    ]

    # Round scores for consistent comparison
    merged = round_scores(merged)
    expected_output = round_scores(expected_output)
    assert merged == expected_output
