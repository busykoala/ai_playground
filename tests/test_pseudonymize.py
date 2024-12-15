import pandas as pd

from pseudo_ai.pseudonymize import pseudonymize

# Sample sensitive data for IBANs and customer numbers
sensitive_data = pd.DataFrame(
    {
        "IBAN": ["DE89370400440532013000", "GB82WEST12345698765432"],
        "CustomerNumber": ["CUS-12345678", "CUS-87654321"],
    }
)


def test_pseudonymize_basic_name_replacement():
    text = "John Doe visited Jane Doe yesterday."
    result = pseudonymize(text, sensitive_data)
    assert "John" not in result["pseudonymized_text"]
    assert "Doe" not in result["pseudonymized_text"]
    assert "Jane" not in result["pseudonymized_text"]
    assert "visited" in result["pseudonymized_text"]
    assert "yesterday." in result["pseudonymized_text"]


def test_pseudonymize_address_replacement():
    text = "Alice lives at 456 Elm Street, Springfield."
    result = pseudonymize(text, sensitive_data)
    assert "Alice" not in result["pseudonymized_text"]
    assert "456" in result["pseudonymized_text"]
    assert "Elm Street" not in result["pseudonymized_text"]
    assert "Springfield" not in result["pseudonymized_text"]
    assert "lives at" in result["pseudonymized_text"]


def test_pseudonymize_case_sensitivity():
    text = "john lives in NEW YORK."
    result = pseudonymize(text, sensitive_data)
    assert "john" not in result["pseudonymized_text"]
    assert "NEW YORK" not in result["pseudonymized_text"]
    assert "lives in" in result["pseudonymized_text"]


def test_pseudonymize_iban_replacement():
    text = "My IBAN is DE89370400440532013000."
    result = pseudonymize(text, sensitive_data)
    pseudonym_map = result["pseudonym_map"]
    assert "DE89370400440532013000" not in result["pseudonymized_text"]
    assert "My IBAN is" in result["pseudonymized_text"]
    assert (
        pseudonym_map["DE89370400440532013000"] in result["pseudonymized_text"]
    )


def test_pseudonymize_customer_number_replacement():
    text = "Customer number CUS-12345678 is valid."
    result = pseudonymize(text, sensitive_data)
    pseudonym_map = result["pseudonym_map"]
    assert "CUS-12345678" not in result["pseudonymized_text"]
    assert "Customer number" in result["pseudonymized_text"]
    assert "is valid." in result["pseudonymized_text"]
    assert pseudonym_map["CUS-12345678"] in result["pseudonymized_text"]


def test_pseudonymize_multiple_entities():
    text = "John and Jane visited Paris and London."
    result = pseudonymize(text, sensitive_data)
    assert "John" not in result["pseudonymized_text"]
    assert "Jane" not in result["pseudonymized_text"]
    assert "Paris" not in result["pseudonymized_text"]
    assert "London" not in result["pseudonymized_text"]
    assert "visited" in result["pseudonymized_text"]


def test_pseudonymize_non_sensitive_text():
    text = "No sensitive terms here."
    result = pseudonymize(text, sensitive_data)
    assert result["pseudonymized_text"] == text


def test_pseudonymize_punctuation_preservation():
    text = "John Doe, who lives in Paris, visited the Eiffel Tower."
    result = pseudonymize(text, sensitive_data)
    assert "John" not in result["pseudonymized_text"]
    assert "Doe" not in result["pseudonymized_text"]
    assert "Paris" not in result["pseudonymized_text"]
    assert "lives in" in result["pseudonymized_text"]
    assert "," in result["pseudonymized_text"]
    assert "." in result["pseudonymized_text"]


def test_pseudonymize_numbers_in_text():
    text = "The address is 456 Main Street and the ZIP is 12345."
    result = pseudonymize(text, sensitive_data)
    assert "Main Street" not in result["pseudonymized_text"]
    assert "456" in result["pseudonymized_text"]
    assert "12345" in result["pseudonymized_text"]
    assert "The address is" in result["pseudonymized_text"]
