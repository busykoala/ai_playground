import hashlib
from typing import Dict
from typing import TypedDict

import pandas as pd
from transformers import pipeline


class PseudonymizeResult(TypedDict):
    pseudonymized_text: str
    pseudonym_map: Dict[str, str]


def preprocess_text(
    text: str, sensitive_data: pd.DataFrame, pseudonym_map: Dict[str, str]
) -> str:
    if "IBAN" in sensitive_data.columns:
        for iban in sensitive_data[
            "IBAN"
        ].dropna():  # Drop NaN values and iterate
            pseudonym = hashlib.sha256(iban.encode("utf-8")).hexdigest()[:10]
            pseudonym_map[iban] = pseudonym
            text = text.replace(iban, pseudonym)

    if "CustomerNumber" in sensitive_data.columns:
        for customer_number in sensitive_data["CustomerNumber"].dropna():
            pseudonym = hashlib.sha256(
                customer_number.encode("utf-8")
            ).hexdigest()[:10]
            pseudonym_map[customer_number] = pseudonym
            text = text.replace(customer_number, pseudonym)

    return text


def pseudonymize(
    text: str, sensitive_data: pd.DataFrame
) -> PseudonymizeResult:
    pseudonym_map: Dict[str, str] = {}

    # Preprocess sensitive terms (IBAN and CustomerNumber)
    text = preprocess_text(text, sensitive_data, pseudonym_map)

    # Load pre-trained NER pipeline
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )

    # Detect entities
    raw_entities = ner(text)
    if not isinstance(raw_entities, list) or not all(
        isinstance(entity, dict) for entity in raw_entities
    ):
        raise ValueError("Unexpected output from NER pipeline")

    # Replace detected entities
    for entity in sorted(raw_entities, key=lambda x: x["start"], reverse=True):
        word = entity["word"]
        pseudonym = hashlib.sha256(word.encode("utf-8")).hexdigest()[:10]
        pseudonym_map[word] = pseudonym
        text = text[: entity["start"]] + pseudonym + text[entity["end"] :]

    return {"pseudonymized_text": text, "pseudonym_map": pseudonym_map}
