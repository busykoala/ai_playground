import pandas as pd

from pseudo_ai.pseudonymize import pseudonymize

# Example sensitive data
sensitive_data = pd.DataFrame(
    {
        "IBAN": ["DE89370400440532013000", None],
        "CustomerNumber": ["CUS-12345678", "CUS-87654321"],
    }
)

# Input text
text = "John Doe lives in Berlin. His IBAN is DE89370400440532013000, and his customer number is CUS-12345678."

# Pseudonymize
result = pseudonymize(text, sensitive_data)

print("Pseudonymized Text:", result["pseudonymized_text"])
print("Pseudonym Map:", result["pseudonym_map"])
