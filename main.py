import pandas as pd

from pseudo_ai.pseudonymize import pseudonymize
from pseudo_ai.pseudonymize import reverse_pseudonymize

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
pseudonymized_text = result["pseudonymized_text"]
pseudonym_map = result["pseudonym_map"]

print("Pseudonymized Text:", pseudonymized_text)
print("Pseudonym Map:", pseudonym_map)

# Reverse pseudonymize
original_text = reverse_pseudonymize(pseudonymized_text, pseudonym_map)

print("Reversed Text:", original_text)
