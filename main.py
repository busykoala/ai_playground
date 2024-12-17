from io import BytesIO

import requests
from PIL import Image

from pseudo_ai.model import Model


def sentiment_analysis(text):
    model = "dbmdz/bert-large-cased-finetuned-conll03-english"
    task = "ner"

    print("\n--- Local Pipeline Results (CPU) ---")
    local_model = Model(
        model,
        task,
        local=True,
    )
    print(local_model.predict(text))

    print("\n--- Remote Pipeline Results (API) ---")
    remote_model = Model(
        model,
        task,
        local=False,
    )
    print(remote_model.predict(text))


def text_generation(text):
    model = "microsoft/phi-2"
    task = "text-generation"

    print("\n--- Local Pipeline Results (CPU) ---")
    local_model = Model(
        model,
        task,
        local=True,
        temperature=0.5,
        do_sample=True,
    )
    print(local_model.predict(text))

    print("\n--- Remote Pipeline Results (API) ---")
    remote_model = Model(
        model,
        task,
        local=False,
        temperature=0.5,
        do_sample=True,
    )
    print(remote_model.predict(text))


def image_to_text():
    model = "Salesforce/blip-image-captioning-base"
    task = "image-to-text"

    image_url = "https://http.cat/418"
    image = Image.open(BytesIO(requests.get(image_url).content))

    print("\n--- Local Pipeline Results (CPU) ---")
    local_model = Model(
        model,
        task,
        local=True,
    )
    print(local_model.predict(image))

    print("\n--- Remote Pipeline Results (API) ---")
    remote_model = Model(
        model,
        task,
        local=False,
    )
    print(remote_model.predict(image))


if __name__ == "__main__":
    text = "The Swiss man has traveled to New York. There he met Anna Wintour."
    sentiment_analysis(text)
    text_generation(text)
    image_to_text()
