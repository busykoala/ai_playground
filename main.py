from pseudo_ai.model import Model

if __name__ == "__main__":
    text = "The Swiss man has traveled to New York. There he met Anna Wintour."

    # ----------------------------------------------
    # Example 1: Sentiment Analysis
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Example 2: Text Generation
    # ----------------------------------------------

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
