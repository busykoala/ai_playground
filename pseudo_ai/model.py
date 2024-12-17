from transformers import pipeline
from dotenv import load_dotenv
import os
import requests
import numpy as np

load_dotenv()

class Model:
    def __init__(self, model: str, task: str, local: bool = False, **kwargs):
        """
        Initialize the model for local or remote inference.
        :param model: Model name (e.g., "gpt2")
        :param task: Task type (e.g., "text-generation")
        :param local: Use local pipeline or remote API
        :param kwargs: Additional arguments for pipeline or API
        """
        self.model = model
        self.task = task
        self.local = local
        self.pipeline_kwargs = kwargs

        self.predict = (
            self.local_pipeline()
            if local
            else lambda text: self.remote_pipeline()(text, **kwargs)
        )

    def local_pipeline(self):
        """
        Return a local Hugging Face pipeline with additional arguments.
        """
        if self.task == "ner":
            return lambda text: self.merge_local_ner_results(pipeline(self.task, model=self.model, device=-1, **self.pipeline_kwargs)(text))
        return lambda text: pipeline(self.task, model=self.model, device=-1, **self.pipeline_kwargs)(text)

    def remote_pipeline(self):
        """
        Return a remote Hugging Face API call with additional arguments.
        """
        api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}

        def call_api(text, **kwargs):
            payload = {"inputs": text}
            payload.update(kwargs)
            response = requests.post(api_url, headers=headers, json=payload)
            return response.json()

        return call_api

    @staticmethod
    def merge_local_ner_results(local_results):
        """
        Merge and unify local NER results into grouped entities.

        :param local_results: List of local pipeline NER results (split words).
        :return: List of grouped NER results (merged entities).
        """
        merged_results = []
        current_entity = None
        current_word = ""
        current_start = None
        current_end = None
        current_scores = []

        for result in local_results:
            entity = result["entity"]
            word = result["word"]
            start = result["start"]
            end = result["end"]
            score = result["score"]

            # Handle subword tokens (e.g., "##tour")
            if word.startswith("##"):
                word = word[2:]  # Remove "##" prefix
                current_word += word
                current_end = end
                current_scores.append(float(score))
            elif entity.startswith("I-") and current_entity == entity.split("-")[-1]:
                current_word += f" {word}"
                current_end = end
                current_scores.append(float(score))
            else:
                # Store the previous entity
                if current_entity:
                    merged_results.append({
                        "entity_group": current_entity,
                        "score": float(np.mean(current_scores)),
                        "word": current_word,
                        "start": current_start,
                        "end": current_end,
                    })

                # Start a new entity
                current_entity = entity.split("-")[-1]  # Extract entity group (e.g., LOC)
                current_word = word
                current_start = start
                current_end = end
                current_scores = [float(score)]

        # Append the last entity
        if current_entity:
            merged_results.append({
                "entity_group": current_entity,
                "score": float(np.mean(current_scores)),
                "word": current_word,
                "start": current_start,
                "end": current_end,
            })

        return merged_results
