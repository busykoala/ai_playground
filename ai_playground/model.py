import base64
import os
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline

from ai_playground.merge_ner_results import merge_local_ner_results

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
            return lambda text: merge_local_ner_results(
                pipeline(
                    self.task,
                    model=self.model,
                    device=-1,
                    **self.pipeline_kwargs,
                )(text),
                text,
            )
        return lambda text: pipeline(
            self.task, model=self.model, device=-1, **self.pipeline_kwargs
        )(text)

    def remote_pipeline(self):
        """
        Return a remote Hugging Face API call with additional arguments.
        """
        api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {
            "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
        }

        def call_api(input_data, **kwargs):
            payload = self._prepare_payload(input_data)
            payload.update(kwargs)
            response = requests.post(api_url, headers=headers, json=payload)
            return response.json()

        return call_api

    @staticmethod
    def _prepare_payload(input_data):
        if isinstance(input_data, Image.Image):
            img_format = input_data.format
            buffered = BytesIO()
            input_data.save(buffered, format=img_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return {"inputs": img_str}
        else:
            return {"inputs": input_data}
