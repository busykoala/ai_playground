# AI Playground

This is my playground to experiment with huggingface models.

![AI Playground](./assets/cover.webp)

## Installation & Setup

```bash
poetry install
```

Add `HUGGINGFACEHUB_API_TOKEN=hf_xxxxxx` to your `.env` file with your
huggingface token.

## Usage

There are some examples in the `main.py` file. You can run it with:

```bash
poetry run python main.py
```

## Tests

```bash
poetry run pytest
```

## Formatting

```bash
poetry run ruff format
poetry run ruff check --fix
```
