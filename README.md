*This project has been created as part of the 42 curriculum by mait-tal.*

# call-me-maybe

## Description

I built `call-me-maybe` to explore how large language models can be guided to produce valid JSON function calls instead of free-form text. The project implements a constrained decoding pipeline that combines an LLM with a custom state machine and token-level trie constraints. The goal is to reliably generate structured function call objects like:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

This is useful for testing whether LLMs can speak the language of computers and execute function-calling workflows without generating malformed JSON.

## Instructions

### Requirements

- Python 3.10 or newer
- `numpy`
- `pydantic`
- Local package `llm_sdk` included in the repository

### Install

From the repository root:

```bash
python -m pip install -r requirements.txt 2>/dev/null || true
python -m pip install -e .
```

If you do not have a `requirements.txt`, install dependencies directly:

```bash
python -m pip install numpy pydantic
```

### Run

I run the project with:

```bash
python -m src
```

By default it reads:

- `data/input/function_definitions.json`
- `data/input/function_calling_tests.json`

and writes results to:

- `data/output/function_calling_results.json`

You can also pass custom paths:

```bash
python -m src --functions_definitions data/input/function_definitions.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Example usage

The project is designed to map natural language prompts to function calls. A few example prompts from `data/input/function_calling_tests.json` are:

- `What is the sum of 2 and 3?`
- `Greet shrek`
- `Reverse the string 'hello'`
- `What is the square root of 16?`
- `Replace all vowels in 'Programming is fun' with asterisks`

A generated output example is:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

## Algorithm explanation

I implemented a constrained decoding approach by defining a JSON grammar state machine and enforcing allowed tokens at each generation step. The pipeline builds two token-level tries: one for valid function names and one for valid parameter names for the chosen function.

The generation loop works like this:

1. Encode the prompt with the LLM tokenizer.
2. Request the next token logits from the model.
3. Restrict the logits to tokens that are valid for the current grammar state.
4. Choose the highest-scoring token among allowed tokens.
5. Transition the parser state based on the chosen token.
6. When the function name is complete, load the corresponding parameter schema and build the parameter trie.

This constrained decoding keeps the LLM output inside a valid JSON function call structure and prevents common errors like missing quotes, extra commas, or invalid parameter names.

## Design decisions

- I used a state machine because JSON generation is inherently sequential and grammar-driven.
- I built tries from tokenizer token IDs to ensure token-level constraints match the model’s vocabulary.
- I separated parsing logic into `Parser`, `Pipeline`, `Models`, and `Utils` so the pipeline remains focused on generation and state tracking.
- I used `pydantic` models for function definitions and pipeline validation, which makes input loading and schema handling more robust.
- I avoided overwriting protected files by checking output paths before writing results.

## Performance analysis

- Accuracy: The constrained decoder is highly reliable for the supported function definitions and sample prompts, because it enforces syntax and valid function/parameter names.
- Speed: Generation is efficient for short function-call JSON outputs. Each prompt completes in a few seconds, depending on the underlying LLM model implementation.
- Reliability: The approach is reliable for the current project scope, but it depends on the model’s tokenizer and the quality of the supplied logits. The state machine ensures structural correctness even if the model’s raw logits suggest invalid tokens.

## Challenges faced

- One challenge was preventing malformed JSON while still allowing the LLM to choose among valid function names and parameters.
- I solved this by creating token-level tries and a strict state transition system that only permits grammatically valid tokens.
- Another challenge was determining parameter types dynamically after the function name is chosen. I handled that by delaying schema loading until the function name has been fully decoded.
- I also had to handle edge cases in prompt input and output file protection to avoid accidental overwrites.

## Testing strategy

I validated the implementation by:

- Running the project on the sample prompts in `data/input/function_calling_tests.json`.
- Verifying that outputs are valid JSON objects with the expected `name` and `parameters` fields.
- Ensuring the state machine transitions correctly through all supported token states.
- Confirming that invalid output paths are rejected before writing.

For future work, I would add unit tests for the state transitions in `src/Pipeline.py`, trie construction in `src/Utils.py`, and file parsing in `src/Parser.py`.

## Resources

- JSON specification: https://www.json.org/json-en.html
- Prompt engineering and function calling with LLMs: general online tutorials and model documentation
- `llm_sdk` model docs: local SDK reference used by this project

### AI usage

I used AI indirectly as part of this project’s topic: the code drives a language model through constrained decoding. I did not use AI to write the core algorithm in the repository; instead, I used the model for token scoring and generation.

The AI model is used for:

- generating logits for the next token during decoding
- supporting tokenization and vocabulary lookup

The rest of the project is implemented in Python, with deterministic structures and grammar rules layered on top of the model.
