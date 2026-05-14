*This project has been created as part of the 42 curriculum by mait-tal.*

# call-me-maybe

## Description

`call-me-maybe` is a constrained decoding framework that guides Large Language Models (LLMs) to generate **valid, structured JSON function calls** from natural language prompts. By combining LLM inference with a state machine and token-level constraints, the system reliably produces properly formatted function call objects without malformed JSON, missing quotes, or invalid parameters.

### Key Features

- **Constrained Decoding**: Grammar-based token filtering at each generation step
- **State Machine Parser**: JSON parsing state machine ensures valid structure throughout generation
- **Token Trie Constraints**: Function names and parameter names restricted to valid options
- **Reliable Output**: Guaranteed valid JSON structure—no more parsing errors
- **Extensible**: Easy to add new functions by updating the function definitions file

### Example Input & Output

**Input prompt:** `"What is the sum of 2 and 3?"`

**Output:**
```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Instructions

### Prerequisites

- Python 3.10+
- Dependencies: `numpy`, `pydantic`
- Local package: `llm_sdk` (included in repository)

### Installation

From the repository root:

```bash
# Install dependencies and package
make install
```


### Running the Pipeline

**Default execution:**
```bash
make run
# or
make
```

This processes:
- **Input:** `data/input/function_definitions.json` (available functions)
- **Input:** `data/input/function_calling_tests.json` (test prompts)
- **Output:** `data/output/function_calling_results.json` (generated function calls)

**Custom file paths:**
```bash
uv run python -m src \
  --functions_definitions path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

**Safety:** The pipeline refuses to overwrite `.py` files and protected configuration files (`pyproject.toml`, `uv.lock`).

## Project Structure

```
call_me_maybe/
├── src/                           # Main source code
│   ├── __main__.py               # Entry point; orchestrates the pipeline
│   ├── Models.py                 # Pydantic models for function definitions
│   ├── Parser.py                 # JSON file parsing and output writing
│   ├── Pipeline.py               # Core constrained decoding logic and state machine
│   └── Utils.py                  # Helper functions (trie building, prompt formatting)
├── llm_sdk/                       # Local LLM SDK for model interaction
├── data/
│   ├── input/
│   │   ├── function_definitions.json    # Available functions
│   │   └── function_calling_tests.json  # Test prompts
│   └── output/
│       └── function_calling_results.json # Generated results
├── Makefile                       # Build and utility commands
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

### Module Overview

| Module | Purpose |
|--------|---------|
| `__main__.py` | Argument parsing, file loading, pipeline orchestration, result saving |
| `Models.py` | `FunctionDefinition` class for type-safe function metadata |
| `Parser.py` | JSON file I/O with comprehensive error handling |
| `Pipeline.py` | State machine FSM, token constraint logic, generation loop |
| `Utils.py` | Trie construction, prompt formatting, schema utilities |

## Test Cases

The project includes example function definitions and test prompts in the data directory:

### Available Functions

All functions are defined in `data/input/function_definitions.json`. Example functions include:
- `fn_add_numbers(a: int, b: int)` → Returns the sum
- `fn_greet(name: str)` → Personalized greeting
- `fn_reverse_string(text: str)` → String reversal
- `fn_sqrt(x: number)` → Square root calculation
- `fn_replace_vowels(text: str, replacement: str)` → Vowel replacement

### Sample Prompts

From `data/input/function_calling_tests.json`:
- `"What is the sum of 2 and 3?"`
- `"Greet shrek"`
- `"Reverse the string 'hello'"`
- `"What is the square root of 16?"`
- `"Replace all vowels in 'Programming is fun' with asterisks"`

### Example Output

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## How It Works

### Architecture Overview

The system uses a **token-constrained generation approach** to ensure valid JSON output:

```
Prompt → Tokenizer → State Machine → Allowed Tokens → LLM Logits → Constrained Sampling → Next Token
                           ↑                                              ↓
                    Track JSON structure                          Valid token only
```

### Constrained Decoding Pipeline

The generation process follows these steps:

1. **Encode Prompt**: Convert the user prompt to token IDs using the model's tokenizer
2. **Compute Logits**: Request the next token logits from the LLM
3. **Build Constraints**: Based on current parser state, construct allowed token set
   - For function names: use trie of valid function names
   - For parameters: use trie of valid parameter names
   - For structure: enforce JSON grammar (brackets, quotes, colons, commas)
4. **Filter Logits**: Mask disallowed tokens with negative infinity in logit space
5. **Sample Token**: Select the highest-scoring token from allowed options
6. **Update State**: Transition the parser state machine based on the chosen token
7. **Repeat**: Continue until `DONE` state is reached

### State Machine

The parser implements a deterministic finite state machine (FSM) that validates JSON structure:

- **START** → `{` → **EXPECT_NAME_KEY_OPEN**
- **EXPECT_NAME_KEY_OPEN** → `"` → **EXPECT_NAME_KEY_BODY** → ... → **EXPECT_NAME_VALUE_BODY**
- **EXPECT_NAME_VALUE_BODY** (constrained to function names via trie) → **EXPECT_COMMA_AFTER_NAME**
- **EXPECT_COMMA_AFTER_NAME** → `,` → **EXPECT_PARAMS_KEY_OPEN**
- ... (parameters object parsing) ...
- **EXPECT_FINAL_OBJECT_CLOSE** → `}` → **DONE**

### Trie-Based Token Constraints

For function names and parameter names, the system builds token-level tries that mirror the vocabulary:

```
Valid function: "fn_add_numbers"
             ↓ tokenize each substring
Token IDs: [fn], [_add], [_numbers]
             ↓
Trie structure: {children: {fn_id: {children: {...}}}}
             ↓
At each step, only valid next tokens are allowed
```

### Key Guarantees

✓ **Valid JSON**: Every generated output is syntactically correct JSON
✓ **Valid Functions**: Function names are restricted to the provided definitions
✓ **Valid Parameters**: Parameter names match the function schema
✓ **Type Safety**: Parameter values respect their declared types (string, number, boolean)
✓ **No Hallucination**: LLM cannot generate parameters or functions that don't exist

## Adding New Functions

To extend the system with new functions:

1. **Add function definition** to `data/input/function_definitions.json`:
   ```json
   {
     "name": "fn_my_function",
     "description": "What this function does",
     "parameters": {
       "param1": {"type": "string"},
       "param2": {"type": "number"}
     },
     "returns": {"type": "string"}
   }
   ```

2. **Update test prompts** in `data/input/function_calling_tests.json` with prompts that exercise your new function.

3. **Run the pipeline** to test:
   ```bash
   make run
   ```

The system automatically rebuilds the function name trie and parameter schema for each new function—no code changes needed!



## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **State Machine** | JSON is fundamentally sequential and grammar-driven; FSM is a natural fit |
| **Token-Level Tries** | Ensures constraints match the model's tokenizer vocabulary exactly |
| **Modular Architecture** | Separates concerns: parsing (`Parser`), generation (`Pipeline`), utilities (`Utils`) |
| **Pydantic Models** | Type safety for function definitions and robust schema validation |
| **Output Safety Checks** | Prevents accidental overwrites of Python source files and configs |

## Performance & Reliability

- **Accuracy**: Near 100% for the supported function set—grammar and valid names are always enforced
- **Speed**: Completes in seconds per prompt (depends on underlying LLM model)
- **Reliability**: State machine guarantees structural correctness; output is always valid JSON

## Known Limitations & Future Improvements

- **Current**: Only supports JSON function calls; could extend to other structured formats (YAML, etc.)
- **Future**: Add nested objects/arrays in parameters for more complex types
- **Future**: Support optional/default parameters in function schemas
- **Future**: Caching for trie construction to speed up repeated generations

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Preventing malformed JSON while preserving LLM choice | Token-level tries + strict state transitions |
| Dynamic parameter schema loading | Delay schema loading until function name is complete |
| Edge cases in file I/O | Comprehensive error handling and path validation |
| Tokenizer mismatch between model and constraints | Build tries from actual model tokenizer output |

## Validation & Testing

**Current validation approach:**
- Run sample prompts from `data/input/function_calling_tests.json`
- Verify output is valid JSON with `name` and `parameters` fields
- Confirm state machine transitions through all parsing states
- Test file protection: verify `.py` files cannot be overwritten


## Resources

- **JSON Spec**: https://www.json.org/
- **Function Calling**: Relevant LLM model documentation and prompt engineering guides
- **LLM SDK**: Local SDK included in the `llm_sdk/` directory
- **State Machines**: General computer science and parsing theory resources

## Author Note

This project explores the intersection of constrained decoding and structured output generation. By combining a grammar-based state machine with token-level constraints, we enable LLMs to reliably produce valid JSON function calls—a key capability for AI-powered systems that need to integrate with software APIs and function libraries.