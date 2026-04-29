from typing import Dict, List
from src.Pipeline import Pipeline
from src.Parser import Parser
from src.Models import FunctionDefinition


def build_prompt(prompt_text: str, functions: List[FunctionDefinition]) -> str:
    functions_text = "\n".join(
        f"- {f.name}(" +
        ", ".join(f"{k}: {v['type']}" for k, v in f.parameters.items()) +
        ")"
        for f in functions
    )

    return f"""
You are a function-calling engine.
Given a user request, select the best matching
function and extract its arguments.

Available functions:
{functions_text}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no extra text.
- If no function matches, return: {{"name": "", "parameters": {{}}}}
- String arguments must be quoted. Number arguments must be unquoted.

Output format:
{{"name": "<function_name>", "parameters": {{<key>: <value>, ...}}}}

User request: {prompt_text}
Answer:
""".strip()


def build_fnname_extractor_prompt(prompt_text: str, functions) -> str:
    return f"""
You are a function-calling engine. Given a user request, return the name of the best matching function.

Available functions:
{functions}

Rules:
- Output ONLY the function name. No explanation, no markdown, no extra text.
- If no function matches, output: none

User request: {prompt_text}
Answer:
""".strip()


def build_args_extractor_prompt(
    prompt_text: str, function: FunctionDefinition
) -> str:
    return f"""
You are a function-calling engine. Given a user request and a function name, extract the arguments for that function.

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no extra text.
- String arguments must be quoted. Number arguments must be unquoted.
- If no arguments, return: {{}}

Function: {function}
User request: {prompt_text}
Answer:
""".strip()
    ...


def main():
    # -- parsing --

    parser = Parser()
    # load functions definitions into model objects
    functions = parser.parse_func_defs()
    # store functions by names for easy access
    functions_by_name: Dict[str, "FunctionDefinition"] = {}
    for func in functions:
        functions_by_name[func.name] = func
    # load prompts into a list
    prompts = parser.parse_prompts()

    # -- generation pipeline --
    pipline = Pipeline(functions=functions)

    name = pipline._stage1_extract_name(prompts[7])
    print(name)

    # -- ouput --
    # format and save ouput


if __name__ == "__main__":
    main()
