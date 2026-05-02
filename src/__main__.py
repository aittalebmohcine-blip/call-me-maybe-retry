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
    pipline = Pipeline(functions_by_name=functions_by_name)

    for prompt in prompts:
        built_prompt = build_prompt(prompt, functions)
        output = pipline.stage1(built_prompt)
        print("Output:\n", output)
        print("-" * 50)

    # -- ouput --
    # format and save ouput


if __name__ == "__main__":
    main()
