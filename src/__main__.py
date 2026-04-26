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
You are a function calling system.

Available functions:
{functions_text}

Return ONLY a JSON object:
{{"name": "...", "parameters": {{...}}}}

User request:
{prompt_text}

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
    pipline = Pipeline()
    inhanced_prompt = build_prompt(prompts[1], functions)
    print(pipline.generate(inhanced_prompt))

    # -- ouput --
    # format and save ouput


if __name__ == "__main__":
    main()
