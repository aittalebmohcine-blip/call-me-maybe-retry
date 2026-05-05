import time
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
Functions:
{functions_text}

Request: {prompt_text}
JSON:
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

    # x = [2, 3, 4, 5, 8, 9, 10]
    # x = [8, 9, 10]
    x = range(11)
    ref_s = time.time()
    for i in x:
        s = time.time()
        built_prompt = build_prompt(prompts[i], functions)
        output = pipline.stage1(built_prompt, max_new_tokens=100)
        print("Output:\n", output)
        print(f"Execution time: {time.time() - s:.2f} seconds")

    print(f"\ntotal time: {time.time() - ref_s:.2f} seconds")
    # for prompt in prompts:
    #    built_prompt = build_prompt(prompt, functions)
    #    output = pipline.stage1(built_prompt)
    #    print("Output:\n", output)
    #    print("-" * 50)

    # -- ouput --
    # format and save ouput


if __name__ == "__main__":
    main()
