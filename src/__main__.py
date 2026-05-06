import time
from typing import Dict
import json

from src.Pipeline import Pipeline
from src.Parser import Parser
from src.Models import FunctionDefinition
from src.Utils import Utils


def main():
    # -- loading and parsing --
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

    ref_s = time.time()
    total_calls = []
    for prompt in prompts:
        result = {}
        result["prompt"] = prompt
        s = time.time()
        built_prompt = Utils.build_prompt(prompt, functions)
        output = pipline.stage1(built_prompt, max_new_tokens=100)
        result.update(json.loads(output))
        total_calls.append(result)
        print("Output:\n", result)
        print(f"Execution time: {time.time() - s:.2f} seconds")

    # --- Save ouput ---
    with open("data/output/output.json", "w") as f:
        json.dump(total_calls, f, indent=2)

    print(f"\ntotal time: {time.time() - ref_s:.2f} seconds")


if __name__ == "__main__":
    main()
