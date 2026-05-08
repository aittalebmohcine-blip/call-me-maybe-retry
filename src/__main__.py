import time
from typing import Dict, Any
import json

from src.Pipeline import Pipeline
from src.Parser import Parser
from src.Models import FunctionDefinition
from src.Utils import Utils


def main() -> None:
    # -- loading and parsing --
    parser: Parser = Parser()

    # load functions definitions into model objects
    functions: list[FunctionDefinition] = parser.parse_func_defs()

    # store functions by names for easy access
    functions_by_name: Dict[str, "FunctionDefinition"] = {}
    for func in functions:
        functions_by_name[func.name] = func

    # load prompts into a list
    prompts: list[str] = parser.parse_prompts()

    # -- generation pipeline --
    pipline: Pipeline = Pipeline(functions_by_name=functions_by_name)

    ref_s: float = time.time()
    total_calls: list[Dict[str, Any]] = []
    for prompt in prompts:
        result: Dict[str, Any] = {}
        result["prompt"] = prompt
        s: float = time.time()
        built_prompt: str = Utils.build_prompt(prompt, functions)
        output: str = pipline.pipline(built_prompt, max_new_tokens=100)
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
