import time
from typing import Dict, Any, List
import json
import argparse

from src.Pipeline import Pipeline
from src.Parser import Parser
from src.Models import FunctionDefinition
from src.Utils import Utils


def main() -> None:
    # input parsing
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--functions_definitions",
        default="data/input/function_definitions.json"
    )

    arg_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )

    arg_parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    args = arg_parser.parse_args()

    # -- loading and parsing files --
    parser: Parser = Parser(
        func_defs_path=args.functions_definitions,
        prompts_path=args.input,
        path_to_ouput_file=args.output
    )

    # load functions definitions into model objects
    functions: List[FunctionDefinition] = parser.parse_func_defs()

    # store functions by names for easy access
    functions_by_name: Dict[str, "FunctionDefinition"] = {}
    for func in functions:
        functions_by_name[func.name] = func

    # load prompts into a list
    prompts: List[str] = parser.parse_prompts()

    # -- generation pipeline --
    pipline: Pipeline = Pipeline(functions_by_name=functions_by_name)

    ref_s: float = time.time()
    total_calls: List[Dict[str, Any]] = []
    for prompt in prompts[:1]:
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
    parser.dump_output(total_calls)

    print(f"\ntotal time: {time.time() - ref_s:.2f} seconds")


if __name__ == "__main__":
    main()
