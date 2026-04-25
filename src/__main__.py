from typing import Dict
from src.Pipeline import Pipeline
from src.Parser import Parser
from src.Models import FunctionDefinition


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
    print(pipline.generate(prompts[0]))

    # -- ouput --
    # format and save ouput


if __name__ == "__main__":
    main()
