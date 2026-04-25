from typing import List
import json

from pydantic import BaseModel, Field

from src.Models import FunctionDefinition


class Parser(BaseModel):
    func_defs_path: str = Field(
        default="data/input/functions_definition.json")
    prompts_path: str = Field(default="data/input/function_calling_tests.json")

    def parse_func_defs(self) -> List["FunctionDefinition"]:
        functions: List["FunctionDefinition"] = []

        # load data from the file
        with open(self.func_defs_path) as f:
            data = json.load(f)

        # extract function definitions objs
        for element in data:
            functions.append(FunctionDefinition(**element))

        return functions

    def parse_prompts(self) -> List[str]:
        prompts: List[str] = []

        # load data from the file
        with open(self.prompts_path) as f:
            data = json.load(f)

        # extract prompts
        for element in data:
            prompts.append(element["prompt"])

        return prompts
