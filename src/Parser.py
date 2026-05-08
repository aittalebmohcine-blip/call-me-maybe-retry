from pathlib import Path
from typing import List, Dict
import json

from pydantic import BaseModel, Field

from src.Models import FunctionDefinition


class Parser(BaseModel):
    func_defs_path: str = Field(
        default="data/input/functions_definition.json")

    prompts_path: str = Field(default="data/input/function_calling_tests.json")

    path_to_ouput_file: str = Field(
        default="data/output/function_calling_result.json")

    def parse_func_defs(self) -> List["FunctionDefinition"]:
        functions: List["FunctionDefinition"] = []

        # load data from the file
        try:
            with open(self.func_defs_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"ERROR: could not find the input file '{self.func_defs_path}'"
            )
        except PermissionError:
            raise ValueError(
                "ERROR: you don't have permission"
                f" to input file '{self.func_defs_path}'"
            )

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {self.func_defs_path}")

        except IsADirectoryError:
            raise ValueError(f"'{self.func_defs_path}' is a directory !")

        except OSError as e:
            raise ValueError(f"File error: {e}")

        # extract function definitions objs
        for element in data:
            functions.append(FunctionDefinition(**element))

        return functions

    def parse_prompts(self) -> List[str]:
        prompts: List[str] = []

        # load data from the file
        try:
            with open(self.prompts_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"ERROR: could not find the input file '{self.prompts_path}'"
            )

        except PermissionError:
            raise ValueError(
                "ERROR: you don't have permission"
                f" to input file '{self.prompts_path}'"
            )

        except IsADirectoryError:
            raise ValueError(f"'{self.prompts_path}' is a directory !")

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {self.prompts_path}")

        except OSError as e:
            raise ValueError(f"File error: {e}")

        # extract prompts
        for element in data:
            prompts.append(element["prompt"])

        return prompts

    def dump_output(self, ouput: List[Dict]) -> None:
        try:
            output_path = Path(self.path_to_ouput_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(ouput, f, indent=2)

        except PermissionError:
            raise ValueError(
                "ERROR: you don't have permission"
                f" to ouput file '{self.path_to_ouput_file}'"
            )

        except IsADirectoryError:
            raise ValueError(f"'{self.path_to_ouput_file}' is a directory !")
