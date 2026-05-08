from pydantic import BaseModel
from typing import List, Any, Dict
from enum import Enum, auto
import json

from src.Parser import FunctionDefinition
from llm_sdk import Small_LLM_Model


model: Small_LLM_Model = Small_LLM_Model()
NEGATIVE_INF: float = float('-inf')

path_to_vocab_file: str = model.get_path_to_vocab_file()
with open(path_to_vocab_file, "r", encoding="utf-8") as f:
    vocab: Dict[str, int] = json.load(f)
# a dict for reverse lookup from token id to token text
id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}


class Utils(BaseModel):
    @staticmethod
    def load_function_schema(
        name: str,
        functions_by_name: Dict[str, FunctionDefinition]
    ) -> Dict[str, Dict[str, str]]:
        return functions_by_name[name].parameters

    @staticmethod
    def build_trie(
        strings: List[str]
    ) -> Dict[str, Dict | bool]:
        trie: Dict[str, Any] = {"children": {}, "terminal": False}

        name: str
        node: Dict[str, Any]
        for name in strings:
            node = trie

            id: int
            for id in model.encode(name)[0].tolist():
                if id not in node["children"]:
                    node["children"][id] = {
                        "children": {}, "terminal": False}

                node = node["children"][id]

            node["terminal"] = True

        return trie

    @staticmethod
    def build_prompt(
        prompt_text: str,
        functions: List[FunctionDefinition]
    ) -> str:
        functions_text: str = "\n".join(
            f"- {f.name}(" +
            ", ".join(f"{k}: {v['type']}" for k, v in f.parameters.items()) +
            ")"
            for f in functions
        )

        formatted_prompt: str = f"""
Functions:
{functions_text}

Request: {prompt_text}
JSON:
""".strip()
        return formatted_prompt


class State(Enum):
    # root
    START = auto()

    # "name" key
    EXPECT_NAME_KEY_OPEN = auto()
    EXPECT_NAME_KEY_BODY = auto()
    EXPECT_NAME_KEY_CLOSE = auto()

    EXPECT_COLON_AFTER_NAME_KEY = auto()

    # "name" value (function name)
    EXPECT_NAME_VALUE_OPEN = auto()
    EXPECT_NAME_VALUE_BODY = auto()
    EXPECT_NAME_VALUE_CLOSE = auto()

    EXPECT_COMMA_AFTER_NAME = auto()

    # "parameters" key
    EXPECT_PARAMS_KEY_OPEN = auto()
    EXPECT_PARAMS_KEY_BODY = auto()
    EXPECT_PARAMS_KEY_CLOSE = auto()

    EXPECT_COLON_AFTER_PARAMS_KEY = auto()

    # parameters object
    EXPECT_PARAMS_OBJECT_OPEN = auto()

    # parameter key
    EXPECT_PARAM_KEY_OPEN = auto()
    EXPECT_PARAM_KEY_BODY = auto()
    EXPECT_PARAM_KEY_CLOSE = auto()

    EXPECT_COLON_AFTER_PARAM_KEY = auto()

    # parameter value
    EXPECT_PARAM_NUM_VALUE_BODY = auto()

    EXPECT_PARAM_VALUE_OPEN = auto()
    EXPECT_PARAM_STRING_VALUE_BODY = auto()
    EXPECT_PARAM_VALUE_CLOSE = auto()

    EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE = auto()

    # final
    EXPECT_FINAL_OBJECT_CLOSE = auto()

    DONE = auto()
