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
    """Utility functions for prompt building and trie construction.
    
    Provides static helper methods for loading function schemas, building
    token-based tries, and formatting prompts for the LLM pipeline.
    """
    @staticmethod
    def load_function_schema(
        name: str,
        functions_by_name: Dict[str, FunctionDefinition]
    ) -> Dict[str, Dict[str, str]]:
        """Load parameter schema for a function by name.
        
        Args:
            name: The name of the function.
            functions_by_name: A dictionary mapping function names to
                FunctionDefinition objects.
                
        Returns:
            The parameters dictionary for the specified function.
            
        Raises:
            KeyError: If the function name is not found in the dictionary.
        """
        return functions_by_name[name].parameters

    @staticmethod
    def build_trie(
        strings: List[str]
    ) -> Dict[str, Dict | bool]:
        """Build a token-level trie from a list of strings.
        
        Constructs a trie where each node represents a token (via the model's
        tokenizer) and terminal nodes mark complete strings in the list.
        Used for constraining token generation during decoding.
        
        Args:
            strings: A list of strings to insert into the trie.
            
        Returns:
            A trie structure represented as nested dictionaries with keys
            being token IDs, 'children' containing child nodes, and 'terminal'
            indicating if this node marks a complete string.
        """
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
        """Format a prompt with available functions for the LLM.
        
        Creates a formatted string that includes a list of available functions
        with their signatures and the user's request, formatted for LLM input.
        
        Args:
            prompt_text: The user's request or prompt.
            functions: List of available function definitions to include.
            
        Returns:
            A formatted prompt string ready for LLM processing, containing
            the function list and the user's request.
        """
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
    """Enum representing states in the JSON parsing state machine.
    
    Defines all possible states during constrained decoding of function
    call JSON output. States track parsing progress through the JSON
    structure (opening/closing braces, key/value pairs, etc.).
    """
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
    EXPECT_PARAM_BOOL_VALUE_BODY = auto()

    EXPECT_PARAM_NUM_VALUE_BODY = auto()

    EXPECT_PARAM_VALUE_OPEN = auto()
    EXPECT_PARAM_STRING_VALUE_BODY = auto()
    EXPECT_PARAM_VALUE_CLOSE = auto()

    EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE = auto()

    # final
    EXPECT_FINAL_OBJECT_CLOSE = auto()

    DONE = auto()
