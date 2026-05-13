from pydantic import BaseModel, Field
from typing import Dict, List, Union, Any, Tuple, Optional

from src.Utils import State, model, Utils, NEGATIVE_INF, id_to_token, vocab
from src.Models import FunctionDefinition


class Pipeline(BaseModel):
    """Orchestrates constrained decoding for JSON-formatted function calls.

    Implements a state machine-based approach to generate valid JSON function
    calls from an LLM by constraining token generation at each step based on
    the current parsing state and grammar rules.

    Attributes:
        functions_by_name: Dictionary mapping functionnames to their
        definitions.
        stack: Stack for tracking nested structures (objects/arrays).
        remaining_prams_counter: Counter for tracking unparsed parameters.
        function_schema: Current function's parameter schema being parsed.
    """
    functions_by_name: Dict[str, "FunctionDefinition"] = Field(
        ...,
        description="List of function definitions"
    )

    stack: List[str] = Field(
        default=[],
        description="Stack to manage nested structures"
    )

    remaining_prams_counter: int = Field(
        default=0,
        description="Counter for remaining parameters to parse"
    )
    function_schema: Dict[str, Dict[str, str]] = Field(
        default={},
        description="Schema for the selected function parameters"
    )

    def _allowed_tokens(
        self,
        state: State,
        cur_state: Dict[str, Any]
    ) -> set[int]:
        """Determine valid token IDs for the current parsing state.

        Based on the current state in the JSON parsing state machine,
        determines
        which tokens are grammatically valid. Returns their token IDs for use
        in logit masking during decoding.

        Args:
            state: The current state in the parsing state machine.
            cur_state: Dictionary tracking current parsing context, including
                the cursor position in the trie.

        Returns:
            A set of token IDs that are valid in this state.
        """
        allowed_strings: List[str] = []

        # root
        if state == State.START:
            allowed_strings = ["{"]

        # ----"name" key-----
        elif state == State.EXPECT_NAME_KEY_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_NAME_KEY_BODY:
            allowed_strings = ["name"]

        elif state == State.EXPECT_NAME_KEY_CLOSE:
            allowed_strings = ['"']
        # --------

        elif state == State.EXPECT_COLON_AFTER_NAME_KEY:
            allowed_strings = [":"]

        # ----"name" value (function name)----
        elif state == State.EXPECT_NAME_VALUE_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_NAME_VALUE_BODY:
            return set(cur_state["cursor"]["children"])

        elif state == State.EXPECT_NAME_VALUE_CLOSE:
            allowed_strings = ['"']
        # --------

        elif state == State.EXPECT_COMMA_AFTER_NAME:
            allowed_strings = [","]

        # ----"parameters" key----
        elif state == State.EXPECT_PARAMS_KEY_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_PARAMS_KEY_BODY:
            allowed_strings = ["parameters"]

        elif state == State.EXPECT_PARAMS_KEY_CLOSE:
            allowed_strings = ['"']
        # --------

        elif state == State.EXPECT_COLON_AFTER_PARAMS_KEY:
            allowed_strings = [":"]

        # ----parameters object----
        elif state == State.EXPECT_PARAMS_OBJECT_OPEN:
            allowed_strings = ["{"]
        # --------

        # ----parameter key----
        elif state == State.EXPECT_PARAM_KEY_OPEN:
            self.remaining_prams_counter -= 1
            allowed_strings = ['"']

        elif state == State.EXPECT_PARAM_KEY_BODY:
            return set(cur_state["cursor"]["children"])

        elif state == State.EXPECT_PARAM_KEY_CLOSE:
            allowed_strings = ['"']

        elif state == State.EXPECT_COLON_AFTER_PARAM_KEY:
            allowed_strings = [":"]
        # --------

        # ----parameter value----
        elif state == State.EXPECT_PARAM_BOOL_VALUE_BODY:
            if self.remaining_prams_counter > 0:
                allowed_strings = ["true", "false", ","]
            else:
                allowed_strings = ["true", "false", "}"]

        elif state == State.EXPECT_PARAM_NUM_VALUE_BODY:
            if self.remaining_prams_counter > 0:
                allowed_strings = list("0123456789-+.eE^,")
            else:
                allowed_strings = list("0123456789-+.eE^}")

        elif state == State.EXPECT_PARAM_VALUE_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE:
            if self.remaining_prams_counter > 0:
                allowed_strings = [","]
            else:
                allowed_strings = ["}"]
        # --------

        # final
        elif state == State.EXPECT_FINAL_OBJECT_CLOSE:
            allowed_strings = ["}"]

        else:
            allowed_strings = []

        # ---- APPLY THE RULE HERE ----
        allowed_token_ids: set[int] = set()

        for text in allowed_strings:
            # or tokenizer equivalent
            token_ids = model.encode(text)[0].tolist()

            # IMPORTANT: flatten and union all tokens
            for tid in token_ids:
                allowed_token_ids.add(tid)

        return allowed_token_ids

    def _transition(
        self,
        state: State,
        token_id: int,
        stack: List[Any],
        cur_state: Dict[str, Any],
        f_names_trie: Dict[str, Union[Dict[str, Any], bool]],
        parameter_trie: Optional[Dict[str, Union[Dict[str, Any], bool]]]
    ) -> Tuple[State, List[Any]]:
        """Transition to the next state based on the current token.

        Implements the state transition logic for the JSON parsing state
        machine.
        Updates the parsing context (stack, cursor position) and returns the
        next state and updated stack.

        Args:
            state: The current parsing state.
            token_id: The token ID of the next generated token.
            stack: The object/array nesting stack.
            cur_state: Current parsing context including trie cursor.
            f_names_trie: Trie of valid function names.
            parameter_trie: Trie of valid parameter names for current function.

        Returns:
            A tuple of (next_state, updated_stack).

        Raises:
            ValueError: If the token is invalid for the current state.
        """
        # token = model.decode(token_id)
        token = id_to_token.get(token_id, "")

        # root
        if state == State.START and token == "{":
            stack.append("OBJECT")
            return State.EXPECT_NAME_KEY_OPEN, stack

        # ---- "name" key ----
        if state == State.EXPECT_NAME_KEY_OPEN and token == '"':
            return State.EXPECT_NAME_KEY_BODY, stack

        if state == State.EXPECT_NAME_KEY_BODY:  # and token == '"name"':
            if "name".endswith(token):
                return State.EXPECT_NAME_KEY_CLOSE, stack
            return State.EXPECT_NAME_KEY_BODY, stack

        if state == State.EXPECT_NAME_KEY_CLOSE and token == '"':
            return State.EXPECT_COLON_AFTER_NAME_KEY, stack

        if state == State.EXPECT_COLON_AFTER_NAME_KEY and token == ":":
            return State.EXPECT_NAME_VALUE_OPEN, stack
        # -------------

        # "name" value (function name)
        if state == State.EXPECT_NAME_VALUE_OPEN and token == '"':
            cur_state["cursor"] = f_names_trie
            return State.EXPECT_NAME_VALUE_BODY, stack

        if state == State.EXPECT_NAME_VALUE_BODY:
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                return State.EXPECT_NAME_VALUE_CLOSE, stack
            return State.EXPECT_NAME_VALUE_BODY, stack

        if state == State.EXPECT_NAME_VALUE_CLOSE and token == '"':
            return State.EXPECT_COMMA_AFTER_NAME, stack

        if state == State.EXPECT_COMMA_AFTER_NAME and token == ",":
            return State.EXPECT_PARAMS_KEY_OPEN, stack
        # -------------

        # "parameters" key
        if state == State.EXPECT_PARAMS_KEY_OPEN and token == '"':
            return State.EXPECT_PARAMS_KEY_BODY, stack

        if state == State.EXPECT_PARAMS_KEY_BODY:
            if "parameters".endswith(token):
                return State.EXPECT_PARAMS_KEY_CLOSE, stack
            return State.EXPECT_PARAMS_KEY_BODY, stack

        if state == State.EXPECT_PARAMS_KEY_CLOSE and token == '"':
            return State.EXPECT_COLON_AFTER_PARAMS_KEY, stack

        if state == State.EXPECT_COLON_AFTER_PARAMS_KEY and token == ":":
            return State.EXPECT_PARAMS_OBJECT_OPEN, stack
        # -------------

        # ---- parameters object ----
        if state == State.EXPECT_PARAMS_OBJECT_OPEN and token == "{":
            stack.append("OBJECT")
            return State.EXPECT_PARAM_KEY_OPEN, stack
        # -------------

        # parameter key
        if state == State.EXPECT_PARAM_KEY_OPEN and token == '"':
            cur_state["cursor"] = parameter_trie
            return State.EXPECT_PARAM_KEY_BODY, stack

        if state == State.EXPECT_PARAM_KEY_BODY:
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = parameter_trie
                return State.EXPECT_PARAM_KEY_CLOSE, stack
            return State.EXPECT_PARAM_KEY_BODY, stack

        if state == State.EXPECT_PARAM_KEY_CLOSE and token == '"':
            return State.EXPECT_COLON_AFTER_PARAM_KEY, stack

        if state == State.EXPECT_COLON_AFTER_PARAM_KEY and token == ":":
            # if param is string move to param value open
            total_params = len(self.function_schema)
            curent_param = list(self.function_schema.keys())[
                total_params - self.remaining_prams_counter - 1]

            if self.function_schema[curent_param]["type"] == "string":
                return State.EXPECT_PARAM_VALUE_OPEN, stack

            elif self.function_schema[curent_param]["type"] in [
                    "number", "integer", "float"
            ]:
                return State.EXPECT_PARAM_NUM_VALUE_BODY, stack

            elif self.function_schema[curent_param]["type"] == "boolean":
                return State.EXPECT_PARAM_BOOL_VALUE_BODY, stack
        # -------------

        # parameter value
        # arg is str
        if state == State.EXPECT_PARAM_VALUE_OPEN and token == '"':
            return State.EXPECT_PARAM_STRING_VALUE_BODY, stack

        # arg is number
        if state == State.EXPECT_PARAM_NUM_VALUE_BODY:

            if token == ",":
                return State.EXPECT_PARAM_KEY_OPEN, stack

            elif token == "}":
                stack.pop()
                return State.EXPECT_FINAL_OBJECT_CLOSE, stack

            return State.EXPECT_PARAM_NUM_VALUE_BODY, stack

        # arg is bool
        if state == State.EXPECT_PARAM_BOOL_VALUE_BODY:

            if token == ",":
                return State.EXPECT_PARAM_KEY_OPEN, stack

            elif token == "}":
                stack.pop()
                return State.EXPECT_FINAL_OBJECT_CLOSE, stack

            return State.EXPECT_PARAM_BOOL_VALUE_BODY, stack

        if state == State.EXPECT_PARAM_STRING_VALUE_BODY:

            if token.endswith('"'):
                return State.EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE, stack

            return State.EXPECT_PARAM_STRING_VALUE_BODY, stack

        if state == State.EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE:

            if token == '"':
                return State.EXPECT_PARAM_KEY_BODY, stack

            if token == ",":
                return State.EXPECT_PARAM_KEY_OPEN, stack

            elif token == "}":
                stack.pop()
                return State.EXPECT_FINAL_OBJECT_CLOSE, stack
        # --------------

        # final
        if state == State.EXPECT_FINAL_OBJECT_CLOSE and token == "}":
            stack.pop()
            return State.DONE, stack

        raise ValueError("Invalid transition")

    def pipline(
        self,
        prompt: str,
    ) -> str:
        """Generate a constrained JSON function call from a prompt.

        Orchestrates the generation pipeline:
        1. Builds tries for function names and parameters
        2. Encodes the prompt to token IDs
        3. Iteratively generates tokens while enforcing JSON grammar
        constraints
        4. Returns the generated JSON function call

        The state machine ensures valid JSON structure throughout generation,
        preventing malformed outputs.

        Args:
            prompt: The formatted prompt containing functions and user request.

        Returns:
            A JSON string representing the function call with structure:
            {"name": "function_name", "parameters": {...}}
        """
        f_names: List[str] = list(self.functions_by_name.keys())
        f_names_trie: Dict[
            str, Union[Dict[str, Any], bool]
        ] = Utils.build_trie(f_names)

        # will build this after we know the function
        parameter_trie: Optional[Dict[str, Union[Dict[str, Any], bool]]] = None

        input_ids: List[int] = model.encode(prompt)[0].tolist()
        generated_ids: List[int] = []

        state: State = State.START
        stack: List[Any] = []
        # for trie traversal, start with keys trie
        cur_state: Dict[str, Any] = {"cursor": f_names_trie}
        current_function_name_ids: List[int] = []
        current_function_name: Optional[str] = None
        function_schema: Optional[Dict[str, Dict[str, str]]] = None
        while True:
            # ids -> logits
            logits: List[float] = model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            if state == State.EXPECT_PARAM_STRING_VALUE_BODY:
                masked_logits: List[float] = logits.copy()
                # logits -> next token id
                next_token_id: int = masked_logits.index(max(masked_logits))
                token: str = id_to_token.get(next_token_id, "")
                if '"' in token:
                    token = token[:token.index('"') + 1]
                    next_token_id = vocab.get(token, 0)
            else:
                allowed_token_ids: set[int] = self._allowed_tokens(
                    state, cur_state)
                masked_logits = [
                    log if idx in allowed_token_ids
                    else NEGATIVE_INF
                    for idx, log in enumerate(logits)
                ]
                # logits -> next token id
                next_token_id = masked_logits.index(max(masked_logits))
            # -----

            # logits -> next token id

            # separate generated ids
            generated_ids.append(next_token_id)

            # next_token_id -> ids
            input_ids.append(next_token_id)

            # save the function name
            if state == State.EXPECT_NAME_VALUE_BODY:
                current_function_name_ids.append(next_token_id)
                if cur_state["cursor"]["children"][next_token_id]["terminal"]:
                    current_function_name = model.decode(
                        current_function_name_ids).strip('"')
            # load the function schema if we just completed the function name
            if function_schema is None and current_function_name is not None:
                function_schema = Utils.load_function_schema(
                    current_function_name, self.functions_by_name)
                self.function_schema = function_schema
                self.remaining_prams_counter = len(function_schema.keys())

                # build the trie for parameters
                parameter_trie = Utils.build_trie(
                    list(function_schema.keys()))

            # update state
            state, stack = self._transition(
                state,
                next_token_id,
                stack,
                cur_state,
                f_names_trie,
                parameter_trie
            )

            states: List[State] = [
                State.EXPECT_NAME_VALUE_BODY,
                State.EXPECT_PARAM_KEY_BODY,
            ]
            children: Dict[int, Any] = cur_state["cursor"]["children"]
            if state in states and next_token_id in children:
                cur_state["cursor"] = children[next_token_id]
            if state == State.DONE:
                break

        return str(model.decode(generated_ids).strip())
