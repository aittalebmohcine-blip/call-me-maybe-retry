from enum import Enum, auto
from pydantic import BaseModel, Field
from typing import Dict
import json

from llm_sdk import Small_LLM_Model

NEGATIVE_INF = float('-inf')
model: Small_LLM_Model = Small_LLM_Model()


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

    # parameter value (string only for now)
    EXPECT_PARAM_VALUE_OPEN = auto()
    EXPECT_PARAM_VALUE_BODY = auto()
    EXPECT_PARAM_VALUE_CLOSE = auto()

    EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE = auto()

    # final
    EXPECT_FINAL_OBJECT_CLOSE = auto()

    DONE = auto()


# ------------------------
# buudling the trie for fast id look up
# ------------------------
path = model.get_path_to_vocab_file()

with open(path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# a dict for reverse lookup from token id to token text
id_to_token = {v: k for k, v in vocab.items()}

#
#
# class TrieNode:
#    def __init__(self):
#        self.children = {}
#        self.token_ids = []
#
#
# class Trie:
#    def __init__(self):
#        self.root = TrieNode()
#
#    def insert(self, token_text, token_id):
#        node = self.root
#
#        for ch in token_text:
#            if ch not in node.children:
#                node.children[ch] = TrieNode()
#
#            node = node.children[ch]
#            node.token_ids.append(token_id)
#
#    def lookup(self, prefix):
#        node = self.root
#
#        for ch in prefix:
#            if ch not in node.children:
#                return []
#
#            node = node.children[ch]
#
#        return node.token_ids
#
#
# trie = Trie()
#
# for token_text, token_id in vocab.items():
#    trie.insert(token_text, token_id)

# --------------------------
# pipline
# --------------------------


class Pipeline(BaseModel):
    functions_by_name: Dict = Field(...,
                                    description="List of function definitions")
    stack: list = Field(
        default=[], description="Stack to manage nested structures")
    remaining_prams_counter: int = Field(
        default=0, description="Counter for remaining parameters to parse")

    def allowed_tokens(self, state, cur_state):
        allowed_strings = []

        # root
        if state == State.START:
            allowed_strings = ["{"]

        # ----"name" key-----
        elif state == State.EXPECT_NAME_KEY_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_NAME_KEY_BODY:
            return cur_state["cursor"]["children"]

        elif state == State.EXPECT_NAME_KEY_CLOSE:
            allowed_strings = ['"']
        # --------

        elif state == State.EXPECT_COLON_AFTER_NAME_KEY:
            allowed_strings = [":"]

        # ----"name" value (function name)----
        elif state == State.EXPECT_NAME_VALUE_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_NAME_VALUE_BODY:
            return cur_state["cursor"]["children"]

        elif state == State.EXPECT_NAME_VALUE_CLOSE:
            allowed_strings = ['"']
        # --------

        elif state == State.EXPECT_COMMA_AFTER_NAME:
            allowed_strings = [","]

        # ----"parameters" key----
        elif state == State.EXPECT_PARAMS_KEY_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_PARAMS_KEY_BODY:
            return cur_state["cursor"]["children"]

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
            return cur_state["cursor"]["children"]

        elif state == State.EXPECT_PARAM_KEY_CLOSE:
            allowed_strings = ['"']

        elif state == State.EXPECT_COLON_AFTER_PARAM_KEY:
            allowed_strings = [":"]
        # --------

        # ----parameter value (string only for now)----
        elif state == State.EXPECT_PARAM_VALUE_OPEN:
            allowed_strings = ['"']

        # elif state == State.EXPECT_PARAM_VALUE_BODY:

        # elif state == State.EXPECT_PARAM_VALUE_CLOSE:
        #    allowed_strings = ['"']

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
        allowed_token_ids = set()

        for text in allowed_strings:
            # or tokenizer equivalent
            token_ids = model.encode(text)[0].tolist()

            # IMPORTANT: flatten and union all tokens
            for tid in token_ids:
                allowed_token_ids.add(tid)

        return allowed_token_ids

    def transition(self, state, token_id, stack, cur_state, root, parameter_trie):
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
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = root
                return State.EXPECT_NAME_KEY_CLOSE, stack
            return State.EXPECT_NAME_KEY_BODY, stack

        if state == State.EXPECT_NAME_KEY_CLOSE and token == '"':
            return State.EXPECT_COLON_AFTER_NAME_KEY, stack

        if state == State.EXPECT_COLON_AFTER_NAME_KEY and token == ":":
            return State.EXPECT_NAME_VALUE_OPEN, stack
        # -------------

        # "name" value (function name)
        if state == State.EXPECT_NAME_VALUE_OPEN and token == '"':
            return State.EXPECT_NAME_VALUE_BODY, stack

        if state == State.EXPECT_NAME_VALUE_BODY:
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = root
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

        if state == State.EXPECT_PARAMS_KEY_BODY:  # and token == '"parameters"':
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = root
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
            return State.EXPECT_PARAM_VALUE_OPEN, stack
        # -------------

        # parameter value (string only for now)
        if state == State.EXPECT_PARAM_VALUE_OPEN and token == '"':
            return State.EXPECT_PARAM_VALUE_BODY, stack

        if state == State.EXPECT_PARAM_VALUE_BODY:
            # if '","' in token:
            #    return State.EXPECT_PARAM_NAME, stack
            # if token.endswith('",'):
            #    return State.EXPECT_PARAM_KEY_OPEN, stack
            if token.endswith('"'):
                return State.EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE, stack
            # if token.endswith('}'):
            #    return State.EXPECT_NEXT_PARAM_OR_CLOSE, stack
            return State.EXPECT_PARAM_VALUE_BODY, stack

        # if state == State.EXPECT_PARAM_VALUE_CLOSE and token == '"':
        #    return State.EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE, stack

        if state == State.EXPECT_NEXT_PARAM_OR_OBJECT_CLOSE:
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

    # -------
    # helper to print trie (for debugging)
    # -------
    def _print_trie_children(self, node, tokenizer, indent):
        children = node["children"]
        keys = list(children.keys())
        for i, token_id in enumerate(keys):
            child = children[token_id]
            is_last = (i == len(keys) - 1)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if tokenizer is not None:
                token_str = tokenizer.decode([token_id])
                label = f"[{token_id}] '{token_str}'"
            else:
                label = f"[{token_id}]"

            marker = " ●" if child["terminal"] else ""
            print(f"{indent}{connector}{label}{marker}")
            self._print_trie_children(child, tokenizer, indent + extension)
    # -------

    def build_trie(self, strings):
        trie = {"children": {}, "terminal": False}

        for name in strings:
            node = trie
            for id in model.encode(name)[0].tolist():
                if id not in node["children"]:
                    node["children"][id] = {
                        "children": {}, "terminal": False}
                node = node["children"][id]
            node["terminal"] = True

        return trie

    def stage1(self, prompt: str, max_new_tokens: int = 30) -> str:

        # build the trie for function names
        valid_strings = list(self.functions_by_name.keys())
        # also allow the parameters key
        valid_strings += ['parameters', 'name']
        root_trie = self.build_trie(valid_strings)

        # will build this after we know the function
        parameter_trie = None

        input_ids = model.encode(prompt)[0].tolist()
        generated_ids = []

        state = State.START
        stack = []
        cur_state = {"cursor": root_trie}
        current_function_name_ids = []
        current_function_name = None
        function_schema = None
        for _ in range(max_new_tokens):
            # ids -> logits
            logits = model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            if state == State.EXPECT_PARAM_VALUE_BODY:
                masked_logits = logits.copy()
                # logits -> next token id
                next_token_id = masked_logits.index(max(masked_logits))
                token = id_to_token.get(next_token_id, "")
                if '"' in token:
                    token = token[:token.index('"') + 1]
                    next_token_id = vocab.get(token, "")
                    # if the next token includes a quote, mask all tokens that can lead to a quote
                    # masked_logits = []
                    # for idx, log in enumerate(logits):
                    #     token = id_to_token.get(idx, "")
                    #     if token != '}' and any(char in token for char in ["'"]):
                    #         masked_logits.append(NEGATIVE_INF)
                    #         continue
                    #     masked_logits.append(log)
            else:
                allowed_token_ids = self.allowed_tokens(
                    state, cur_state)
                # if state is State.EXPECT_NAME_KEY and model.encode('"name"')[0][0].item() in allowed_token_ids:
                #    # if "name" is allowed, prioritize it
                #    allowed_token_ids = {model.encode('"name"')[0][0].item()}
                masked_logits = [
                    log if idx in allowed_token_ids
                    else NEGATIVE_INF
                    for idx, log in enumerate(logits)
                ]
                # logits -> next token id
                next_token_id = masked_logits.index(max(masked_logits))
            # -----

            # logits -> next token id
            # next_token_id = masked_logits.index(max(masked_logits))
            # print state allowed tokens and next token for debugging
            # print(f"State: {state.name}", "next token:",
            #      model.decode([next_token_id]))
            print(f"State: {state.name}", "allowed tokens:", [model.decode(
                [tid]) for tid in allowed_token_ids], "next token:", model.decode([next_token_id]))

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
                function_schema = self.load_function_schema(
                    current_function_name)
                self.remaining_prams_counter = len(function_schema.keys())

                # build the trie for parameters
                parameter_trie = self.build_trie(
                    list(function_schema.keys()))

            # update state
            state, stack = self.transition(
                state,
                next_token_id,
                stack,
                cur_state,
                root_trie,
                parameter_trie
            )

            states = [State.EXPECT_NAME_KEY_BODY,
                      State.EXPECT_NAME_VALUE_BODY,
                      State.EXPECT_PARAMS_KEY_BODY,
                      State.EXPECT_PARAM_KEY_BODY,
                      State.EXPECT_PARAM_VALUE_BODY,
                      ]
            children = cur_state["cursor"]["children"]
            if state in states and next_token_id in children:
                cur_state["cursor"] = children[next_token_id]
            if state == State.DONE:
                break

        return model.decode(generated_ids).strip()

    def load_function_schema(self, name: str):
        return self.functions_by_name[name].parameters
