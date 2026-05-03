from enum import Enum, auto
from pydantic import BaseModel, Field
from typing import Dict

from llm_sdk import Small_LLM_Model

NEGATIVE_INF = float('-inf')
model: Small_LLM_Model = Small_LLM_Model()


class State(Enum):
    START = auto()

    EXPECT_NAME_KEY = auto()
    EXPECT_COLON_1 = auto()
    EXPECT_NAME_VALUE = auto()

    EXPECT_COMMA = auto()

    EXPECT_PARAMETERS_KEY = auto()
    EXPECT_COLON_2 = auto()

    EXPECT_PARAMETERS_OPEN = auto()
    EXPECT_PARAM_KEY_OPEN = auto()
    EXPECT_PARAM_NAME = auto()
    EXPECT_PARAM_KEY_CLOSE = auto()
    EXPECT_PARAM_COLON = auto()
    EXPECT_PARAM_VALUE = auto()
    EXPECT_NEXT_PARAM_OR_CLOSE = auto()

    EXPECT_FINAL_CLOSE = auto()

    DONE = auto()


# ------------------------
# buudling the trie for fast id look up
# ------------------------
# path = model.get_path_to_vocab_file()
#
# with open(path, "r", encoding="utf-8") as f:
#    vocab = json.load(f)
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

    def allowed_tokens(self, state, stack, cur_state, function_schema):
        allowed_strings = []

        if state == State.START:
            allowed_strings = ["{"]

        elif state == State.EXPECT_NAME_KEY:
            # all tokens that can continue from here
            return cur_state["cursor"]["children"]
            # allowed_strings = ['"name"']

        elif state == State.EXPECT_COLON_1:
            allowed_strings = [":"]

        elif state == State.EXPECT_NAME_VALUE:
            # function names as strings
            return cur_state["cursor"]["children"]
            # allowed_strings = list(self.functions_by_name.keys())

        elif state == State.EXPECT_COMMA:
            allowed_strings = [","]

        elif state == State.EXPECT_PARAMETERS_KEY:
            return cur_state["cursor"]["children"]
            # allowed_strings = ['"parameters"']

        elif state == State.EXPECT_COLON_2:
            allowed_strings = [":"]

        elif state == State.EXPECT_PARAMETERS_OPEN:
            allowed_strings = ["{"]

        elif state == State.EXPECT_PARAM_KEY_OPEN:
            allowed_strings = ['"']

        elif state == State.EXPECT_NEXT_PARAM_OR_CLOSE:
            allowed_strings = ["}"]

        elif state == State.EXPECT_FINAL_CLOSE:
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

    def transition(self, state, token_id, stack, cur_state, root):
        token = model.decode(token_id)

        if state == State.START and token == "{":
            stack.append("OBJECT")
            return State.EXPECT_NAME_KEY, stack

        if state == State.EXPECT_NAME_KEY:  # and token == '"name"':
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = root
                return State.EXPECT_COLON_1, stack
            return State.EXPECT_NAME_KEY, stack

        if state == State.EXPECT_COLON_1 and token == ":":
            return State.EXPECT_NAME_VALUE, stack

        if state == State.EXPECT_NAME_VALUE:
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = root
                return State.EXPECT_COMMA, stack
            return State.EXPECT_NAME_VALUE, stack

        if state == State.EXPECT_COMMA and token == ",":
            return State.EXPECT_PARAMETERS_KEY, stack

        if state == State.EXPECT_PARAMETERS_KEY:  # and token == '"parameters"':
            if cur_state["cursor"]["children"][token_id]["terminal"]:
                cur_state["cursor"] = root
                return State.EXPECT_COLON_2, stack
            return State.EXPECT_PARAMETERS_KEY, stack

        if state == State.EXPECT_COLON_2 and token == ":":
            return State.EXPECT_PARAMETERS_OPEN, stack

        if state == State.EXPECT_PARAMETERS_OPEN and token == "{":
            stack.append("OBJECT")
            return State.EXPECT_PARAM_KEY_OPEN, stack
            # return State.EXPECT_PARAMETERS_CLOSE, stack

        if state == State.EXPECT_PARAM_KEY_OPEN and token == '"':
            return State.EXPECT_NEXT_PARAM_OR_CLOSE, stack

        if state == State.EXPECT_NEXT_PARAM_OR_CLOSE and token == "}":
            stack.pop()
            return State.EXPECT_FINAL_CLOSE, stack

        if state == State.EXPECT_FINAL_CLOSE and token == "}":
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
        valid_strings = [f'"{name}"' for name in self.functions_by_name.keys()]
        # also allow the parameters key
        valid_strings += ['"parameters"', '"name"']
        root_trie = self.build_trie(valid_strings)

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
            allowed_token_ids = self.allowed_tokens(
                state, stack, cur_state, function_schema)
            if state is State.EXPECT_NAME_KEY and model.encode('"name"')[0][0].item() in allowed_token_ids:
                # if "name" is allowed, prioritize it
                allowed_token_ids = {model.encode('"name"')[0][0].item()}
            masked_logits = [
                log if idx in allowed_token_ids
                else NEGATIVE_INF
                for idx, log in enumerate(logits)
            ]
            # -----

            # logits -> next token id
            next_token_id = masked_logits.index(max(masked_logits))

            # separate generated ids
            generated_ids.append(next_token_id)

            # next_token_id -> ids
            input_ids.append(next_token_id)

            # save the function name
            if state == State.EXPECT_NAME_VALUE:
                current_function_name_ids.append(next_token_id)
                if cur_state["cursor"]["children"][next_token_id]["terminal"]:
                    current_function_name = model.decode(
                        current_function_name_ids).strip('"')
                    print("Selected function:", current_function_name)
            # load the function schema if we just completed the function name
            if function_schema is None and current_function_name is not None:
                function_schema = self.load_function_schema(
                    current_function_name)
                print("Function schema:", function_schema)

            # update state
            state, stack = self.transition(
                state, next_token_id, stack, cur_state, root_trie)

            states = [State.EXPECT_NAME_KEY,
                      State.EXPECT_NAME_VALUE, State.EXPECT_PARAMETERS_KEY]
            children = cur_state["cursor"]["children"]
            if state in states and next_token_id in children:
                cur_state["cursor"] = children[next_token_id]
            if state == State.DONE:
                break

        return model.decode(generated_ids).strip()

    def load_function_schema(self, name: str):
        return self.functions_by_name[name].parameters
