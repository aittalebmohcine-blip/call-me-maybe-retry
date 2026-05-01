from enum import Enum, auto
from pydantic import BaseModel, Field
from typing import Dict
import json

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
    EXPECT_PARAMETERS_CLOSE = auto()

    EXPECT_FINAL_CLOSE = auto()

    DONE = auto()


# ------------------------
# buudling the trie for fast id look up
# ------------------------
path = model.get_path_to_vocab_file()

with open(path, "r", encoding="utf-8") as f:
    vocab = json.load(f)


class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_ids = []


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_text, token_id):
        node = self.root

        for ch in token_text:
            if ch not in node.children:
                node.children[ch] = TrieNode()

            node = node.children[ch]
            node.token_ids.append(token_id)

    def lookup(self, prefix):
        node = self.root

        for ch in prefix:
            if ch not in node.children:
                return []

            node = node.children[ch]

        return node.token_ids


trie = Trie()

for token_text, token_id in vocab.items():
    trie.insert(token_text, token_id)

# --------------------------
# pipline
# --------------------------


class Pipeline(BaseModel):
    functions_by_name: Dict = Field(...,
                                    description="List of function definitions")

    stack: list = Field(
        default=[], description="Stack to manage nested structures")

    def allowed_tokens(self, state, stack):
        allowed_strings = []

        if state == State.START:
            allowed_strings = ["{"]

        elif state == State.EXPECT_NAME_KEY:
            allowed_strings = ['"name"']

        elif state == State.EXPECT_COLON_1:
            allowed_strings = [":"]

        elif state == State.EXPECT_NAME_VALUE:
            # function names as strings
            allowed_strings = list(self.functions_by_name.keys())

        elif state == State.EXPECT_COMMA:
            allowed_strings = [","]

        elif state == State.EXPECT_PARAMETERS_KEY:
            allowed_strings = ['"parameters"']

        elif state == State.EXPECT_COLON_2:
            allowed_strings = [":"]

        elif state == State.EXPECT_PARAMETERS_OPEN:
            allowed_strings = ["{"]

        elif state == State.EXPECT_PARAMETERS_CLOSE:
            allowed_strings = ["}"]

        elif state == State.EXPECT_FINAL_CLOSE:
            allowed_strings = ["}"]

        else:
            allowed_strings = []

        # ---- APPLY THE RULE HERE ----
        allowed_token_ids = set()

        for text in allowed_strings:
            # or tokenizer equivalent
            token_ids = model.encode(text).tolist()

            # IMPORTANT: flatten and union all tokens
            for tid in token_ids:
                allowed_token_ids.add(tid)

        return allowed_token_ids

    def transition(self, state, token_id, stack):
        token = model.decode(token_id)

        if state == State.START and token == "{":
            stack.append("OBJECT")
            return State.EXPECT_NAME_KEY, stack

        if state == State.EXPECT_NAME_KEY:  # and token == '"name"':
            return State.EXPECT_COLON_1, stack

        if state == State.EXPECT_COLON_1 and token == ":":
            return State.EXPECT_NAME_VALUE, stack

        if state == State.EXPECT_NAME_VALUE:
            return State.EXPECT_COMMA, stack

        if state == State.EXPECT_COMMA and token == ",":
            return State.EXPECT_PARAMETERS_KEY, stack

        if state == State.EXPECT_PARAMETERS_KEY:  # and token == '"parameters"':
            return State.EXPECT_COLON_2, stack

        if state == State.EXPECT_COLON_2 and token == ":":
            return State.EXPECT_PARAMETERS_OPEN, stack

        if state == State.EXPECT_PARAMETERS_OPEN and token == "{":
            stack.append("OBJECT")
            return State.EXPECT_PARAMETERS_CLOSE, stack

        if state == State.EXPECT_PARAMETERS_CLOSE and token == "}":
            stack.pop()
            return State.EXPECT_FINAL_CLOSE, stack

        if state == State.EXPECT_FINAL_CLOSE and token == "}":
            stack.pop()
            return State.DONE, stack

        raise ValueError("Invalid transition")

    def stage1(self, prompt: str, max_new_tokens: int = 30) -> str:

        # build the trie for function names
        valid_strings = list(self.functions_by_name.keys())
        # also allow the parameters key
        valid_strings += ['"parameters"', '"name']
        trie = {"children": {}, "terminal": False}
        node = trie

        for name in valid_strings:
            for id in model.encode(name).tolist():
                if id not in node["children"]:
                    node["children"][id] = {
                        "children": {}, "terminal": False}
                node = node["children"][id]
                node["terminal"] = True
            node = trie

        input_ids = model.encode(prompt)[0].tolist()
        generated_ids = []

        state = State.START
        stack = []
        for _ in range(max_new_tokens):
            # ids -> logits
            logits = model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            allowed_token_ids = self.allowed_tokens(state, stack)
            masked_logits = [
                log if idx in allowed_token_ids else NEGATIVE_INF for idx, log in enumerate(logits)]
            # -----

            # logits -> next token id
            next_token_id = logits.index(max(masked_logits))

            # separate generated ids
            generated_ids.append(next_token_id)

            # next_token_id -> ids
            input_ids.append(next_token_id)

            # update state
            state, stack = self.transition(state, next_token_id, stack)

        return model.decode(generated_ids).strip()
