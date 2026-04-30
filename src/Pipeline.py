from pydantic import BaseModel, Field
from typing import Dict
import json

from llm_sdk import Small_LLM_Model

NEGATIVE_INF = float('-inf')
model: Small_LLM_Model = Small_LLM_Model()

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

    def stage1(self, prompt: str, max_new_tokens: int = 30) -> str:
        input_ids = model.encode(prompt)[0].tolist()
        generated_ids = []

        for _ in range(max_new_tokens):
            # ids -> logits
            logits = model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            # -----

            # logits -> next token id
            next_token_id = logits.index(max(logits))

            # separate generated ids
            generated_ids.append(next_token_id)

            # next_token_id -> ids
            input_ids.append(next_token_id)

        return model.decode(generated_ids).strip()
