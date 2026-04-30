from pydantic import BaseModel, Field
from typing import Dict, Set, Any

from llm_sdk import Small_LLM_Model

NEGATIVE_INF = float('-inf')

# Trie node: maps token_id → child node, plus is_terminal flag
# {"children": {token_id: TrieNode}, "terminal": bool}
TrieNode = dict

model: Small_LLM_Model = Small_LLM_Model()
NUM_TOKEN_IDS = set(model.encode("0123456789.-+eE\n")[0].tolist())
FORBIDDEN_STR_TOKEN_IDS = {364, 330}


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
