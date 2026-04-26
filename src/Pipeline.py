from pydantic import BaseModel
from llm_sdk import Small_LLM_Model
from src.ConstrainedDecoding import ConstrainedDecoding

NEG_INF = float('-inf')


class Pipeline(BaseModel):
    model: Small_LLM_Model = Small_LLM_Model()
    tools = ConstrainedDecoding()

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        # prompt -> tokens -> ids
        input_ids = self.model.encode(prompt)[0].tolist()

        # forbidden (temparary)
        eos_id = self.model._tokenizer.eos_token_id
        # for separating genrated from input ids
        generated_ids = []

        # start at root each time you begin a new "token"
        trie_cursor = self.tools.trie
        for _ in range(max_new_tokens):
            # ids -> logits
            logits = self.model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            allowed_ids = self.tools.get_allowed_ids_at_trie_node(trie_cursor)
            masked = [
                logit if i in allowed_ids else NEG_INF
                for i, logit in enumerate(logits)
            ]
            # -----

            # logits -> next token id
            next_token_id = masked.index(max(masked))
            # stop if eos
            if next_token_id == eos_id:
                break
            # separate generated ids
            generated_ids.append(next_token_id)
            # next_token_id -> ids
            input_ids.append(next_token_id)
            # advance the trie cursor
            trie_cursor = trie_cursor["children"][next_token_id]
            # If we completed a valid string, reset cursor to root
            if trie_cursor["termianl"]:
                trie_cursor = self.tools.trie

        return self.model.decode(generated_ids)
