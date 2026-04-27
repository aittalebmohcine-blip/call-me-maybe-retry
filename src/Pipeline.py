from typing import List
from llm_sdk import Small_LLM_Model
from src.ConstrainedDecoding import ConstrainedDecoding

NEG_INF = float('-inf')


class Pipeline():
    def __init__(self) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.tools: ConstrainedDecoding = ConstrainedDecoding(self.model)

    @staticmethod
    def pick_next_token(logits: List[float], allowed_ids: set[int]) -> int | None:
        masked = [
            logit if i in allowed_ids else NEG_INF
            for i, logit in enumerate(logits)
        ]

        best_logit = max(masked)

        if best_logit == NEG_INF:
            return None

        return masked.index(best_logit)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        # prompt -> tokens -> ids
        input_ids = self.model.encode(prompt)[0].tolist()

        # forbidden (temparary)
        eos_id = self.model._tokenizer.eos_token_id
        # for separating genrated from input ids
        generated_ids = []

        # start at root each time you begin a new "token"
        state = {
            "stack": [],          # tracks nesting: "object", "array"
            "phase": "START",     # where inside the current level you are
            "last": None          # last completed string
        }
        trie_cursor = self.tools.trie
        for _ in range(max_new_tokens):
            # ids -> logits
            logits = self.model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            allowed_strings = self.tools.get_allowed_strings(
                state)  # state machine
            allowed_ids = self.tools.get_allowed_ids(
                state, trie_cursor)
            # masked = [
            #    logit if i in allowed_ids else NEG_INF
            #    for i, logit in enumerate(logits)
            # ]
            # -----

            # logits -> next token id
            next_token_id = self.pick_next_token(
                logits, allowed_ids)  # masked.index(max(masked))
            if next_token_id is None:
                # handle explicitly, don't silently continue
                raise ValueError(
                    f"No valid token available at phase='{state['phase']}', "
                    f"allowed_strings={allowed_strings}"
                )
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
            if trie_cursor["terminal"]:
                # what string did we just finish
                completed_string = self.model.decode(next_token_id)
                # advance state machine
                state = self.tools.transition(state, completed_string)
                trie_cursor = self.tools.trie

        return self.model.decode(generated_ids)
