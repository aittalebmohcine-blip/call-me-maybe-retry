from pydantic_core.core_schema import model_field
from llm_sdk import Small_LLM_Model


class Pipeline:
    def __init__(self):
        self.model = Small_LLM_Model()

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        # prompt -> tokens -> ids
        input_ids = self.model.encode(prompt)[0].tolist()

        # forbidden (temparary)
        eos_id = self.model._tokenizer.eos_token_id
        # for separating genrated from input ids
        generated_ids = []

        for _ in range(max_new_tokens):
            # ids -> logits
            logits = self.model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            # -----

            # logits -> next token id
            next_token_id = logits.index(max(logits))
            # stop if eos
            if next_token_id == eos_id:
                break
            # separate generated ids
            generated_ids.append(next_token_id)
            # next_token_id -> ids
            input_ids.append(next_token_id)

        return self.model.decode(generated_ids)
