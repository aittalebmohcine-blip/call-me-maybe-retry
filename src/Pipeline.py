from pydantic import BaseModel, Field
from typing import Dict, Set, Any

from llm_sdk import Small_LLM_Model

NEGATIVE_INF = float('-inf')

# Trie node: maps token_id → child node, plus is_terminal flag
# {"children": {token_id: TrieNode}, "terminal": bool}
TrieNode = dict

model: Small_LLM_Model = Small_LLM_Model()


class Pipeline(BaseModel):
    functions_by_name: Dict = Field(...,
                                    description="List of function definitions")

    def _build_numeric_token_ids(self) -> Set[int]:
        numeric_tokens = "0123456789.-+eE "
        token_ids = set()
        for token in numeric_tokens:
            ids = model.encode(token)[0].tolist()
            token_ids.update(ids)
        return token_ids

    def build_trie(self, string_to_ids: dict[str, list[int]]) -> TrieNode:
        root = {"children": {}, "terminal": False}
        for _, ids in string_to_ids.items():
            node = root
            for token_id in ids:
                if token_id not in node["children"]:
                    node["children"][token_id] = {
                        "children": {}, "terminal": False}
                node = node["children"][token_id]
            node["terminal"] = True
        return root

    def get_allowed_ids_at_trie_node(self, node: TrieNode) -> set[int]:
        """Which token IDs are valid next steps from this trie node?"""
        return set(node["children"].keys())

    # ------------------------------------------------------------------
    # Stage 1: greedy constrained decode → function name
    # ------------------------------------------------------------------

    def _stage1_extract_name(self, prompt: str, max_new_tokens: int = 50) -> str:
        valid_function_names = [f for f in self.functions_by_name.keys()]

        extract_fnname_prompt = f"""
You are a function-calling engine. Given a user request, return the name of the best matching function.

Available functions:
{self.functions_by_name.values()}

Rules:
- Output ONLY the function name. No explanation, no markdown, no extra text.
- If no function matches, output: none

User request: {prompt}
Answer:
        """.strip()

        string_to_ids = {name: model.encode(
            name)[0].tolist() for name in valid_function_names}
        trie = self.build_trie(string_to_ids)

        # prompt -> tokens -> ids
        input_ids = model.encode(extract_fnname_prompt)[0].tolist()

        # for separating genrated from input ids
        generated_ids = []

        trie_cursor = trie
        for _ in range(max_new_tokens):
            # ids -> logits
            logits = model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            allowed_ids = self.get_allowed_ids_at_trie_node(trie_cursor)
            masked_logits = [
                logits[i] if i in allowed_ids else NEGATIVE_INF for i in range(len(logits))]
            # -----

            # logits -> next token id
            next_token_id = logits.index(max(masked_logits))

            # separate generated ids
            generated_ids.append(next_token_id)

            # next_token_id -> ids
            input_ids.append(next_token_id)

            # Advance the trie cursor
            trie_cursor = trie_cursor["children"][next_token_id]

            # If we completed a valid string, reset cursor to root
            if trie_cursor["terminal"]:
                break
                # trie_cursor = trie

        return model.decode(generated_ids).strip()

    # ------------------------------------------------------------------
    # Stage 2: per-parameter constrained decode → argument values
    # ------------------------------------------------------------------

    def _stage2_extract_args(self, prompt: str, fn_name: str) -> Dict[str, Any]:
        # extract parameters schema for the selected function
        fn_def = self.functions_by_name[fn_name]
        params = fn_def.parameters
        extracted = {}

        for param_name, param_spec in params.items():
            param_type = param_spec["type"]

            sub_prompt = (
                f"You are a parameter extractor for a function call.\n"
                f"Function: '{fn_name}'\n"
                f"Parameters to fill: {list(params.keys())}\n"
                f"User request: \"{prompt}\"\n\n"
                # f"Extract ONLY the value for the parameter '{param_name}'.\n"
                f"- Do not output the other parameters.\n"
                # f"- Do not output the parameter name, just the raw value.\n"
                f"- Do not perform any calculation.\n"
                f"Example: if the request is 'sum of 1 and 2' and param is 'b', output: 20\n"
                f"Value of '{param_name}':"
            )

            if param_type == "number":
                value_str = self._extract_string_arg(
                    sub_prompt, self._build_numeric_token_ids())
                try:
                    extracted[param_name] = int(
                        value_str) if value_str.isdigit() else float(value_str)
                except ValueError:
                    # or handle error as needed
                    extracted[param_name] = value_str
            else:
                extracted[param_name] = "..."
        return extracted

    def _extract_string_arg(self, prompt: str, allowed_token_ids: Set[int], max_new_tokens: int = 5) -> str:
        # prompt -> tokens -> ids
        input_ids = model.encode(prompt)[0].tolist()

        # for separating genrated from input ids
        generated_ids = []

        for _ in range(max_new_tokens):
            # ids -> logits
            logits = model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            masked_logits = [
                logits[i] if i in allowed_token_ids else NEGATIVE_INF for i in range(len(logits))]
            # -----

            # logits -> next token id
            next_token_id = logits.index(max(masked_logits))
            eos_id = model._tokenizer.eos_token_id
            if next_token_id == eos_id:
                break
            # separate generated ids
            generated_ids.append(next_token_id)
            # next_token_id -> ids
            input_ids.append(next_token_id)

        return model.decode(generated_ids).strip()
