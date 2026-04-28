from pydantic import BaseModel, Field

from llm_sdk import Small_LLM_Model

NEGATIVE_INF = float('-inf')


class Pipeline(BaseModel):
    model: Small_LLM_Model = Field(default=Small_LLM_Model())

    # Trie node: maps token_id → child node, plus is_terminal flag
    TrieNode = dict  # {"children": {token_id: TrieNode}, "terminal": bool}

    def build_trie(self, string_to_ids: dict[str, list[int]]) -> TrieNode:
        root = {"children": {}, "terminal": False}
        for s, ids in string_to_ids.items():
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

    def _stage1_extract_name(self, prompt: str, functions, max_new_tokens: int = 50) -> str:
        valid_function_names = [f.name for f in functions]

        extract_fnname_prompt = f"""
        you are a function-calling engine.
        Given a user request, return the name of the best matching function.
        prompt: {prompt}
        functions: {valid_function_names}
        """.strip()

        string_to_ids = {name: self.model.encode(
            name)[0].tolist() for name in valid_function_names}
        trie = self.build_trie(string_to_ids)

        # prompt -> tokens -> ids
        input_ids = self.model.encode(extract_fnname_prompt)[0].tolist()

        # for separating genrated from input ids
        generated_ids = []

        trie_cursor = trie
        for _ in range(max_new_tokens):
            # ids -> logits
            logits = self.model.get_logits_from_input_ids(input_ids)

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

        return self.model.decode(generated_ids).strip()

    def _stage2_(self, prompt: str, functions, max_new_tokens: int = 50) -> str:
        # prompt -> tokens -> ids
        input_ids = self.model.encode(prompt)[0].tolist()

        # for separating genrated from input ids
        generated_ids = []

        for _ in range(max_new_tokens):
            # ids -> logits
            logits = self.model.get_logits_from_input_ids(input_ids)

            # -----
            # logits -> filter (inject constrained decoding)
            #
            #
            # -----

            # logits -> next token id
            next_token_id = logits.index(max(logits))
            # separate generated ids
            generated_ids.append(next_token_id)
            # next_token_id -> ids
            input_ids.append(next_token_id)

        return self.model.decode(generated_ids).strip()
