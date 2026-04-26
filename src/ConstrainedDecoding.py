from typing import Dict, List
import json


class ConstrainedDecoding():
    def __init__(self, model) -> None:
        pass
        self._model = model
        self._vocab_path: str = self._model.get_path_to_vocab_file()

        with open(self._vocab_path, "r", encoding="utf-8") as f:
            vocab: Dict[str, int] = json.load(f)
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}

        VALID_STRINGS: List[str] = [
            "{", "}", ":", ",", '"',
            "fn_add_numbers", "fn_greet", "fn_reverse_string",
            "fn_get_square_root", "fn_substitute_string_with_regex",
            "int", "float", "str", "bool",
            "true", "false", "null",
        ]

        string_to_ids: Dict[str, List[int]] = {}
        for s in VALID_STRINGS:
            ids = self._model.encode(s)[0].tolist()
            if ids is not None:
                string_to_ids[s] = ids
            else:
                print(f"Warning: '{s}' could not be tokenized cleanly")

        self.trie = self.build_trie(string_to_ids)

    # ---------------------
    #
    # Trie node: maps token_id → child node, plus is_terminal flag
    TrieNode = dict  # {"children": {token_id: TrieNode}, "terminal": bool}

    @staticmethod
    def build_trie(string_to_ids: Dict[str, List[int]]) -> TrieNode:
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

    @staticmethod
    def get_allowed_ids_at_trie_node(node: TrieNode) -> set[int]:
        """Which token IDs are valid next steps from this trie node?"""
        return set(node["children"].keys())
    #
    # ---------------------
