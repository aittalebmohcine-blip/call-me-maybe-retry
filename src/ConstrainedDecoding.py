from typing import Dict, List, Set
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
            "true", "false", "null", "name",
        ]

        self.string_to_ids: Dict[str, List[int]] = {}
        for s in VALID_STRINGS:
            ids = self._model.encode(s)[0].tolist()
            if ids is not None:
                self.string_to_ids[s] = ids
            else:
                print(f"Warning: '{s}' could not be tokenized cleanly")

        self.trie = self.build_trie(self.string_to_ids)

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
    #
    # convert allowed strings → allowed token IDs via your trie

    def get_allowed_ids(self, state: dict, trie_cursor: dict) -> set[int]:

        # Case 1: mid-string — trie cursor is ahead of root
        # the state machine already approved this string when it started
        # just follow the trie
        if trie_cursor is not self.trie:
            return self.get_allowed_ids_at_trie_node(trie_cursor)

        # Case 2: at a string boundary — ask the state machine
        # then map each allowed string to its FIRST token only
        allowed_strings = self.get_allowed_strings(state)

        allowed_ids = set()
        for s in allowed_strings:
            if s in self.string_to_ids:
                first_id = self.string_to_ids[s][0]
                allowed_ids.add(first_id)

        # intersect with trie root's children just to be safe
        return allowed_ids & self.get_allowed_ids_at_trie_node(trie_cursor)

    @staticmethod
    def get_allowed_strings(state: dict) -> Set[str]:
        phase = state["phase"]

        if phase == "START":
            return {"{"}

        if phase == "EXPECT_KEY":
            return {'"'}

        if phase == "INSIDE_KEY":
            # your fn names
            return {
                "name",
            }

        if phase == "EXPECT_CLOSING_QUOTE":
            return {'"'}

        if phase == "EXPECT_COLON":
            return {":"}

        if phase == "EXPECT_VALUE":
            return {
                "fn_add_numbers",
                "fn_greet",
                "fn_reverse_string",
                "fn_get_square_root",
                "fn_substitute_string_with_regex"
            }

        if phase == "AFTER_VALUE":
            return {",", "}"}

    @staticmethod
    def transition(state: Dict, completed: str) -> Dict:
        phase = state["phase"]

        if phase == "START" and completed == "{":
            state["stack"].append("object")
            state["phase"] = "EXPECT_KEY"

        elif phase == "EXPECT_KEY" and completed == '"':
            state["phase"] = "INSIDE_KEY"

        elif phase == "INSIDE_KEY":
            state["last_key"] = completed
            state["phase"] = "EXPECT_CLOSING_QUOTE"

        elif phase == "EXPECT_CLOSING_QUOTE" and completed == '"':
            state["phase"] = "EXPECT_COLON"

        elif phase == "EXPECT_COLON" and completed == ":":
            state["phase"] = "EXPECT_VALUE"

        elif phase == "EXPECT_VALUE":
            state["phase"] = "AFTER_VALUE"

        elif phase == "AFTER_VALUE":
            if completed == ",":
                state["phase"] = "EXPECT_KEY"
            elif completed == "}":
                state["stack"].pop()
                state["phase"] = "DONE"

        return state
