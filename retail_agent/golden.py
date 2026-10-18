from __future__ import annotations

import json
import re
from importlib.resources import files


TOKEN_RE = re.compile(r"[a-z0-9]+")


class GoldenKnowledge:
    def __init__(self) -> None:
        path = files("retail_agent.data").joinpath("golden.json")
        self.examples: list[dict[str, str]] = json.loads(path.read_text())

    def retrieve(self, question: str, limit: int = 2) -> list[dict[str, str]]:
        query_tokens = set(TOKEN_RE.findall(question.casefold()))
        scored = []
        for example in self.examples:
            text = " ".join(example.values()).casefold()
            tokens = set(TOKEN_RE.findall(text))
            score = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
            scored.append((score, example))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [example for score, example in scored[:limit] if score > 0]
