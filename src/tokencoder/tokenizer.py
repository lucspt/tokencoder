import json
from base64 import b64decode
from typing import Any
from pathlib import Path

from tiktoken.core import Encoding

from .types import PathLike
from .patterns import DEFAULT_REGEX_PATTERN


class Tokenizer(Encoding):
    @staticmethod
    def from_file(
        fp: PathLike,
        *,
        special_tokens: dict[str, int] = {},
        **tiktoken_kwargs: Any,
    ) -> "Tokenizer":
        fp = str(fp)
        fp = fp if fp.endswith(".json") else f"{fp}.json"
        with open(fp, "r") as f:
            data = json.load(f)
        mergeable_ranks = {b64decode(k): v for k, v in data["vocab"].items()}
        name = tiktoken_kwargs.get("name")
        return Tokenizer(
            name=name if name else Path(fp).stem,
            mergeable_ranks=mergeable_ranks,
            pat_str=DEFAULT_REGEX_PATTERN,
            special_tokens=(data.get("special_tokens", {}) | special_tokens),
            **tiktoken_kwargs,
        )
