import json
from base64 import b64decode
from typing import Any
from pathlib import Path

from tiktoken.core import Encoding

from .types import PathLike


class Tokenizer(Encoding):
    @staticmethod
    def from_file(
        fp: PathLike,
        *,
        special_tokens: dict[str, int] = {},
        **tiktoken_kwargs: Any,
    ) -> "Tokenizer":
        """Load a tokenizer from `fp`.

        Args:
            fp (PathLike): A `Path` or string pointing to the tokenizer file.
                If the path does not end with ".json" it will be added automatically.
            special_tokens (dict[str, int]): A mapping of special tokens to their
                token id.
            tiktoken_kwargs: Keyword arguments to use when constructing the tiktoken
                `Encoding` class.

        Raises:
            `FileNotFoundError`: When `fp` does not exist.

        Returns:
            `Tokenizer`: the Tokenizer object.
        """
        fp = str(fp)
        fp = fp if fp.endswith(".json") else f"{fp}.json"
        with open(fp, "r") as f:
            data = json.load(f)
        mergeable_ranks = {b64decode(k): v for k, v in data["vocab"].items()}
        name = tiktoken_kwargs.get("name")
        return Tokenizer(
            name=name if name else Path(fp).stem,
            mergeable_ranks=mergeable_ranks,
            pat_str=data["regex_pattern"],
            special_tokens=(data.get("special_tokens", {}) | special_tokens),
            **tiktoken_kwargs,
        )
