import json
import base64
import warnings
from typing import TypeVar, Iterable, Optional
from pathlib import Path

import regex  # type: ignore

from .types import Pair, Decoder, PathLike
from .patterns import DEFAULT_REGEX_PATTERN

FilePath = TypeVar("FilePath", bound=PathLike)


class TokenizerTrainer:
    """A tokenizer trainer class.
    Creates and trains a new tokenizer for encoding and decoding sequences.

    Attributes:
        special_tokens (Iterable[str], optional): Optional. The special tokens to create a tokenizer with.
        regex_pattern (Pattern[str]): A `regex` compiled pattern used for chunking
            text sequences during training.
    """

    def __init__(
        self,
        name: str,
        special_tokens: Optional[Iterable[str]] = None,
        regex_pattern_string: str = DEFAULT_REGEX_PATTERN,
    ) -> None:
        """Initialize a `TokenizerTrainer` class.

        This class allows you to train and save a `tiktoken` compatible tokenizer.

        Args:
            special_tokens (Iterable[str], optional): The special tokens that will be used to create the tokenizer.
            regex_pattern_string (str): A regex pattern string to used to parse and chunk text data before training.
                Unless you have a specific regex pattern you would like to train on, the default one is sufficient.
        """
        self.name = name
        self.regex_pattern = regex_pattern_string
        self.special_tokens = set(special_tokens) if special_tokens else None

    @staticmethod
    def count_pairs(
        tokens: list[int], counts: Optional[dict[Pair, int]] = None
    ) -> dict[Pair, int]:
        """Count the frequency of integer id pairs gen a list of integers.

        Args:
            tokens (list[int]): The integer ids.
            counts: (dict[Pair, int], Optional): initial mapping of pairs to their counts.
                Further counting performed by this function will mutate the mapping.
        """
        counts = {} if counts is None else counts
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(tokens: list[int], pair: Pair, idx: int) -> list[int]:
        """Replace every occurence of `pair` in `tokens` with `idx`.

        Args:
            tokens (list[int]): The tokens to transform.
            pair (Pair): The pair to replace.
            idx (int): The integer to replace it with
        """
        res, length, i = [], len(tokens), 0
        while i < length:
            if i + 1 < length and (tokens[i], tokens[i + 1]) == pair:
                res.append(idx)
                i += 2
            else:
                res.append(tokens[i])
                i += 1
        return res

    def __assert_file_valid(self, file: PathLike) -> None:
        if Path(file).exists():
            raise FileExistsError(
                f"The file {file}, which is where training would save the tokenizer, "
                "already exists. Training will be aborted to avoid overwriting this file. "
                f"Consider specifying a different `save_dir` argument or tokenizer name to "
                "avoid this error"
            )

    def train(
        self,
        *,
        text: str,
        vocab_size: int,
        save_dir: Optional[PathLike] = None,
    ) -> str:
        """Train a save a tokenizer to `filepath`.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired vocabulary size. Should be an integer greater than 256.
                Traning continues until this number is reached.
            filepath (PathLike, optional): A string or `Path` file to save the tokenizer to. No file extension is
                added to this value.
            save_dir (FilePath): A string or `Path` specifying the directory to save the tokenizer to.
        """
        base_size = 2**8
        if vocab_size < base_size:
            raise ValueError(
                f"The `vocab_size` parameter must be greater than {base_size}, "
                f"but instead got {vocab_size}. Please specify a valid `vocab_size` "
                "to be able to train a tokenizer."
            )

        dirname = Path(save_dir) if save_dir else Path.cwd()

        filepath = dirname / f"{self.name}.json"
        self.__assert_file_valid(filepath)

        chunks: list[list[int]] = [
            list(c.encode("utf-8")) for c in regex.findall(self.regex_pattern, text)
        ]
        decoder = {i: bytes([i]) for i in range(base_size)}

        while (nth_merge := len(decoder)) < vocab_size:
            counts: dict[Pair, int] = {}
            for c in chunks:
                self.count_pairs(c, counts)

            if not counts:  # we've already done all possible merges
                max_size = len(decoder)
                warnings.warn(
                    "NOTE: The given text for tokenizer training"
                    f"is too short for the specified vocab_size, {vocab_size}. "
                    f"The max vocab size for the text is {max_size}, "
                    f"and thus the tokenizer's vocab size will be {max_size}",
                    category=UserWarning,
                )
                break

            pair = max(counts, key=counts.get)  # type: ignore
            chunks = [self.merge(c, pair, nth_merge) for c in chunks]
            decoder[nth_merge] = decoder[pair[0]] + decoder[pair[1]]

        self.decoder = decoder
        return self.save(filepath=filepath, decoder=decoder)

    def save(self, filepath: PathLike, decoder: Decoder) -> str:
        """Serialize and save the tokenizer to `filepath`.
        This file can then be used to generate a create a new `tiktoken` tokenizer
        with the `Tokenizer.from_file` method.

        Returns:
            The file holding the tokenizer's data.
        """
        self.__assert_file_valid(filepath)
        vocab = dict(
            (base64.b64encode(v).decode("utf-8"), k) for k, v in decoder.items()
        )
        data = {
            "name": self.name,
            "regex_pattern": self.regex_pattern,
            "vocab": vocab,
        }
        if self.special_tokens is not None:
            start_idx = len(self.decoder)
            data["special_tokens"] = dict(
                zip(
                    self.special_tokens,
                    range(start_idx, start_idx + len(self.special_tokens)),
                )
            )

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return str(filepath)
