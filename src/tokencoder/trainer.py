import json
import base64
import warnings
import itertools
from typing import TypeVar, Optional
from pathlib import Path
from collections.abc import Iterable, Sequence, Generator

import regex  # type: ignore

from .types import Pair, Decoder, PathLike
from .patterns import DEFAULT_REGEX_PATTERN

FilePath = TypeVar("FilePath", bound=PathLike)
ChunkSize = Optional[int]
ChunkSizes = Optional[int | list[Optional[int]]]


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
        special_tokens: Optional[Sequence[str]] = None,
        regex_pattern_string: str = DEFAULT_REGEX_PATTERN,
    ) -> None:
        """Initialize a `TokenizerTrainer` class.

        This class allows you to train and save a `tiktoken` compatible tokenizer.

        Args:
            special_tokens (Sequence[str], optional): The special tokens that will be used to create the tokenizer.
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

    def _warn_vocab_size_not_possible(
        self, desired_vocab_size: int, max_vocab_size: int
    ) -> None:
        warnings.warn(
            "NOTE: The given text for tokenizer training "
            f"is too short for the specified vocab_size, {desired_vocab_size}. "
            f"The max vocab size for the text is {max_vocab_size}, "
            f"and thus the tokenizer's vocab size will be {max_vocab_size}",
            category=UserWarning,
        )

    def build_decoder(self, vocab_size: int) -> Decoder:
        base_size = 2**8
        if vocab_size < base_size:
            raise ValueError(
                f"The `vocab_size` parameter must be greater than {base_size}, "
                f"but instead got {vocab_size}. Please specify a valid `vocab_size` "
                "to be able to train a tokenizer."
            )
        decoder = {i: bytes([i]) for i in range(base_size)}
        return decoder

    def get_merged_chunks_generator(
        self, chunks: Iterable[list[int]], pair: Pair, idx: int
    ) -> Generator[list[int], None, None]:
        for c in chunks:
            yield self.merge(c, pair, idx)

    def _train_on_chunks(
        self,
        *,
        chunks: Iterable[list[int]],
        decoder: Decoder,
        vocab_size: int,
    ) -> Decoder:
        """Perform byte pair merging on `text` and return the resulting vocab."""
        while (nth_merge := len(decoder)) < vocab_size:
            counts: dict[Pair, int] = {}
            chunks_to_count, chunks_to_merge = itertools.tee(chunks, 2)
            for c in chunks_to_count:
                self.count_pairs(c, counts)

            if not counts:
                break

            pair = max(counts, key=counts.get)  # type: ignore
            chunks = self.get_merged_chunks_generator(chunks_to_merge, pair, nth_merge)
            decoder[nth_merge] = decoder[pair[0]] + decoder[pair[1]]

        return decoder

    @staticmethod
    def read_file_chunks(
        fp: PathLike, chunksize: ChunkSize = -1
    ) -> Generator[list[int], None, None]:
        """Read a file in `chunksize` chunks."""
        with open(fp, "r") as f:
            while chunk := f.read(chunksize):
                rchunks = regex.findall(DEFAULT_REGEX_PATTERN, chunk)
                for c in rchunks:
                    yield list(c.encode("utf-8"))

    def files_to_chunks(
        self, files: Iterable[PathLike], chunksize: ChunkSizes = -1
    ) -> itertools.chain[list[int]]:
        """Given an iterable of files, extract all of their chunks"""
        files_set = set(files)
        if not isinstance(chunksize, list):
            chunksize = [chunksize] * len(files_set)
        return itertools.chain(
            *tuple(
                self.read_file_chunks(f, chunksize[i]) for i, f in enumerate(files_set)
            )
        )

    def _train_on_files(
        self,
        *,
        files: Iterable[PathLike],
        vocab_size: int,
        decoder: Decoder,
        chunksize: ChunkSizes = -1,
    ) -> Decoder:
        """Train a tokenizer on the given `files`"""
        file_chunks = self.files_to_chunks(files, chunksize=chunksize)
        decoder = self._train_on_chunks(
            chunks=file_chunks,
            decoder=decoder,
            vocab_size=vocab_size,
        )
        return decoder

    def train(
        self,
        *,
        vocab_size: int,
        files: Optional[Iterable[PathLike]] = None,
        text: Optional[str] = None,
        save_dir: Optional[PathLike] = None,
        file_read_chunksize: ChunkSizes = -1,
    ) -> str:
        """Train a save a tokenizer to `filepath`.

        One of `files` or `text` must be given.

        **Tip**:
            Use the `files` argument if your text corpus is too large to fit into memory. The contents of
            each file will be loaded in chunks and used for training. You can modify the chunksize with
            the `file_read_chunksize` parameter.

        Args:
            vocab_size (int): The desired vocabulary size. Should be an integer greater than 256.
                Traning continues until this number is reached.
            text (str, optional): The text to train on. Must be specified if `files` is not passed, otherwise optional.
            files (Iterable[PathLike], optional): The files with text content to train in.
                Must be specified if `text` is not passed, otherwise optional.
            save_dir (PathLike, optional): A `str` or `Path` directory to save the tokenizer data to.
            file_read_chunksize (int): The size of each chunk that the files will be read with
                when loaded. Defaults to `-1` (the fulll file).
        """
        if not files and not text:
            raise ValueError("You must specify either `files` or `text` to train on.")

        dirname = Path(save_dir) if save_dir else Path.cwd()
        filepath = dirname / f"{self.name}.json"
        self.__assert_file_valid(filepath)
        decoder = self.build_decoder(vocab_size=vocab_size)
        if files:
            decoder = self._train_on_files(
                files=files,
                decoder=decoder,
                vocab_size=vocab_size,
                chunksize=file_read_chunksize,
            )
        elif text:
            decoder = self._train_on_chunks(
                chunks=[
                    list(c.encode("utf-8"))
                    for c in regex.findall(self.regex_pattern, text)
                ],
                decoder=decoder,
                vocab_size=vocab_size,
            )
        if len(decoder) < vocab_size:
            self._warn_vocab_size_not_possible(
                desired_vocab_size=vocab_size, max_vocab_size=len(decoder)
            )  # we've already done all possible merges

        vocab = dict(
            (base64.b64encode(v).decode("utf-8"), k) for k, v in decoder.items()
        )
        if len(vocab) != len(decoder):
            self._warn_vocab_size_not_possible(
                desired_vocab_size=vocab_size, max_vocab_size=len(vocab)
            )
            vocab = dict(zip(vocab.keys(), range(len(vocab))))
        return self.save(filepath=filepath, vocab=vocab)

    def save(self, filepath: PathLike, vocab: dict[str, int]) -> str:
        """Serialize and save the tokenizer to `filepath`.
        This file can then be used to generate a create a new `tiktoken` tokenizer
        with the `Tokenizer.from_file` method.

        Returns:
            The file holding the tokenizer's data.
        """
        self.__assert_file_valid(filepath)
        data = {
            "name": self.name,
            "regex_pattern": self.regex_pattern,
            "vocab": vocab,
        }
        if self.special_tokens is not None:
            start_idx = len(vocab)
            data["special_tokens"] = dict(
                zip(
                    self.special_tokens,
                    range(start_idx, start_idx + len(self.special_tokens)),
                )
            )

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return str(filepath)
