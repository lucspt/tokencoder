import json
import string
from typing import Callable
from pathlib import Path

import pytest
from tiktoken import Encoding

from tokencoder.tokenizer import Tokenizer


def test_subclasses_encoding() -> None:
    assert issubclass(Tokenizer, Encoding)


@pytest.fixture(scope="module")
def text() -> str:
    return "\n".join(
        [
            string.ascii_letters,
            string.digits,
            string.octdigits,
            string.whitespace,
            string.punctuation,
        ]
    )


@pytest.mark.parametrize(
    "name,special_tokens,",
    [("one", {"<|endoftext|>", "<|fim_start|>"}), ("two", set())],
)
def test_from_file(
    name: str,
    special_tokens: set[str],
    train_tokenizer: Callable[..., Path],
) -> None:
    pth = train_tokenizer(
        special_tokens=special_tokens,
        name=name,
        text=Path(__file__).read_text() * 10,
        vocab_size=270,
    )
    tokenizer = Tokenizer.from_file(pth)
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.name == name
    assert tokenizer.special_tokens_set == special_tokens


def test_from_file_without_extension(
    tmp_path: Path, train_tokenizer: Callable[..., Path]
) -> None:
    name = "test"
    train_tokenizer(name=name, save_dir=tmp_path)
    assert isinstance(Tokenizer.from_file(tmp_path / name), Tokenizer)


def test_encode(train_tokenizer: Callable[..., Path], text: str) -> None:
    pth = train_tokenizer(name="encode")
    tokenizer = Tokenizer.from_file(pth)
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    for x in tokens:
        assert isinstance(x, int)


def test_decode(train_tokenizer: Callable[..., Path], text: str) -> None:
    pth = train_tokenizer(name="encode")
    tokenizer = Tokenizer.from_file(pth)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_eot_token(train_tokenizer: Callable[..., Path]) -> None:
    pth = train_tokenizer(name="eot_id", special_tokens={"<|endoftext|>"})
    with open(pth, "r") as f:
        data = json.load(f)
    tokenizer = Tokenizer.from_file(pth)
    assert tokenizer.eot_token == data["special_tokens"].get("<|endoftext|>")
