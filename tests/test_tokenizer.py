from typing import Callable
from pathlib import Path

import pytest
from tiktoken import Encoding

from tokencoder.tokenizer import Tokenizer


def test_subclasses_encoding() -> None:
    assert issubclass(Tokenizer, Encoding)


@pytest.mark.parametrize(
    "name,special_tokens,", [("one", {"<|endoftext|>", "<|fim_start|>"}), ("two", {})]
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


def test_from_file_without_extension(
    tmp_path: Path, train_tokenizer: Callable[..., Path]
) -> None:
    name = "test"
    train_tokenizer(name=name, save_dir=tmp_path)
    assert isinstance(Tokenizer.from_file(tmp_path / name), Tokenizer)
