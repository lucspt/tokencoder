from typing import Any, Callable, Iterable, Generator
from pathlib import Path

import pytest

from tokencoder.trainer import TokenizerTrainer


@pytest.fixture
def train_tokenizer(
    tmp_path: Path,
) -> Generator[Callable[..., Path], None, None]:
    def _train(
        *, name: str, special_tokens: Iterable[str] = {}, **train_kwargs: Any
    ) -> Path:
        trainer = TokenizerTrainer(name=name, special_tokens=special_tokens)
        args: dict[str, Any] = {
            "text": Path(__file__).read_text() * 10,
            "vocab_size": 2**8 + 1,
            "save_dir": tmp_path,
        } | train_kwargs
        return Path(trainer.train(**args))

    yield _train
